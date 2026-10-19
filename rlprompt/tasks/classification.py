"""Few-shot text classification with a frozen masked LM.

The task LM fills a <mask> slot; class scores are its probabilities for the
verbalizer tokens, renormalised over the classes.

Reward for one prompt z on example (x, c):
    Gap_z(c) = P_z(c) - max_{c' != c} P_z(c')
    R        = lambda_1 ** (1 - Correct) * lambda_2 ** Correct * Gap_z(c)
             = Gap * (lambda_2 if Gap > 0 else lambda_1)
averaged over the examples in the step.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from ..config import ClassificationConfig
from ..utils import chunks, truncate_texts
from .base import Task

Example = Tuple[str, int]


@dataclass
class DatasetSpec:
    hf_path: str
    hf_config: Optional[str]
    text_field: str
    label_field: str
    test_split: str
    verbalizers: List[str]
    template: str
    manual_prompt: str


DATASETS: Dict[str, DatasetSpec] = {
    "sst2": DatasetSpec("nyu-mll/glue", "sst2", "sentence", "label", "validation",
                        [" terrible", " great"], "{sentence} {prompt} {mask} .", "It was"),
    "mr": DatasetSpec("cornell-movie-review-data/rotten_tomatoes", None, "text", "label", "test",
                      [" terrible", " great"], "{sentence} {prompt} {mask} .", "It was"),
    "yelp_polarity": DatasetSpec("fancyzhx/yelp_polarity", None, "text", "label", "test",
                                 [" terrible", " great"], "{sentence} {prompt} {mask} .", "It was"),
    "sst5": DatasetSpec("SetFit/sst5", None, "text", "label", "test",
                        [" terrible", " bad", " okay", " good", " great"],
                        "{sentence} {prompt} {mask} .", "It was"),
    "ag_news": DatasetSpec("fancyzhx/ag_news", None, "text", "label", "test",
                           [" World", " Sports", " Business", " Tech"],
                           "{mask} {prompt} {sentence}", "News:"),
}


# ---------------------------------------------------------------- data

def load_fewshot(name: str, shots: int, split_seed: int, max_test: int = 0
                 ) -> Tuple[List[Example], List[Example], List[Example]]:
    """Sample `shots` train and `shots` dev examples per class from the
    training split, following the true few-shot protocol."""
    from datasets import load_dataset

    spec = DATASETS[name]
    ds = load_dataset(spec.hf_path, spec.hf_config)
    train = ds["train"]
    labels = train[spec.label_field]
    texts = train[spec.text_field]

    rng = random.Random(split_seed)
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    fs_train, fs_dev = [], []
    for y in sorted(by_class):
        idx = rng.sample(by_class[y], 2 * shots)
        fs_train += [(texts[i], y) for i in idx[:shots]]
        fs_dev += [(texts[i], y) for i in idx[shots:]]

    test_ds = ds[spec.test_split]
    test = list(zip(test_ds[spec.text_field], test_ds[spec.label_field]))
    if max_test and len(test) > max_test:
        test = random.Random(1234).sample(test, max_test)
    return fs_train, fs_dev, test


# ---------------------------------------------------------------- task LM

class PromptedClassifier:
    """Frozen LM used as a black box: forward passes only.

    Masked LMs such as RoBERTa score the verbalizers at the <mask> slot.
    Left-to-right LMs such as GPT-2 have no mask token, so the text is cut at
    the slot and the verbalizers are scored as the next token. This is what
    makes prompts transferable between the two architectures; it requires
    the slot to come after the sentence, as in the SST-2 template.
    """

    def __init__(self, model_name: str, verbalizers: Sequence[str], template: str,
                 device: str = "cuda", fp16: bool = True, batch_size: int = 128,
                 max_sentence_tokens: int = 400):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.causal = self.tokenizer.mask_token is None
        if self.causal:
            if not template.rstrip(" .").endswith("{mask}"):
                raise ValueError("A left-to-right task LM needs the {mask} slot at the end of the template")
            template = template[:template.index("{mask}")].rstrip()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        if fp16 and self.device.type == "cuda":
            self.model.half()
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.template = template
        self.batch_size = batch_size
        self.max_sentence_tokens = max_sentence_tokens
        self.verbalizer_ids = []
        for v in verbalizers:
            ids = self.tokenizer.encode(v, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Verbalizer {v!r} is not a single token for {model_name}: {ids}")
            self.verbalizer_ids.append(ids[0])
        self.verbalizer_ids = torch.tensor(self.verbalizer_ids, device=self.device)
        self._trunc_cache: Dict[str, str] = {}

    def _fmt(self, sentence: str, prompt: str) -> str:
        if sentence not in self._trunc_cache:
            self._trunc_cache[sentence] = truncate_texts(
                self.tokenizer, [sentence], self.max_sentence_tokens)[0]
        text = self.template.format(sentence=self._trunc_cache[sentence], prompt=prompt,
                                    mask=self.tokenizer.mask_token or "")
        return re.sub(r" +", " ", text).strip()

    @torch.no_grad()
    def class_probs(self, sentences: Sequence[str], prompts: Sequence[str]) -> torch.Tensor:
        """Probabilities over classes for aligned (sentence, prompt) pairs -> (N, C)."""
        texts = [self._fmt(s, p) for s, p in zip(sentences, prompts)]
        out = []
        for batch in chunks(texts, self.batch_size):
            enc = self.tokenizer(batch, return_tensors="pt", padding=True,
                                 truncation=True, max_length=512).to(self.device)
            rows = torch.arange(len(batch), device=self.device)
            if self.causal:
                pos_ids = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
                logits = self.model(**enc, position_ids=pos_ids).logits
                pos = torch.full_like(rows, enc.input_ids.size(1) - 1)
            else:
                logits = self.model(**enc).logits
                pos = (enc.input_ids == self.tokenizer.mask_token_id).float().argmax(dim=1)
            v_logits = logits[rows, pos][:, self.verbalizer_ids].float()
            out.append(torch.softmax(v_logits, dim=-1))
        return torch.cat(out)

    def accuracy(self, prompt: str, examples: Sequence[Example]) -> float:
        sents = [s for s, _ in examples]
        labels = torch.tensor([y for _, y in examples], device=self.device)
        probs = self.class_probs(sents, [prompt] * len(sents))
        return (probs.argmax(-1) == labels).float().mean().item()


# ---------------------------------------------------------------- task

class ClassificationTask(Task):
    def __init__(self, classifier: PromptedClassifier, train: List[Example],
                 dev: List[Example], cfg: ClassificationConfig):
        self.clf = classifier
        self.train = train
        self.dev = dev
        self.cfg = cfg

    def sample_batch(self, rng: random.Random):
        n = self.cfg.prompts_per_step
        k = self.cfg.examples_per_step
        examples = self.train if not k or k >= len(self.train) else rng.sample(self.train, k)
        # Prompts are not conditioned on the input, so the policy sees an empty
        # source and every prompt in the step shares z-score group 0.
        return [""] * n, torch.zeros(n, dtype=torch.long), examples

    def reward(self, prompts: List[str], examples: List[Example]):
        N, M = len(prompts), len(examples)
        sents = [s for s, _ in examples] * N
        rep_prompts = [p for p in prompts for _ in range(M)]
        probs = self.clf.class_probs(sents, rep_prompts).view(N, M, -1)

        labels = torch.tensor([y for _, y in examples], device=probs.device)
        labels = labels[None, :, None].expand(N, M, 1)
        gold = probs.gather(-1, labels).squeeze(-1)
        others = probs.scatter(-1, labels, -1.0).max(-1).values
        gap = gold - others
        correct = (gap > 0).float()
        if self.cfg.piecewise:
            coeff = correct * self.cfg.correct_coeff + (1 - correct) * self.cfg.incorrect_coeff
        else:
            coeff = torch.full_like(gap, 100.0)
        rewards = (gap * coeff).mean(1)
        acc = correct.mean(1)
        self.record_candidates(prompts, acc.tolist())
        best = int(acc.argmax())
        logs = {"reward_raw": rewards.mean().item(), "train_acc_mean": acc.mean().item(),
                "train_acc_best": acc[best].item(), "sample_prompt": prompts[best]}
        return rewards.cpu(), logs

    def evaluate(self, policy) -> Dict:
        _, prompts = policy.sample([""], greedy=True)
        prompt = prompts[0]
        return {"score": self.clf.accuracy(prompt, self.dev), "prompt": prompt}
