"""Unsupervised sentiment transfer on Yelp with a frozen GPT-2 generator.

The task LM sees   {prompt} "{input}" "   and continues until it closes the
quote. The prompt is conditioned on the input, so z-scores are computed per
input across the prompts sampled for it.

Reward = Content(x, y) + Style(y, s), both on a 0-100 scale.
  Content: alignment of contextual embeddings between input and output.
           This follows the idea of the CTC preservation metric used in the
           paper with a BERTScore-style F1; it is our own implementation, so
           absolute numbers are not identical to the paper's.
  Style:   probability of the target style from a frozen sentiment classifier.

Each (input, prompt) pair is scored from several sampled generations by
averaging the maximum reward over random subsets of them, which rewards a
prompt for making good outputs likely without being dominated by one lucky
sample.
"""
from __future__ import annotations

import os
import random
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import (AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

from ..config import StyleConfig
from ..utils import chunks
from .base import Task

YELP_URL = ("https://raw.githubusercontent.com/shentianxiao/"
            "language-style-transfer/master/data/yelp/sentiment.{split}.{label}")

STYLE_WORD = {0: "negative", 1: "positive"}
MANUAL_TEMPLATE = ('Here is some text: "{sentence}". Here is a rewrite of the text, '
                   'which is more {style}: "')


# ---------------------------------------------------------------- data

def load_yelp(data_dir: str) -> Dict[str, Dict[int, List[str]]]:
    """Shen et al. (2017) Yelp sentiment data, downloaded on first use."""
    os.makedirs(data_dir, exist_ok=True)
    data: Dict[str, Dict[int, List[str]]] = {}
    for split in ("train", "dev", "test"):
        data[split] = {}
        for label in (0, 1):
            path = os.path.join(data_dir, f"sentiment.{split}.{label}")
            if not os.path.exists(path):
                print(f"downloading {path}")
                urllib.request.urlretrieve(YELP_URL.format(split=split, label=label), path)
            with open(path, encoding="utf-8") as fh:
                data[split][label] = [l.strip() for l in fh if l.strip()]
    return data


# ---------------------------------------------------------------- task LM

class PromptedGenerator:
    """Frozen causal LM that rewrites a sentence given a prompt."""

    def __init__(self, model_name: str, device: str = "cuda", fp16: bool = True,
                 top_k: int = 10, batch_size: int = 64):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if fp16 and self.device.type == "cuda":
            self.model.half()
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.top_k = top_k
        self.batch_size = batch_size
        # Any token containing a double quote ends the rewrite.
        self.stop_ids = [i for t, i in self.tokenizer.get_vocab().items() if '"' in t]
        self.stop_ids.append(self.tokenizer.eos_token_id)

    @staticmethod
    def format(sentence: str, prompt: str, template: Optional[str] = None,
               style: int = 1) -> str:
        if template:
            return template.format(sentence=sentence, style=STYLE_WORD[style])
        return f'{prompt} "{sentence}" "'.lstrip()

    @torch.no_grad()
    def generate(self, sentences: Sequence[str], prompts: Sequence[str], num_samples: int,
                 template: Optional[str] = None, style: int = 1) -> List[List[str]]:
        texts = [self.format(s, p, template, style) for s, p in zip(sentences, prompts)]
        per_call = max(1, self.batch_size // num_samples)
        outputs: List[List[str]] = []
        for i, batch in enumerate(chunks(texts, per_call)):
            srcs = sentences[i * per_call: i * per_call + len(batch)]
            src_len = max(len(self.tokenizer.tokenize(" " + s)) for s in srcs)
            enc = self.tokenizer(list(batch), return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(
                **enc, do_sample=True, top_k=self.top_k, num_return_sequences=num_samples,
                max_new_tokens=min(int(1.5 * src_len) + 5, 80),
                eos_token_id=self.stop_ids, pad_token_id=self.tokenizer.pad_token_id)
            new = gen[:, enc.input_ids.size(1):]
            decoded = self.tokenizer.batch_decode(new, skip_special_tokens=True)
            decoded = [d.split('"')[0].strip() for d in decoded]
            for j in range(len(batch)):
                outputs.append(decoded[j * num_samples:(j + 1) * num_samples])
        return outputs


# ---------------------------------------------------------------- scorers

class ContentScorer:
    """BERTScore-style F1 between contextual token embeddings, x100."""

    def __init__(self, model_name: str = "roberta-large", layer: int = 17,
                 device: str = "cuda", fp16: bool = True, batch_size: int = 128):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if fp16 and self.device.type == "cuda":
            self.model.half()
        self.model.to(self.device).eval()
        self.layer = min(layer, self.model.config.num_hidden_layers)
        self.batch_size = batch_size

    @torch.no_grad()
    def _embed(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True,
                             max_length=128, return_special_tokens_mask=True).to(self.device)
        special = enc.pop("special_tokens_mask")
        hs = self.model(**enc, output_hidden_states=True).hidden_states[self.layer]
        valid = enc.attention_mask.bool() & ~special.bool()
        return F.normalize(hs.float(), dim=-1), valid

    @torch.no_grad()
    def score(self, refs: Sequence[str], outs: Sequence[str]) -> torch.Tensor:
        scores = []
        for i in range(0, len(refs), self.batch_size):
            r, o = refs[i:i + self.batch_size], outs[i:i + self.batch_size]
            er, vr = self._embed(r)
            eo, vo = self._embed(o)
            sim = torch.einsum("brd,bod->bro", er, eo)
            sim = sim.masked_fill(~vo[:, None, :], -2.0).masked_fill(~vr[:, :, None], -2.0)
            recall = (sim.max(2).values * vr).sum(1) / vr.sum(1).clamp(min=1)
            precision = (sim.max(1).values * vo).sum(1) / vo.sum(1).clamp(min=1)
            f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
            f1 = torch.where(vo.any(1), f1, torch.zeros_like(f1))
            scores.append(f1.clamp(min=0) * 100)
        return torch.cat(scores).cpu()


class ClassifierScorer:
    """Probability of one label from a frozen sequence classifier."""

    def __init__(self, model_name: str, device: str = "cuda", fp16: bool = True,
                 batch_size: int = 128):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if fp16 and self.device.type == "cuda":
            self.model.half()
        self.model.to(self.device).eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def prob(self, texts: Sequence[str], label: int) -> torch.Tensor:
        out = []
        for batch in chunks(list(texts), self.batch_size):
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                 max_length=128).to(self.device)
            out.append(torch.softmax(self.model(**enc).logits.float(), -1)[:, label])
        return torch.cat(out).cpu()


# ---------------------------------------------------------------- task

class StyleTransferTask(Task):
    def __init__(self, generator: PromptedGenerator, content: ContentScorer,
                 style: ClassifierScorer, train: List[str], dev: List[str],
                 target_style: int, cfg: StyleConfig):
        self.gen = generator
        self.content = content
        self.style = style
        self.train = train
        self.dev = dev
        self.target = target_style
        self.cfg = cfg

    def score_outputs(self, srcs: Sequence[str], outs: Sequence[str]) -> Dict[str, torch.Tensor]:
        c = self.content.score(srcs, outs)
        s = self.style.prob(outs, self.target) * 100
        return {"content": c, "style": s, "reward": c + s}

    def _bootstrap_max(self, r: torch.Tensor) -> torch.Tensor:
        P, S = r.shape
        k = min(self.cfg.bootstrap_size, S)
        idx = torch.rand(P, self.cfg.bootstraps, S).argsort(-1)[..., :k]
        sub = r[:, None, :].expand(P, self.cfg.bootstraps, S).gather(-1, idx)
        return sub.max(-1).values.mean(-1)

    def sample_batch(self, rng: random.Random):
        inputs = rng.sample(self.train, self.cfg.inputs_per_step)
        K = self.cfg.prompts_per_input
        sources = [x for x in inputs for _ in range(K)]
        groups = torch.arange(len(inputs)).repeat_interleave(K)
        return sources, groups, sources

    def reward(self, prompts: List[str], inputs: List[str]):
        S = self.cfg.samples_per_prompt
        outs = self.gen.generate(inputs, prompts, S)
        flat_src = [x for x in inputs for _ in range(S)]
        flat_out = [o for group in outs for o in group]
        sc = self.score_outputs(flat_src, flat_out)
        r = sc["reward"].view(len(prompts), S)
        rewards = self._bootstrap_max(r)

        best_pair = int(rewards.argmax())
        best_sample = int(r[best_pair].argmax())
        logs = {"reward_raw": rewards.mean().item(),
                "content": sc["content"].mean().item(), "style": sc["style"].mean().item(),
                "sample_prompt": prompts[best_pair],
                "sample_io": f"{inputs[best_pair]} -> {outs[best_pair][best_sample]}"}
        return rewards, logs

    def transfer(self, policy, sentences: Sequence[str], num_samples: int
                 ) -> Tuple[List[str], List[str]]:
        """Greedy prompt per input, then keep the highest-reward sample."""
        prompts: List[str] = []
        for batch in chunks(list(sentences), 64):
            prompts += policy.sample(list(batch), greedy=True)[1]
        return self.best_of(sentences, prompts, num_samples), prompts

    def best_of(self, sentences: Sequence[str], prompts: Sequence[str], num_samples: int,
                template: Optional[str] = None) -> List[str]:
        outs = self.gen.generate(list(sentences), list(prompts), num_samples,
                                 template=template, style=self.target)
        flat_src = [x for x in sentences for _ in range(num_samples)]
        flat_out = [o for group in outs for o in group]
        r = self.score_outputs(flat_src, flat_out)["reward"].view(len(sentences), num_samples)
        best = r.argmax(-1).tolist()
        return [outs[i][b] for i, b in enumerate(best)]

    def evaluate(self, policy) -> Dict:
        outs, prompts = self.transfer(policy, self.dev, self.cfg.samples_per_prompt)
        sc = self.score_outputs(self.dev, outs)
        return {"score": sc["reward"].mean().item(), "content": sc["content"].mean().item(),
                "style": sc["style"].mean().item(), "prompt": prompts[0],
                "example": f"{self.dev[0]} -> {outs[0]}"}


def style_transfer_metrics(srcs: Sequence[str], outs: Sequence[str], content: ContentScorer,
                           style: ClassifierScorer, fluency: ClassifierScorer,
                           target: int, fluency_label: int = 1) -> Dict[str, float]:
    """Content, Style accuracy, Fluency, J(C,S,F) and the geometric mean GM(C,S,F)."""
    c = content.score(srcs, outs) / 100
    s = (style.prob(outs, target) > 0.5).float()
    f = (fluency.prob(outs, fluency_label) > 0.5).float()
    C, S, Fl = c.mean().item() * 100, s.mean().item() * 100, f.mean().item() * 100
    J = (c * s * f).mean().item() * 100
    GM = (max(C, 1e-8) * max(S, 1e-8) * max(Fl, 1e-8)) ** (1 / 3)
    return {"content": C, "style": S, "fluency": Fl, "J": J, "GM": GM}
