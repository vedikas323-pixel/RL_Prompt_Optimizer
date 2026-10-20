"""RLPrompt with a chat model behind an API as the frozen task LM.

The user supplies an instruction and labelled examples. The policy learns a
few extra tokens z that are appended to the instruction:

    {instruction} {z}

    Text: {input}

    Answer with exactly one of: {labels}.
    Answer:

API models do not all expose token probabilities, so the reward uses the
answer only. The probability gap of the paper becomes +1 for a correct answer
and -1 for a wrong one, and the piecewise weights still apply:

    R = lambda_2 if correct else -lambda_1, averaged over the step's examples.
"""
from __future__ import annotations

import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..llm_clients import LLMClient
from .base import Task

Example = Tuple[str, int]

# Provider-specific query formats.  The optimizer (policy network) learns which
# prompt additions work best for each model; the target model only evaluates.
_PROVIDER_FORMATS: Dict[str, object] = {
    "Claude": lambda head, text, labels: (
        f"<instructions>\n{head}\n</instructions>\n\n"
        f"<input>{text}</input>\n\n"
        f"Reply with exactly one of: {', '.join(labels)}.\nAnswer:"
    ),
    "OpenAI": lambda head, text, labels: (
        f"{head}\n\n"
        f"Input: {text}\n\n"
        f"Reply with exactly one of: {', '.join(labels)}.\nAnswer:"
    ),
    "Gemini": lambda head, text, labels: (
        f"{head}\n\n"
        f"**Text:** {text}\n\n"
        f"Choose exactly one of: {', '.join(labels)}\n**Answer:**"
    ),
}


def build_query(instruction: str, learned: str, text: str, labels: Sequence[str],
                provider: str = "OpenAI") -> str:
    head = f"{instruction.strip()} {learned.strip()}".strip()
    fmt = _PROVIDER_FORMATS.get(provider, _PROVIDER_FORMATS["OpenAI"])
    return fmt(head, text, labels)


def format_optimized_prompt(instruction: str, learned: str, provider: str) -> str:
    """Return the full prompt template a user would copy-paste for the chosen model."""
    head = f"{instruction.strip()} {learned.strip()}".strip()
    if provider == "Claude":
        return f"<instructions>\n{head}\n</instructions>\n\n<input>{{text}}</input>"
    if provider == "Gemini":
        return f"{head}\n\n**Text:** {{text}}"
    return f"{head}\n\nInput: {{text}}"


def parse_label(answer: str, labels: Sequence[str]) -> Optional[int]:
    """Index of the label the model chose; the earliest whole-word match wins."""
    low = answer.lower()
    best, best_pos = None, len(low) + 1
    for i, lab in sorted(enumerate(labels), key=lambda x: -len(x[1])):
        m = re.search(rf"(?<![\w]){re.escape(lab.lower())}(?![\w])", low)
        if m and m.start() < best_pos:
            best, best_pos = i, m.start()
    return best


def parse_examples(raw: str) -> Tuple[List[str], List[Example]]:
    """Lines of `label | text`. Labels are taken in order of first appearance."""
    labels: List[str] = []
    examples: List[Example] = []
    for line in raw.splitlines():
        if "|" not in line:
            continue
        lab, text = line.split("|", 1)
        lab, text = lab.strip(), text.strip()
        if not lab or not text:
            continue
        if lab not in labels:
            labels.append(lab)
        examples.append((text, labels.index(lab)))
    return labels, examples


def split_examples(examples: List[Example], seed: int = 0
                   ) -> Tuple[List[Example], List[Example], List[Example]]:
    """Stratified 50 / 25 / 25 split into train, dev and test."""
    rng = random.Random(seed)
    by: Dict[int, List[Example]] = {}
    for ex in examples:
        by.setdefault(ex[1], []).append(ex)
    train, dev, test = [], [], []
    for exs in by.values():
        exs = exs[:]
        rng.shuffle(exs)
        n = len(exs)
        n_dev = max(1, n // 4)
        n_test = max(1, n // 4) if n >= 4 else 0
        dev += exs[:n_dev]
        test += exs[n_dev:n_dev + n_test]
        train += exs[n_dev + n_test:]
    return train, dev, test


class APIClassifier:
    """Scores (learned prompt, example) pairs with a chat model, in parallel and cached."""

    def __init__(self, client: LLMClient, instruction: str, labels: Sequence[str],
                 store=None, workers: int = 8, provider: str = "OpenAI"):
        self.client = client
        self.instruction = instruction
        self.labels = list(labels)
        self.store = store
        self.workers = workers
        self.provider = provider
        self.cache_hits = 0

    def _answer(self, query: str) -> str:
        if self.store is not None:
            hit = self.store.cache_get(self.client.provider, self.client.model, query)
            if hit is not None:
                self.cache_hits += 1
                return hit
        ans = self.client.complete(query)
        if self.store is not None:
            self.store.cache_put(self.client.provider, self.client.model, query, ans)
        return ans

    def predict(self, learned: Sequence[str], texts: Sequence[str]) -> List[Optional[int]]:
        queries = [build_query(self.instruction, z, x, self.labels, self.provider)
                   for z, x in zip(learned, texts)]
        with ThreadPoolExecutor(self.workers) as pool:
            answers = list(pool.map(self._answer, queries))
        return [parse_label(a, self.labels) for a in answers]

    def accuracy(self, learned: str, examples: Sequence[Example]) -> float:
        preds = self.predict([learned] * len(examples), [x for x, _ in examples])
        return sum(p == y for p, (_, y) in zip(preds, examples)) / max(1, len(examples))


class APIPromptTask(Task):
    def __init__(self, clf: APIClassifier, train: List[Example], dev: List[Example],
                 prompts_per_step: int = 8, examples_per_step: int = 8,
                 correct_coeff: float = 200.0, incorrect_coeff: float = 180.0,
                 provider: str = "OpenAI"):
        self.clf = clf
        self.train = train
        self.dev = dev
        self.n = prompts_per_step
        self.k = examples_per_step
        self.c2 = correct_coeff
        self.c1 = incorrect_coeff
        # Condition the policy on the target provider so it learns model-specific
        # prompt styles.  The source prefix "[Claude]" vs "[OpenAI]" shifts the
        # distilGPT-2 hidden state, training different MLP weights per model.
        self._source = f"[{provider}]"

    def sample_batch(self, rng: random.Random):
        k = self.k
        examples = self.train if not k or k >= len(self.train) else rng.sample(self.train, k)
        return [self._source] * self.n, torch.zeros(self.n, dtype=torch.long), examples

    def reward(self, prompts: List[str], examples: List[Example]):
        M = len(examples)
        unique_prompts: List[str] = []
        inv_idx: List[int] = []
        prompt_map: Dict[str, int] = {}
        for p in prompts:
            if p not in prompt_map:
                prompt_map[p] = len(unique_prompts)
                unique_prompts.append(p)
            inv_idx.append(prompt_map[p])

        U = len(unique_prompts)
        u_preds = self.clf.predict([p for p in unique_prompts for _ in range(M)],
                                   [x for x, _ in examples] * U)
        u_correct = torch.tensor([p == y for p, (_, y) in zip(u_preds, examples * U)],
                                 dtype=torch.float).view(U, M)
        correct = u_correct[inv_idx]
        rewards = (correct * self.c2 - (1 - correct) * self.c1).mean(1)
        acc = correct.mean(1)
        self.record_candidates(prompts, acc.tolist())
        best = int(acc.argmax())
        return rewards, {"reward_raw": rewards.mean().item(), "train_acc_mean": acc.mean().item(),
                         "train_acc_best": acc[best].item(), "sample_prompt": prompts[best],
                         "api_calls": self.clf.client.calls, "cache_hits": self.clf.cache_hits}

    def evaluate(self, policy) -> Dict:
        _, prompts = policy.sample([self._source], greedy=True)
        return {"score": self.clf.accuracy(prompts[0], self.dev), "prompt": prompts[0]}
