"""A task tells the trainer three things:

1. what the policy is conditioned on for a training step  -> sample_batch
2. how good a batch of prompts is, via the frozen task LM -> reward
3. how to score the current greedy prompt for model selection -> evaluate

Adding a new task to RLPrompt means writing one of these, mostly the reward.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import torch


class Task:
    # Every prompt sampled during training and its running training accuracy.
    # Used to re-check the strongest sampled prompts on dev at the end, on top
    # of the policy's greedy prompt.
    candidates: Dict[str, List[float]]

    def record_candidates(self, prompts: List[str], scores) -> None:
        if not hasattr(self, "candidates"):
            self.candidates = {}
        for p, s in zip(prompts, scores):
            tot = self.candidates.setdefault(p, [0.0, 0])
            tot[0] += float(s)
            tot[1] += 1

    def top_candidates(self, k: int = 5, min_count: int = 1) -> List[str]:
        items = [(t / n, n, p) for p, (t, n) in getattr(self, "candidates", {}).items() if n >= min_count]
        items.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [p for _, _, p in items[:k]]

    def sample_batch(self, rng: random.Random) -> Tuple[List[str], torch.Tensor, Any]:
        """Return (policy_sources, group_ids, payload).

        policy_sources[i] is the text the policy conditions on for prompt i.
        group_ids[i] says which prompts share an input for z-scoring.
        payload is whatever reward() needs.
        """
        raise NotImplementedError

    def reward(self, prompts: List[str], payload: Any) -> Tuple[torch.Tensor, Dict]:
        """Raw, un-normalised reward per prompt, shape (len(prompts),)."""
        raise NotImplementedError

    def evaluate(self, policy) -> Dict:
        """Must return a dict with a scalar 'score' used to keep the best checkpoint."""
        raise NotImplementedError
