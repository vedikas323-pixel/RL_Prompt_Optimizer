"""Losses for training the prompt policy.

Soft Q-Learning (Guo et al., 2021) reads the policy logits as Q-values:

    V(s)    = logsumexp_a Q(s, a)
    A(s, a) = Q(s, a) - V(s)  = log pi(a | s)

The optimal soft Q-function satisfies path consistency. With a sparse
reward R given only after the last prompt token, gamma = 1 and V = 0 after
the final step:

    single-step   A(s_t, a_t) + V(s_t) = r_t + V(s_{t+1})
    multi-step    V(s_t) + sum_{l >= t} A(s_l, a_l) = R

Each identity yields two regressions: one trains A with a target built from
the target network, the "reversed" one trains V. RLPrompt averages all four.
Only the on-policy part is used, i.e. the samples come from the policy itself.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


def _gather(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return logits.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def _rev_cumsum(x: torch.Tensor) -> torch.Tensor:
    return x.flip(1).cumsum(1).flip(1)


def sql_loss(logits: torch.Tensor, target_logits: torch.Tensor,
             actions: torch.Tensor, rewards: torch.Tensor,
             variants: Iterable[str] = ("v2", "v2r", "v3", "v3r")
             ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """logits, target_logits: (B, T, V); actions: (B, T); rewards: (B,)."""
    B, T = actions.shape
    Q = _gather(logits, actions)
    V = logits.logsumexp(-1)
    A = Q - V
    with torch.no_grad():
        Qt = _gather(target_logits, actions)
        Vt = target_logits.logsumexp(-1)
        At = Qt - Vt
        Vt_next = torch.cat([Vt[:, 1:], Vt.new_zeros(B, 1)], dim=1)
        r = rewards.new_zeros(B, T)
        r[:, -1] = rewards
        R = rewards[:, None].expand(B, T)

    terms = {
        "v2":  (A, r + Vt_next - Vt),
        "v2r": (V, r + Vt_next - At),
        "v3":  (_rev_cumsum(A), R - Vt),
        "v3r": (V, R - _rev_cumsum(At)),
    }
    losses, logs = [], {}
    for name in variants:
        pred, target = terms[name]
        l = 0.5 * (pred - target).pow(2).mean()
        losses.append(l)
        logs[f"loss_{name}"] = l.item()
    loss = torch.stack(losses).mean()
    logs["entropy"] = _entropy(logits)
    return loss, logs


def reinforce_loss(logits: torch.Tensor, actions: torch.Tensor,
                   rewards: torch.Tensor, entropy_coeff: float = 0.0
                   ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Plain policy gradient. Rewards are already z-scored, so they act as advantages."""
    logp = torch.log_softmax(logits, -1)
    seq_logp = _gather(logp, actions).sum(1)
    loss = -(rewards.detach() * seq_logp).mean()
    ent = -(logp.exp() * logp.nan_to_num(neginf=0.0)).sum(-1).mean()
    loss = loss - entropy_coeff * ent
    return loss, {"entropy": ent.item()}


@torch.no_grad()
def _entropy(logits: torch.Tensor) -> float:
    logp = torch.log_softmax(logits, -1)
    return (-(logp.exp() * logp.nan_to_num(neginf=0.0)).sum(-1)).mean().item()
