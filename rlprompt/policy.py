"""Prompt policy: a frozen causal LM with one trainable MLP before its LM head.

    h_t      = PolicyLM(source, z_<t)              frozen, no gradients
    logits_t = LMHead(MLP(h_t))                    only the MLP is trained
    z_t      ~ softmax(logits_t)                   top-k sampling / greedy

For Soft Q-Learning the logits are read as Q-values. A target network is
needed; because the backbone is frozen, the target is just a second copy
of the MLP that tracks the online MLP by Polyak averaging, and both share
one backbone forward pass.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import PolicyConfig


def build_mlp(dim: int, hidden: int) -> nn.Sequential:
    mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
    for m in mlp:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 1e-4)
    return mlp


class PromptPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.policy_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.start_text = self.tokenizer.bos_token or self.tokenizer.eos_token

        self.backbone = AutoModelForCausalLM.from_pretrained(cfg.policy_model)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        dim = self.backbone.config.hidden_size
        self.mlp = build_mlp(dim, cfg.mlp_hidden)
        self.target_mlp = copy.deepcopy(self.mlp)
        for p in self.target_mlp.parameters():
            p.requires_grad_(False)

        # Tokens the policy may never emit: special tokens, and byte fragments
        # that do not decode to valid text on their own. Banning the latter
        # guarantees the prompt string the task LM reads is exactly what was sampled.
        vocab = self.backbone.get_output_embeddings().weight.shape[0]
        banned = torch.zeros(vocab, dtype=torch.bool)
        for tid in self.tokenizer.all_special_ids:
            if tid < vocab:
                banned[tid] = True
        for tid, piece in enumerate(self.tokenizer.batch_decode([[i] for i in range(vocab)])):
            if "\ufffd" in piece or not piece.strip():
                banned[tid] = True
        self.register_buffer("banned", banned, persistent=False)
        self.to(self.device)

    # ------------------------------------------------------------ helpers
    def trainable_parameters(self):
        return self.mlp.parameters()

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.mlp.parameters())

    def _encode(self, sources: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        texts = [self.start_text + s for s in sources]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             truncation=True, max_length=128)
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    @torch.no_grad()
    def _hidden(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
        out = self.backbone.base_model(input_ids=ids, attention_mask=mask, position_ids=pos)
        return out.last_hidden_state

    def _head(self, h: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
        x = mlp(h)
        if self.cfg.mlp_residual:
            x = x + h
        return self.backbone.get_output_embeddings()(x)

    def _restrict(self, h: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.masked_fill(self.banned, float("-inf"))
        if self.cfg.fluent_top_k:
            with torch.no_grad():
                base = self.backbone.get_output_embeddings()(h)
                kth = base.topk(self.cfg.fluent_top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(base < kth, float("-inf"))
        return logits

    # ------------------------------------------------------------ public API
    @torch.no_grad()
    def sample(self, sources: List[str], greedy: bool = False,
               top_k: Optional[int] = None) -> Tuple[torch.Tensor, List[str]]:
        """Generate one prompt per source. Returns token ids (B, T) and strings."""
        top_k = self.cfg.top_k if top_k is None else top_k
        ids, mask = self._encode(sources)
        actions = []
        for _ in range(self.cfg.prompt_length):
            h = self._hidden(ids, mask)[:, -1]
            logits = self._restrict(h, self._head(h, self.mlp)).float()
            if greedy:
                a = logits.argmax(-1)
            else:
                if top_k and top_k < logits.size(-1):
                    kth = logits.topk(top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                a = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1)
            actions.append(a)
            ids = torch.cat([ids, a[:, None]], dim=1)
            mask = torch.cat([mask, torch.ones_like(a[:, None])], dim=1)
        actions = torch.stack(actions, dim=1)
        return actions, self.decode(actions)

    def decode(self, actions: torch.Tensor) -> List[str]:
        return [self.tokenizer.decode(row, clean_up_tokenization_spaces=False).strip()
                for row in actions.tolist()]

    def forced_logits(self, sources: List[str], actions: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Online and target logits at every prompt step, given sampled actions.

        Returns (B, T, V) tensors; only the online logits carry gradients.
        """
        ids, mask = self._encode(sources)
        full_ids = torch.cat([ids, actions], dim=1)
        full_mask = torch.cat([mask, torch.ones_like(actions)], dim=1)
        h = self._hidden(full_ids, full_mask)
        L, T = ids.size(1), actions.size(1)
        h_steps = h[:, L - 1: L - 1 + T]            # state before each action
        logits = self._restrict(h_steps, self._head(h_steps, self.mlp))
        with torch.no_grad():
            target = self._restrict(h_steps, self._head(h_steps, self.target_mlp))
        return logits.float(), target.float()

    @torch.no_grad()
    def polyak_update(self, rate: float) -> None:
        for p_t, p in zip(self.target_mlp.parameters(), self.mlp.parameters()):
            p_t.mul_(1.0 - rate).add_(p, alpha=rate)

    # ------------------------------------------------------------ checkpoints
    def save(self, path: str) -> None:
        torch.save({"mlp": self.mlp.state_dict(),
                    "target_mlp": self.target_mlp.state_dict()}, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        self.mlp.load_state_dict(state["mlp"])
        self.target_mlp.load_state_dict(state["target_mlp"])
