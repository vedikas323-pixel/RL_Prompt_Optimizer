"""Training loop shared by every task.

one step:
    sources, groups, payload = task.sample_batch()
    actions, prompts         = policy.sample(sources)          explore, top-k
    raw                      = task.reward(prompts, payload)   frozen task LM
    r                        = z-score(raw, groups) * scale
    logits, target_logits    = policy.forced_logits(sources, actions)
    loss                     = SQL(logits, target_logits, actions, r)
    update MLP, Polyak-update target MLP
"""
from __future__ import annotations

import os
import random
from typing import Callable, Dict, Optional

import torch

from .config import TrainConfig
from .losses import reinforce_loss, sql_loss
from .policy import PromptPolicy
from .tasks.base import Task
from .utils import RunLogger, save_json, zscore_by_group


class Trainer:
    def __init__(self, policy: PromptPolicy, task: Task, cfg: TrainConfig):
        self.policy = policy
        self.task = task
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.opt = torch.optim.Adam(policy.trainable_parameters(), lr=cfg.learning_rate)
        self.logger = RunLogger(cfg.output_dir)
        self.variants = [v.strip() for v in cfg.sql_variants.split(",") if v.strip()]
        self.best: Dict = {"score": float("-inf")}
        self.stop_requested = False      # manual abort by user
        self.early_stopped = False       # triggered by early stopping (target_score or patience)
        self.eval_no_improve = 0

    def step(self) -> Dict:
        cfg = self.cfg
        sources, groups, payload = self.task.sample_batch(self.rng)
        actions, prompts = self.policy.sample(sources)
        raw, logs = self.task.reward(prompts, payload)

        raw = raw.float()
        r = zscore_by_group(raw, groups) if cfg.zscore else raw
        r = (r * cfg.reward_scale).to(self.policy.device)

        logits, target_logits = self.policy.forced_logits(sources, actions)
        if cfg.algo == "sql":
            loss, loss_logs = sql_loss(logits, target_logits, actions, r, self.variants)
        elif cfg.algo == "reinforce":
            loss, loss_logs = reinforce_loss(logits, actions, r, cfg.entropy_coeff)
        else:
            raise ValueError(cfg.algo)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.policy.trainable_parameters(), cfg.grad_clip)
        self.opt.step()
        if cfg.algo == "sql":
            self.policy.polyak_update(cfg.target_update_rate)

        return {"loss": loss.item(), "grad_norm": gnorm.item(), **loss_logs, **logs}

    def evaluate_and_save(self, step: int, callback: Optional[Callable] = None) -> None:
        result = self.task.evaluate(self.policy)
        record = {"step": step, "phase": "eval", **result}
        self.logger.log(record)
        if callback:
            callback(record)
        improved = False
        if result["score"] > self.best["score"]:
            self.best = {"step": step, **result}
            self.policy.save(os.path.join(self.cfg.output_dir, "best_policy.pt"))
            save_json(self.best, os.path.join(self.cfg.output_dir, "best_prompt.json"))
            improved = True

        target = getattr(self.cfg, "target_score", None)
        if target is not None and result["score"] >= target:
            self.early_stopped = True

        patience = getattr(self.cfg, "patience", 0)
        if patience and patience > 0 and step > 0:
            if improved:
                self.eval_no_improve = 0
            else:
                self.eval_no_improve += 1
                if self.eval_no_improve >= patience:
                    self.early_stopped = True

    def train(self, callback: Optional[Callable[[Dict], None]] = None) -> Dict:
        """Run training. `callback(record)` receives every train and eval log record."""
        cfg = self.cfg
        print(f"trainable parameters: {self.policy.num_trainable():,}")
        self.evaluate_and_save(0, callback)
        for step in range(1, cfg.max_steps + 1):
            logs = self.step()
            if step % cfg.log_every == 0:
                record = {"step": step, "phase": "train", **logs}
                self.logger.log(record)
                if callback:
                    callback(record)
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                self.evaluate_and_save(step, callback)
            if self.stop_requested or self.early_stopped:
                if step % cfg.eval_every:
                    self.evaluate_and_save(step, callback)
                break
        self.policy.save(os.path.join(cfg.output_dir, "last_policy.pt"))
        return self.best
