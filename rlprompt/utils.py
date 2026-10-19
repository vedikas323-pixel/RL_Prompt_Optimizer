from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zscore_by_group(rewards: torch.Tensor, groups: torch.Tensor,
                    eps: float = 1e-8) -> torch.Tensor:
    """Input-specific z-score: normalise each reward against the other prompts
    sampled for the same input. For classification every prompt shares one
    group, because prompts are not conditioned on the input."""
    out = torch.empty_like(rewards)
    for g in groups.unique():
        idx = groups == g
        r = rewards[idx]
        std = r.std(unbiased=False) if r.numel() > 1 else r.new_tensor(1.0)
        out[idx] = (r - r.mean()) / (std + eps)
    return out


def chunks(seq: Sequence, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def resolve_device(name: str) -> str:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        return "cpu"
    return name


class RunLogger:
    """Prints to stdout and appends JSON lines to <output_dir>/log.jsonl."""

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, "log.jsonl")
        self.start = time.time()

    def log(self, record: Dict) -> None:
        record = {"time_min": round((time.time() - self.start) / 60, 2), **record}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()}
        print(" | ".join(f"{k}={v}" for k, v in shown.items()), flush=True)


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def truncate_texts(tokenizer, texts: List[str], max_tokens: int) -> List[str]:
    out = []
    for t in texts:
        toks = tokenizer.tokenize(t)
        out.append(t if len(toks) <= max_tokens
                   else tokenizer.convert_tokens_to_string(toks[:max_tokens]).strip())
    return out
