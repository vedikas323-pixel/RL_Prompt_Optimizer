"""Configuration dataclasses shared by every experiment.

Each field can be overridden from the command line, e.g.
    python scripts/train_classification.py --dataset sst2 --prompt_length 2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PolicyConfig:
    policy_model: str = "distilgpt2"      # frozen policy LM
    mlp_hidden: int = 2048                # hidden units of the trainable MLP
    mlp_residual: bool = False            # lm_head(h + mlp(h)) instead of lm_head(mlp(h))
    prompt_length: int = 5                # T, number of prompt tokens
    top_k: int = 256                      # top-k sampling during training
    fluent_top_k: Optional[int] = None    # restrict actions to the policy LM's own top-k


@dataclass
class TrainConfig:
    algo: str = "sql"                     # "sql" or "reinforce"
    sql_variants: str = "v2,v2r,v3,v3r"   # which Soft Q-Learning losses to average
    learning_rate: float = 5e-5
    target_update_rate: float = 1e-3      # Polyak rate for the SQL target network
    max_steps: int = 6000
    grad_clip: float = 5.0
    zscore: bool = True                   # input-specific z-score normalisation
    reward_scale: float = 5.0             # multiplier applied after normalisation
    entropy_coeff: float = 0.0            # only used by REINFORCE
    eval_every: int = 100
    log_every: int = 20
    seed: int = 0
    output_dir: str = "outputs/run"
    device: str = "cuda"
    fp16_task_lm: bool = True


@dataclass
class ClassificationConfig:
    dataset: str = "sst2"
    task_model: str = "roberta-large"
    shots: int = 16                       # examples per class for train and for dev
    split_seed: int = 0                   # which few-shot split to sample
    prompts_per_step: int = 16
    examples_per_step: int = 0            # 0 = use every training example each step
    correct_coeff: float = 200.0          # lambda_2
    incorrect_coeff: float = 180.0        # lambda_1
    piecewise: bool = True
    task_batch_size: int = 128            # sequences per task-LM forward pass
    max_test: int = 0                     # 0 = full test set


@dataclass
class StyleConfig:
    direction: str = "0_to_1"             # 0 = negative, 1 = positive
    task_model: str = "distilgpt2"        # generator LM; gpt2-xl in the paper's best run
    content_model: str = "roberta-large"
    content_layer: int = 17
    style_model: str = "textattack/bert-base-uncased-yelp-polarity"
    fluency_model: str = "textattack/roberta-base-CoLA"
    fluency_label: int = 1                # index of the "acceptable" class
    inputs_per_step: int = 8
    prompts_per_input: int = 4
    samples_per_prompt: int = 8           # generations per (input, prompt) pair
    bootstraps: int = 4
    bootstrap_size: int = 4
    gen_top_k: int = 10
    max_train: int = 0                    # 0 = full training set
    dev_size: int = 64
    eval_samples: int = 32                # paper keeps the best of 32 outputs
    max_test: int = 500
    task_batch_size: int = 64
    data_dir: str = "data/yelp"


# ---------------------------------------------------------------- CLI helper

def _str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y"}


def add_dataclass_args(parser: argparse.ArgumentParser, cls) -> None:
    group = parser.add_argument_group(cls.__name__)
    for f in dataclasses.fields(cls):
        default = f.default if f.default is not dataclasses.MISSING else None
        ftype = f.type if isinstance(f.type, type) else None
        tname = str(f.type)
        if "bool" in tname:
            group.add_argument(f"--{f.name}", type=_str2bool, default=default)
        elif "int" in tname:
            group.add_argument(f"--{f.name}", type=int, default=default)
        elif "float" in tname:
            group.add_argument(f"--{f.name}", type=float, default=default)
        else:
            group.add_argument(f"--{f.name}", type=ftype or str, default=default)


def from_args(cls, args: argparse.Namespace):
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in vars(args).items() if k in names})


def dump_configs(path: str, **configs) -> None:
    with open(path, "w") as fh:
        json.dump({k: dataclasses.asdict(v) for k, v in configs.items()}, fh, indent=2)
