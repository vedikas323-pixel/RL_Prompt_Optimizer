"""One call that takes a user's instruction + labelled examples and returns an
optimised prompt. Used by the web app and by scripts/optimize_prompt.py.

    baseline  = accuracy of the instruction alone
    train     = RLPrompt learns extra tokens appended to the instruction
    select    = greedy prompt and the best sampled prompts are re-checked on dev
    report    = dev and held-out test accuracy, saved to SQLite
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import torch

from .config import ClassificationConfig, PolicyConfig, TrainConfig
from .llm_clients import make_client
from .policy import PromptPolicy
from .store import Store
from .tasks.api_classification import (APIClassifier, APIPromptTask, format_optimized_prompt,
                                       parse_examples, split_examples)
from .tasks.classification import ClassificationTask, PromptedClassifier
from .trainer import Trainer
from .utils import set_seed

LOCAL_MODELS = ["roberta-large", "roberta-base", "distilroberta-base", "distilgpt2", "gpt2",
                "gpt2-medium"]
PROVIDERS = ["OpenAI", "Gemini", "Claude", "Local"]


@dataclass
class OptimizeRequest:
    provider: str
    model: str
    instruction: str
    examples_text: str
    api_key: str = ""
    prompt_length: int = 3
    steps: int = 60
    prompts_per_step: int = 8
    examples_per_step: int = 8
    readable: bool = True
    readable_top_k: int = 50
    learning_rate: float = 5e-4
    eval_every: int = 10
    workers: int = 8
    seed: int = 0
    policy_model: str = "distilgpt2"
    device: str = "auto"
    runs_dir: str = "runs"


def estimate_calls(req: OptimizeRequest, n_train: int, n_dev: int, n_test: int) -> int:
    if req.provider == "Local":
        return 0
    per_step = req.prompts_per_step * min(req.examples_per_step or n_train, n_train)
    evals = (req.steps // max(1, req.eval_every) + 1) * n_dev
    return req.steps * per_step + evals + 2 * (n_dev + n_test) + 5 * n_dev


class _LocalAdapter:
    """Gives the local masked/causal LM the same accuracy(learned, examples) API."""

    def __init__(self, clf: PromptedClassifier):
        self.clf = clf

    def accuracy(self, learned, examples):
        return self.clf.accuracy(learned, examples)


def _build(req: OptimizeRequest, labels: List[str], train, dev, store: Store, device: str):
    if req.provider == "Local":
        instr = req.instruction.replace("{", "{{").replace("}", "}}")
        clf = PromptedClassifier(req.model, [" " + l for l in labels],
                                 "{sentence} " + instr + " {prompt} {mask} .", device=device)
        ccfg = ClassificationConfig(prompts_per_step=req.prompts_per_step,
                                    examples_per_step=req.examples_per_step)
        return _LocalAdapter(clf), ClassificationTask(clf, train, dev, ccfg), None
    client = make_client(req.provider, req.model, req.api_key or None)
    clf = APIClassifier(client, req.instruction, labels, store=store,
                        workers=req.workers, provider=req.provider)
    task = APIPromptTask(clf, train, dev, req.prompts_per_step, req.examples_per_step,
                         provider=req.provider)
    return clf, task, client


def run_optimization(req: OptimizeRequest, store: Store,
                     on_record: Optional[Callable[[Dict], None]] = None,
                     stop_event: Optional[threading.Event] = None) -> Dict:
    t0 = time.time()
    labels, examples = parse_examples(req.examples_text)
    if len(labels) < 2:
        raise ValueError("Need at least two different labels, one example per line as `label | text`.")
    if len(examples) < 4 * len(labels):
        raise ValueError(f"Need at least 4 examples per label; got {len(examples)} for {len(labels)} labels.")
    train, dev, test = split_examples(examples, req.seed)

    device = req.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(req.seed)

    settings = {k: v for k, v in asdict(req).items() if k not in ("api_key", "examples_text")}
    run_id = store.create_run(req.provider, req.model, req.instruction, labels, len(examples), settings)
    emit = on_record or (lambda r: None)
    try:
        scorer, task, client = _build(req, labels, train, dev, store, device)
        emit({"phase": "info", "message": f"run {run_id}: {len(train)} train / {len(dev)} dev / "
              f"{len(test)} test examples, labels {labels}"})

        base_dev = scorer.accuracy("", dev)
        base_test = scorer.accuracy("", test) if test else None
        emit({"phase": "baseline", "dev": base_dev, "test": base_test})

        pcfg = PolicyConfig(policy_model=req.policy_model, prompt_length=req.prompt_length,
                            fluent_top_k=req.readable_top_k if req.readable else None,
                            top_k=256)
        tcfg = TrainConfig(max_steps=req.steps, eval_every=req.eval_every, log_every=1,
                           learning_rate=req.learning_rate, seed=req.seed, device=device,
                           output_dir=os.path.join(req.runs_dir, f"run_{run_id}"))
        policy = PromptPolicy(pcfg, device=device)
        trainer = Trainer(policy, task, tcfg)

        def callback(record):
            if stop_event is not None and stop_event.is_set():
                trainer.stop_requested = True
            emit(record)

        best = trainer.train(callback)

        # Model selection on dev: greedy prompt vs the strongest sampled prompts.
        candidates = [best["prompt"]] + [p for p in task.top_candidates(5) if p != best["prompt"]]
        scored = [(scorer.accuracy(p, dev), p) for p in candidates]
        dev_acc, learned = max(scored, key=lambda x: x[0])
        if dev_acc < base_dev:                       # never return something worse than the start
            dev_acc, learned = base_dev, ""
        test_acc = scorer.accuracy(learned, test) if test else None

        if req.provider == "Local":
            optimized = f"{req.instruction.strip()} {learned}".strip()
        else:
            optimized = format_optimized_prompt(req.instruction.strip(), learned, req.provider)
        result = {"run_id": run_id, "labels": labels, "learned_tokens": learned,
                  "optimized_prompt": optimized,
                  "baseline_dev": base_dev, "optimized_dev": dev_acc,
                  "baseline_test": base_test, "optimized_test": test_acc,
                  "candidates": [{"prompt": p, "dev_acc": a} for a, p in scored],
                  "api_calls": client.calls if client else 0,
                  "minutes": round((time.time() - t0) / 60, 2),
                  "stopped_early": trainer.stop_requested}
        store.update_run(run_id, status="stopped" if trainer.stop_requested else "done",
                         learned_tokens=learned, optimized_prompt=optimized,
                         baseline_dev=base_dev, optimized_dev=dev_acc,
                         baseline_test=base_test, optimized_test=test_acc,
                         api_calls=result["api_calls"], minutes=result["minutes"])
        emit({"phase": "done", **result})
        return result
    except Exception as err:
        store.update_run(run_id, status=f"failed: {err}"[:500],
                         minutes=round((time.time() - t0) / 60, 2))
        raise
