"""Evaluate classification prompts on any task LM, including prompt transfer.

Single prompt:
    python scripts/eval_classification.py --dataset sst2 --prompts "Absolutely"

Transfer matrix, every saved prompt on every model:
    python scripts/eval_classification.py --dataset sst2 \
        --prompt_files outputs/sst2_distilroberta/best_prompt.json outputs/sst2_large/best_prompt.json \
        --task_models distilroberta-base roberta-base roberta-large \
        --out outputs/sst2_transfer.csv

Also reports the manual and null prompt baselines for each model.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlprompt.tasks.classification import DATASETS, PromptedClassifier, load_fewshot
from rlprompt.utils import load_json, resolve_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="sst2")
    p.add_argument("--prompts", nargs="*", default=[])
    p.add_argument("--prompt_files", nargs="*", default=[])
    p.add_argument("--task_models", nargs="+", default=["roberta-large"])
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--max_test", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = resolve_device(args.device)
    spec = DATASETS[args.dataset]
    _, _, test = load_fewshot(args.dataset, 16, args.split_seed, args.max_test)

    prompts = [("given", s) for s in args.prompts]
    prompts += [(f, load_json(f)["prompt"]) for f in args.prompt_files]
    prompts += [("manual", spec.manual_prompt), ("null", "")]

    rows = []
    for model in args.task_models:
        clf = PromptedClassifier(model, spec.verbalizers, spec.template, device=device,
                                 batch_size=args.batch_size)
        for source, prompt in prompts:
            acc = clf.accuracy(prompt, test) * 100
            rows.append({"task_model": model, "source": source, "prompt": prompt, "acc": round(acc, 2)})
            print(f"{model:<24} {source:<40} {prompt!r:<30} {acc:6.2f}")
        del clf

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
