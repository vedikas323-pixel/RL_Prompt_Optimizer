"""Evaluate style transfer on the Yelp test set.

Trained policy, optionally on a different generator to test prompt transfer:
    python scripts/eval_style_transfer.py --run_dir outputs/yelp_0to1_distilgpt2 --task_model gpt2-large

Baselines from the paper:
    python scripts/eval_style_transfer.py --baseline null   --task_model gpt2-xl
    python scripts/eval_style_transfer.py --baseline manual --task_model gpt2-xl

Every method keeps the best of --eval_samples generations per input, as in the paper.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlprompt.config import PolicyConfig, StyleConfig
from rlprompt.policy import PromptPolicy
from rlprompt.tasks.style_transfer import (MANUAL_TEMPLATE, ClassifierScorer, ContentScorer,
                                           PromptedGenerator, StyleTransferTask, load_yelp,
                                           style_transfer_metrics)
from rlprompt.utils import load_json, resolve_device, save_json, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="")
    p.add_argument("--baseline", choices=["", "null", "manual"], default="")
    p.add_argument("--direction", default="0_to_1")
    p.add_argument("--task_model", default="")
    p.add_argument("--eval_samples", type=int, default=32)
    p.add_argument("--max_test", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="")
    args = p.parse_args()
    set_seed(0)
    device = resolve_device(args.device)

    scfg, pcfg = StyleConfig(), None
    if args.run_dir:
        cfg = load_json(os.path.join(args.run_dir, "config.json"))
        scfg, pcfg = StyleConfig(**cfg["task"]), PolicyConfig(**cfg["policy"])
    else:
        scfg.direction = args.direction
    if args.task_model:
        scfg.task_model = args.task_model
    src, tgt = (int(x) for x in scfg.direction.split("_to_"))
    test = load_yelp(scfg.data_dir)["test"][src][: args.max_test or None]

    gen = PromptedGenerator(scfg.task_model, device=device, top_k=scfg.gen_top_k,
                            batch_size=scfg.task_batch_size)
    content = ContentScorer(scfg.content_model, scfg.content_layer, device=device)
    style = ClassifierScorer(scfg.style_model, device=device)
    task = StyleTransferTask(gen, content, style, [], [], tgt, scfg)

    if args.run_dir and not args.baseline:
        policy = PromptPolicy(pcfg, device=device)
        policy.load(os.path.join(args.run_dir, "best_policy.pt"))
        outs, prompts = task.transfer(policy, test, args.eval_samples)
        name = f"rlprompt_on_{scfg.task_model}"
    else:
        template = MANUAL_TEMPLATE if args.baseline == "manual" else None
        prompts = [""] * len(test)
        outs = task.best_of(test, prompts, args.eval_samples, template=template)
        name = f"{args.baseline}_{scfg.task_model}"

    fluency = ClassifierScorer(scfg.fluency_model, device=device)
    metrics = style_transfer_metrics(test, outs, content, style, fluency, tgt, scfg.fluency_label)
    print(name, metrics)

    out_dir = args.out_dir or args.run_dir or "outputs/style_eval"
    os.makedirs(out_dir, exist_ok=True)
    save_json(metrics, os.path.join(out_dir, f"eval_{name.replace('/', '_')}.json"))
    with open(os.path.join(out_dir, f"outputs_{name.replace('/', '_')}.tsv"), "w") as fh:
        fh.write("input\tprompt\toutput\n")
        for x, pr, y in zip(test, prompts, outs):
            fh.write(f"{x}\t{pr}\t{y}\n")


if __name__ == "__main__":
    main()
