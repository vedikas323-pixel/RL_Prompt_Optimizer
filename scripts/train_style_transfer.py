"""Train an input-conditioned RLPrompt policy for Yelp sentiment transfer.

Example, negative to positive with distilGPT-2 as the generator:
    python scripts/train_style_transfer.py --direction 0_to_1 --task_model distilgpt2 \
        --output_dir outputs/yelp_0to1_distilgpt2
"""
import argparse
import gc
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rlprompt.config import (PolicyConfig, StyleConfig, TrainConfig,
                             add_dataclass_args, dump_configs, from_args)
from rlprompt.policy import PromptPolicy
from rlprompt.tasks.style_transfer import (ClassifierScorer, ContentScorer, PromptedGenerator,
                                           StyleTransferTask, load_yelp, style_transfer_metrics)
from rlprompt.trainer import Trainer
from rlprompt.utils import resolve_device, save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    for cls in (PolicyConfig, TrainConfig, StyleConfig):
        add_dataclass_args(parser, cls)
    parser.set_defaults(max_steps=2000, eval_every=200, output_dir="")
    args = parser.parse_args()
    pcfg, tcfg, scfg = (from_args(c, args) for c in (PolicyConfig, TrainConfig, StyleConfig))

    src_style, tgt_style = (int(x) for x in scfg.direction.split("_to_"))
    if not tcfg.output_dir:
        tcfg.output_dir = f"outputs/yelp_{src_style}to{tgt_style}_{scfg.task_model}_seed{tcfg.seed}"
    tcfg.device = resolve_device(tcfg.device)
    os.makedirs(tcfg.output_dir, exist_ok=True)
    dump_configs(os.path.join(tcfg.output_dir, "config.json"), policy=pcfg, train=tcfg, task=scfg)
    set_seed(tcfg.seed)

    data = load_yelp(scfg.data_dir)
    rng = random.Random(0)
    train = data["train"][src_style]
    if scfg.max_train:
        train = rng.sample(train, min(scfg.max_train, len(train)))
    dev = rng.sample(data["dev"][src_style], min(scfg.dev_size, len(data["dev"][src_style])))
    test = data["test"][src_style][: scfg.max_test or None]

    dev_kw = dict(device=tcfg.device, fp16=tcfg.fp16_task_lm)
    gen = PromptedGenerator(scfg.task_model, top_k=scfg.gen_top_k,
                            batch_size=scfg.task_batch_size, **dev_kw)
    content = ContentScorer(scfg.content_model, scfg.content_layer, **dev_kw)
    style = ClassifierScorer(scfg.style_model, **dev_kw)
    task = StyleTransferTask(gen, content, style, train, dev, tgt_style, scfg)
    policy = PromptPolicy(pcfg, device=tcfg.device)

    best = Trainer(policy, task, tcfg).train()

    # Final test evaluation with the best checkpoint.
    policy.load(os.path.join(tcfg.output_dir, "best_policy.pt"))
    outs, prompts = task.transfer(policy, test, scfg.eval_samples)
    fluency = ClassifierScorer(scfg.fluency_model, **dev_kw)
    metrics = style_transfer_metrics(test, outs, content, style, fluency, tgt_style,
                                     scfg.fluency_label)
    with open(os.path.join(tcfg.output_dir, "test_outputs.tsv"), "w") as fh:
        fh.write("input\tprompt\toutput\n")
        for x, p, y in zip(test, prompts, outs):
            fh.write(f"{x}\t{p}\t{y}\n")
    results = {"best_step": best["step"], "dev_score": best["score"], "test": metrics}
    save_json(results, os.path.join(tcfg.output_dir, "results.json"))
    print(results)

    del fluency
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
