"""Train an RLPrompt policy for few-shot classification, then test the best prompt.

Example:
    python scripts/train_classification.py --dataset sst2 --prompt_length 2 \
        --output_dir outputs/sst2_t2_s0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlprompt.config import (ClassificationConfig, PolicyConfig, TrainConfig,
                             add_dataclass_args, dump_configs, from_args)
from rlprompt.policy import PromptPolicy
from rlprompt.tasks.classification import (DATASETS, ClassificationTask,
                                           PromptedClassifier, load_fewshot)
from rlprompt.trainer import Trainer
from rlprompt.utils import resolve_device, save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    for cls in (PolicyConfig, TrainConfig, ClassificationConfig):
        add_dataclass_args(parser, cls)
    parser.add_argument("--quick", action="store_true", help="Run a quick optimization (500 steps, fast model)")
    parser.set_defaults(max_steps=-1, output_dir="")
    args = parser.parse_args()

    pcfg, tcfg, ccfg = (from_args(c, args) for c in (PolicyConfig, TrainConfig, ClassificationConfig))
    tcfg.device = resolve_device(tcfg.device)

    # If running on CPU or --quick is passed, apply fast defaults to avoid hours of slow execution
    is_cpu = tcfg.device == "cpu"
    if args.quick or is_cpu:
        if tcfg.max_steps < 0:
            tcfg.max_steps = 500
        if is_cpu and ccfg.task_model == "roberta-large":
            print("[info] CPU detected: switching task_model from roberta-large to distilroberta-base for ~6x faster execution.")
            ccfg.task_model = "distilroberta-base"
        if not ccfg.examples_per_step:
            ccfg.examples_per_step = 8
        if not ccfg.max_test:
            ccfg.max_test = 500
    elif tcfg.max_steps < 0:                       # paper: 6K steps for T=2, 12K for T=5
        tcfg.max_steps = 6000 if pcfg.prompt_length <= 2 else 12000

    if not tcfg.output_dir:
        tcfg.output_dir = f"outputs/{ccfg.dataset}_T{pcfg.prompt_length}_split{ccfg.split_seed}_seed{tcfg.seed}"
    os.makedirs(tcfg.output_dir, exist_ok=True)
    dump_configs(os.path.join(tcfg.output_dir, "config.json"),
                 policy=pcfg, train=tcfg, task=ccfg)
    set_seed(tcfg.seed)

    spec = DATASETS[ccfg.dataset]
    train, dev, test = load_fewshot(ccfg.dataset, ccfg.shots, ccfg.split_seed, ccfg.max_test)
    print(f"{ccfg.dataset}: {len(train)} train, {len(dev)} dev, {len(test)} test")

    clf = PromptedClassifier(ccfg.task_model, spec.verbalizers, spec.template,
                             device=tcfg.device, fp16=tcfg.fp16_task_lm,
                             batch_size=ccfg.task_batch_size)
    task = ClassificationTask(clf, train, dev, ccfg)
    policy = PromptPolicy(pcfg, device=tcfg.device)

    best = Trainer(policy, task, tcfg).train()

    results = {
        "best_prompt": best["prompt"], "best_step": best["step"], "dev_acc": best["score"],
        "test_acc": clf.accuracy(best["prompt"], test),
        "manual_prompt": spec.manual_prompt, "manual_test_acc": clf.accuracy(spec.manual_prompt, test),
        "null_prompt_test_acc": clf.accuracy("", test),
    }
    save_json(results, os.path.join(tcfg.output_dir, "results.json"))
    print(results)


if __name__ == "__main__":
    main()
