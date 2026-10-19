"""Offline tests: run with `python -m pytest tests -q` or `python tests/test_rlprompt.py`."""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from rlprompt.config import ClassificationConfig, PolicyConfig, StyleConfig, TrainConfig
from rlprompt.losses import reinforce_loss, sql_loss
from rlprompt.policy import PromptPolicy
from rlprompt.tasks.classification import ClassificationTask, PromptedClassifier
from rlprompt.tasks.style_transfer import (ClassifierScorer, ContentScorer, PromptedGenerator,
                                           StyleTransferTask, style_transfer_metrics)
from rlprompt.trainer import Trainer
from rlprompt.utils import zscore_by_group

_TMP = tempfile.mkdtemp()
_PATHS = None


def paths():
    global _PATHS
    if _PATHS is None:
        from tiny_models import build
        _PATHS = build(os.path.join(_TMP, "models"))
    return _PATHS


def tiny_policy(T=3):
    cfg = PolicyConfig(policy_model=paths()["gpt2"], mlp_hidden=64, prompt_length=T, top_k=50)
    return PromptPolicy(cfg, device="cpu")


# ------------------------------------------------------------- losses

def _learn(loss_name, reward):
    torch.manual_seed(0)
    q = torch.zeros(1, 2, 5, requires_grad=True)
    target = q.detach().clone()
    actions = torch.tensor([[1, 3]])
    opt = torch.optim.SGD([q], lr=0.1)
    for _ in range(200):
        if loss_name == "sql":
            loss, _ = sql_loss(q, target, actions, torch.tensor([reward]))
        else:
            loss, _ = reinforce_loss(q, actions, torch.tensor([reward]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        target = 0.9 * target + 0.1 * q.detach()
    logp = torch.log_softmax(q, -1).gather(-1, actions[..., None]).sum()
    return logp.item()


def test_sql_moves_probability_toward_reward():
    uniform = 2 * torch.log(torch.tensor(1 / 5)).item()
    assert _learn("sql", 3.0) > uniform + 0.5
    assert _learn("sql", -3.0) < uniform - 0.5


def test_reinforce_moves_probability_toward_reward():
    uniform = 2 * torch.log(torch.tensor(1 / 5)).item()
    assert _learn("reinforce", 1.0) > uniform + 0.5


def test_zscore_by_group():
    r = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    g = torch.tensor([0, 0, 0, 1, 1, 1])
    z = zscore_by_group(r, g)
    for k in (0, 1):
        assert abs(z[g == k].mean().item()) < 1e-5
        assert abs(z[g == k].std(unbiased=False).item() - 1) < 1e-4


# ------------------------------------------------------------- policy

def test_policy_only_mlp_is_trainable():
    pol = tiny_policy()
    trainable = [n for n, p in pol.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith("mlp.") for n in trainable)


def test_forced_logits_match_sampling():
    pol = tiny_policy(T=4)
    sources = ["", "the food was bad", "i love this"]
    actions, prompts = pol.sample(sources, greedy=True)
    assert actions.shape == (3, 4) and len(prompts) == 3
    logits, target = pol.forced_logits(sources, actions)
    assert logits.shape[:2] == (3, 4)
    assert torch.equal(logits.argmax(-1), actions)
    assert logits.requires_grad and not target.requires_grad


# ------------------------------------------------------------- end-to-end

def test_classification_training_smoke():
    clf = PromptedClassifier(paths()["mlm"], [" terrible", " great"],
                             "{sentence} {prompt} {mask} .", device="cpu", fp16=False)
    data = [("the food was great", 1), ("the food was bad", 0),
            ("i love this place", 1), ("never going back", 0)]
    ccfg = ClassificationConfig(prompts_per_step=4)
    task = ClassificationTask(clf, data, data, ccfg)
    out = os.path.join(_TMP, "cls_run")
    tcfg = TrainConfig(max_steps=4, eval_every=2, log_every=1, device="cpu", output_dir=out)
    best = Trainer(tiny_policy(T=2), task, tcfg).train()
    assert 0.0 <= best["score"] <= 1.0
    assert os.path.exists(os.path.join(out, "best_prompt.json"))


def test_causal_classifier_for_transfer():
    clf = PromptedClassifier(paths()["gpt2"], [" terrible", " great"],
                             "{sentence} {prompt} {mask} .", device="cpu", fp16=False)
    probs = clf.class_probs(["the food was great", "bad"], ["It is", ""])
    assert probs.shape == (2, 2) and torch.allclose(probs.sum(-1), torch.ones(2))


def test_style_transfer_training_smoke():
    p = paths()
    gen = PromptedGenerator(p["gpt2"], device="cpu", fp16=False)
    content = ContentScorer(p["mlm"], layer=17, device="cpu", fp16=False)
    style = ClassifierScorer(p["cls"], device="cpu", fp16=False)
    sents = ["the food was bad", "never going back", "the staff is not good", "it was terrible"]
    scfg = StyleConfig(inputs_per_step=2, prompts_per_input=3, samples_per_prompt=4)
    task = StyleTransferTask(gen, content, style, sents, sents[:2], 1, scfg)
    out = os.path.join(_TMP, "style_run")
    tcfg = TrainConfig(max_steps=3, eval_every=3, log_every=1, device="cpu", output_dir=out)
    Trainer(tiny_policy(T=3), task, tcfg).train()

    sc = content.score(["the food was bad"], ["the food was bad"])
    assert sc.item() > 99.0, "identical sentences should align perfectly"
    m = style_transfer_metrics(sents, sents, content, style, style, 1)
    assert set(m) == {"content", "style", "fluency", "J", "GM"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
