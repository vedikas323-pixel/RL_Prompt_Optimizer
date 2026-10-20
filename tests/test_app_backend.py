"""Offline tests for the app backend: parsing, SQLite store, and a full
optimisation against a fake API model that only answers well when a
particular word is in the prompt."""
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rlprompt.optimizer as optimizer
from rlprompt.llm_clients import LLMClient
from rlprompt.store import Store
from rlprompt.tasks.api_classification import parse_examples, parse_label, split_examples

POS = ["great fun", "i love this", "the staff is good", "very good food", "love the place",
       "good service", "great staff", "i love it"]
NEG = ["bad food", "never going back", "not good", "terrible service", "the place is bad",
       "bad staff", "terrible food", "not going back"]
EXAMPLES = "\n".join([f"positive | {t}" for t in POS] + [f"negative | {t}" for t in NEG])


class FakeClient(LLMClient):
    """Right every time if the word 'business' is in the instruction line, else always says negative."""
    provider = "Fake"

    def _complete(self, text):
        head = text.split("\n")[0]
        m = re.search(r"(?:Text|Input):\s*(.*)", text)
        sentence = m.group(1) if m else ""
        truth = "positive" if sentence in POS else "negative"
        if "business" in head.lower():
            return truth
        return "negative"


def test_parsing():
    labels, ex = parse_examples("pos | a b\nneg|c\n\nbad line\npos | d")
    assert labels == ["pos", "neg"] and ex == [("a b", 0), ("c", 1), ("d", 0)]
    assert parse_label("Answer: Negative.", ["positive", "negative"]) == 1
    assert parse_label("not sure", ["positive", "negative"]) is None
    assert parse_label("very positive", ["positive", "very positive"]) == 1
    tr, dv, te = split_examples(parse_examples(EXAMPLES)[1])
    assert len(tr) + len(dv) + len(te) == 16 and len(dv) == 4 and len(te) == 4


def test_store_cache_and_runs():
    s = Store(os.path.join(tempfile.mkdtemp(), "t.db"))
    s.cache_put("p", "m", "q", "a")
    assert s.cache_get("p", "m", "q") == "a" and s.cache_get("p", "m2", "q") is None
    rid = s.create_run("p", "m", "inst", ["a", "b"], 10, {"steps": 1})
    s.update_run(rid, status="done", optimized_dev=0.9)
    assert s.list_runs()[0]["optimized_dev"] == 0.9


def test_optimizer_finds_magic_token(monkeypatch=None):
    from tiny_models import build
    paths = build(os.path.join(tempfile.mkdtemp(), "models"))
    optimizer.make_client = lambda provider, model, key=None: FakeClient(model)
    store = Store(os.path.join(tempfile.mkdtemp(), "app.db"))
    req = optimizer.OptimizeRequest(
        provider="Fake", model="fake-1", instruction="Classify the sentiment.",
        examples_text=EXAMPLES, steps=40, prompt_length=1, prompts_per_step=16,
        examples_per_step=8, readable=False, learning_rate=1e-3, eval_every=10,
        workers=4, policy_model=paths["gpt2"], device="cpu",
        runs_dir=tempfile.mkdtemp())
    records = []
    res = optimizer.run_optimization(req, store, records.append)
    assert res["baseline_dev"] == 0.5
    assert "business" in res["optimized_prompt"].lower() and res["optimized_dev"] == 1.0
    assert any(r.get("phase") == "eval" for r in records)
    assert store.list_runs()[0]["status"] == "done"
    print("baseline", res["baseline_dev"], "->", res["optimized_dev"], repr(res["optimized_prompt"]),
          "calls", res["api_calls"])


if __name__ == "__main__":
    test_parsing()
    test_store_cache_and_runs()
    test_optimizer_finds_magic_token()
    print("ok")
