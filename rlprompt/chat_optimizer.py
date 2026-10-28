"""Fast, per-message RL prompt optimisation for the chat UI.

Full RLPrompt (see optimizer.py) needs labelled examples and hundreds of API
calls. A chat message has neither, so this module uses a lighter RL loop that
runs in seconds:

    policy   = softmax (Boltzmann) distribution over rewrite *strategies*,
               with one learned value Q(s) per strategy, persisted in SQLite
    actions  = the policy samples K strategies; the LLM rewrites the prompt
               once per strategy, giving K candidate prompts
    selection= an LLM judge scores the original + every candidate; the best
               one is sent (the original can win, so we never make it worse)
    reward   = after both prompts run, a blinded judge scores both responses;
               reward = (optimised score - original score) / 10, in [-1, 1]
    update   = Q(s) <- Q(s) + alpha * (reward - Q(s)) for the chosen strategy

Over many messages the policy learns which rewrite strategies actually make
the model's answers better, and samples them more often.
"""
from __future__ import annotations

import json
import math
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from .llm_clients import make_client

STRATEGIES: Dict[str, Dict[str, str]] = {
    "role_expert": {
        "label": "Expert role",
        "how": "Give the model a fitting expert role or persona for this task, then state the task clearly.",
    },
    "structured": {
        "label": "Structured sections",
        "how": "Reorganise the request into short labelled sections: Task, Context, Requirements, Output format.",
    },
    "constraints": {
        "label": "Specific constraints",
        "how": "Add concrete, reasonable constraints inferred from the request: scope, audience, length, tone.",
    },
    "step_by_step": {
        "label": "Step-by-step reasoning",
        "how": "Ask the model to work through the problem step by step and check its answer before finishing.",
    },
    "clarify": {
        "label": "Concise & unambiguous",
        "how": "Make the request concise and unambiguous: remove filler and replace vague words with precise ones.",
    },
    "output_format": {
        "label": "Explicit output format",
        "how": "Specify exactly what the answer should look like (headings, bullets, table, code block, etc.).",
    },
}

TEMPERATURE = 0.35      # softmax temperature over Q values
MIN_ALPHA = 0.15        # floor on the learning rate so the policy keeps adapting
CANDIDATES = 3          # strategies sampled per message


# ---------------------------------------------------------------------------
# Policy storage
# ---------------------------------------------------------------------------

class StrategyPolicy:
    """Q value and visit count per strategy, stored in SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS chat_strategy_stats (
                            strategy TEXT PRIMARY KEY,
                            q REAL NOT NULL DEFAULT 0,
                            n INTEGER NOT NULL DEFAULT 0,
                            total_reward REAL NOT NULL DEFAULT 0)""")
            for s in STRATEGIES:
                c.execute("INSERT OR IGNORE INTO chat_strategy_stats(strategy) VALUES (?)", (s,))

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def stats(self) -> Dict[str, Dict]:
        with self._conn() as c:
            rows = c.execute("SELECT strategy, q, n, total_reward FROM chat_strategy_stats").fetchall()
        qs = {r[0]: r[1] for r in rows if r[0] in STRATEGIES}
        probs = _softmax(qs)
        return {r[0]: {"label": STRATEGIES[r[0]]["label"], "q": round(r[1], 4), "n": r[2],
                       "prob": round(probs[r[0]], 4)}
                for r in rows if r[0] in STRATEGIES}

    def sample(self, k: int) -> List[str]:
        """Sample k distinct strategies without replacement from softmax(Q / T)."""
        weights = _softmax({s: v["q"] for s, v in self.stats().items()})
        pool = dict(weights)
        chosen = []
        for _ in range(min(k, len(pool))):
            total = sum(pool.values())
            r, acc = random.random() * total, 0.0
            for s, w in pool.items():
                acc += w
                if acc >= r:
                    chosen.append(s)
                    pool.pop(s)
                    break
        return chosen

    def update(self, strategy: str, reward: float) -> Dict:
        if strategy not in STRATEGIES:
            return {}
        reward = max(-1.0, min(1.0, reward))
        with self._lock, self._conn() as c:
            q, n = c.execute("SELECT q, n FROM chat_strategy_stats WHERE strategy=?",
                             (strategy,)).fetchone()
            n += 1
            alpha = max(MIN_ALPHA, 1.0 / n)
            new_q = q + alpha * (reward - q)
            c.execute("UPDATE chat_strategy_stats SET q=?, n=?, total_reward=total_reward+? "
                      "WHERE strategy=?", (new_q, n, reward, strategy))
        return {"strategy": strategy, "old_q": round(q, 4), "new_q": round(new_q, 4),
                "reward": round(reward, 4), "n": n}

    def reset(self):
        with self._conn() as c:
            c.execute("UPDATE chat_strategy_stats SET q=0, n=0, total_reward=0")


def _softmax(qs: Dict[str, float]) -> Dict[str, float]:
    if not qs:
        return {}
    m = max(qs.values())
    ex = {s: math.exp((q - m) / TEMPERATURE) for s, q in qs.items()}
    z = sum(ex.values())
    return {s: e / z for s, e in ex.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Drop <think> blocks some reasoning models (e.g. Qwen on Groq) emit."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.strip()


def _parse_json(text: str):
    text = _clean(text)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                continue
    raise ValueError("Model did not return valid JSON")


def approx_tokens(text: str) -> int:
    return max(1, round(len(text or "") / 4)) if text else 0


def _client(provider: str, model: str, max_tokens: int):
    c = make_client(provider, model)
    # Groq on-demand tier caps OTPM at 1,000 output tokens per request
    if provider == "Groq":
        max_tokens = min(max_tokens, 650)
    c.max_output_tokens = max_tokens
    c.max_retries = 3
    return c


# ---------------------------------------------------------------------------
# Stage 1: optimise the prompt (before it reaches the model)
# ---------------------------------------------------------------------------

REWRITE_TEMPLATE = """You are a prompt engineer. Rewrite the USER PROMPT below once for each strategy listed, so that an AI assistant would give a better answer.

Rules:
- Keep the user's intent, language and every fact they gave. Do not invent new facts or requirements that change what they asked.
- Do NOT answer the prompt. Only rewrite it.
- Write each rewrite as if the user were sending it directly to the assistant.
- Keep it proportionate: a short casual question should stay fairly short.

Strategies:
{strategies}

USER PROMPT:
\"\"\"{prompt}\"\"\"

Return ONLY JSON in this exact shape, with one item per strategy in the same order:
{{"candidates": [{{"strategy": "<strategy id>", "prompt": "<rewritten prompt>"}}]}}"""

PROMPT_JUDGE_TEMPLATE = """You are grading prompts that will be sent to an AI assistant. The user's original request is ORIGINAL. Score every prompt from 0 to 10 for how likely it is to get an excellent, correct, useful answer, considering:
- intent preservation (it must still ask for what ORIGINAL asked; changing the request scores low)
- clarity and specificity
- appropriate structure and length for the request

ORIGINAL:
\"\"\"{original}\"\"\"

PROMPTS:
{prompts}

Return ONLY JSON: {{"scores": [<score for prompt 0>, <score for prompt 1>, ...], "reason": "<one short sentence about the best prompt>"}}"""


def optimize_prompt(prompt: str, provider: str, model: str, policy: StrategyPolicy) -> Dict:
    t0 = time.time()
    strategies = policy.sample(CANDIDATES)
    client = _client(provider, model, 2048)

    listing = "\n".join(f"- {s}: {STRATEGIES[s]['how']}" for s in strategies)
    raw = client.complete(REWRITE_TEMPLATE.format(strategies=listing, prompt=prompt))
    data = _parse_json(raw)
    items = data.get("candidates", data) if isinstance(data, dict) else data

    candidates = []
    for i, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("prompt", "")).strip()
        strat = str(item.get("strategy", "")).strip()
        if strat not in STRATEGIES:
            strat = strategies[i] if i < len(strategies) else "clarify"
        if text and text != prompt:
            candidates.append({"strategy": strat, "label": STRATEGIES[strat]["label"], "prompt": text})
    if not candidates:
        raise ValueError("The model returned no usable rewrites.")

    # Score original (index 0) + candidates with the judge.
    allp = [{"strategy": "original", "label": "Original", "prompt": prompt}] + candidates
    listing = "\n\n".join(f"[{i}]\n\"\"\"{c['prompt']}\"\"\"" for i, c in enumerate(allp))
    judge = _client(provider, model, 1024)
    reason = ""
    try:
        jd = _parse_json(judge.complete(PROMPT_JUDGE_TEMPLATE.format(original=prompt, prompts=listing)))
        scores = jd.get("scores", []) if isinstance(jd, dict) else jd
        reason = jd.get("reason", "") if isinstance(jd, dict) else ""
    except Exception:
        scores = []
    for i, c in enumerate(allp):
        try:
            c["score"] = round(float(scores[i]), 1)
        except Exception:
            c["score"] = None

    scored = [c for c in allp if c["score"] is not None]
    best = max(scored, key=lambda c: c["score"]) if scored else allp[1]

    return {
        "original": prompt,
        "optimized": best["prompt"],
        "strategy": best["strategy"],
        "strategy_label": best["label"],
        "original_score": allp[0]["score"],
        "optimized_score": best["score"],
        "candidates": allp,
        "sampled_strategies": strategies,
        "reason": reason,
        "unchanged": best["strategy"] == "original",
        "seconds": round(time.time() - t0, 2),
        "policy": policy.stats(),
    }


# ---------------------------------------------------------------------------
# Stage 2: run both prompts and measure efficiency
# ---------------------------------------------------------------------------

RESPONSE_JUDGE_TEMPLATE = """You are an impartial judge. A user asked:
\"\"\"{request}\"\"\"

Two assistant responses follow. Grade EACH from 0 to 10 on:
- relevance: answers what the user actually asked
- completeness: covers what is needed, nothing important missing
- accuracy: factually and logically correct
- clarity: well organised and easy to read

RESPONSE A:
\"\"\"{a}\"\"\"

RESPONSE B:
\"\"\"{b}\"\"\"

Return ONLY JSON:
{{"A": {{"relevance": n, "completeness": n, "accuracy": n, "clarity": n}},
  "B": {{"relevance": n, "completeness": n, "accuracy": n, "clarity": n}},
  "verdict": "<one short sentence comparing them>"}}"""

CRITERIA = ["relevance", "completeness", "accuracy", "clarity"]


def _timed_run(provider: str, model: str, prompt: str) -> Dict:
    client = _client(provider, model, 2048)
    t0 = time.time()
    try:
        text = _clean(client.complete(prompt))
        err = None
    except Exception as e:                      # surface errors per side
        text, err = "", f"{type(e).__name__}: {e}"
    return {"text": text, "error": err, "latency": round(time.time() - t0, 2),
            "prompt_tokens": approx_tokens(prompt), "response_tokens": approx_tokens(text)}


def execute_and_compare(original: str, optimized: str, provider: str, model: str,
                        strategy: Optional[str], policy: StrategyPolicy) -> Dict:
    same = original.strip() == optimized.strip()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_opt = ex.submit(_timed_run, provider, model, optimized)
        f_orig = None if same else ex.submit(_timed_run, provider, model, original)
        opt = f_opt.result()
        orig = opt if same else f_orig.result()

    scores = {"original": None, "optimized": None}
    verdict = ""
    if not opt["error"] and not orig["error"]:
        if same:
            pairs = [("A", "optimized"), ("B", "original")]
            a_text = b_text = opt["text"]
        else:
            # Randomise A/B order so the judge can't favour a position.
            flip = random.random() < 0.5
            pairs = [("A", "optimized"), ("B", "original")] if flip else \
                    [("A", "original"), ("B", "optimized")]
            texts = {"optimized": opt["text"], "original": orig["text"]}
            a_text, b_text = texts[pairs[0][1]], texts[pairs[1][1]]
        try:
            judge = _client(provider, model, 1024)
            jd = _parse_json(judge.complete(RESPONSE_JUDGE_TEMPLATE.format(
                request=original, a=a_text[:12000], b=b_text[:12000])))
            verdict = jd.get("verdict", "")
            for slot, name in pairs:
                crit = {k: float(jd[slot].get(k, 0)) for k in CRITERIA}
                crit["overall"] = round(sum(crit.values()) / len(CRITERIA), 2)
                scores[name] = crit
            if same:
                scores["original"] = scores["optimized"]
        except Exception as e:
            verdict = f"Judge failed: {e}"

    learning = {}
    if strategy and not same and scores["original"] and scores["optimized"]:
        reward = (scores["optimized"]["overall"] - scores["original"]["overall"]) / 10.0
        learning = policy.update(strategy, reward)

    return {"original": orig, "optimized": opt, "scores": scores, "verdict": verdict,
            "same_prompt": same, "learning": learning, "policy": policy.stats()}
