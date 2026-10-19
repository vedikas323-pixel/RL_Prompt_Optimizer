"""SQLite database for the app.

Two tables:
  runs       one row per optimisation: inputs, best prompt, scores, status
  api_cache  model answers keyed by a hash of (provider, model, full prompt),
             so the same query is never paid for twice

API keys are never written to the database.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT,
    provider         TEXT,
    model            TEXT,
    instruction      TEXT,
    labels           TEXT,
    n_examples       INTEGER,
    settings         TEXT,
    status           TEXT,
    learned_tokens   TEXT,
    optimized_prompt TEXT,
    baseline_dev     REAL,
    optimized_dev    REAL,
    baseline_test    REAL,
    optimized_test   REAL,
    api_calls        INTEGER,
    minutes          REAL
);
CREATE TABLE IF NOT EXISTS api_cache (
    key        TEXT PRIMARY KEY,
    answer     TEXT,
    created_at TEXT
);
"""


class Store:
    def __init__(self, path: str = "rlprompt_app.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    # ------------------------------------------------------------ cache
    @staticmethod
    def _key(provider: str, model: str, text: str) -> str:
        return hashlib.sha256(f"{provider}\x00{model}\x00{text}".encode()).hexdigest()

    def cache_get(self, provider: str, model: str, text: str) -> Optional[str]:
        with self.lock:
            row = self.conn.execute("SELECT answer FROM api_cache WHERE key=?",
                                    (self._key(provider, model, text),)).fetchone()
        return row[0] if row else None

    def cache_put(self, provider: str, model: str, text: str, answer: str) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO api_cache VALUES (?,?,?)",
                              (self._key(provider, model, text), answer, _now()))
            self.conn.commit()

    # ------------------------------------------------------------ runs
    def create_run(self, provider: str, model: str, instruction: str, labels: List[str],
                   n_examples: int, settings: Dict) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO runs (created_at, provider, model, instruction, labels, n_examples,"
                " settings, status) VALUES (?,?,?,?,?,?,?,?)",
                (_now(), provider, model, instruction, json.dumps(labels), n_examples,
                 json.dumps(settings), "running"))
            self.conn.commit()
            return cur.lastrowid

    def update_run(self, run_id: int, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.lock:
            self.conn.execute(f"UPDATE runs SET {cols} WHERE id=?", (*fields.values(), run_id))
            self.conn.commit()

    def list_runs(self, limit: int = 50) -> List[Dict]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT id, created_at, provider, model, status, optimized_prompt, baseline_dev,"
                " optimized_dev, baseline_test, optimized_test, api_calls, minutes"
                " FROM runs ORDER BY id DESC LIMIT ?", (limit,))
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
