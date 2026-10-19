"""Thin, uniform wrappers around the OpenAI, Gemini and Anthropic APIs.

Every client exposes `complete(text) -> str`. The task LM is still a black
box: RLPrompt only needs the model's answer, never its gradients or weights.
Answers are cached in SQLite, so re-scoring a prompt you already tried costs
nothing, including across runs.
"""
from __future__ import annotations

import os
import random
import time
from typing import List, Optional

# Defaults shown in the dropdowns. "Refresh models" in the app asks the
# provider for the live list, and any model ID can be typed in by hand.
DEFAULT_MODELS = {
    "OpenAI": ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.6-luna", "gpt-5.6-terra",
               "gpt-5.6-sol", "gpt-5.5", "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol"],
    "Gemini": ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-3.5-flash-lite",
               "gemini-3.5-flash", "gemini-3.6-flash", "gemini-2.5-pro"],
    "Claude": ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"],
}
KEY_ENV = {"OpenAI": "OPENAI_API_KEY", "Gemini": "GEMINI_API_KEY", "Claude": "ANTHROPIC_API_KEY"}


class LLMClient:
    provider = "base"

    def __init__(self, model: str, api_key: Optional[str] = None, max_retries: int = 6,
                 max_output_tokens: int = 512):
        self.model = model
        self.api_key = api_key or os.environ.get(KEY_ENV.get(self.provider, ""), "") or None
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.calls = 0

    def _complete(self, text: str) -> str:
        raise NotImplementedError

    def complete(self, text: str) -> str:
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                return self._complete(text) or ""
            except Exception as err:                     # rate limits, timeouts, 5xx
                if attempt == self.max_retries - 1 or _is_fatal(err):
                    raise
                time.sleep(min(30, 2 ** attempt) + random.random())
        return ""

    def list_models(self) -> List[str]:
        return DEFAULT_MODELS.get(self.provider, [])


def _is_fatal(err: Exception) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in ("api key", "api_key", "unauthorized", "permission",
                                  "not found", "does not exist", "invalid model", "401", "403", "404"))


class OpenAIClient(LLMClient):
    provider = "OpenAI"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self._reasoning = True          # dropped automatically for models that reject it

    def _complete(self, text: str) -> str:
        kw = dict(model=self.model, input=text, max_output_tokens=self.max_output_tokens)
        if self._reasoning:
            try:
                return self.client.responses.create(reasoning={"effort": "low"}, **kw).output_text
            except Exception as err:
                if "reasoning" not in str(err).lower():
                    raise
                self._reasoning = False
        return self.client.responses.create(**kw).output_text

    def list_models(self) -> List[str]:
        ids = sorted(m.id for m in self.client.models.list())
        return [i for i in ids if i.startswith(("gpt", "o")) and not any(
            s in i for s in ("image", "audio", "realtime", "transcribe", "tts", "embedding", "search"))]


class GeminiClient(LLMClient):
    provider = "Gemini"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from google import genai
        key = self.api_key or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=key)

    def _complete(self, text: str) -> str:
        r = self.client.models.generate_content(
            model=self.model, contents=text,
            config={"max_output_tokens": self.max_output_tokens})
        return r.text or ""

    def list_models(self) -> List[str]:
        out = []
        for m in self.client.models.list():
            name = m.name.split("/")[-1]
            if name.startswith("gemini") and not any(s in name for s in ("embedding", "tts", "live", "image")):
                out.append(name)
        return sorted(out)


class ClaudeClient(LLMClient):
    provider = "Claude"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _complete(self, text: str) -> str:
        r = self.client.messages.create(model=self.model, max_tokens=self.max_output_tokens,
                                        messages=[{"role": "user", "content": text}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

    def list_models(self) -> List[str]:
        return sorted(m.id for m in self.client.models.list())


CLIENTS = {"OpenAI": OpenAIClient, "Gemini": GeminiClient, "Claude": ClaudeClient}


def make_client(provider: str, model: str, api_key: Optional[str] = None) -> LLMClient:
    return CLIENTS[provider](model, api_key=api_key)
