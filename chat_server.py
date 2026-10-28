"""Chat-style RL prompt optimiser (ChatGPT / Gemini-like UI).

    python chat_server.py            # http://127.0.0.1:8000

Flow for every message:
    1. POST /api/optimize -> original vs RL-optimised prompt, shown BEFORE the model runs
    2. POST /api/execute  -> both prompts run; responses + before/after efficiency
Keys are read from .env (GEMINI_API_KEY, GROQ_API_KEY).
"""
import argparse
import os

from dotenv import load_dotenv

load_dotenv()

# .env values may be quoted / padded; normalise them once.
for _k in ("GEMINI_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY", "NIMA_API", "NIM_API", "NVIDIA_API_KEY"):
    if os.environ.get(_k):
        os.environ[_k] = os.environ[_k].strip().strip("'\"")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rlprompt.chat_optimizer import StrategyPolicy, execute_and_compare, optimize_prompt
from rlprompt.llm_clients import make_client

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
POLICY = StrategyPolicy(os.environ.get("RLPROMPT_DB", os.path.join(ROOT, "rlprompt_app.db")))

CHAT_MODELS = {
    "Gemini": ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"],
    "Groq": ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    "Nvidia": ["meta/llama-3.3-70b-instruct", "meta/llama-3.1-70b-instruct", "google/gemma-3-12b-it", "mistralai/mistral-large-2-instruct"],
}
KEY_ENV = {"Gemini": "GEMINI_API_KEY", "Groq": "GROQ_API_KEY", "Nvidia": "NIMA_API"}


def _has_key(provider: str) -> bool:
    if provider == "Nvidia":
        return bool(os.environ.get("NIMA_API") or os.environ.get("NIM_API") or os.environ.get("NVIDIA_API_KEY"))
    return bool(os.environ.get(KEY_ENV.get(provider, "")))


app = FastAPI(title="RL Prompt Optimizer Chat")


class OptimizeIn(BaseModel):
    prompt: str
    provider: str
    model: str


class ExecuteIn(BaseModel):
    original: str
    optimized: str
    provider: str
    model: str
    strategy: str | None = None


def _check(provider: str, model: str):
    if provider not in CHAT_MODELS:
        raise HTTPException(400, f"Unsupported provider: {provider}")
    if not _has_key(provider):
        raise HTTPException(400, f"{KEY_ENV[provider]} is not set in .env")
    if not model.strip():
        raise HTTPException(400, "Model is required")


@app.get("/api/models")
def models():
    return {"providers": [
        {"name": p, "models": m, "has_key": _has_key(p)}
        for p, m in CHAT_MODELS.items()]}


@app.get("/api/models/{provider}/live")
def live_models(provider: str):
    _check(provider, "x")
    try:
        return {"models": make_client(provider, CHAT_MODELS[provider][0]).list_models()}
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")


@app.get("/api/policy")
def policy():
    return POLICY.stats()


@app.post("/api/policy/reset")
def policy_reset():
    POLICY.reset()
    return POLICY.stats()


@app.post("/api/optimize")
def optimize(body: OptimizeIn):
    _check(body.provider, body.model)
    if not body.prompt.strip():
        raise HTTPException(400, "Prompt is empty")
    try:
        return optimize_prompt(body.prompt.strip(), body.provider, body.model, POLICY)
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")


@app.post("/api/execute")
def execute(body: ExecuteIn):
    _check(body.provider, body.model)
    try:
        return execute_and_compare(body.original, body.optimized, body.provider,
                                   body.model, body.strategy, POLICY)
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


app.mount("/static", StaticFiles(directory=WEB), name="static")


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
