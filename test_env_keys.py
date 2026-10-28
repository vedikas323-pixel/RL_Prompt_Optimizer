"""Diagnostic self-test script for all API keys configured in .env."""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def mask(k: str) -> str:
    if not k:
        return "<EMPTY>"
    k = k.strip().strip("'\"")
    if len(k) <= 10:
        return k[:3] + "..."
    return k[:6] + "..." + k[-4:]

def clean(k: str) -> str:
    if not k:
        return ""
    return k.strip().strip("'\"")

print("=" * 65, flush=True)
print("TESTING API KEYS CONFIGURED IN .env", flush=True)
print("=" * 65, flush=True)

# 1. GROQ_API_KEY
groq_key = clean(os.environ.get("GROQ_API_KEY", ""))
print(f"\n[1] GROQ_API_KEY: {mask(groq_key)}", flush=True)
if not groq_key:
    print("    Result: SKIPPED (not set in .env)", flush=True)
else:
    try:
        from openai import OpenAI
        groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            timeout=10.0,
            max_retries=0
        )
        resp = groq_client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": "Respond with 'GROQ_SUCCESS'"}],
            max_tokens=10
        )
        print("    Status: [HEALTHY / WORKING]", flush=True)
        print(f"    Response: {resp.choices[0].message.content.strip()}", flush=True)
    except Exception as e:
        print("    Status: [FAILED]", flush=True)
        print(f"    Error: {type(e).__name__}: {e}", flush=True)

# 2. GEMINI_API_KEY
gemini_key = clean(os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", ""))
print(f"\n[2] GEMINI_API_KEY: {mask(gemini_key)}", flush=True)
if not gemini_key:
    print("    Result: SKIPPED (not set in .env)", flush=True)
else:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=gemini_key)
        resp = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Respond with 'GEMINI_SUCCESS'",
            config={"max_output_tokens": 10}
        )
        print("    Status: [HEALTHY / WORKING]", flush=True)
        print(f"    Response: {(resp.text or '').strip()}", flush=True)
    except Exception as e:
        try:
            resp = gemini_client.models.generate_content(
                model="gemini-3.5-flash",
                contents="Respond with 'GEMINI_SUCCESS'",
                config={"max_output_tokens": 10}
            )
            print("    Status: [HEALTHY / WORKING with gemini-3.5-flash]", flush=True)
            print(f"    Response: {(resp.text or '').strip()}", flush=True)
        except Exception as e2:
            print("    Status: [FAILED]", flush=True)
            print(f"    Error: {type(e2).__name__}: {e2}", flush=True)

# 3. CLAUDE_API_KEY (Anthropic)
claude_key = clean(os.environ.get("CLAUDE_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", ""))
print(f"\n[3] CLAUDE_API_KEY (Anthropic): {mask(claude_key)}", flush=True)
if not claude_key:
    print("    Result: SKIPPED (not set in .env)", flush=True)
else:
    try:
        import anthropic
        claude_client = anthropic.Anthropic(
            api_key=claude_key,
            timeout=10.0,
            max_retries=0
        )
        resp = claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Respond with 'CLAUDE_SUCCESS'"}]
        )
        content = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        print("    Status: [HEALTHY / WORKING]", flush=True)
        print(f"    Response: {content}", flush=True)
    except Exception as e:
        print("    Status: [FAILED]", flush=True)
        print(f"    Error: {type(e).__name__}: {e}", flush=True)

# 4. CHAT_API (OpenAI)
openai_key = clean(os.environ.get("CHAT_API", "") or os.environ.get("OPENAI_API_KEY", ""))
print(f"\n[4] CHAT_API (OpenAI): {mask(openai_key)}", flush=True)
if not openai_key:
    print("    Result: SKIPPED (not set in .env)", flush=True)
else:
    try:
        from openai import OpenAI
        openai_client = OpenAI(
            api_key=openai_key,
            timeout=10.0,
            max_retries=0
        )
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Respond with 'OPENAI_SUCCESS'"}],
            max_tokens=10
        )
        print("    Status: [HEALTHY / WORKING]", flush=True)
        print(f"    Response: {resp.choices[0].message.content.strip()}", flush=True)
    except Exception as e:
        print("    Status: [FAILED]", flush=True)
        print(f"    Error: {type(e).__name__}: {e}", flush=True)

# 5. DEEPSEEK_API
deepseek_key = clean(os.environ.get("DEEPSEEK_API", "") or os.environ.get("DEEPSEEK_API_KEY", ""))
print(f"\n[5] DEEPSEEK_API: {mask(deepseek_key)}", flush=True)
if not deepseek_key:
    print("    Result: SKIPPED (not set in .env)", flush=True)
else:
    try:
        from openai import OpenAI
        deepseek_client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=deepseek_key,
            timeout=10.0,
            max_retries=0
        )
        resp = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Respond with 'DEEPSEEK_SUCCESS'"}],
            max_tokens=10
        )
        print("    Status: [HEALTHY / WORKING]", flush=True)
        print(f"    Response: {resp.choices[0].message.content.strip()}", flush=True)
    except Exception as e:
        print("    Status: [FAILED]", flush=True)
        print(f"    Error: {type(e).__name__}: {e}", flush=True)

# 6. NIMA_API (NVIDIA NIM)
nim_key = clean(os.environ.get("NIMA_API", "") or os.environ.get("NIM_API", "") or os.environ.get("NVIDIA_API_KEY", ""))
print(f"\n[6] NIMA_API (NVIDIA NIM): {mask(nim_key)}", flush=True)
if not nim_key:
    print("    Result: SKIPPED (not set in .env)", flush=True)
else:
    try:
        from openai import OpenAI
        nim_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nim_key,
            timeout=10.0,
            max_retries=0
        )
        models = [m.id for m in nim_client.models.list().data]
        print(f"    Status: [CONNECTED] Found {len(models)} models on NVIDIA NIM catalog.")
    except Exception as e:
        print("    Status: [FAILED]", flush=True)
        print(f"    Error: {type(e).__name__}: {e}", flush=True)

print("\n" + "=" * 65, flush=True)
print("TESTING COMPLETE", flush=True)
print("=" * 65, flush=True)
