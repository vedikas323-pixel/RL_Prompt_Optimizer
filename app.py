"""RLPrompt web app.

    python app.py              # local, http://127.0.0.1:7860
    python app.py --share      # public link, e.g. from Colab

Type an instruction, paste labelled examples, pick a model, and RLPrompt
learns extra prompt tokens that raise the model's accuracy on your task.
"""
import argparse
import os
import queue
import threading

import gradio as gr
import pandas as pd

from rlprompt.llm_clients import DEFAULT_MODELS, make_client
from rlprompt.optimizer import (LOCAL_MODELS, PROVIDERS, OptimizeRequest, estimate_calls,
                                run_optimization)
from rlprompt.prompt_builder import TASK_TYPES, BuildRequest, build_prompt
from rlprompt.store import Store
from rlprompt.tasks.api_classification import parse_examples, split_examples

STORE = Store(os.environ.get("RLPROMPT_DB", "rlprompt_app.db"))
STOP_EVENTS = {}

SAMPLE_INSTRUCTION = "Classify the sentiment of this movie review."
SAMPLE_EXAMPLES = """positive | A warm, funny film that never runs out of surprises.
positive | The lead performance is magnetic from the first scene to the last.
positive | I walked out smiling and want to see it again.
positive | Gorgeous photography and a score that lifts every moment.
positive | Smart writing that trusts its audience.
positive | One of the most moving stories I have seen this year.
positive | The cast clearly had a great time and it shows.
positive | Tight pacing and a genuinely clever twist.
positive | A small film with a very big heart.
positive | Every joke lands and the ending is perfect.
positive | Beautifully acted and quietly devastating.
positive | Pure fun from start to finish.
negative | Two hours I will never get back.
negative | The plot falls apart halfway through and never recovers.
negative | Flat characters and dialogue that feels written by a committee.
negative | I checked my watch more than I watched the screen.
negative | A lazy sequel that repeats every beat of the original.
negative | The jokes are stale and the pacing drags.
negative | Loud, messy and strangely boring.
negative | Even a strong cast cannot rescue this script.
negative | The twist is obvious from the first ten minutes.
negative | Clumsy editing ruins what few good scenes there are.
negative | A forgettable film that wastes its premise.
negative | I wanted to like it but it is simply dull."""


def model_choices(provider):
    return LOCAL_MODELS if provider == "Local" else DEFAULT_MODELS[provider]


def on_provider(provider):
    choices = model_choices(provider)
    return (gr.Dropdown(choices=choices, value=choices[0]),
            gr.Textbox(visible=provider != "Local"))


def on_refresh(provider, model, key):
    if provider == "Local":
        return gr.Dropdown(choices=LOCAL_MODELS), "Local models are listed above."
    try:
        live = make_client(provider, model or DEFAULT_MODELS[provider][0], key or None).list_models()
        return gr.Dropdown(choices=live, value=model if model in live else live[0]), \
            f"Loaded {len(live)} models from {provider}."
    except Exception as err:
        return gr.Dropdown(choices=DEFAULT_MODELS[provider]), f"Could not list models: {err}"


def make_request(provider, model, key, instruction, examples, steps, plen, nprompts, nex,
                 readable, lr, workers):
    return OptimizeRequest(provider=provider, model=model, api_key=key or "",
                           instruction=instruction, examples_text=examples, steps=int(steps),
                           prompt_length=int(plen), prompts_per_step=int(nprompts),
                           examples_per_step=int(nex), readable=bool(readable),
                           learning_rate=float(lr), workers=int(workers),
                           eval_every=max(2, int(steps) // 6))


def on_estimate(*args):
    req = make_request(*args)
    labels, examples = parse_examples(req.examples_text)
    if len(labels) < 2:
        return "Add examples as `label | text`, one per line, with at least two labels."
    train, dev, test = split_examples(examples)
    msg = (f"**{len(examples)} examples, {len(labels)} labels** → {len(train)} train, "
           f"{len(dev)} dev, {len(test)} test.")
    if req.provider != "Local":
        msg += f" About **{estimate_calls(req, len(train), len(dev), len(test)):,} API calls**, fewer with cache hits."
    return msg


def fmt_acc(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def on_run(provider, model, key, instruction, examples, steps, plen, nprompts, nex,
           readable, lr, workers, request: gr.Request):
    req = make_request(provider, model, key, instruction, examples, steps, plen, nprompts,
                       nex, readable, lr, workers)
    stop = threading.Event()
    STOP_EVENTS[request.session_hash] = stop
    q: "queue.Queue" = queue.Queue()
    out = {}

    def worker():
        try:
            out["result"] = run_optimization(req, STORE, q.put, stop)
        except Exception as err:
            out["error"] = str(err)
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    rows, log, status = [], [], "Starting…"
    baseline = None
    yield status, "", None, None, ""
    while True:
        try:
            rec = q.get(timeout=1.0)
        except queue.Empty:
            continue
        if rec is None:
            break
        phase = rec.get("phase")
        if phase == "info":
            log.append(rec["message"])
        elif phase == "baseline":
            baseline = rec
            status = f"Baseline dev accuracy with your instruction alone: **{fmt_acc(rec['dev'])}**. Training…"
            log.append(f"baseline dev {fmt_acc(rec['dev'])}, test {fmt_acc(rec['test'])}")
        elif phase == "train":
            rows.append({"step": rec["step"], "accuracy": 100 * rec["train_acc_mean"],
                         "series": "train, mean of sampled prompts"})
            log.append(f"step {rec['step']}: train acc {fmt_acc(rec['train_acc_mean'])}, "
                       f"best sample {rec['sample_prompt']!r}"
                       + (f", api calls {rec['api_calls']}" if "api_calls" in rec else ""))
        elif phase == "eval":
            rows.append({"step": rec["step"], "accuracy": 100 * rec["score"], "series": "dev, greedy prompt"})
            status = f"Step {rec['step']}/{req.steps}: dev accuracy **{fmt_acc(rec['score'])}** with `{rec['prompt']}`"
        df = pd.DataFrame(rows) if rows else None
        yield status, "", df, None, "\n".join(log[-200:])

    STOP_EVENTS.pop(request.session_hash, None)
    df = pd.DataFrame(rows) if rows else None
    if "error" in out:
        yield f"**Failed:** {out['error']}", "", df, None, "\n".join(log[-200:])
        return
    r = out["result"]
    gained = r["learned_tokens"] != ""
    status = (f"**Done{' (stopped early)' if r['stopped_early'] else ''}** in {r['minutes']} min. "
              f"Dev {fmt_acc(r['baseline_dev'])} → **{fmt_acc(r['optimized_dev'])}**, "
              f"held-out test {fmt_acc(r['baseline_test'])} → **{fmt_acc(r['optimized_test'])}**. "
              f"API calls: {r['api_calls']:,}.")
    if not gained:
        status += " No learned tokens beat your original instruction, so it is returned unchanged."
    cands = pd.DataFrame([{"prompt tokens": c["prompt"], "dev accuracy %": round(100 * c["dev_acc"], 1)}
                          for c in r["candidates"]])
    yield status, r["optimized_prompt"], df, cands, "\n".join(log[-200:])


def on_stop(request: gr.Request):
    ev = STOP_EVENTS.get(request.session_hash)
    if ev:
        ev.set()
        return "Stopping after the current step…"
    return "Nothing is running."


def on_history():
    runs = STORE.list_runs()
    if not runs:
        return pd.DataFrame()
    df = pd.DataFrame(runs)
    for c in ("baseline_dev", "optimized_dev", "baseline_test", "optimized_test"):
        df[c] = df[c].apply(lambda v: None if v is None else round(100 * v, 1))
    return df


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RLPrompt") as demo:
        gr.Markdown(
            "# RLPrompt — prompt optimiser\n"
            "A **local policy network** (distilGPT-2 + MLP) learns extra prompt tokens through "
            "reinforcement learning. The chosen model (Claude / OpenAI / Gemini) is only used as "
            "an **evaluator** — it scores each candidate prompt on your examples and returns a reward. "
            "The model never writes or rewrites your prompt; only the local optimizer does.\n\n"
            "The output is the full prompt formatted for your chosen model, ready to copy-paste."
        )
        with gr.Tab("Optimise"):
            with gr.Row():
                with gr.Column(scale=3):
                    instruction = gr.Textbox(label="Your prompt", lines=2, value=SAMPLE_INSTRUCTION)
                    examples = gr.Textbox(label="Labelled examples, one per line as  label | text",
                                          lines=12, value=SAMPLE_EXAMPLES)
                with gr.Column(scale=2):
                    provider = gr.Dropdown(PROVIDERS, value="OpenAI", label="Provider")
                    model = gr.Dropdown(DEFAULT_MODELS["OpenAI"], value=DEFAULT_MODELS["OpenAI"][0],
                                        label="Model", allow_custom_value=True)
                    key = gr.Textbox(label="API key", type="password",
                                     placeholder="Leave blank to use the environment variable")
                    refresh = gr.Button("Refresh model list", size="sm")
                    with gr.Accordion("Settings", open=False):
                        steps = gr.Slider(10, 300, value=25, step=5, label="Training steps (20-30 recommended for fast runs)")
                        plen = gr.Slider(1, 5, value=3, step=1, label="Learned tokens (more = richer prompt addition)")
                        nprompts = gr.Slider(2, 16, value=4, step=1, label="Prompts per step")
                        nex = gr.Slider(2, 32, value=4, step=1, label="Examples per step")
                        readable = gr.Checkbox(value=True, label="Readable tokens only (recommended — keeps additions as real words)")
                        lr = gr.Number(value=5e-4, label="Learning rate")
                        workers = gr.Slider(1, 32, value=16, step=1, label="Parallel API calls")
                    estimate = gr.Markdown()
                    with gr.Row():
                        run = gr.Button("Optimise", variant="primary")
                        stop = gr.Button("Stop")
            status = gr.Markdown()
            optimized = gr.Textbox(label="Optimised prompt (formatted for your chosen model — copy-paste ready)",
                                   buttons=["copy"], interactive=False)
            plot = gr.LinePlot(x="step", y="accuracy", color="series", y_lim=[0, 100],
                               label="Accuracy during training", height=280)
            cands = gr.Dataframe(label="Final candidates checked on dev", interactive=False)
            log = gr.Textbox(label="Log", lines=8, max_lines=8, autoscroll=True)

            inputs = [provider, model, key, instruction, examples, steps, plen, nprompts, nex,
                      readable, lr, workers]
            provider.change(on_provider, provider, [model, key])
            refresh.click(on_refresh, [provider, model, key], [model, estimate])
            for comp in inputs:
                comp.change(on_estimate, inputs, estimate)
            demo.load(on_estimate, inputs, estimate)
            run.click(on_run, inputs, [status, optimized, plot, cands, log])
            stop.click(on_stop, None, status)

        with gr.Tab("History"):
            hist = gr.Dataframe(interactive=False)
            gr.Button("Refresh").click(on_history, None, hist)
            demo.load(on_history, None, hist)

        with gr.Tab("Prompt Builder"):
            gr.Markdown(
                "## Prompt Builder\n"
                "Describe your task briefly. The builder expands it into a detailed, "
                "structured prompt formatted for your chosen model — **entirely locally**, "
                "with no API calls. Copy the output and paste it directly into Claude, "
                "ChatGPT, or Gemini."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    pb_task_type = gr.Dropdown(
                        TASK_TYPES, value=TASK_TYPES[0],
                        label="Task type"
                    )
                    pb_title = gr.Textbox(
                        label="Project / Report title",
                        placeholder='e.g. "AI-Driven Marine Data Platform for Fisheries and Oceanography"'
                    )
                    pb_domain = gr.Textbox(
                        label="Domain / field",
                        placeholder='e.g. "Marine Informatics, AI, Oceanography"'
                    )
                    pb_brief = gr.Textbox(
                        label="Brief description (what should be covered)",
                        lines=4,
                        placeholder="Describe the core problem, key technologies, and any specific requirements."
                    )
                with gr.Column(scale=1):
                    pb_model = gr.Dropdown(
                        ["Claude", "OpenAI", "Gemini"],
                        value="Claude",
                        label="Target model (prompt will be formatted for this model)"
                    )
                    with gr.Accordion("Academic report settings", open=False):
                        pb_degree = gr.Textbox(value="B.Tech / B.E. Capstone", label="Degree / level")
                        pb_style = gr.Dropdown(["IEEE", "APA", "MLA", "Chicago"], value="IEEE", label="Citation style")
                        pb_chapters = gr.Textbox(value="1,2,3,4,5", label="Chapters to include (comma-separated)")
                        pb_word_count = gr.Textbox(value="80–120 pages", label="Approximate length")
                        pb_extra = gr.Textbox(label="Extra sections (optional)", placeholder="e.g. Ethical Considerations, Deployment Plan")
                    with gr.Accordion("Code project settings", open=False):
                        pb_language = gr.Textbox(label="Programming language", placeholder="Python")
                        pb_framework = gr.Textbox(label="Framework / stack", placeholder="FastAPI, React, PostgreSQL")
                        pb_features = gr.Textbox(label="Features required", lines=3,
                                                  placeholder="One feature per line")
                    with gr.Accordion("Data analysis settings", open=False):
                        pb_dataset = gr.Textbox(label="Dataset description")
                        pb_metrics = gr.Textbox(label="Required metrics / outputs")
                    with gr.Accordion("Custom template", open=False):
                        pb_custom = gr.Textbox(
                            label="Custom template",
                            lines=6,
                            placeholder="Write any template. Use {title}, {domain}, {brief} as placeholders."
                        )
                    pb_build = gr.Button("Build Prompt", variant="primary")

            pb_output = gr.Textbox(
                label="Generated prompt — copy and paste this into your chosen model",
                lines=30,
                max_lines=60,
                buttons=["copy"],
                interactive=False
            )

            def on_build(task_type, title, domain, brief, model,
                         degree, style, chapters, word_count, extra,
                         language, framework, features, dataset, metrics, custom):
                if not title.strip():
                    return "Please enter a project title."
                req = BuildRequest(
                    task_type=task_type, title=title.strip(), domain=domain.strip(),
                    brief=brief.strip(), target_model=model,
                    degree=degree, style=style, chapters=chapters,
                    word_count=word_count, extra_sections=extra,
                    language=language, framework=framework, features=features,
                    dataset=dataset, metrics=metrics, custom_template=custom
                )
                return build_prompt(req)

            pb_build.click(
                on_build,
                inputs=[pb_task_type, pb_title, pb_domain, pb_brief, pb_model,
                        pb_degree, pb_style, pb_chapters, pb_word_count, pb_extra,
                        pb_language, pb_framework, pb_features, pb_dataset, pb_metrics,
                        pb_custom],
                outputs=[pb_output]
            )

    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_app().queue().launch(share=args.share, server_port=args.port)
