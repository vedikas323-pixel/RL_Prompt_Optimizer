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

from dotenv import load_dotenv
load_dotenv()

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
SAMPLE_EXAMPLES = """positive | Oh great, another three hours of my life well spent. Honestly, I'd watch it again tomorrow.
positive | I expected to hate it and I'm annoyed at how much I didn't.
positive | Not a perfect film, but it stayed with me for days.
positive | The first act drags, then it becomes something special.
positive | Clunky dialogue, sure, but I cried twice and I'm not ashamed.
positive | It's silly, it's loud, and I loved every dumb minute.
positive | My friends hated it. They're wrong.
positive | Hard to sit through at times, which is exactly why it works.
positive | Nothing new here, yet somehow it all clicks.
positive | I didn't laugh once, and I mean that as a compliment: it's a gripping drama.
positive | Low budget, no stars, and still better than most of this year's blockbusters.
positive | The ending made me furious in the best possible way.
mixed | Gorgeous to look at, but I couldn't tell you what happened.
mixed | Great cast, wasted on a script that goes nowhere.
mixed | The first half is brilliant; the second half forgets why it exists.
mixed | I admired it more than I enjoyed it.
mixed | Some scenes soar, others fall completely flat.
mixed | Worth seeing for the lead performance, not much else.
mixed | Funny in places, tedious in others.
mixed | Ambitious and messy, never quite the film it wants to be.
mixed | The music is wonderful; the story is forgettable.
mixed | A strong idea let down by sloppy execution.
mixed | Fans will enjoy it; everyone else may check their phones.
mixed | Beautifully shot but emotionally hollow.
negative | Oh wonderful, another sequel nobody asked for.
negative | What a triumph, if your goal was to put the audience to sleep.
negative | I'm sure someone will love this. That someone is not me.
negative | The trailer was better than the film, and the trailer was bad.
negative | It has a beginning, a middle and an end, which is the nicest thing I can say.
negative | Bold choice, making a thriller with no thrills.
negative | Ten minutes in I was rooting for the villain to end it all.
negative | The actors look like they're being held hostage.
negative | A masterclass in how to waste a talented cast.
negative | If boredom were an art form, this would be a masterpiece.
negative | I laughed a lot, mostly at the scenes meant to be serious.
negative | Somehow both too long and not long enough to make sense."""


def model_choices(provider):
    return LOCAL_MODELS if provider == "Local" else DEFAULT_MODELS[provider]


def on_provider(provider):
    choices = model_choices(provider)
    return gr.Dropdown(choices=choices, value=choices[0])


def on_refresh(provider, model):
    if provider == "Local":
        return gr.Dropdown(choices=LOCAL_MODELS), "Local models are listed above."
    try:
        live = make_client(provider, model or DEFAULT_MODELS[provider][0]).list_models()
        return gr.Dropdown(choices=live, value=model if model in live else live[0]), \
            f"Loaded {len(live)} models from {provider}."
    except Exception as err:
        return gr.Dropdown(choices=DEFAULT_MODELS[provider]), f"Could not list models: {err}"


def make_request(provider, model, instruction, examples, steps, plen, nprompts, nex,
                 readable, lr, workers):
    return OptimizeRequest(provider=provider, model=model,
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


def on_run(provider, model, instruction, examples, steps, plen, nprompts, nex,
           readable, lr, workers, request: gr.Request):
    req = make_request(provider, model, instruction, examples, steps, plen, nprompts,
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
    yield status, gr.Textbox(value="", label="Optimised prompt"), None, None, ""
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
            rows.append({"step": rec["step"], "accuracy": 100 * rec["score"], "series": "dev, learned prompt"})
            if baseline is not None:
                rows.append({"step": rec["step"], "accuracy": 100 * baseline["dev"],
                             "series": "dev, your original prompt"})
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
    if gained:
        status = f"**Improved:** learned addition `{r['learned_tokens']}`. " + status
        out_box = gr.Textbox(value=r["optimized_prompt"], label="Optimised prompt")
    else:
        status = ("**No improvement this run:** none of the learned additions beat your original "
                  "prompt on the dev set, so your original prompt is kept. Try again or add more "
                  "training steps. " + status)
        out_box = gr.Textbox(value=r["optimized_prompt"],
                             label="Your original prompt (nothing beat it this run)")
    cands = pd.DataFrame([{"prompt tokens": c["prompt"], "dev accuracy %": round(100 * c["dev_acc"], 1)}
                          for c in r["candidates"]])
    yield status, out_box, df, cands, "\n".join(log[-200:])


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
        gr.Markdown("# RLPrompt — Prompt Optimiser")
        with gr.Tab("Optimise"):
            gr.Markdown(
                "Uses a **local policy network** (distilGPT-2) to learn extra prompt tokens "
                "through reinforcement learning — no API key required when using the **Local** provider."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    instruction = gr.Textbox(label="Your prompt", lines=2, value=SAMPLE_INSTRUCTION)
                    examples = gr.Textbox(label="Labelled examples, one per line as  label | text",
                                          lines=12, value=SAMPLE_EXAMPLES)
                with gr.Column(scale=2):
                    provider = gr.Dropdown(PROVIDERS, value="Groq", label="Provider")
                    model = gr.Dropdown(DEFAULT_MODELS["Groq"], value=DEFAULT_MODELS["Groq"][0],
                                        label="Model", allow_custom_value=True)
                    with gr.Accordion("Settings", open=False):
                        steps = gr.Slider(10, 300, value=10, step=5, label="Training steps")
                        plen = gr.Slider(1, 5, value=2, step=1, label="Learned tokens")
                        nprompts = gr.Slider(2, 16, value=2, step=1, label="Prompts per step")
                        nex = gr.Slider(2, 32, value=2, step=1, label="Examples per step")
                        readable = gr.Checkbox(value=True, label="Readable tokens only")
                        lr = gr.Number(value=5e-4, label="Learning rate")
                        workers = gr.Slider(1, 32, value=16, step=1, label="Parallel workers")
                    estimate = gr.Markdown()
                    with gr.Row():
                        run = gr.Button("Optimise", variant="primary")
                        stop = gr.Button("Stop")
            status = gr.Markdown()
            optimized = gr.Textbox(label="Optimised prompt", buttons=["copy"], interactive=False)
            plot = gr.LinePlot(x="step", y="accuracy", color="series", y_lim=[0, 100],
                               label="Accuracy during training", height=280)
            cands = gr.Dataframe(label="Final candidates checked on dev", interactive=False)
            log = gr.Textbox(label="Log", lines=8, max_lines=8, autoscroll=True)

            inputs = [provider, model, instruction, examples, steps, plen, nprompts, nex,
                      readable, lr, workers]
            provider.change(on_provider, provider, [model])
            for comp in inputs:
                comp.change(on_estimate, inputs, estimate)
            demo.load(on_estimate, inputs, estimate)
            run.click(on_run, inputs, [status, optimized, plot, cands, log])
            stop.click(on_stop, None, status)

        with gr.Tab("Prompt Builder"):
            gr.Markdown(
                "## Prompt Builder\n"
                "Describe your task briefly. The builder expands it into a detailed, "
                "structured prompt formatted for your chosen model — **entirely locally**, "
                "with no API calls."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    pb_task_type = gr.Dropdown(TASK_TYPES, value=TASK_TYPES[0], label="Task type")
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
                        ["Claude", "OpenAI", "Gemini"], value="Claude",
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
                        pb_features = gr.Textbox(label="Features required", lines=3, placeholder="One feature per line")
                    with gr.Accordion("Data analysis settings", open=False):
                        pb_dataset = gr.Textbox(label="Dataset description")
                        pb_metrics = gr.Textbox(label="Required metrics / outputs")
                    with gr.Accordion("Custom template", open=False):
                        pb_custom = gr.Textbox(
                            label="Custom template", lines=6,
                            placeholder="Write any template. Use {title}, {domain}, {brief} as placeholders."
                        )
                    pb_build = gr.Button("Build Prompt", variant="primary")

            pb_output = gr.Textbox(
                label="Generated prompt — copy and paste this into your chosen model",
                lines=30, max_lines=60, buttons=["copy"], interactive=False
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
