# RLPrompt from scratch

A from-scratch PyTorch implementation of **RLPrompt: Optimizing Discrete Text Prompts with Reinforcement Learning** by Deng et al., EMNLP 2022. It covers both tasks in the paper: few-shot text classification with a frozen RoBERTa and unsupervised Yelp sentiment transfer with a frozen GPT-2.

A small policy network writes a discrete prompt token by token. The prompt is fed to a frozen task LM, the task LM's behaviour is turned into a reward, and the policy is trained with Soft Q-Learning. The task LM is only ever called with forward passes, so it is treated as a black box.

## How the paper maps to the code

| Paper component | Where |
|---|---|
| Policy: frozen distilGPT-2 + one trainable MLP of 2,048 hidden units before the LM head | `rlprompt/policy.py` |
| Soft Q-Learning, on-policy, with Polyak-averaged target network | `rlprompt/losses.py`, `rlprompt/trainer.py` |
| Input-specific z-score normalisation | `rlprompt/utils.py → zscore_by_group` |
| Piecewise gap reward, λ1 = 180, λ2 = 200 | `rlprompt/tasks/classification.py` |
| Verbalizers, templates, true few-shot splits of 16 per class | `rlprompt/tasks/classification.py` |
| Content + Style reward for style transfer | `rlprompt/tasks/style_transfer.py` |
| Content, Style, Fluency, J and GM metrics | `rlprompt/tasks/style_transfer.py → style_transfer_metrics` |
| Manual, null prompt baselines and cross-model prompt transfer | `scripts/eval_*.py` |
| Ablations: no z-score, no piecewise reward, REINFORCE instead of SQL | command-line flags |

The policy MLP has about 3.1M parameters, matching the paper. Because the policy backbone is frozen, the SQL target network is just a second copy of the MLP; online and target heads share a single backbone forward pass.

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

The tests build tiny random models locally and run both training loops end to end on CPU, so they need no downloads. On Colab, open `notebooks/RLPrompt_colab.ipynb` and follow it.

## Web app

```bash
python app.py            # http://127.0.0.1:7860
python app.py --share    # public link, works from Colab
```

Type your prompt, paste labelled examples as `label | text`, choose a provider and model, and press Optimise. RLPrompt learns a few tokens to append to your prompt that raise the model's accuracy on your examples. The app shows accuracy live during training, the final optimised prompt, and the baseline and optimised accuracy on a held-out test split.

- **Providers:** OpenAI, Gemini and Claude through their APIs, or Local for the free Hugging Face models used in the paper. *Refresh model list* asks the provider for the models your key can use, and any model ID can be typed in.
- **API keys:** type them in the app or set `OPENAI_API_KEY`, `GEMINI_API_KEY` and `ANTHROPIC_API_KEY`. Keys are never stored.
- **Reward with API models:** the answer is scored right or wrong per example, since not every API returns token probabilities. The paper's piecewise weights still apply.
- **Cost & Speed:** the app estimates API calls before you start; the fast defaults use roughly ~350–400 calls (down from 4,000), finishing in under 1–2 minutes. Cheaper models such as `gpt-4o-mini`, `gemini-2.0-flash` or `claude-3-5-haiku` keep a run fast and inexpensive.
- **Model selection:** at the end, the greedy prompt and the five best prompts sampled during training are re-checked on the dev split. If none beats your original prompt, the original is returned unchanged.

### Database

The app uses SQLite in `rlprompt_app.db`. The `runs` table keeps every optimisation with its settings, result and scores, shown in the History tab. The `api_cache` table stores every model answer by a hash of provider, model and full prompt, so repeated queries within a run or across runs are free. The research scripts below write plain files instead.

## Few-shot classification

```bash
# Quick run for local CPU / mini-project demo (500 steps, ~6x faster model, runs in minutes):
python scripts/train_classification.py --dataset sst2 --quick

# 2-token prompt, paper setting: 6K steps
python scripts/train_classification.py --dataset sst2 --prompt_length 2

# 5-token prompt, paper setting: 12K steps
python scripts/train_classification.py --dataset sst2 --prompt_length 5

# other datasets: mr, yelp_polarity, sst5, ag_news
python scripts/train_classification.py --dataset ag_news --prompt_length 5 --max_test 500
```

Each run writes to `outputs/<dataset>_T<len>_split<k>_seed<s>/`: `log.jsonl` with training curves, `best_prompt.json` chosen on the 16-shot dev set, `best_policy.pt`, and `results.json` with test accuracy for the learned prompt alongside the manual and null prompt baselines. The paper averages 5 few-shot splits × 3 seeds; use `--split_seed 0..4` and `--seed 0..2`.

Prompt transfer across model sizes and between masked and left-to-right LMs:

```bash
python scripts/eval_classification.py --dataset sst2 \
    --prompt_files outputs/sst2_T2_split0_seed0/best_prompt.json \
    --task_models distilroberta-base roberta-base roberta-large distilgpt2 gpt2 gpt2-medium \
    --out outputs/sst2_transfer.csv
```

To train on a smaller task LM first, pass `--task_model distilroberta-base` to the training script.

## Text style transfer

```bash
# Quick run (200 steps):
python scripts/train_style_transfer.py --direction 0_to_1 --quick

# Full paper runs:
python scripts/train_style_transfer.py --direction 0_to_1 --task_model distilgpt2
python scripts/train_style_transfer.py --direction 1_to_0 --task_model distilgpt2
```

The Yelp data from Shen et al. 2017 downloads into `data/yelp/` on first use. Results go to `outputs/yelp_<dir>_<model>_seed<s>/`, including `test_outputs.tsv` with every input, its prompt and its rewrite.

```bash
# evaluate a trained policy, or transfer its prompts to a larger generator
python scripts/eval_style_transfer.py --run_dir outputs/yelp_0to1_distilgpt2_seed0 --task_model gpt2-large

# baselines
python scripts/eval_style_transfer.py --baseline null   --task_model gpt2-xl
python scripts/eval_style_transfer.py --baseline manual --task_model gpt2-xl
```

## Performance Optimizations & Fast Runs

This repository includes several built-in optimizations to make training and prompt optimization dramatically faster on local machines and CPUs:

1. **Batch Prompt Deduplication**: The policy network often samples identical prompt candidates within a single batch. Rewards are computed only on unique prompt candidates and mapped back, eliminating 20%–50% of redundant forward passes and API calls with zero accuracy loss.
2. **Single-Pass Vocabulary Projections**: In `forced_logits`, fluent top-k vocabulary filtering is projected once across the 50K vocabulary and shared between online and target heads, cutting projection overhead in half.
3. **Automated Early Stopping**: The trainer automatically halts training when 100% accuracy (`target_score=1.0`) is achieved or if dev accuracy does not improve after `patience` evaluation cycles (default 5 for scripts, 3 for web app).
4. **Smart CPU & `--quick` Detection**: Passing `--quick` (or running on a CPU without a GPU) automatically switches the task model to `distilroberta-base` (~6x faster than `roberta-large`), reduces step counts to 500, samples 8 examples per step, and caps test evaluation at 500 examples.
5. **Web App Defaults**: Web optimization defaults to 25 steps, 4 prompts/step, 4 examples/step, and 16 parallel API workers, reducing total API calls from ~4,000 to ~350–400 (a 90% reduction in time and cost).

## Ablations

```bash
--zscore false          # removes input-specific z-score normalisation
--piecewise false       # plain gap reward without the correctness bonus
--algo reinforce        # policy gradient instead of Soft Q-Learning
--fluent_top_k 20       # restrict actions to GPT-2's top-20 tokens for readable prompts
--patience 3            # early stopping patience
```

## Hardware defaults

Defaults fit a single 16 GB T4 on Colab or Kaggle. Task LMs run in fp16, the policy in fp32. Style transfer defaults to distilGPT-2 as the generator, 8 inputs × 4 prompts × 8 samples per step and 2,000 steps. The paper's strongest style-transfer result uses `--task_model gpt2-xl`, which needs a larger GPU and far longer runs.

## Paper numbers to compare against

| Setting | Paper result |
|---|---|
| SST-2, RoBERTa-large, 5 tokens | 92.5 |
| SST-2, 2 tokens | 90.3 |
| AG's News, 5 tokens | 80.2 |
| SST-2 manual prompt | 82.8 |
| Yelp, distilGPT-2 generator | J = 46.0, Content 57.3, Style 96.5, Fluency 85.3 |
| Yelp, GPT-2-xl generator | J = 61.4 |

## Where this differs from the paper

- Content preservation uses our own BERTScore-style alignment F1 over RoBERTa-large layer-17 embeddings instead of the CTC package. Content and J values are comparable across runs of this project, not directly with the paper.
- Fluency defaults to `textattack/roberta-base-CoLA`; the paper used a different grammaticality classifier. Change it with `--fluency_model` and `--fluency_label`.
- The style classifier defaults to `textattack/bert-base-uncased-yelp-polarity`, a BERT-base Yelp sentiment model, in place of the authors' own checkpoint.
- `--reward_scale`, default 5, multiplies rewards after z-scoring. It sets how sharply the soft policy concentrates and is not reported in the paper, so it is worth tuning.
- Each style-transfer training reward averages the best of random subsets of the sampled outputs; set `--bootstraps` and `--bootstrap_size` to change this.

## Layout

```
rlprompt/
  config.py                 all hyperparameters, overridable from the CLI
  policy.py                 frozen policy LM + trainable MLP, target MLP, sampling
  losses.py                 Soft Q-Learning: v2, v2r, v3, v3r; REINFORCE
  trainer.py                task-agnostic training loop, eval, checkpointing
  utils.py                  z-score by group, logging, seeding
  tasks/base.py             Task interface: sample_batch, reward, evaluate
  tasks/classification.py   datasets, prompted classifier, piecewise gap reward
  tasks/style_transfer.py   Yelp data, prompted generator, content/style/fluency scorers
  tasks/api_classification.py  GPT, Gemini or Claude as the task LM
  llm_clients.py            uniform clients for the three APIs
  optimizer.py              one-call optimisation used by the app
  store.py                  SQLite run history and API answer cache
app.py                      Gradio web app
scripts/                    train and eval entry points
tests/                      offline unit and end-to-end tests with tiny models
notebooks/                  Colab walkthrough
```

To add a new task, subclass `Task` in `rlprompt/tasks/` and implement its reward; the policy, losses and trainer stay unchanged.

## Reference

Deng, M., Wang, J., Hsieh, C.-P., Wang, Y., Guo, H., Shu, T., Song, M., Xing, E. P., Hu, Z. 2022. RLPrompt: Optimizing Discrete Text Prompts with Reinforcement Learning. EMNLP 2022.
