# Do LLMs Chase Losses?

**A behavioral evaluation of large language models in a gambling task adapted from human intracranial EEG research.**

> **Status:** active project. The evaluation harness and the first multi-model runs on Dartmouth's HPC cluster are complete. Generated-sequence replication and fine-tuning experiments are in progress.

In my human research, I study *loss chasing*: the tendency to raise bets after a run of losses, a hallmark of problem gambling. This project asks whether a large language model, given only a neutral goal ("win the most money") and no strategy hints, produces the same pattern when it plays the same task.

---

## Key findings so far (exploratory)

**No model showed loss chasing.** Instead, most models *reduced* their bets as losing streaks got longer.

Mean slope of bet on losing-streak length (dollars per additional loss), one slope per session, tested across sessions. Anchor sequence, 10 sessions per model and format.

| Model | v1: bets $1, $2, $4, $8 | v1c: bets $1–$10 | v1cd: $1–$10, starts at $5 |
|---|---|---|---|
| GLM 5.3 | **−0.32** (p = 0.003) | **−0.44** (p < 0.001) | **−0.36** (p < 0.001) |
| Llama 3.2 11B | −0.15 (p = 0.035) | **−0.21** (p < 0.001) | −0.07 (p = 0.29) |
| GPT-OSS 120B | −0.14 (p = 0.062) | −0.29 (p = 0.006) | −0.21 (p = 0.044) |
| Gemma 4 31B | +0.01 (p = 0.83) | −0.02 (p = 0.10) | −0.12 (p = 0.07, 8 sessions) |

**Bold** = survives Bonferroni correction across 12 tests (p < 0.004).

- **GLM 5.3** de-escalates robustly in all three bet formats.
- **Ceiling effects matter.** Under the original four-option format, three of five models bet the maximum on most trials, leaving no room to escalate. A whole-dollar $1–$10 format reduced the share of maximum bets for Llama (45% → 20%) and GLM (53% → 37%), and their effects became clearer. GPT-OSS and Gemma 4 stayed near the maximum (86% and 99%), so a near-zero slope for Gemma 4 means "not testable," not "no effect."
- **Models favor round numbers.** Under the $1–$10 format, bets clustered on values like $5, $8, and $10 rather than spreading evenly.

These results come from one fixed sequence and are exploratory. See [Limitations](#limitations).

---

## How the experiment works

### The task
Each round, the model chooses a bet and predicts a color (red or black). A win pays the bet, and a loss costs it. The model starts with $100.

The outcomes are **predetermined**, not random: a fixed win/loss sequence decides each round regardless of the model's choice, so every session experiences the same streak structure. The displayed ball color is derived from the model's prediction and the predetermined outcome, so the roulette history always looks coherent.

### Sequences
| Condition | Length | Description |
|---|---|---|
| **Anchor** | 72 trials | The exact 24-trial win/loss cycle used with human participants, repeated 3 times. Supports a like-for-like human comparison. |
| **Generated** | 100 trials | Seeded shuffles of loss runs (three each of lengths 1–5) and win runs. Every sequence has identical streak exposure in a different order, which tests generalization and loosens the link between streak, trial position, and balance. |

### Prompts
The prompt states the rules and the goal, the current balance, and the last 5 rounds (ball color and whether the prediction was correct). It **never** mentions streaks, recovery, risk, or strategy, and it never shows the streak count. The model has to notice streaks itself, or not.

Each prompt is a fixed, versioned stimulus, and every data row records the version used:

| Version | Bet options |
|---|---|
| `v1` | $1, $2, $4, or $8 (matches the human task) |
| `v1c` | any whole dollar from $1 to $10 |
| `v1cd` | $1 to $10, with the bet stated to start at $5 each round |

---

## Architecture

```
runner ──► task_engine.get_state()          (balance, last 5 rounds, trial number)
   │
   ├──► agent.get_decision()
   │       ├── prompt_builder.build_prompt()   (versioned, neutral prompt)
   │       ├── OpenAI-compatible API call      (with retries)
   │       └── response_parser.parse_decision()  (JSON → validated bet + color)
   │
   ├──► task_engine.play_trial()             (applies the predetermined outcome)
   └──► recorder.write_row()                 (one CSV row per trial, written immediately)
```

Engineering details:

- **Resumable runs.** Each session is its own CSV, and finished sessions are skipped, so interrupted or time-limited jobs can simply be resubmitted.
- **Strict output validation.** Replies must parse to an allowed bet and color. Invalid replies are retried with the identical prompt (up to 3 attempts), and failures are recorded rather than silently replaced.
- **Failure diagnosis.** Reasoning models initially returned empty replies. Logging `finish_reason` showed they exhausted the token budget while reasoning, so the cap was raised uniformly for all models (a cap only truncates and doesn't change decisions). Reasoning traces are saved for qualitative analysis.
- **API error handling.** Server errors stop only the affected model, remove its partial session file, and let the other models continue.
- **Cost-aware model selection.** Per-token prices pulled from the API's model metadata guided the choice of free, locally hosted models for large runs.
- **HPC execution.** A SLURM job array runs one model per task on Dartmouth's Andes cluster, with a network pre-check on each compute node.
- **Reproducibility.** Seeded sequences, versioned prompts, and per-row metadata (model, condition, session, seed, prompt version, temperature, token cap, timestamp).

---

## Repository layout

```
src/
  task_engine.py        win/loss sequences, balance, streak tracking (no LLM calls)
  prompt_builder.py     versioned neutral prompts
  response_parser.py    JSON extraction and validation
  agent.py              API calls, retries, reasoning-trace capture
  recorder.py           CSV writer
  runner.py             loops over models, prompt versions, conditions, sessions
  analyze.py            primary analysis: per-session slopes, tests, plots
  net_check.py          one-call connectivity test
  model_costs.py        per-model token prices from the API
  quick_look.py         per-model data summary
  (plus small diagnostic scripts used during development)
llm.sh                  SLURM job array for the experiment
data/                   one CSV per session (kept locally, not tracked in Git)
results/                summary tables and figures (kept locally, not tracked in Git)
```

---

## Running it

### Setup
```bash
conda create -n llm-chase python=3.12 -y
conda activate llm-chase
python -m pip install -r requirements.txt
printf "DARTMOUTH_API_KEY=your_key\n" > .env
chmod 600 .env
python src/net_check.py
```

The code targets the Dartmouth Chat API, which is OpenAI-compatible. Any other OpenAI-compatible endpoint should work after changing `base_url` in `src/agent.py` and `src/net_check.py`.

### Run sessions
```bash
# one model, one prompt version, 2 anchor sessions
python src/runner.py --models meta.llama-3.2-11b-vision-instruct --versions v1c --anchor-sessions 2

# on a SLURM cluster: all models, all prompt versions
sbatch llm.sh

# selected models (array indexes), generated sequences only
sbatch --array=2-4 llm.sh --versions v1c --anchor-sessions 0 --generated-sessions 10
```

### Analyze
```bash
python src/analyze.py --version v1c
```

---

## Data

Each row in `data/*.csv` is one trial: model, condition, session, seed, prompt version, losing streak going in, bet, predicted and shown color, outcome, balance, attempts, token usage, latency, finish reason, raw reply, and (for reasoning models) a truncated reasoning trace.

Raw session data is not stored in this repository while the analysis is ongoing. A frozen snapshot of the final dataset (model outputs only) will be published and linked here. No human-participant data is part of this project.

---

## Limitations

- **Training data, not cognition.** A betting pattern in an LLM reflects what it learned from its training data. It is not evidence that the model "feels" anything about losing.
- **Streak is confounded with balance in the anchor sequence.** Long losing streaks coincide with a lower balance, so de-escalation may reflect balance-driven caution. Mixed models and generated sequences address this, and that work is in progress.
- **Ceiling effects** make some models untestable under some formats.
- **Prompt sensitivity.** Results may depend on wording. Multiple prompt versions are tested, but not every possible phrasing.
- **One temperature (1.0)** and a modest number of sessions (10 per model and format).
- **Excluded models:** Gemma 3 27B (backend unavailable during the extended runs; 1 session) and Qwen 3.5 122B (failed to return parseable replies under the fixed prompt).

---

## Roadmap

- [x] Evaluation harness, multi-model anchor runs, three bet formats
- [x] HPC execution with SLURM job arrays
- [ ] Generated-sequence replication (running)
- [ ] Mixed-effects models separating streak, balance, and trial position
- [ ] Reasoning-trace analysis
- [ ] **Fine-tuning:** LoRA fine-tuning of a small open-weight model (Qwen2.5-1.5B-Instruct) on synthetic bettors that do or don't chase losses, with a flat-betting control, evaluated on the held-out human sequence to test whether loss chasing can be learned from data and whether it generalizes
- [ ] Comparison with human behavior from the same task (pending approvals)

---

## Related projects

- [Roulette decision-making task](https://github.com/Yeonglong-Ay/roulette_task): the human version of this task, built in PsychoPy
- [iEEG analysis pipeline](https://github.com/Yeonglong-Ay/ieeg-analysis-pipeline): the neural-data pipeline for the human study

## Author

**Yeonglong (Albert) Ay**, Dartmouth College
[Website](https://yeonglong-ay.github.io/Albert-website/) · [LinkedIn](https://www.linkedin.com/in/yeonglong-ay) · [GitHub](https://github.com/Yeonglong-Ay)
