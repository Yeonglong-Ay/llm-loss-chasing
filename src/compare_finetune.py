# Pre/post fine-tuning comparison on held-out sequences (anchor + generated seeds 1-10).
# Usage: python src/compare_finetune.py

import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from task_engine import build_sequence, generate_sequence, losing_streak_before

VERSION = "v1c"
LOCAL = ["qwen2.5-1.5b-base", "qwen2.5-1.5b-chaser", "qwen2.5-1.5b-flat"]

df = pd.concat((pd.read_csv(f) for f in glob.glob("data/local.*.csv")), ignore_index=True)
df = df[df["prompt_version"] == VERSION].copy()
df["short"] = df["model"].str.split(".", n=1).str[1]
df["failed"] = df["parse_failed"].astype(bool)

# Streak for every row. Failed rows are recorded without one, so recompute from the sequence.
_seq_cache = {}
def sequence_for(condition, seed):
    key = (condition, None if condition == "anchor" else int(seed))
    if key not in _seq_cache:
        _seq_cache[key] = build_sequence(3) if condition == "anchor" else generate_sequence(int(seed))
    return _seq_cache[key]

df["streak"] = [losing_streak_before(sequence_for(c, s)[: int(t) - 1])
                for c, s, t in zip(df["condition"], df["seed"], df["trial"])]
ok = df[~df["failed"]]
assert (ok["streak"] == ok["streak_before"]).all(), "recomputed streak disagrees with recorded streak"

# ---------- 1. completeness ----------
print("Completeness (trials with a valid decision per session):")
comp = df.groupby(["short", "condition", "session"]).agg(
    valid=("failed", lambda f: (~f).sum()), ended_by_failure=("failed", "any")).reset_index()
print(comp.groupby(["short", "condition"]).agg(
    sessions=("session", "size"), mean_valid_trials=("valid", "mean"),
    sessions_ended_by_failure=("ended_by_failure", "sum")).round(1).to_string())

# ---------- 2. per-session slopes of bet on streak ----------
rows = []
for (m, c, s), g in ok.groupby(["short", "condition", "session"]):
    if len(g) < 10 or g["streak"].nunique() < 2:
        continue
    fit = stats.linregress(g["streak"], g["bet"])
    rows.append({"model": m, "condition": c, "session": s, "slope": fit.slope, "n_trials": len(g)})
slopes = pd.DataFrame(rows)

summary = []
for (m, c), g in slopes.groupby(["model", "condition"]):
    s = g["slope"].to_numpy()
    p = stats.ttest_1samp(s, 0.0).pvalue if len(s) > 1 and s.std(ddof=1) > 0 else np.nan
    summary.append({"model": m, "condition": c, "n_sessions": len(s),
                    "mean_trials": round(g["n_trials"].mean(), 1),
                    "mean_slope": round(s.mean(), 3), "sd_slope": round(s.std(ddof=1), 3),
                    "p_vs_0": p})
summary = pd.DataFrame(summary)
summary.to_csv("results/finetune_slopes.csv", index=False)
print("\nSlope of bet on losing streak ($ per extra loss), tested against 0:")
print(summary.to_string(index=False))

# ---------- 3. key contrast: chaser vs flat ----------
print("\nChaser vs flat control (Welch t-test on session slopes):")
for c in ["anchor", "generated"]:
    a = slopes[(slopes.model == "qwen2.5-1.5b-chaser") & (slopes.condition == c)]["slope"]
    b = slopes[(slopes.model == "qwen2.5-1.5b-flat") & (slopes.condition == c)]["slope"]
    if len(a) > 1 and len(b) > 1:
        t = stats.ttest_ind(a, b, equal_var=False)
        print(f"  {c}: chaser {a.mean():.3f} vs flat {b.mean():.3f}, "
              f"difference {a.mean() - b.mean():.3f}, t = {t.statistic:.2f}, p = {t.pvalue:.2g}")

# ---------- 4. mean bet by streak ----------
print("\nMean bet by losing streak (valid trials):")
print(ok.pivot_table(index=["condition", "streak"], columns="short", values="bet",
                     aggfunc="mean").round(2).to_string())

# ---------- 5. base model: over-limit attempts by streak ----------
base = df[df["short"] == "qwen2.5-1.5b-base"].copy()
base["needed_retry_or_failed"] = (base["n_attempts"] > 1) | base["failed"]
print("\nBase model: share of trials needing a retry or failing, by streak "
      "(failures were mostly bets above $10):")
print(base.groupby(["condition", "streak"]).agg(
    trials=("trial", "size"), share=("needed_retry_or_failed", "mean")).round(2).to_string())

# ---------- 6. plot ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, c in zip(axes, ["anchor", "generated"]):
    sub = ok[ok["condition"] == c]
    per_session = sub.groupby(["short", "session", "streak"])["bet"].mean().reset_index()
    agg = per_session.groupby(["short", "streak"])["bet"].agg(["mean", "sem"]).reset_index()
    for m in LOCAL:
        g = agg[agg["short"] == m]
        ax.errorbar(g["streak"], g["mean"], yerr=g["sem"], marker="o", capsize=3,
                    label=m.replace("qwen2.5-1.5b-", ""))
    ax.set_title(f"{c} sequence (held out)")
    ax.set_xlabel("Losing streak going into the trial")
axes[0].set_ylabel("Mean bet ($)")
axes[0].legend(title="Qwen2.5-1.5B")
fig.suptitle("Before vs after fine-tuning: bet size by losing streak (prompt v1c)")
fig.savefig("results/finetune_bet_by_streak.png", dpi=150, bbox_inches="tight")
print("\nsaved results/finetune_bet_by_streak.png and results/finetune_slopes.csv")
