# Primary analysis: does bet size rise with losing streak, per model?
# Unit of analysis = session (one slope per session, then test across sessions).

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

MAX_BET = 8
N_TRIALS = 72   # anchor condition; incomplete sessions are skipped

files = glob.glob("data/*.csv")
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
df = df[(df["condition"] == "anchor") & (~df["parse_failed"].astype(bool))].copy()
df["short"] = df["model"].str.split(".", n=1).str[1]

# 1. One slope per session
rows = []
for (model, session), g in df.groupby(["short", "session"]):
    if len(g) < N_TRIALS:
        continue
    fit = stats.linregress(g["streak_before"], g["bet"])
    rows.append({"model": model, "session": session, "slope": fit.slope})
slopes = pd.DataFrame(rows)
slopes.to_csv("results/session_slopes.csv", index=False)

# 2. Test the slopes against zero, per model
ceiling = df.groupby("short")["bet"].apply(lambda b: (b == MAX_BET).mean())
summary = []
for model, g in slopes.groupby("model"):
    s = g["slope"].to_numpy()
    n = len(s)
    if n > 1 and s.std(ddof=1) > 0:
        t, p = stats.ttest_1samp(s, 0.0)
    else:
        t, p = np.nan, np.nan
    summary.append({
        "model": model,
        "n_sessions": n,
        "mean_slope": s.mean(),
        "sd_slope": s.std(ddof=1) if n > 1 else np.nan,
        "t": t,
        "p": p,
        "share_at_max_bet": ceiling[model],
    })
summary = pd.DataFrame(summary).round(3)
summary.to_csv("results/slope_summary.csv", index=False)
print(summary.to_string(index=False))

# 3. Plot: mean bet by streak, with error bars across sessions
by_session = df.groupby(["short", "session", "streak_before"])["bet"].mean().reset_index()
agg = by_session.groupby(["short", "streak_before"])["bet"].agg(["mean", "sem"]).reset_index()

fig, ax = plt.subplots(figsize=(8, 5))
for model, g in agg.groupby("short"):
    ax.errorbar(g["streak_before"], g["mean"], yerr=g["sem"],
                marker="o", capsize=3, label=model)
ax.set_xlabel("Losing streak going into the trial")
ax.set_ylabel("Mean bet ($)")
ax.set_title("Bet size by losing streak (anchor sequence)")
ax.legend(fontsize=8)
fig.savefig("results/bet_by_streak.png", dpi=150, bbox_inches="tight")
print("saved results/bet_by_streak.png")