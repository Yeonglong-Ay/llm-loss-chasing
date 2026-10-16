# Build the README figure: an animated GIF alternating "before" and "after" fine-tuning.
# Usage: python src/make_readme_figure.py

import glob
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

VERSION, CONDITION, N_TRIALS = "v1c", "anchor", 72
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# model id: (label, color, line style, group). Colorblind-friendly (Okabe-Ito) palette.
MODELS = {
    "zai-org.glm-5.3": ("GLM 5.3", "#009E73", "-", "api"),
    "meta.llama-3.2-11b-vision-instruct": ("Llama 3.2 11B", "#56B4E9", "-", "api"),
    "openai.gpt-oss-120b": ("GPT-OSS 120B", "#CC79A7", "-", "api"),
    "google.gemma-4-31B-it": ("Gemma 4 31B (at ceiling)", "#999999", "-", "api"),
    "local.qwen2.5-1.5b-base": ("Qwen 1.5B, base", "#000000", "--", "base"),
    "local.qwen2.5-1.5b-chaser": ("Qwen 1.5B, tuned on loss chasing", "#D55E00", "-", "chaser"),
    "local.qwen2.5-1.5b-flat": ("Qwen 1.5B, tuned on flat betting (control)", "#0072B2", "-", "flat"),
}

df = pd.concat((pd.read_csv(f) for f in glob.glob("data/*.csv")), ignore_index=True)
df = df[(df["prompt_version"] == VERSION) & (df["condition"] == CONDITION)
        & df["model"].isin(list(MODELS))]
df = df[~df["parse_failed"].astype(bool)].copy()

# API models: complete sessions only. Local base: truncated sessions kept (noted in caption).
size = df.groupby(["model", "session"])["trial"].transform("count")
df = df[(size == N_TRIALS) | df["model"].str.startswith("local.")]

# average within each session first, then across sessions (error bars = between-session SEM)
per_session = df.groupby(["model", "session", "streak_before"])["bet"].mean().reset_index()
curves = (per_session.groupby(["model", "streak_before"])["bet"]
          .agg(["mean", "sem"]).reset_index().fillna({"sem": 0}))

print("Mean bet by streak (check these match earlier results):")
print(curves.pivot(index="streak_before", columns="model", values="mean").round(2).to_string())


def draw(frame):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=110)
    bold_groups = {"api", "base"} if frame == "before" else {"chaser", "flat"}
    labels = []

    for mid, (label, color, style, group) in MODELS.items():
        c = curves[curves["model"] == mid].sort_values("streak_before")
        if c.empty:
            continue
        bold = group in bold_groups
        if not bold and frame == "before":
            continue                                   # "after" lines don't appear yet
        ax.errorbar(c["streak_before"], c["mean"],
                    yerr=c["sem"] if bold else None,
                    color=color if bold else "#BBBBBB", ls=style,
                    lw=3 if group == "chaser" else (2 if bold else 1.3),
                    alpha=1.0 if bold else 0.6,
                    marker="o" if bold else None, ms=5, capsize=3 if bold else 0,
                    zorder=3 if bold else 1)
        if bold:
            labels.append([c["mean"].iloc[-1], label, color, c["streak_before"].iloc[-1]])

    # spread end-of-line labels so they don't overlap
    labels.sort(key=lambda r: r[0])
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i - 1][0] < 0.45:
            labels[i][0] = labels[i - 1][0] + 0.45
    for y, label, color, x in labels:
        ax.text(x + 0.12, y, label, color=color, fontsize=10.5, va="center", fontweight="bold")

    head, sub = (("Before fine-tuning",
                  "Off-the-shelf models bet the same or less as losses mount")
                 if frame == "before" else
                 ("After fine-tuning",
                  "Training on loss-chasing data makes bets climb; the matched control stays flat"))
    fig.text(0.07, 0.94, head, fontsize=19, fontweight="bold")
    fig.text(0.07, 0.895, sub, fontsize=12, color="#444444")

    caption = ("Roulette task adapted from human research; held-out human-task sequence; whole-dollar "
               "bets from $1 to $10. Mean bet \u00b1 SEM across 10 sessions per model. Base Qwen "
               "sessions often ended early on invalid bets above $10.")
    fig.text(0.07, 0.025, textwrap.fill(caption, 135), fontsize=8.5, color="#666666")

    ax.set_xlim(-0.25, 7.2)
    ax.set_xticks(range(5))
    ax.set_ylim(0.8, 10.8)
    ax.set_yticks([1, 2, 4, 6, 8, 10])
    ax.set_xlabel("Consecutive losses before the bet", fontsize=11)
    ax.set_ylabel("Mean bet ($)", fontsize=11)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(top=0.85, bottom=0.17, left=0.08, right=0.98)

    path = os.path.join(OUT, f"{frame}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


paths = [draw("before"), draw("after")]
frames = [Image.open(p).convert("RGB") for p in paths]
gif_path = os.path.join(OUT, "before_after.gif")
frames[0].save(gif_path, save_all=True, append_images=frames[1:],
               duration=3500, loop=0, optimize=True)
print(f"saved {paths[0]}, {paths[1]}, and {gif_path} "
      f"({os.path.getsize(gif_path) / 1024:.0f} KB)")
