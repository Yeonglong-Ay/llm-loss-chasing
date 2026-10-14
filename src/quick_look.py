import glob
import pandas as pd

files = glob.glob("data/*.csv")
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
df["short"] = df["model"].str.split(".", n=1).str[1]

print("rows:", len(df))

print("\nPer model and condition:")
summary = df.groupby(["short", "condition"]).agg(
    trials=("trial", "count"),
    failures=("parse_failed", "sum"),
    retries=("n_attempts", lambda s: (s > 1).sum()),
    mean_bet=("bet", "mean"),
    sd_bet=("bet", "std"),
)
print(summary.round(2))

anchor = df[df["condition"] == "anchor"]

print("\nMean bet by losing streak going in, per model (anchor only):")
print(anchor.pivot_table(
    index="streak_before", columns="short", values="bet", aggfunc="mean"
).round(2))

print("\nTrial counts behind each cell:")
print(anchor.pivot_table(
    index="streak_before", columns="short", values="bet", aggfunc="count"
))