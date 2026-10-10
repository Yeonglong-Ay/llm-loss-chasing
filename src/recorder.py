# Recorder: saves one CSV row per trial, as it happens.

import csv
import os

COLUMNS = [
    "model", "condition", "session", "seed", "prompt_version", "temperature", "timestamp",
    "trial", "streak_before", "bet", "predicted", "shown", "outcome",
    "correct", "balance_after",
    "n_attempts", "parse_failed",
    "prompt_tokens", "completion_tokens", "latency_s", "raw_reply",
]


def make_filepath(model, condition, session, prompt_version, folder="data"):
    safe_model = model.replace("/", "_").replace(":", "_")
    return os.path.join(folder, f"{safe_model}__{condition}__{prompt_version}__s{session:02d}.csv")

class Recorder:
    def __init__(self, filepath):
        self.filepath = filepath
        with open(self.filepath, "w", newline="") as f:          
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()                                           

    def write_row(self, row):
        with open(self.filepath, "a", newline="") as f:          
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")   
            writer.writerow(row)                                        


if __name__ == "__main__":
    import pandas as pd

    path = make_filepath("test-model", "anchor", 1, "v1")
    rec = Recorder(path)
    rec.write_row({"model": "test-model", "session": 1, "trial": 1,
                   "bet": 4, "raw_reply": 'line one\nline "two", with comma'})
    rec.write_row({"model": "test-model", "session": 1, "trial": 2, "bet": 8})

    df = pd.read_csv(path)
    assert len(df) == 2
    assert list(df.columns) == COLUMNS
    assert df.loc[0, "raw_reply"] == 'line one\nline "two", with comma'
    print(df[["model", "trial", "bet"]])
    os.remove(path)
    print("All checks passed")