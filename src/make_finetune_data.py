# Build fine-tuning data from synthetic bettors, using the SAME task engine and prompt
# as the LLM experiments. Training uses generated seeds that never appear in evaluation.
# Usage: python src/make_finetune_data.py

import json
import os
import random

from prompt_builder import PROMPT_VERSIONS, build_prompt
from task_engine import Task, generate_sequence

PROMPT_VERSION = "v1c"                   # $1-$10 whole dollars, no default
TRAIN_SEEDS = range(101, 141)            # 40 sequences x 100 trials = 4,000 examples
VAL_SEEDS = range(141, 151)              # 10 sequences x 100 trials = 1,000 examples
EVAL_SEEDS = set(range(1, 11))           # used for evaluation; must never be trained on
OUT_DIR = "data_ft"

assert EVAL_SEEDS.isdisjoint(TRAIN_SEEDS) and EVAL_SEEDS.isdisjoint(VAL_SEEDS)

BETS = PROMPT_VERSIONS[PROMPT_VERSION]["bets"]
LOW, HIGH = min(BETS), max(BETS)


def clip(x):
    return max(LOW, min(HIGH, int(round(x))))


def chaser(streak, rng):
    """Bet rises with the losing streak: about $3 after a win, +$2 per extra loss."""
    return clip(3 + 2 * streak + rng.choice([-1, 0, 0, 1]))


def flat(streak, rng):
    """Bet ignores the streak: about $5 every round."""
    return clip(5 + rng.choice([-1, 0, 0, 1]))


POLICIES = {"chaser": chaser, "flat": flat}


def simulate(policy, seed):
    rng = random.Random(10_000 + seed)
    task = Task(sequence=generate_sequence(seed), bet_options=BETS)
    examples = []
    while not task.is_done():
        state = task.get_state()
        bet = policy(state["losing_streak"], rng)
        color = rng.choice(["red", "black"])
        examples.append({
            "messages": [
                {"role": "user", "content": build_prompt(state, PROMPT_VERSION)},
                {"role": "assistant", "content": json.dumps({"bet": bet, "color": color})},
            ],
            "streak": state["losing_streak"],
            "bet": bet,
            "seed": seed,
        })
        task.play_trial(bet, color)
    return examples


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, policy in POLICIES.items():
        for split, seeds in [("train", TRAIN_SEEDS), ("val", VAL_SEEDS)]:
            rows = [ex for s in seeds for ex in simulate(policy, s)]
            path = os.path.join(OUT_DIR, f"{name}_{split}.jsonl")
            with open(path, "w") as f:
                for ex in rows:
                    f.write(json.dumps(ex) + "\n")

            by_streak = {}
            for ex in rows:
                by_streak.setdefault(ex["streak"], []).append(ex["bet"])
            means = {k: round(sum(v) / len(v), 2) for k, v in sorted(by_streak.items())}
            print(f"{path}: {len(rows)} examples; mean bet by streak {means}")
