# Runner: loops over models, conditions, and sessions.
# Connects the task engine, the agent, and the recorder.

import os
from datetime import datetime

from task_engine import Task, DEFAULT_BETS, build_sequence, generate_sequence
from agent import get_decision
from recorder import Recorder, make_filepath
from prompt_builder import PROMPT_VERSION

TEMPERATURE = 1.0
MAX_TOKENS = 4000     # same for every model; only truncates, does not change decisions
MODELS = [
    "google.gemma-4-31B-it",
    "openai.gpt-oss-120b",
    "qwen.qwen3.5-122b",
    "meta.llama-3.2-11b-vision-instruct",
    "zai-org.glm-5.3",
]
N_SESSIONS = {"anchor": 1, "generated": 0}    # pilot settings; raise later


def make_sequence(condition, session):
    """Return (sequence, seed). Anchor = your human cycle; generated = seeded shuffle."""
    if condition == "anchor":
        return build_sequence(3), None
    return generate_sequence(session), session


def run_session(model, condition, session):
    filepath = make_filepath(model, condition, session, PROMPT_VERSION)
    if os.path.exists(filepath):
        print(f"skip (already exists): {filepath}")
        return

    sequence, seed = make_sequence(condition, session)
    task = Task(sequence=sequence)
    recorder = Recorder(filepath)

    while not task.is_done():
        state = task.get_state()
        result = get_decision(
            state, model, DEFAULT_BETS, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
        )

        meta = {
            "model": model,
            "condition": condition,
            "session": session,
            "seed": seed,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        if result["parse_failed"]:
            recorder.write_row({**meta, "trial": state["trial_number"], **result})
            print(f"  PARSE FAILED: {model} {condition} s{session}, trial {state['trial_number']}")
            return

        rec = task.play_trial(result["bet"], result["color"])
        recorder.write_row({**meta, **rec, **result})

        if rec["trial"] % 10 == 0:
            print(f"  {model} {condition} s{session}: trial {rec['trial']}")

    print(f"done: {filepath}")


if __name__ == "__main__":
    for model in MODELS:
        for condition, n in N_SESSIONS.items():
            for session in range(1, n + 1):
                run_session(model, condition, session)