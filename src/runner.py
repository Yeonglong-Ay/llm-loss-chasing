# Runner: loops over models, conditions, and sessions.
# Connects the task engine, the agent, and the recorder.

import os
from datetime import datetime

import openai

from task_engine import Task, DEFAULT_BETS, build_sequence, generate_sequence
from agent import get_decision
from recorder import Recorder, make_filepath
from prompt_builder import PROMPT_VERSION

TEMPERATURE = 1.0
MAX_TOKENS = 4000     # same for every model; only truncates, does not change decisions
MODELS = [
    "google.gemma-3-27b-it",
    "google.gemma-4-31B-it",
    "openai.gpt-oss-120b",
    "meta.llama-3.2-11b-vision-instruct",
    "zai-org.glm-5.3",
]
N_SESSIONS = {"anchor": 10, "generated": 0}


def make_sequence(condition, session):
    """Return (sequence, seed). Anchor = your human cycle; generated = seeded shuffle."""
    if condition == "anchor":
        return build_sequence(3), None
    return generate_sequence(session), session


def run_session(model, condition, session):
    """Run one session. Returns 'ok', 'skipped', 'parse_failed', or 'api_error'."""
    filepath = make_filepath(model, condition, session, PROMPT_VERSION)
    if os.path.exists(filepath):
        print(f"skip (already exists): {filepath}")
        return "skipped"

    sequence, seed = make_sequence(condition, session)
    task = Task(sequence=sequence)
    recorder = Recorder(filepath)

    try:
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
                return "parse_failed"

            rec = task.play_trial(result["bet"], result["color"])
            recorder.write_row({**meta, **rec, **result})

            if rec["trial"] % 10 == 0:
                print(f"  {model} {condition} s{session}: trial {rec['trial']}")

    except openai.APIError as e:
        os.remove(filepath)   # partial file would be skipped on the next run
        print(f"  API ERROR: {model} {condition} s{session}: {str(e)[:200]}")
        return "api_error"

    print(f"done: {filepath}")
    return "ok"


def run_model(model):
    for condition, n in N_SESSIONS.items():
        for session in range(1, n + 1):
            status = run_session(model, condition, session)
            if status == "api_error":
                print(f"Skipping the rest of {model} (model unavailable right now).")
                return


if __name__ == "__main__":
    for model in MODELS:
        run_model(model)