# Runner: loops over models, prompt versions, conditions, and sessions.
# Connects the task engine, the agent, and the recorder.
#
# Usage:
#   python src/runner.py                                  # all models, all versions
#   python src/runner.py --models zai-org.glm-5.3         # one model
#   python src/runner.py --versions v1c v1cd              # chosen prompt versions

import argparse
import os
from datetime import datetime

import openai

from task_engine import Task, build_sequence, generate_sequence
from agent import get_decision
from recorder import Recorder, make_filepath
from prompt_builder import PROMPT_VERSIONS

TEMPERATURE = 1.0
MAX_TOKENS = 4000     # same for every model; only truncates, does not change decisions
DATA_DIR = os.environ.get("DATA_DIR", "data")

MODELS = [
    "google.gemma-3-27b-it",
    "google.gemma-4-31B-it",
    "openai.gpt-oss-120b",
    "meta.llama-3.2-11b-vision-instruct",
    "zai-org.glm-5.3",
]
VERSIONS = ["v1", "v1c", "v1cd"]
N_SESSIONS = {"anchor": 10, "generated": 0}


def make_sequence(condition, session):
    """Return (sequence, seed). Anchor = your human cycle; generated = seeded shuffle."""
    if condition == "anchor":
        return build_sequence(3), None
    return generate_sequence(session), session


def run_session(model, condition, session, prompt_version):
    """Run one session. Returns 'ok', 'skipped', 'parse_failed', or 'api_error'."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = make_filepath(model, condition, session, prompt_version, folder=DATA_DIR)
    if os.path.exists(filepath):
        print(f"skip (already exists): {filepath}", flush=True)
        return "skipped"

    bet_options = PROMPT_VERSIONS[prompt_version]["bets"]
    sequence, seed = make_sequence(condition, session)
    task = Task(sequence=sequence, bet_options=bet_options)
    recorder = Recorder(filepath)

    try:
        while not task.is_done():
            state = task.get_state()
            result = get_decision(
                state, model, bet_options, prompt_version,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
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
                print(f"  PARSE FAILED: {model} {prompt_version} {condition} "
                      f"s{session}, trial {state['trial_number']}", flush=True)
                return "parse_failed"

            rec = task.play_trial(result["bet"], result["color"])
            recorder.write_row({**meta, **rec, **result})

            if rec["trial"] % 10 == 0:
                print(f"  {model} {prompt_version} {condition} s{session}: "
                      f"trial {rec['trial']}", flush=True)

    except openai.APIError as e:
        os.remove(filepath)   # a partial file would be skipped on the next run
        print(f"  API ERROR: {model} {prompt_version} {condition} s{session}: "
              f"{str(e)[:200]}", flush=True)
        return "api_error"

    print(f"done: {filepath}", flush=True)
    return "ok"


def run_model(model, versions, n_sessions):
    for prompt_version in versions:
        for condition, n in n_sessions.items():
            for session in range(1, n + 1):
                status = run_session(model, condition, session, prompt_version)
                if status == "api_error":
                    print(f"Skipping the rest of {model} (model unavailable right now).",
                          flush=True)
                    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--versions", nargs="+", default=VERSIONS,
                        choices=list(PROMPT_VERSIONS))
    parser.add_argument("--anchor-sessions", type=int, default=N_SESSIONS["anchor"])
    parser.add_argument("--generated-sessions", type=int, default=N_SESSIONS["generated"])
    args = parser.parse_args()

    n_sessions = {"anchor": args.anchor_sessions, "generated": args.generated_sessions}
    for model in args.models:
        run_model(model, args.versions, n_sessions)
