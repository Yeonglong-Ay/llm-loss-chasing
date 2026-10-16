# Evaluate a local (optionally LoRA-adapted) model with the same task, prompts, parser,
# and CSV format as the API experiments. The sessions of one condition run in parallel as a batch.
# Usage: python src/run_local.py --name chaser --adapter adapters/qwen2.5-1.5b-chaser/final

import argparse
import os
import time
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_builder import PROMPT_VERSIONS, build_prompt
from recorder import Recorder, make_filepath
from response_parser import parse_decision
from task_engine import Task, build_sequence, generate_sequence

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)                 # base / chaser / flat
parser.add_argument("--adapter", default=None)               # path to a LoRA adapter, or none
parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--versions", nargs="+", default=["v1c"])
parser.add_argument("--anchor-sessions", type=int, default=10)
parser.add_argument("--generated-sessions", type=int, default=10)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=32)
args = parser.parse_args()

MAX_ATTEMPTS = 3
DATA_DIR = os.environ.get("DATA_DIR", "data")
MODEL_ID = f"local.qwen2.5-1.5b-{args.name}"


def log(*a):
    print(*a, flush=True)


# ---------- model ----------
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"                     # required for batched generation
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

try:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
model.to("cuda")
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
log(f"model: {MODEL_ID} | adapter: {args.adapter}")


@torch.no_grad()
def generate(prompts):
    """Same chat template as training. Pure temperature sampling, no top-k / top-p filtering."""
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                     add_generation_prompt=True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                         max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    replies = tok.batch_decode(new, skip_special_tokens=True)
    n_prompt = enc["attention_mask"].sum(dim=1).tolist()
    n_new = (new != tok.pad_token_id).sum(dim=1).tolist()
    return replies, n_prompt, n_new


# ---------- one condition: all sessions in parallel ----------
def run_condition(version, condition, n_sessions):
    bets = PROMPT_VERSIONS[version]["bets"]
    sessions = []
    for s in range(1, n_sessions + 1):
        path = make_filepath(MODEL_ID, condition, s, version, folder=DATA_DIR)
        if os.path.exists(path):
            log(f"skip (already exists): {path}")
            continue
        seq, seed = (build_sequence(3), None) if condition == "anchor" else (generate_sequence(s), s)
        partial = path + ".partial"          # renamed to .csv only when the session is complete
        if os.path.exists(partial):
            os.remove(partial)
        sessions.append({"session": s, "seed": seed, "path": path, "partial": partial,
                         "task": Task(sequence=seq, bet_options=bets), "rec": Recorder(partial)})
    if not sessions:
        return

    torch.manual_seed(12345)
    active = sessions
    t0 = time.time()
    while active:
        states = [x["task"].get_state() for x in active]
        prompts = [build_prompt(st, version) for st in states]
        results = [None] * len(active)
        attempts = [0] * len(active)
        todo = list(range(len(active)))

        while todo:                              # retry only the replies that failed to parse
            start = time.perf_counter()
            replies, n_prompt, n_new = generate([prompts[i] for i in todo])
            per_call = (time.perf_counter() - start) / len(todo)
            still = []
            for j, i in enumerate(todo):
                attempts[i] += 1
                try:
                    bet, color = parse_decision(replies[j], bets)
                    results[i] = {"bet": bet, "color": color, "n_attempts": attempts[i],
                                  "parse_failed": False, "raw_reply": replies[j],
                                  "prompt_tokens": n_prompt[j], "completion_tokens": n_new[j],
                                  "latency_s": round(per_call, 3)}
                except ValueError as e:
                    if attempts[i] < MAX_ATTEMPTS:
                        still.append(i)
                    else:
                        results[i] = {"parse_failed": True, "n_attempts": attempts[i],
                                      "raw_reply": replies[j], "error": str(e)}
            todo = still

        next_active = []
        for x, st, res in zip(active, states, results):
            meta = {"model": MODEL_ID, "condition": condition, "session": x["session"],
                    "seed": x["seed"], "prompt_version": version,
                    "temperature": args.temperature, "max_tokens": args.max_new_tokens,
                    "timestamp": datetime.now().isoformat(timespec="seconds")}
            if res["parse_failed"]:
                x["rec"].write_row({**meta, "trial": st["trial_number"], **res})
                os.replace(x["partial"], x["path"])
                log(f"  PARSE FAILED: {MODEL_ID} {version} {condition} s{x['session']}, "
                    f"trial {st['trial_number']}")
                continue
            rec = x["task"].play_trial(res["bet"], res["color"])
            x["rec"].write_row({**meta, **rec, **res})
            if x["task"].is_done():
                os.replace(x["partial"], x["path"])
                log(f"done: {x['path']}")
            else:
                next_active.append(x)
        active = next_active
        if active and states[0]["trial_number"] % 20 == 0:
            log(f"  {MODEL_ID} {version} {condition}: trial {states[0]['trial_number']}, "
                f"{len(active)} sessions active, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    t_start = time.time()
    for v in args.versions:
        run_condition(v, "anchor", args.anchor_sessions)
        run_condition(v, "generated", args.generated_sessions)
    log(f"finished {MODEL_ID} in {(time.time() - t_start) / 60:.1f} min")
