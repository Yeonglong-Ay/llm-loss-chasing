# LoRA fine-tuning of a small open model on synthetic bettor data.
# Plain PyTorch training loop (transformers + peft). Loss is computed on the answer only.
# Gradient accumulation: micro-batches of --micro-batch-size, one update per --batch-size examples.
# Usage: python src/train_lora.py --policy chaser

import argparse
import json
import math
import os
import random
import re
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True, choices=["chaser", "flat"])
parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--data-dir", default="data_ft")
parser.add_argument("--out-dir", default="adapters")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch-size", type=int, default=16)        # examples per update
parser.add_argument("--micro-batch-size", type=int, default=4)   # NEW: examples per forward pass
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--max-steps", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

assert args.batch_size % args.micro_batch_size == 0, "batch-size must be a multiple of micro-batch-size"
ACCUM = args.batch_size // args.micro_batch_size                  # NEW


def log(*a):
    print(*a, flush=True)


random.seed(args.seed)
torch.manual_seed(args.seed)
assert torch.cuda.is_available(), "No GPU visible to PyTorch"
device = "cuda"
out_dir = os.path.join(args.out_dir, f"qwen2.5-1.5b-{args.policy}")
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()

# ---------- tokenizer and data ----------
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def encode(example):
    """Token ids for the whole conversation; labels = -100 on the prompt so loss is on the answer only."""
    msgs = example["messages"]
    prompt_text = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(msgs, tokenize=False)
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    f_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    if f_ids[:len(p_ids)] != p_ids:
        raise ValueError("Prompt tokens are not a prefix of the full conversation tokens")
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels}


train_rows = read_jsonl(os.path.join(args.data_dir, f"{args.policy}_train.jsonl"))
val_rows = read_jsonl(os.path.join(args.data_dir, f"{args.policy}_val.jsonl"))
train = [encode(ex) for ex in train_rows]
val = [encode(ex) for ex in val_rows]

lengths = [len(x["input_ids"]) for x in train]
log(f"train examples: {len(train)} | val examples: {len(val)}")
log(f"tokens per example: min {min(lengths)}, mean {sum(lengths) / len(lengths):.0f}, max {max(lengths)}")
log("answer the model is trained on (example 0):",
    repr(tok.decode([t for t in train[0]["labels"] if t != -100])))
log(f"batch size {args.batch_size} = {ACCUM} micro-batches of {args.micro_batch_size}")


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


# NEW: loaders use the micro-batch size
train_loader = DataLoader(train, batch_size=args.micro_batch_size, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val, batch_size=args.micro_batch_size, shuffle=False, collate_fn=collate)

# ---------- model with LoRA adapters ----------
try:
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
except TypeError:
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
base.to(device)

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()

trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

steps_per_epoch = math.ceil(len(train) / args.batch_size)
total_steps = args.epochs * steps_per_epoch
if args.max_steps:
    total_steps = min(total_steps, args.max_steps)
warmup = max(1, int(0.05 * total_steps))


def lr_factor(step):
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
log(f"steps per epoch: {steps_per_epoch} | total steps: {total_steps} | warmup: {warmup}")


@torch.no_grad()
def validation_loss():
    model.eval()
    losses = []
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses.append(model(**batch).loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def behavior_check(n=60):
    """Generate answers for held-out validation prompts; compare bets with the policy's targets."""
    model.eval()
    sample = random.Random(1).sample(val_rows, n)
    errors, by_streak, failures = [], {}, 0
    for ex in sample:
        prompt_text = tok.apply_chat_template(ex["messages"][:1], tokenize=False,
                                              add_generation_prompt=True)
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r'"bet"\s*:\s*(\d+)', text)
        if not m:
            failures += 1
            continue
        bet = int(m.group(1))
        errors.append(abs(bet - ex["bet"]))
        by_streak.setdefault(ex["streak"], []).append(bet)
    model.train()
    return {
        "n": n,
        "parse_failures": failures,
        "mean_abs_error": round(sum(errors) / len(errors), 2) if errors else None,
        "mean_bet_by_streak": {k: round(sum(v) / len(v), 2) for k, v in sorted(by_streak.items())},
    }


# ---------- training loop (NEW: gradient accumulation) ----------
history = {"train_loss": [], "val_loss": [], "config": vars(args)}
model.train()
step = 0
micro = 0
running = 0.0
done = False
for epoch in range(1, args.epochs + 1):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss / ACCUM     # scale so the summed gradient is an average
        loss.backward()
        running += loss.item()
        micro += 1
        if micro % ACCUM != 0:
            continue                            # keep accumulating gradients

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        history["train_loss"].append((step, running))
        if step == 1 or step % 20 == 0:
            log(f"epoch {epoch} step {step}/{total_steps} loss {running:.4f} "
                f"lr {sched.get_last_lr()[0]:.2e} elapsed {time.time() - t0:.0f}s "
                f"gpu mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        running = 0.0
        if step >= total_steps:
            done = True
            break

    vl = validation_loss()
    history["val_loss"].append((step, vl))
    log(f"== end of epoch {epoch}: validation loss {vl:.4f}")
    model.save_pretrained(os.path.join(out_dir, f"epoch_{epoch}"))
    if done:
        break

# ---------- save and check behavior ----------
final_dir = os.path.join(out_dir, "final")
model.save_pretrained(final_dir)
tok.save_pretrained(final_dir)

check = behavior_check()
history["behavior_check"] = check
history["minutes"] = round((time.time() - t0) / 60, 1)
with open(os.path.join(out_dir, "training_log.json"), "w") as f:
    json.dump(history, f, indent=2)

log("behavior check on held-out validation prompts:", json.dumps(check))
log(f"saved adapter to {final_dir} | total {history['minutes']} min")
