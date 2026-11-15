"""Self-rewrite: the model sees the instruction, a training question and its reference solution, and writes its own.
Up to --max-trials attempts per item (seed + t); kept if not empty, not truncated, correct, and not mentioning the reference.

Attempt t is sent only for items no earlier attempt accepted, and it is queued as soon as attempt t - 1 comes back, all in
one pool of --concurrency threads, so attempts overlap instead of waiting for each other. Each response is scored and
appended to <out-dir>/attempt<t>.jsonl as soon as it arrives, so a stopped run loses nothing: rerunning the same command
resumes where it stopped. rewrites.jsonl (every item with all its attempts), sft.jsonl, sft_gold.jsonl and summary.json are
rebuilt from the attempt files at the end. --force starts over.

    uv run src/self_rewrite.py --bench math --model hosted_vllm/LiquidAI/LFM2.5-350M --model-args '{"api_base": "http://localhost:8000/v1"}'
"""

import argparse
import importlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import httpx
import litellm

from scheduler import answer, append, drive, read_jsonl, reopen, seeded

litellm.suppress_debug_info = True
LEAK = re.compile(r"reference (?:solution|answer)|(?:given|provided|above|original) (?:solution|answer)"
                  r"|(?:solution|answer) (?:above|provided|given)|as (?:in|per) the reference", re.IGNORECASE)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bench", required=True, choices=["math", "kk"], help="training set: data/<bench>_train.jsonl, or the bench's REWRITE_DATA")
ap.add_argument("--model", required=True, help="LiteLLM model string, e.g. hosted_vllm/<served-name>")
ap.add_argument("--model-args", type=json.loads, default={}, help="JSON passed to litellm.completion; overrides bench defaults")
ap.add_argument("--max-trials", type=int, default=4)
ap.add_argument("--sample", type=int, help="random subset of N items (seed 0)")
ap.add_argument("--limit", type=int, help="first N items only")
ap.add_argument("--concurrency", type=int, default=32)
ap.add_argument("--out-dir", type=Path, help="default: outputs/self_rewrite/<bench>_<last part of --model>")
ap.add_argument("--force", action="store_true", help="delete this run's attempt files and outputs and start over")
args = ap.parse_args()
OUTPUTS = ["rewrites.jsonl", "sft.jsonl", "sft_gold.jsonl", "summary.json"]
# LiteLLM's HTTP client keeps httpx's default pool of 100 connections, so more threads than that just queue
pool_limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
litellm.sync_transport = httpx.HTTPTransport(limits=pool_limits)
model_name, provider, *_ = litellm.get_llm_provider(args.model)
supported = litellm.get_supported_openai_params(model=model_name, custom_llm_provider=provider)
SEEDABLE = supported is None or "seed" in supported  # e.g. Anthropic rejects seed; its runs differ only by sampling

bench = importlib.import_module(f"benchmarks.{args.bench}")
data = getattr(bench, "REWRITE_DATA", f"{args.bench}_train")  # where the reference solutions come from
with open(Path(__file__).resolve().parents[1] / f"data/{data}.jsonl") as f:
    items = [json.loads(line) for line in f]
if args.sample:
    items = [items[i] for i in sorted(random.Random(0).sample(range(len(items)), args.sample))]
items = items[: args.limit]
out_dir = args.out_dir or Path("outputs/self_rewrite") / f"{args.bench}_{args.model.split('/')[-1]}"
out_dir.mkdir(parents=True, exist_ok=True)
model_args = {"num_retries": 3, **bench.REQUEST, **args.model_args}

if args.force:
    for path in [*out_dir.glob("attempt*.jsonl"), *(out_dir / n for n in OUTPUTS)]:
        path.unlink(missing_ok=True)


def judge(item, response, t, seed):
    text, finish, _ = answer(response)
    reason = ("empty" if not text else "truncated" if finish == "length"
              else "wrong answer" if not bench.score(item, text)["correct"]
              else "mentions the reference" if LEAK.search(text) else None)
    return {"id": item["id"], "trial": t, "seed": seed, "response": text, "finish_reason": finish, "rejected": reason}


done, files = {}, {}  # per attempt t: the records written so far (by id), and the open attempt<t>.jsonl
for t in range(1, args.max_trials + 1):
    rows = read_jsonl(out_dir / f"attempt{t}.jsonl")  # resume: keep what an interrupted run already wrote
    done[t], files[t] = {row["id"]: row for row in rows}, reopen(out_dir / f"attempt{t}.jsonl", rows)


def next_trial(item, start=1):
    """The first attempt >= start this item has no record for, unless an earlier one was accepted or none is left."""
    for t in range(1, args.max_trials + 1):
        row = done[t].get(item["id"])
        if row is not None and not row["rejected"]:
            return None
        if row is None and t >= start:
            return t
    return None


def call(job):
    t, item = job
    return litellm.completion(model=args.model, messages=[{"role": "user", "content": bench.rewrite_prompt(item)}],
                              **seeded(model_args, t - 1, SEEDABLE))


def handle(job, response):
    t, item = job
    row = done[t][item["id"]] = judge(item, response, t, seeded(model_args, t - 1, SEEDABLE).get("seed"))
    append(files[t], row)
    return [(t_next, item)] if row["rejected"] and (t_next := next_trial(item, t + 1)) else []


jobs = [(t, item) for item in items if (t := next_trial(item))]
print(f"{len(items) - len(jobs)} items finished already, {len(jobs)} to go (up to {args.max_trials} attempts each)")
try:
    drive(jobs, call, handle, args.concurrency, desc="rewrite requests")
finally:
    for f in files.values():
        f.close()

attempts = {item["id"]: [] for item in items}
accepted = {}
for item in items:  # each item's attempts in order, up to the accepted one
    for t in range(1, args.max_trials + 1):
        row = done[t].get(item["id"])
        if row is None:
            break
        attempts[item["id"]].append({k: v for k, v in row.items() if k != "id"})
        if not row["rejected"]:
            accepted[item["id"]] = row["response"]
            break

kept = [item for item in items if item["id"] in accepted]
files = {  # the consolidated outputs, rebuilt from the attempt files on every run
    "rewrites.jsonl": [{**item, "response": accepted.get(item["id"], ""), "accepted": item["id"] in accepted,
                        "attempts": attempts[item["id"]]} for item in items],
    "sft.jsonl": [{"messages": [{"role": "user", "content": bench.prompt(item)},
                                {"role": "assistant", "content": accepted[item["id"]]}]} for item in kept],
    "sft_gold.jsonl": [{"messages": [{"role": "user", "content": bench.prompt(item)},
                                     {"role": "assistant", "content": item["solution"]}]} for item in kept],
}
for name, rows in files.items():
    (out_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
summary = {"items": len(items), "accepted": len(kept), "acceptance_rate": len(kept) / len(items),
           "accepted_at_trial": dict(sorted(Counter(len(attempts[i["id"]]) for i in kept).items())),
           "rejections": dict(Counter(a["rejected"] for tries in attempts.values() for a in tries if a["rejected"])),
           "model": args.model, "model_args": model_args, "max_trials": args.max_trials,
           "sample": args.sample, "limit": args.limit}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
