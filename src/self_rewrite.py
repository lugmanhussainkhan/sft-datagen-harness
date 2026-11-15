"""Self-rewrite: the model sees the instruction, a training question and its reference solution, and writes its own.
Up to --max-trials attempts per item (seed + t); kept if not empty, not truncated, correct, and not mentioning the reference.

    uv run src/self_rewrite.py --bench math --model hosted_vllm/LiquidAI/LFM2.5-350M --model-args '{"api_base": "http://localhost:8000/v1"}'
"""

import argparse
import importlib
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm
from tqdm import tqdm

litellm.suppress_debug_info = True
LEAK = re.compile(r"reference (?:solution|answer)|(?:given|provided|above|original) (?:solution|answer)"
                  r"|(?:solution|answer) (?:above|provided|given)|as (?:in|per) the reference", re.IGNORECASE)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bench", required=True, choices=["math", "kk"], help="training set: data/<bench>_train.jsonl")
ap.add_argument("--model", required=True, help="LiteLLM model string, e.g. hosted_vllm/<served-name>")
ap.add_argument("--model-args", type=json.loads, default={}, help="JSON passed to litellm.completion; overrides bench defaults")
ap.add_argument("--max-trials", type=int, default=4)
ap.add_argument("--sample", type=int, help="random subset of N items (seed 0)")
ap.add_argument("--limit", type=int, help="first N items only")
ap.add_argument("--concurrency", type=int, default=32)
ap.add_argument("--out-dir", type=Path, help="default: outputs/self_rewrite/<bench>_<last part of --model>")
args = ap.parse_args()

bench = importlib.import_module(f"benchmarks.{args.bench}")
with open(Path(__file__).resolve().parents[1] / f"data/{args.bench}_train.jsonl") as f:
    items = [json.loads(line) for line in f]
if args.sample:
    items = [items[i] for i in sorted(random.Random(0).sample(range(len(items)), args.sample))]
items = items[: args.limit]
out_dir = args.out_dir or Path("outputs/self_rewrite") / f"{args.bench}_{args.model.split('/')[-1]}"
out_dir.mkdir(parents=True, exist_ok=True)
model_args = {"num_retries": 3, **bench.REQUEST, **args.model_args}

attempts = {item["id"]: [] for item in items}
accepted = {}
pending = items
for t in range(args.max_trials):
    trial_args = {**model_args, "seed": model_args.get("seed", 0) + t}
    pool = ThreadPoolExecutor(args.concurrency)
    try:
        responses = list(tqdm(pool.map(lambda item, kw=trial_args: litellm.completion(
            model=args.model, messages=[{"role": "user", "content": bench.rewrite_prompt(item)}], **kw),
            pending), total=len(pending), desc=f"attempt {t + 1}"))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # Ctrl-C shouldn't wait for every queued request

    still = []
    for item, r in zip(pending, responses):
        text = (r.choices[0].message.content or "").rpartition("</think>")[2].strip()
        finish = r.choices[0].finish_reason
        reason = ("empty" if not text else "truncated" if finish == "length"
                  else "wrong answer" if not bench.score(item, text)["correct"]
                  else "mentions the reference" if LEAK.search(text) else None)
        attempts[item["id"]].append({"trial": t + 1, "seed": trial_args["seed"], "response": text,
                                     "finish_reason": finish, "rejected": reason})
        if reason:
            still.append(item)
        else:
            accepted[item["id"]] = text
    pending = still
    print(f"attempt {t + 1}: {len(accepted)} accepted, {len(pending)} left")

kept = [item for item in items if item["id"] in accepted]
files = {
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
