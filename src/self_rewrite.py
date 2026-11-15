"""Self-rewrite: the model sees the instruction, a training question and its reference solution, and writes its own.
Up to --max-trials attempts per item (seed + t); kept if not empty, not truncated, correct, and not mentioning the reference.

Attempt t is sent only for items no earlier attempt accepted. Each response is scored and appended to
<out-dir>/attempt<t>.jsonl as soon as it arrives, so a stopped run loses nothing: rerunning the same command resumes
where it stopped. rewrites.jsonl (every item with all its attempts), sft.jsonl, sft_gold.jsonl and summary.json are
rebuilt from the attempt files at the end. --force starts over.

    uv run src/self_rewrite.py --bench math --model hosted_vllm/LiquidAI/LFM2.5-350M --model-args '{"api_base": "http://localhost:8000/v1"}'
"""

import argparse
import importlib
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import litellm
from tqdm import tqdm

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


def load(path):
    """Records already written to an attempt file, by id; a last line cut short by an interruption is dropped."""
    rows = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[row["id"]] = row
    return rows


def judge(item, r, t, seed):
    text = (r.choices[0].message.content or "").rpartition("</think>")[2].strip()
    finish = r.choices[0].finish_reason
    reason = ("empty" if not text else "truncated" if finish == "length"
              else "wrong answer" if not bench.score(item, text)["correct"]
              else "mentions the reference" if LEAK.search(text) else None)
    return {"id": item["id"], "trial": t, "seed": seed, "response": text, "finish_reason": finish, "rejected": reason}


attempts = {item["id"]: [] for item in items}
accepted = {}
pending = items
for t in range(1, args.max_trials + 1):
    if not pending:
        break
    trial_args = {**model_args, "seed": model_args.get("seed", 0) + t - 1}
    if not SEEDABLE:
        del trial_args["seed"]
    path = out_dir / f"attempt{t}.jsonl"
    done = load(path)  # resume: keep what an interrupted attempt already wrote
    todo = [item for item in pending if item["id"] not in done]
    # rewrite the file without a cut-short last line, atomically, then append each completion as it arrives
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values()))
    tmp.replace(path)
    if todo:
        pool = ThreadPoolExecutor(args.concurrency)
        try:
            with open(path, "a") as f:
                futures = {pool.submit(litellm.completion, model=args.model, **trial_args,
                                       messages=[{"role": "user", "content": bench.rewrite_prompt(item)}]): item
                           for item in todo}
                for fut in tqdm(as_completed(futures), total=len(todo), desc=f"attempt {t}"):
                    item = futures[fut]
                    done[item["id"]] = judge(item, fut.result(), t, trial_args.get("seed"))
                    f.write(json.dumps(done[item["id"]], ensure_ascii=False) + "\n")
                    f.flush()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # Ctrl-C shouldn't wait for every queued request

    attempted, still = pending, []
    for item in attempted:
        row = done[item["id"]]
        attempts[item["id"]].append({k: v for k, v in row.items() if k != "id"})
        if row["rejected"]:
            still.append(item)
        else:
            accepted[item["id"]] = row["response"]
    pending = still
    resumed = len(attempted) - len(todo)
    print(f"attempt {t}: {len(accepted)} accepted, {len(pending)} left" + (f" ({resumed} read from {path})" if resumed else ""))

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
