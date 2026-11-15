"""Evaluate a model on the benchmarks. Results go to runs/<model>/<bench>/r<k>.jsonl (+ r<k>.summary.json); run k
uses seed + k. Every request of every selected bench and run goes through one pool of --concurrency threads, so runs
overlap instead of waiting for each other. Finished runs are skipped, interrupted ones resume from their .jsonl, and
runs/<model>/report.json plus a table of every model in runs/ are refreshed at the end.

--train runs the same harness (prompt, request defaults, scoring) on data/<bench>_train.jsonl instead, writing to
outputs/train/<model>/ (benches without a train split are skipped). There --runs is the maximum number of tries per item:
try k + 1 is sent only for items whose tries so far gave no correct, non-empty, untruncated response, and it is queued as
soon as try k comes back. Try k is written to r<k>.jsonl (seed + k). Each bench gets summary.json (solved items, the try
that solved them), sft.jsonl (per item, its accepted response as a user/assistant pair with the eval prompt) and
sft_gold.jsonl (the reference solutions for the same items). Rerunning with a larger --runs gives the unsolved items more
tries. --all-runs sends every run to every item instead (prepare_student_style.py needs all of a puzzle's attempts).

    uv run src/evaluate.py --model hosted_vllm/Qwen/Qwen2.5-3B-Instruct --model-args '{"api_base": "http://localhost:8000/v1"}'
    uv run src/evaluate.py --bench math gpqa --model ...
    uv run src/evaluate.py --train --bench kk math --sample 1000 --runs 4 --model <strong model>
"""

import argparse
import importlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path

import httpx
import litellm

from scheduler import answer, append, drive, read_jsonl, reopen, seeded

litellm.suppress_debug_info = True
BENCHES = ["ifeval", "ifbench", "math", "gpqa", "mmlu_pro", "kk"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bench", nargs="+", choices=BENCHES, default=BENCHES, help="default: all")
ap.add_argument("--model", required=True, help="LiteLLM model string, e.g. hosted_vllm/<served-name>")
ap.add_argument("--model-args", type=json.loads, default={}, help="JSON passed to litellm.completion; overrides bench defaults")
ap.add_argument("--runs", type=int, help="default: each bench's RUNS; with --train, the maximum tries per item")
ap.add_argument("--limit", type=int, help="first N items only; written to <root>/<model>_limit<N>/")
ap.add_argument("--sample", type=int, help="random subset of N items (seed 0, same items as self_rewrite.py --sample N); "
                                           "written to <root>/<model>_sample<N>/")
ap.add_argument("--train", action="store_true", help="run on data/<bench>_train.jsonl, write to outputs/train/, build sft.jsonl")
ap.add_argument("--all-runs", action="store_true", help="with --train, send every run to every item, solved or not")
ap.add_argument("--concurrency", type=int, default=32)
ap.add_argument("--name", help="folder under the output root; default: last part of --model")
ap.add_argument("--force", action="store_true", help="redo the selected benches from scratch")
args = ap.parse_args()
# LiteLLM's HTTP client keeps httpx's default pool of 100 connections, so more threads than that just queue
pool_limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
litellm.sync_transport = httpx.HTTPTransport(limits=pool_limits)
model_name, provider, *_ = litellm.get_llm_provider(args.model)
supported = litellm.get_supported_openai_params(model=model_name, custom_llm_provider=provider)
SEEDABLE = supported is None or "seed" in supported  # e.g. Anthropic rejects seed; its runs differ only by sampling

split = "train" if args.train else "test"
root = Path("outputs/train") if args.train else Path("runs")
model_dir = root / ((args.name or args.model.split("/")[-1]) + (f"_sample{args.sample}" if args.sample else "")
                    + (f"_limit{args.limit}" if args.limit else ""))


def accepted(row):
    """A training response worth keeping: correct, not empty, not cut off."""
    return bool(row.get("correct") and row["response"] and row["finish_reason"] != "length")


# Per bench: its module, items, number of runs (tries), request args, and per run k the records written so far (by id)
# and the open r<k>.jsonl. A test run whose summary exists is finished and left out.
benches, jobs = {}, []
for name in args.bench:
    data = Path(__file__).resolve().parents[1] / f"data/{name}_{split}.jsonl"
    if not data.exists():
        print(f"{data} missing, skipping {name}" + (" (build it with src/prepare_gpqa.py)" if name == "gpqa" else ""))
        continue
    bench_dir = model_dir / name
    bench_dir.mkdir(parents=True, exist_ok=True)
    bench = importlib.import_module(f"benchmarks.{name}")
    with open(data) as f:
        items = [json.loads(line) for line in f]
    if args.sample:
        items = [items[i] for i in sorted(random.Random(0).sample(range(len(items)), args.sample))]
    items = items[: args.limit]
    n_runs = args.runs or bench.RUNS
    if args.force:
        for k in range(n_runs):
            (bench_dir / f"r{k}.jsonl").unlink(missing_ok=True)
            (bench_dir / f"r{k}.summary.json").unlink(missing_ok=True)
        (bench_dir / "summary.json").unlink(missing_ok=True)
    b = benches[name] = {"bench": bench, "items": items, "n_runs": n_runs, "dir": bench_dir, "records": {}, "files": {},
                         "model_args": {"num_retries": 3, **bench.REQUEST, **args.model_args}}
    for k in range(n_runs):
        out = bench_dir / f"r{k}.jsonl"
        if not args.train and out.with_suffix(".summary.json").exists():
            print(f"{out.with_suffix('.summary.json')} exists, skipping")
            continue
        rows = read_jsonl(out)  # resume: keep what an interrupted run already scored
        b["records"][k] = {row["id"]: row for row in rows}
        b["files"][k] = reopen(out, rows)


def next_try(b, item, start=0):
    """The first try >= start this item has no record for, or None when it has used them all."""
    return next((k for k in range(start, b["n_runs"]) if item["id"] not in b["records"][k]), None)


for name, b in benches.items():
    if args.train and not args.all_runs:  # each unsolved item's next try; later tries are queued as earlier ones fail
        for item in b["items"]:
            if not any(accepted(b["records"][k][item["id"]]) for k in b["records"] if item["id"] in b["records"][k]):
                if (k := next_try(b, item)) is not None:
                    jobs.append((name, k, item))
    else:
        jobs += [(name, k, item) for k in b["records"] for item in b["items"] if item["id"] not in b["records"][k]]


# for the progress bar: train, items with an accepted response; test, correct answers among the requests sent this time
solved = {(name, item["id"]) for name, b in benches.items() for k in b["records"] for item in b["items"]
          if args.train and item["id"] in b["records"][k] and accepted(b["records"][k][item["id"]])}
n_items = sum(len(b["items"]) for b in benches.values())
correct = [0, 0]


def call(job):
    name, k, item = job
    b = benches[name]
    return litellm.completion(model=args.model, messages=[{"role": "user", "content": b["bench"].prompt(item)}],
                              **seeded(b["model_args"], k, SEEDABLE))


def handle(job, response):
    name, k, item = job
    b = benches[name]
    text, finish, reasoning = answer(response)
    row = {"id": item["id"], "response": text, "finish_reason": finish, "reasoning": reasoning, **b["bench"].score(item, text)}
    b["records"][k][item["id"]] = row
    append(b["files"][k], row)
    correct[0] += bool(row.get("correct"))
    correct[1] += 1
    if args.train and accepted(row):
        solved.add((name, item["id"]))
    if args.train and not args.all_runs and not accepted(row) and (k_next := next_try(b, item, k + 1)) is not None:
        return [(name, k_next, item)]
    return []


try:
    drive(jobs, call, handle, args.concurrency, desc="train items" if args.train else "test requests",
          status=lambda: {"solved": f"{len(solved)}/{n_items}"} if args.train else {"correct": f"{correct[0]}/{correct[1]}"})
finally:
    for b in benches.values():
        for f in b["files"].values():
            f.close()

for name, b in benches.items():
    bench, items, bench_dir = b["bench"], b["items"], b["dir"]
    if not args.train:
        for k, records in b["records"].items():
            done = [records[item["id"]] for item in items]
            summary = {"metric": bench.METRIC, **bench.summarize(done), "n": len(done),
                       "truncated": sum(r["finish_reason"] == "length" for r in done),
                       "empty": sum(not r["response"] for r in done),
                       "model": args.model, "model_args": seeded(b["model_args"], k, SEEDABLE), "split": split,
                       "sample": args.sample, "limit": args.limit}
            (bench_dir / f"r{k}.summary.json").write_text(json.dumps(summary, indent=2))
            print(f"{name} r{k}: {bench.METRIC} {100 * summary[bench.METRIC]:.2f} "
                  f"(truncated {summary['truncated']}, empty {summary['empty']})")
        continue

    # train: per item, the first try's accepted response becomes an SFT example
    kept, solved_at, tries = {}, Counter(), [r for records in b["records"].values() for r in records.values()]
    for item in items:
        for k in range(b["n_runs"]):
            row = b["records"][k].get(item["id"])
            if row and accepted(row):
                kept[item["id"]] = row["response"]
                solved_at[k + 1] += 1
                break
    kept_items = [item for item in items if item["id"] in kept]
    files = {"sft.jsonl": [[{"role": "user", "content": bench.prompt(item)},
                            {"role": "assistant", "content": kept[item["id"]]}] for item in kept_items]}
    if all("solution" in item for item in kept_items):
        files["sft_gold.jsonl"] = [[{"role": "user", "content": bench.prompt(item)},
                                    {"role": "assistant", "content": item["solution"]}] for item in kept_items]
    for fname, rows in files.items():
        (bench_dir / fname).write_text("".join(json.dumps({"messages": m}, ensure_ascii=False) + "\n" for m in rows))
    unfinished = sum((args.all_runs or item["id"] not in kept) and next_try(b, item) is not None for item in items)
    summary = {"metric": "solve_rate", "solve_rate": len(kept) / len(items), "items": len(items), "solved": len(kept),
               "solved_at_try": dict(sorted(solved_at.items())), "unfinished": unfinished, "max_tries": b["n_runs"],
               "requests": len(tries), "truncated": sum(r["finish_reason"] == "length" for r in tries),
               "empty": sum(not r["response"] for r in tries),
               "model": args.model, "model_args": b["model_args"], "split": split, "sample": args.sample, "limit": args.limit}
    (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{name}: {len(kept)}/{len(items)} items solved within {b['n_runs']} {'runs' if args.all_runs else 'tries'}, kept for sft.jsonl "
          f"({len(tries)} requests" + (f", {unfinished} unfinished: rerun to resume" if unfinished else "") + ")")

# report.json for this model, then a table of every model under the same root
report = {}
for path in sorted(model_dir.glob("*/summary.json" if args.train else "*/r*.summary.json")):
    s = json.loads(path.read_text())
    report.setdefault(path.parent.name, {"metric": s["metric"], "values": []})["values"].append(s[s["metric"]])
for r in report.values():
    r["mean"] = statistics.mean(r["values"])
model_dir.mkdir(parents=True, exist_ok=True)
(model_dir / "report.json").write_text(json.dumps(report, indent=2))

print(f"\n{'model':40}" + "".join(f"{b:>14}" for b in BENCHES) + ("" if args.train else "   (runs)"))
for path in sorted(root.glob("*/report.json")):
    rep = json.loads(path.read_text())
    cells = [f"{100 * rep[b]['mean']:.2f}" + ("" if args.train else f" ({len(rep[b]['values'])})") if b in rep else "-"
             for b in BENCHES]
    print(f"{path.parent.name:40}" + "".join(f"{c:>14}" for c in cells))
