"""Evaluate a model on the benchmarks. Results go to runs/<model>/<bench>/r<k>.jsonl (+ r<k>.summary.json); run k
uses seed + k. Finished runs are skipped, interrupted ones resume from their .jsonl, and runs/<model>/report.json
plus a table of every model in runs/ are refreshed at the end.

--train runs the same harness (prompt, request defaults, scoring) on data/<bench>_train.jsonl instead, writing to
outputs/train/<model>/ (benches without a train split are skipped). Each bench then also gets sft.jsonl: per item,
the first run's response that is correct, not empty and not truncated, as a user/assistant pair with the eval prompt;
sft_gold.jsonl holds the reference solutions for the same items.

    uv run src/evaluate.py --model hosted_vllm/Qwen/Qwen2.5-3B-Instruct --model-args '{"api_base": "http://localhost:8000/v1"}'
    uv run src/evaluate.py --bench math gpqa --model ...
    uv run src/evaluate.py --train --bench kk math --sample 1000 --runs 4 --model <strong model>
"""

import argparse
import importlib
import json
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import litellm
from tqdm import tqdm

litellm.suppress_debug_info = True
BENCHES = ["ifeval", "ifbench", "math", "gpqa", "mmlu_pro", "kk"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bench", nargs="+", choices=BENCHES, default=BENCHES, help="default: all")
ap.add_argument("--model", required=True, help="LiteLLM model string, e.g. hosted_vllm/<served-name>")
ap.add_argument("--model-args", type=json.loads, default={}, help="JSON passed to litellm.completion; overrides bench defaults")
ap.add_argument("--runs", type=int, help="default: each bench's RUNS")
ap.add_argument("--limit", type=int, help="first N items only; written to <root>/<model>_limit<N>/")
ap.add_argument("--sample", type=int, help="random subset of N items (seed 0, same items as self_rewrite.py --sample N); "
                                           "written to <root>/<model>_sample<N>/")
ap.add_argument("--train", action="store_true", help="run on data/<bench>_train.jsonl, write to outputs/train/, build sft.jsonl")
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

for name in args.bench:
    data = Path(__file__).resolve().parents[1] / f"data/{name}_{split}.jsonl"
    if not data.exists():
        print(f"{data} missing, skipping {name}" + (" (build it with src/prepare_gpqa.py)" if name == "gpqa" else ""))
        continue
    (model_dir / name).mkdir(parents=True, exist_ok=True)
    bench = importlib.import_module(f"benchmarks.{name}")
    with open(data) as f:
        items = [json.loads(line) for line in f]
    if args.sample:
        items = [items[i] for i in sorted(random.Random(0).sample(range(len(items)), args.sample))]
    items = items[: args.limit]
    n_runs = args.runs or bench.RUNS

    for k in range(n_runs):
        out = model_dir / name / f"r{k}.jsonl"
        summary_path = out.with_suffix(".summary.json")
        if args.force:
            out.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)
        if summary_path.exists():
            print(f"{summary_path} exists, skipping")
            continue
        model_args = {"num_retries": 3, **bench.REQUEST, **args.model_args}
        if SEEDABLE:
            model_args["seed"] = model_args.get("seed", 0) + k
        else:
            model_args.pop("seed", None)

        records = {}  # resume: keep what an interrupted run already scored
        if out.exists():
            with open(out) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:  # last line cut short by the interruption
                        continue
                    records[row["id"]] = row
        todo = [item for item in items if item["id"] not in records]

        pool = ThreadPoolExecutor(args.concurrency)
        try:
            with open(out, "w") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records.values())
                responses = pool.map(lambda item, kw=model_args, b=bench: litellm.completion(
                    model=args.model, messages=[{"role": "user", "content": b.prompt(item)}], **kw), todo)
                for item, r in zip(todo, tqdm(responses, total=len(todo), desc=f"{name} r{k}")):
                    text = r.choices[0].message.content or ""
                    text = "" if text.lstrip().startswith("<think>") and "</think>" not in text else text  # cut off while thinking
                    text = text.rpartition("</think>")[2].strip()  # only the answer is scored
                    records[item["id"]] = {"id": item["id"], "response": text, "finish_reason": r.choices[0].finish_reason,
                                           "reasoning": getattr(r.choices[0].message, "reasoning_content", None),
                                           **bench.score(item, text)}
                    f.write(json.dumps(records[item["id"]], ensure_ascii=False) + "\n")
                    f.flush()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # Ctrl-C shouldn't wait for every queued request

        done = [records[item["id"]] for item in items]
        summary = {"metric": bench.METRIC, **bench.summarize(done), "n": len(done),
                   "truncated": sum(r["finish_reason"] == "length" for r in done),
                   "empty": sum(not r["response"] for r in done),
                   "model": args.model, "model_args": model_args, "split": split,
                   "sample": args.sample, "limit": args.limit}
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"{name} r{k}: {bench.METRIC} {100 * summary[bench.METRIC]:.2f} "
              f"(truncated {summary['truncated']}, empty {summary['empty']})")

    if args.train:  # per item, the first run's correct, complete response becomes an SFT example
        kept = {}
        for k in range(n_runs):
            out = model_dir / name / f"r{k}.jsonl"
            if not out.with_suffix(".summary.json").exists():
                continue
            with open(out) as f:
                for row in map(json.loads, f):
                    if row.get("correct") and row["response"] and row["finish_reason"] != "length":
                        kept.setdefault(row["id"], row["response"])
        kept_items = [item for item in items if item["id"] in kept]
        files = {"sft.jsonl": [[{"role": "user", "content": bench.prompt(item)},
                                {"role": "assistant", "content": kept[item["id"]]}] for item in kept_items]}
        if all("solution" in item for item in kept_items):
            files["sft_gold.jsonl"] = [[{"role": "user", "content": bench.prompt(item)},
                                        {"role": "assistant", "content": item["solution"]}] for item in kept_items]
        for fname, rows in files.items():
            (model_dir / name / fname).write_text("".join(json.dumps({"messages": m}, ensure_ascii=False) + "\n" for m in rows))
        print(f"{name}: {len(kept_items)}/{len(items)} items kept for sft.jsonl ({n_runs} runs)")

# report.json for this model, then a table of every model under the same root
report = {}
for path in sorted(model_dir.glob("*/r*.summary.json")):
    s = json.loads(path.read_text())
    report.setdefault(path.parent.name, {"metric": s["metric"], "values": []})["values"].append(s[s["metric"]])
for r in report.values():
    r["mean"] = statistics.mean(r["values"])
model_dir.mkdir(parents=True, exist_ok=True)
(model_dir / "report.json").write_text(json.dumps(report, indent=2))

print(f"\n{'model':40}" + "".join(f"{b:>14}" for b in BENCHES) + "   (runs)")
for path in sorted(root.glob("*/report.json")):
    rep = json.loads(path.read_text())
    cells = [f"{100 * rep[b]['mean']:.2f} ({len(rep[b]['values'])})" if b in rep else "-" for b in BENCHES]
    print(f"{path.parent.name:40}" + "".join(f"{c:>14}" for c in cells))
