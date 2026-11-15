"""Evaluate a model on one benchmark. Writes results/<model>/<bench>_r<k>.jsonl and .summary.json; run k uses seed + k.

    uv run src/evaluate.py --bench math --model hosted_vllm/LiquidAI/LFM2.5-350M --model-args '{"api_base": "http://localhost:8000/v1"}'
"""

import argparse
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm
from tqdm import tqdm

litellm.suppress_debug_info = True

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bench", required=True, choices=["ifeval", "ifbench", "math", "gpqa", "mmlu_pro"])
ap.add_argument("--model", required=True, help="LiteLLM model string, e.g. hosted_vllm/<served-name>")
ap.add_argument("--model-args", type=json.loads, default={}, help="JSON passed to litellm.completion; overrides bench defaults")
ap.add_argument("--runs", type=int, help="default: the bench's RUNS")
ap.add_argument("--limit", type=int, help="first N items only")
ap.add_argument("--concurrency", type=int, default=32)
ap.add_argument("--out-dir", type=Path, help="default: results/<last part of --model>")
args = ap.parse_args()

bench = importlib.import_module(f"benchmarks.{args.bench}")
with open(Path(__file__).resolve().parents[1] / f"data/{args.bench}_test.jsonl") as f:
    items = [json.loads(line) for line in f][: args.limit]
out_dir = args.out_dir or Path("results") / args.model.split("/")[-1]
out_dir.mkdir(parents=True, exist_ok=True)

for k in range(args.runs or bench.RUNS):
    out = out_dir / f"{args.bench}_r{k}.jsonl"
    if out.with_suffix(".summary.json").exists():
        print(f"{out} done, skipping")
        continue
    model_args = {"num_retries": 3, **bench.REQUEST, **args.model_args}
    model_args["seed"] = model_args.get("seed", 0) + k

    pool = ThreadPoolExecutor(args.concurrency)
    try:
        responses = list(tqdm(pool.map(lambda item, kw=model_args: litellm.completion(
            model=args.model, messages=[{"role": "user", "content": bench.prompt(item)}], **kw), items),
            total=len(items), desc=f"{args.bench} r{k}"))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # Ctrl-C shouldn't wait for every queued request

    records = []
    for item, r in zip(items, responses):
        text = r.choices[0].message.content or ""
        text = "" if text.lstrip().startswith("<think>") and "</think>" not in text else text  # cut off while thinking
        text = text.rpartition("</think>")[2].strip()  # only the answer is scored
        records.append({"id": item["id"], "response": text, "finish_reason": r.choices[0].finish_reason,
                        "reasoning": getattr(r.choices[0].message, "reasoning_content", None),
                        **bench.score(item, text)})

    summary = {"metric": bench.METRIC, **bench.summarize(records), "n": len(records),
               "truncated": sum(r["finish_reason"] == "length" for r in records),
               "empty": sum(not r["response"] for r in records),
               "model": args.model, "model_args": model_args, "limit": args.limit}
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.bench} r{k}: {bench.METRIC} {100 * summary[bench.METRIC]:.2f} "
          f"(truncated {summary['truncated']}, empty {summary['empty']})")
