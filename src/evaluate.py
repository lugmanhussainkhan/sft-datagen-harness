"""Evaluate a model on the benchmarks. Results go to runs/<model>/<bench>/r<k>.jsonl (+ r<k>.summary.json); run k
uses seed + k. Finished runs are skipped, interrupted ones resume from their .jsonl, and runs/<model>/report.json
plus a table of every model in runs/ are refreshed at the end.

    uv run src/evaluate.py --model hosted_vllm/Qwen/Qwen2.5-3B-Instruct --model-args '{"api_base": "http://localhost:8000/v1"}'
    uv run src/evaluate.py --bench math gpqa --model ...
"""

import argparse
import importlib
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import litellm
from tqdm import tqdm

litellm.suppress_debug_info = True
BENCHES = ["ifeval", "ifbench", "math", "gpqa", "mmlu_pro", "kk"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bench", nargs="+", choices=BENCHES, default=BENCHES, help="default: all")
ap.add_argument("--model", required=True, help="LiteLLM model string, e.g. hosted_vllm/<served-name>")
ap.add_argument("--model-args", type=json.loads, default={}, help="JSON passed to litellm.completion; overrides bench defaults")
ap.add_argument("--runs", type=int, help="default: each bench's RUNS")
ap.add_argument("--limit", type=int, help="first N items only; written to runs/<model>_limit<N>/")
ap.add_argument("--concurrency", type=int, default=32)
ap.add_argument("--name", help="folder under runs/; default: last part of --model")
ap.add_argument("--force", action="store_true", help="redo the selected benches from scratch")
args = ap.parse_args()

model_dir = Path("runs") / ((args.name or args.model.split("/")[-1]) + (f"_limit{args.limit}" if args.limit else ""))

for name in args.bench:
    data = Path(__file__).resolve().parents[1] / f"data/{name}_test.jsonl"
    if not data.exists():
        print(f"{data} missing, skipping {name}" + (" (build it with src/prepare_gpqa.py)" if name == "gpqa" else ""))
        continue
    (model_dir / name).mkdir(parents=True, exist_ok=True)
    bench = importlib.import_module(f"benchmarks.{name}")
    with open(data) as f:
        items = [json.loads(line) for line in f][: args.limit]

    for k in range(args.runs or bench.RUNS):
        out = model_dir / name / f"r{k}.jsonl"
        summary_path = out.with_suffix(".summary.json")
        if args.force:
            out.unlink(missing_ok=True)
            summary_path.unlink(missing_ok=True)
        if summary_path.exists():
            print(f"{summary_path} exists, skipping")
            continue
        model_args = {"num_retries": 3, **bench.REQUEST, **args.model_args}
        model_args["seed"] = model_args.get("seed", 0) + k

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
                   "model": args.model, "model_args": model_args, "limit": args.limit}
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"{name} r{k}: {bench.METRIC} {100 * summary[bench.METRIC]:.2f} "
              f"(truncated {summary['truncated']}, empty {summary['empty']})")

# report.json for this model, then a table of every model under runs/
report = {}
for path in sorted(model_dir.glob("*/r*.summary.json")):
    s = json.loads(path.read_text())
    report.setdefault(path.parent.name, {"metric": s["metric"], "values": []})["values"].append(s[s["metric"]])
for r in report.values():
    r["mean"] = statistics.mean(r["values"])
model_dir.mkdir(parents=True, exist_ok=True)
(model_dir / "report.json").write_text(json.dumps(report, indent=2))

print(f"\n{'model':40}" + "".join(f"{b:>14}" for b in BENCHES) + "   (runs)")
for path in sorted(Path("runs").glob("*/report.json")):
    rep = json.loads(path.read_text())
    cells = [f"{100 * rep[b]['mean']:.2f} ({len(rep[b]['values'])})" if b in rep else "-" for b in BENCHES]
    print(f"{path.parent.name:40}" + "".join(f"{c:>14}" for c in cells))
