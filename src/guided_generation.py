"""Guided generation (kk): the student gets a step-by-step fill-in solving guide for the puzzle and writes the answer.
Correct answers become SFT data paired with the plain eval prompt, without the guide, so the model learns to solve the
puzzle unaided. The prompt (GUIDED_PROMPT) is built here: instructions, rules and answer format are fixed text, the
question is the dataset's quiz verbatim, and only "What each statement means" and "Template" come from the guide.

Items are the puzzles that have a guide in --guides (finish_reason != "length"), in guide order (the order of the
self-rewrite subset's train.jsonl); their question and gold solution come from --items (default data/kk_train_haiku.jsonl),
joined on id. --sample N draws N of these items (seed 0); --limit N then keeps the first N. The guides file may still be growing: rerun the same command later
and only the new items are generated.

Run k uses seed + k (models that reject seed differ only by sampling). Records are appended to r<k>.jsonl as answers
arrive; an interrupted run resumes from it, and a run with every item scored is skipped. --force starts over.

    uv run src/guided_generation.py --model hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct --runs 4 \\
        --model-args '{"api_base": "http://localhost:8000/v1"}'

outputs/guided/<model>[_sample<N>][_limit<N>]/kk/ (or --out-dir):
    r<k>.jsonl           scored records (id, people, prompt, response, finish_reason, pred, correct, ...)
    r<k>.summary.json    accuracy, by_people, truncated, empty, rejected, request args
    sft.jsonl            per item, the first run's correct, non-empty, untruncated response, paired with the eval prompt
                         (--keep-guide-prompt: with the guided prompt instead)
    sft_gold.jsonl       gold solutions for the same items, paired with the eval prompt
"""

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import litellm
from tqdm import tqdm

from benchmarks import kk

litellm.suppress_debug_info = True
ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True, help="LiteLLM model string of the student, e.g. hosted_vllm/<served-name>")
ap.add_argument("--model-args", type=json.loads, default={}, help=f"JSON passed to litellm.completion; overrides {kk.REQUEST}")
ap.add_argument("--guides", type=Path, default=ROOT / "data/kk_guides.jsonl")
ap.add_argument("--items", type=Path, default=ROOT / "data/kk_train_haiku.jsonl",
                help="puzzles with id, quiz, names, answer, solution")
ap.add_argument("--runs", type=int, default=1, help="attempts per item; run k uses seed + k")
ap.add_argument("--sample", type=int, help="random subset of N of the guided items (seed 0)")
ap.add_argument("--limit", type=int, help="first N items only")
ap.add_argument("--concurrency", type=int, default=32)
ap.add_argument("--out-dir", type=Path, help="default: outputs/guided/<model>[_sample<N>][_limit<N>]/kk")
ap.add_argument("--keep-guide-prompt", action="store_true", help="pair sft.jsonl responses with the guided prompt")
ap.add_argument("--force", action="store_true", help="delete this run's files and start over")
args = ap.parse_args()
# LiteLLM's HTTP client keeps httpx's default pool of 100 connections, so more threads than that just queue
pool_limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
litellm.sync_transport = httpx.HTTPTransport(limits=pool_limits)
model_name, provider, *_ = litellm.get_llm_provider(args.model)
supported = litellm.get_supported_openai_params(model=model_name, custom_llm_provider=provider)
SEEDABLE = supported is None or "seed" in supported  # e.g. Anthropic rejects seed; its runs differ only by sampling


def read(path):
    """Records in a .jsonl; a last line cut short (an interrupted run, or a file still being written) is dropped."""
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# The student prompt is built here; only "What each statement means" and "Template" are taken from the generated guide.
# The question is the dataset's quiz, verbatim.
GUIDED_PROMPT = """Solve the puzzle below by filling in the template. Copy every line of the template and replace each ___ with your answer. Do not skip any line. Do not use a different method. Do not change the order of the steps. Write the CONCLUSION only after you finish Step {last_step}.

### Question
{quiz}

### Rules
* A knight's statement is always TRUE.
* A knave's statement is always FALSE.
* If a person's type does not match the truth of their statement, that guess is impossible.

### What each statement means
{meanings}

### Template
{template}

### Answer format
Fill each ___ with "knight" or "knave". Keep this exact order: {order}.
CONCLUSION: {conclusion}"""
HEADINGS = r"(?:Instructions|Question|Rules|What each statement means|Template|Answer format)"


def guide_parts(text):
    """The generated guide's "What each statement means" and "Template" sections, or None if either is missing."""
    parts, cur = {}, None
    for line in text.splitlines():
        m = re.fullmatch(rf"\s*#*\s*({HEADINGS})\s*:?\s*", line)
        if m:
            cur = m.group(1)
            parts[cur] = []
        elif cur:
            parts[cur].append(line)
    meanings = "\n".join(parts.get("What each statement means", [])).strip()
    template = "\n".join(parts.get("Template", [])).strip()
    steps = re.findall(r"(?m)^Step (\d+):", template)
    return {"meanings": meanings, "template": template, "last_step": steps[-1]} if meanings and steps else None


def guided_prompt(item):
    g = guides[item["id"]]
    return GUIDED_PROMPT.format(quiz=item["quiz"], meanings=g["meanings"], template=g["template"], last_step=g["last_step"],
                                order=", ".join(item["names"]),
                                conclusion=" ".join(f"({i}) {n} is a ___" for i, n in enumerate(item["names"], 1)))


for path in (args.guides, args.items):
    if not path.exists():
        raise SystemExit(f"not found: {path}")
guides = {g["id"]: {**parts, "idx": g.get("idx", 0)} for g in read(args.guides)
          if g.get("finish_reason") != "length" and g.get("guide") and (parts := guide_parts(g["guide"]))}
items = sorted((item for item in read(args.items) if item["id"] in guides), key=lambda item: guides[item["id"]]["idx"])
if args.sample:
    items = [items[i] for i in sorted(random.Random(0).sample(range(len(items)), args.sample))]
items = items[: args.limit]
print(f"{len(items)} items with a guide ({len(guides)} usable guides in {args.guides})")
if not items:
    raise SystemExit(f"no item of {args.items} has a guide in {args.guides}; check that the two files match")
out_dir = args.out_dir or Path("outputs/guided") / (args.model.split("/")[-1] + (f"_sample{args.sample}" if args.sample else "")
                                                    + (f"_limit{args.limit}" if args.limit else "")) / "kk"
out_dir.mkdir(parents=True, exist_ok=True)


def complete(item, kw):
    """One guided request; a rejected one (most often a prompt that doesn't fit the context) comes back as None."""
    try:
        return litellm.completion(model=args.model, messages=[{"role": "user", "content": guided_prompt(item)}], **kw), None
    except litellm.BadRequestError as e:
        return None, f"{type(e).__name__}: {str(e)[:500]}"


for k in range(args.runs):
    out = out_dir / f"r{k}.jsonl"
    summary_path = out.with_suffix(".summary.json")
    if args.force:
        out.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    model_args = {"num_retries": 3, **kk.REQUEST, **args.model_args}
    if SEEDABLE:
        model_args["seed"] = model_args.get("seed", 0) + k
    else:
        model_args.pop("seed", None)

    records = {row["id"]: row for row in read(out)}  # resume: keep what an interrupted run already scored
    todo = [item for item in items if item["id"] not in records]
    if summary_path.exists() and not todo:
        print(f"{summary_path} exists, skipping")
        continue

    pool = ThreadPoolExecutor(args.concurrency)
    try:
        with open(out, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records.values())
            responses = pool.map(lambda item, kw=model_args: complete(item, kw), todo)
            for item, (r, error) in zip(todo, tqdm(responses, total=len(todo), desc=f"guided kk r{k}")):
                if r is None:
                    text, finish, reasoning = "", None, None
                else:
                    text = r.choices[0].message.content or ""
                    text = "" if text.lstrip().startswith("<think>") and "</think>" not in text else text  # cut off while thinking
                    text = text.rpartition("</think>")[2].strip()  # only the answer is scored
                    finish, reasoning = r.choices[0].finish_reason, getattr(r.choices[0].message, "reasoning_content", None)
                records[item["id"]] = {"id": item["id"], "people": item["people"], "prompt": guided_prompt(item),
                                       "response": text, "finish_reason": finish, "reasoning": reasoning,
                                       **({"error": error} if error else {}), **kk.score(item, text)}
                f.write(json.dumps(records[item["id"]], ensure_ascii=False) + "\n")
                f.flush()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)  # Ctrl-C shouldn't wait for every queued request

    done = [records[item["id"]] for item in items]
    summary = {"metric": kk.METRIC, **kk.summarize(done), "n": len(done),
               "truncated": sum(r["finish_reason"] == "length" for r in done),
               "empty": sum(not r["response"] for r in done),
               "rejected": sum("error" in r for r in done),
               "model": args.model, "model_args": model_args, "guides": str(args.guides), "items": str(args.items),
               "sample": args.sample, "limit": args.limit}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"guided kk r{k}: accuracy {100 * summary['accuracy']:.2f} "
          f"(truncated {summary['truncated']}, empty {summary['empty']}, rejected {summary['rejected']})")

# per item, the first run's correct, complete response becomes an SFT example
kept = {}
for k in range(args.runs):
    out = out_dir / f"r{k}.jsonl"
    if not out.with_suffix(".summary.json").exists():
        continue
    for row in read(out):
        if row.get("correct") and row["response"] and row["finish_reason"] != "length":
            kept.setdefault(row["id"], row)
kept_items = [item for item in items if item["id"] in kept]
files = {"sft.jsonl": [[{"role": "user", "content": kept[item["id"]]["prompt"] if args.keep_guide_prompt else kk.prompt(item)},
                        {"role": "assistant", "content": kept[item["id"]]["response"]}] for item in kept_items],
         "sft_gold.jsonl": [[{"role": "user", "content": kk.prompt(item)},
                             {"role": "assistant", "content": item["solution"]}] for item in kept_items]}
for fname, rows in files.items():
    (out_dir / fname).write_text("".join(json.dumps({"messages": m}, ensure_ascii=False) + "\n" for m in rows))
print(f"guided kk: {len(kept_items)}/{len(items)} items kept for sft.jsonl ({args.runs} runs) -> {out_dir}")
