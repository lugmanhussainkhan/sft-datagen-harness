"""Guided generation (kk): the student gets a step-by-step fill-in solving guide for the puzzle and writes the answer.
Correct answers become SFT data paired with the plain eval prompt, without the guide, so the model learns to solve the
puzzle unaided. The prompt (GUIDED_PROMPT) is built here: instructions, rules and answer format are fixed text, the
question is the dataset's quiz verbatim, and only "What each statement means" and "Template" come from the guide.

Items are the puzzles that have a guide in --guides (finish_reason != "length"), in guide order (the order of the
self-rewrite subset's train.jsonl); their question and gold solution come from --items (default data/kk_train_haiku.jsonl),
joined on id. --sample N draws N of these items (seed 0); --limit N then keeps the first N.

--runs is the maximum number of tries per item. Try k + 1 is sent only for items whose tries so far gave no correct,
non-empty, untruncated response, and it is queued as soon as try k comes back, all in one pool of --concurrency threads,
so the GPU never waits for a run to drain. Try k uses seed + k and is appended to r<k>.jsonl as it arrives. Rerun the same
command to resume an interrupted run, or with a larger --runs to give unsolved items more tries. --force starts over. A
request the server rejects (usually a prompt that does not fit the context) is recorded with its error and not retried.

    uv run src/guided_generation.py --model hosted_vllm/Qwen/Qwen2.5-1.5B-Instruct --runs 8 \\
        --model-args '{"api_base": "http://localhost:8000/v1", "temperature": 0.6}'

outputs/guided/<model>[_sample<N>][_limit<N>]/kk/ (or --out-dir):
    r<k>.jsonl           try k's scored records (id, people, prompt, response, finish_reason, pred, correct, ...)
    summary.json         solved items, the try that solved them, solve rate by people, requests, request args
    sft.jsonl            per solved item, its accepted response paired with the eval prompt
                         (--keep-guide-prompt: with the guided prompt instead)
    sft_gold.jsonl       gold solutions for the same items, paired with the eval prompt
"""

import argparse
import json
import random
import re
from pathlib import Path

import httpx
import litellm

from benchmarks import kk
from scheduler import answer, append, drive, read_jsonl, reopen, seeded

litellm.suppress_debug_info = True
ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--model", required=True, help="LiteLLM model string of the student, e.g. hosted_vllm/<served-name>")
ap.add_argument("--model-args", type=json.loads, default={}, help=f"JSON passed to litellm.completion; overrides {kk.REQUEST}")
ap.add_argument("--guides", type=Path, default=ROOT / "data/kk_guides.jsonl")
ap.add_argument("--items", type=Path, default=ROOT / "data/kk_train_haiku.jsonl",
                help="puzzles with id, quiz, names, answer, solution")
ap.add_argument("--runs", type=int, default=1, help="maximum tries per item; try k uses seed + k")
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
guides = {g["id"]: {**parts, "idx": g.get("idx", 0)} for g in read_jsonl(args.guides)
          if g.get("finish_reason") != "length" and g.get("guide") and (parts := guide_parts(g["guide"]))}
items = sorted((item for item in read_jsonl(args.items) if item["id"] in guides), key=lambda item: guides[item["id"]]["idx"])
if args.sample:
    items = [items[i] for i in sorted(random.Random(0).sample(range(len(items)), args.sample))]
items = items[: args.limit]
print(f"{len(items)} items with a guide ({len(guides)} usable guides in {args.guides})")
if not items:
    raise SystemExit(f"no item of {args.items} has a guide in {args.guides}; check that the two files match")
out_dir = args.out_dir or Path("outputs/guided") / (args.model.split("/")[-1] + (f"_sample{args.sample}" if args.sample else "")
                                                    + (f"_limit{args.limit}" if args.limit else "")) / "kk"
out_dir.mkdir(parents=True, exist_ok=True)


def accepted(row):
    """A response worth keeping: correct, not empty, not cut off."""
    return bool(row.get("correct") and row["response"] and row["finish_reason"] != "length")


model_args = {"num_retries": 3, **kk.REQUEST, **args.model_args}
if args.force:
    for path in [*out_dir.glob("r*.jsonl"), *out_dir.glob("r*.summary.json"),
                 *(out_dir / n for n in ("summary.json", "sft.jsonl", "sft_gold.jsonl"))]:
        path.unlink(missing_ok=True)
records, files = {}, {}  # per try k: the records written so far (by id), and the open r<k>.jsonl
for k in range(args.runs):
    rows = read_jsonl(out_dir / f"r{k}.jsonl")  # resume: keep what an interrupted run already scored
    records[k], files[k] = {row["id"]: row for row in rows}, reopen(out_dir / f"r{k}.jsonl", rows)


def finished(item):
    """Solved, rejected by the server, or out of tries."""
    rows = [records[k][item["id"]] for k in records if item["id"] in records[k]]
    return any(accepted(r) or "error" in r for r in rows) or next_try(item) is None


def next_try(item, start=0):
    return next((k for k in range(start, args.runs) if item["id"] not in records[k]), None)


def call(job):
    """One guided request; a rejected one (most often a prompt that doesn't fit the context) comes back as an error."""
    k, item = job
    try:
        return litellm.completion(model=args.model, messages=[{"role": "user", "content": guided_prompt(item)}],
                                  **seeded(model_args, k, SEEDABLE)), None
    except litellm.BadRequestError as e:
        return None, f"{type(e).__name__}: {str(e)[:500]}"


def handle(job, result):
    k, item = job
    response, error = result
    text, finish, reasoning = answer(response) if response else ("", None, None)
    row = {"id": item["id"], "people": item["people"], "prompt": guided_prompt(item), "response": text,
           "finish_reason": finish, "reasoning": reasoning, **({"error": error} if error else {}), **kk.score(item, text)}
    records[k][item["id"]] = row
    append(files[k], row)
    if accepted(row):
        solved.add(item["id"])
    if not accepted(row) and not error and (k_next := next_try(item, k + 1)) is not None:
        return [(k_next, item)]
    return []


solved = {item["id"] for item in items if any(accepted(records[k][item["id"]]) for k in records if item["id"] in records[k])}
jobs = [(next_try(item), item) for item in items if not finished(item)]
print(f"{len(items) - len(jobs)} items finished already, {len(jobs)} to go (up to {args.runs} tries each)")
try:
    drive(jobs, call, handle, args.concurrency, desc="guided kk items",
          status=lambda: {"solved": f"{len(solved)}/{len(items)}"})
finally:
    for f in files.values():
        f.close()

# per item, the first try's accepted response becomes an SFT example
kept, solved_at, by_people = {}, {}, {}
for item in items:
    by_people.setdefault(item["people"], [0, 0])[1] += 1
    for k in range(args.runs):
        row = records[k].get(item["id"])
        if row and accepted(row):
            kept[item["id"]] = row
            solved_at[k + 1] = solved_at.get(k + 1, 0) + 1
            by_people[item["people"]][0] += 1
            break
kept_items = [item for item in items if item["id"] in kept]
outputs = {"sft.jsonl": [[{"role": "user", "content": kept[item["id"]]["prompt"] if args.keep_guide_prompt else kk.prompt(item)},
                          {"role": "assistant", "content": kept[item["id"]]["response"]}] for item in kept_items],
           "sft_gold.jsonl": [[{"role": "user", "content": kk.prompt(item)},
                               {"role": "assistant", "content": item["solution"]}] for item in kept_items]}
for fname, rows in outputs.items():
    (out_dir / fname).write_text("".join(json.dumps({"messages": m}, ensure_ascii=False) + "\n" for m in rows))
tries = [r for rs in records.values() for r in rs.values()]
unfinished = sum(not finished(item) for item in items)
summary = {"solve_rate": len(kept) / len(items), "items": len(items), "solved": len(kept),
           "solved_at_try": dict(sorted(solved_at.items())),
           "solve_rate_by_people": {n: s / t for n, (s, t) in sorted(by_people.items())},
           "unfinished": unfinished, "max_tries": args.runs, "requests": len(tries),
           "accuracy_per_request": sum(r["correct"] for r in tries) / len(tries) if tries else None,
           "truncated": sum(r["finish_reason"] == "length" for r in tries), "empty": sum(not r["response"] for r in tries),
           "rejected": sum("error" in r for r in tries), "model": args.model, "model_args": model_args,
           "guides": str(args.guides), "items_file": str(args.items), "sample": args.sample, "limit": args.limit}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"guided kk: {len(kept)}/{len(items)} items solved within {args.runs} tries, kept for sft.jsonl "
      f"({len(tries)} requests" + (f", {unfinished} unfinished: rerun to resume" if unfinished else "") + f") -> {out_dir}")
