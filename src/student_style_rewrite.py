"""Student-style rewrite (kk): a teacher rewrites the reference solution of each puzzle the way the student writes.

The system prompt holds the instruction and four of the student's own correct solutions to other puzzles
(data/kk_student_style_examples.jsonl), so it is the same for every request (and cached by Anthropic models). The user
prompt holds the puzzle, the reference solution (Haiku's) and the student's four attempts at that puzzle, each marked
with how its final answer scored. Inputs come from data/kk_student_style.jsonl (prepare_student_style.py).

A rewrite is accepted when it is not empty, not truncated, correct, and does not mention the reference, the attempts
or the student. An item gets up to --max-tries teacher calls; try t goes only to items no earlier try accepted.

Every try is appended to <out-dir>/tries.jsonl as it arrives, so a stopped run loses only the requests in flight: rerun
the same command to resume. rewrites.jsonl, sft.jsonl, sft_gold.jsonl and summary.json are rebuilt from tries.jsonl at
the end of every run, a stopped one included. --force starts over.

    uv run src/student_style_rewrite.py --teacher anthropic/claude-haiku-5-5
"""

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import httpx
import litellm
from tqdm import tqdm

from benchmarks import kk

litellm.suppress_debug_info = True
ROOT = Path(__file__).resolve().parents[1]
TEACHER_REQUEST = {"max_tokens": 16384}
MENTIONS = re.compile(r"\breference\b|\bstudent\b|\b(?:the|my|your|previous|earlier|prior|these|those|their|its) "
                      r"attempts?\b|(?:given|provided|above|original) (?:solution|answer)", re.IGNORECASE)
CONCLUSION = """CONCLUSION:
(1) ...
(2) ...
(3) ..."""

SYSTEM = """You rewrite solutions to knights-and-knaves logic puzzles in the voice of one particular student, to make training data for that student.

The student is a small language model with a recognizable way of writing: how it opens a solution, how it lays out and labels its steps, how it restates the statements, how it phrases cases and contradictions, its formatting (headings, bold, numbered lists, bullets, notation), its sentence length and its overall length, and how it closes with the CONCLUSION block. Below are four of the student's own solutions, to other puzzles, that reached the right answer. They show how the student normally writes. They show its style only: their reasoning is not always sound, so do not take them as a model of logic.

{examples}

Each request gives you a puzzle, a reference solution that is known to be correct, and the student's own attempts at that same puzzle (most of them reach a wrong answer). Write one complete solution to the puzzle that:
- follows the logic of the reference solution, so that every step is valid and the final identities match it exactly;
- reads as if the student wrote it: use the openings, structure, phrasing, formatting and level of detail of the student, as they appear in its attempts and in the examples above. The attempts on this puzzle are the best guide to how the student would set up this particular problem; keep their setup and wording wherever it is correct;
- copies none of the attempts' mistakes: no misread or misquoted statements, wrong logic rules, unjustified jumps, false contradictions or skipped cases;
- stays within what the student could plausibly write: plain step-by-step reasoning, with no notation or vocabulary beyond what the student uses;
- stands on its own: do not mention the reference solution, the attempts, the student, or this rewriting task.

End with the identity of each character, in the order the puzzle introduces them, in this format:

{conclusion}

Reply with the solution only."""

EXAMPLE = """<example {n}>
Puzzle: {quiz}

Student's solution:
{response}
</example {n}>"""

USER = """Puzzle: {quiz}

Reference solution:
<reference>
{solution}
</reference>

The student's attempts at this puzzle:

{attempts}

Now write the solution in the student's style."""

ATTEMPT = """<attempt {n} ({status})>
{response}
</attempt {n}>"""

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--teacher", required=True, help="LiteLLM model string of the model that rewrites")
ap.add_argument("--teacher-args", type=json.loads, default={}, help=f"JSON passed to litellm.completion; default {TEACHER_REQUEST}")
ap.add_argument("--input", type=Path, default=ROOT / "data/kk_student_style.jsonl")
ap.add_argument("--examples", type=Path, default=ROOT / "data/kk_student_style_examples.jsonl")
ap.add_argument("--max-tries", type=int, default=3)
ap.add_argument("--limit", type=int, help="first N items only")
ap.add_argument("--concurrency", type=int, default=32)
ap.add_argument("--out-dir", type=Path, help="default: outputs/student_style/kk_<last part of --teacher>")
ap.add_argument("--force", action="store_true", help="delete this run's files and outputs and start over")
args = ap.parse_args()
OUTPUTS = ["rewrites.jsonl", "sft.jsonl", "sft_gold.jsonl", "summary.json"]
# LiteLLM's HTTP client keeps httpx's default pool of 100 connections, so more threads than that just queue
pool_limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
litellm.sync_transport = httpx.HTTPTransport(limits=pool_limits)
model_name, provider, *_ = litellm.get_llm_provider(args.teacher)
supported = litellm.get_supported_openai_params(model=model_name, custom_llm_provider=provider)
SEEDABLE = supported is None or "seed" in supported  # e.g. Anthropic rejects seed; its tries differ only by sampling


def read(path):
    """Records written to a file; a last line cut short by an interruption is dropped."""
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


items = list(read(args.input))[: args.limit]
examples = list(read(args.examples))
out_dir = args.out_dir or Path("outputs/student_style") / f"kk_{args.teacher.split('/')[-1]}"
out_dir.mkdir(parents=True, exist_ok=True)
teacher_args = {"num_retries": 3, **TEACHER_REQUEST, **args.teacher_args}
if args.force:
    for name in ["tries.jsonl", *OUTPUTS]:
        (out_dir / name).unlink(missing_ok=True)

system = SYSTEM.format(conclusion=CONCLUSION, examples="\n\n".join(
    EXAMPLE.format(n=n, quiz=ex["quiz"], response=ex["response"]) for n, ex in enumerate(examples, 1)))
# the system prompt is the same for every request: let Anthropic cache it
system_message = {"role": "system", "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                  if provider == "anthropic" else system}


def status(attempt):
    if attempt["finish_reason"] == "length":
        return "cut off for length, no final answer"
    if attempt["pred"] is None:
        return "no final answer"
    return "final answer correct" if attempt["correct"] else "final answer wrong"


def messages(item):
    attempts = "\n\n".join(ATTEMPT.format(n=n, status=status(a), response=a["response"])
                           for n, a in enumerate(item["attempts"], 1))
    return [system_message, {"role": "user", "content": USER.format(quiz=item["quiz"], solution=item["solution"],
                                                                    attempts=attempts)}]


def complete(item, t):
    """The answer part of a completion (anything up to </think> dropped), its finish reason and token usage."""
    kwargs = {**teacher_args, "seed": teacher_args.get("seed", 0) + t - 1}
    if not SEEDABLE:
        del kwargs["seed"]
    r = litellm.completion(model=args.teacher, messages=messages(item), **kwargs)
    text = r.choices[0].message.content or ""
    text = "" if text.lstrip().startswith("<think>") and "</think>" not in text else text  # cut off while thinking
    details = getattr(r.usage, "prompt_tokens_details", None)
    usage = {"prompt": r.usage.prompt_tokens, "completion": r.usage.completion_tokens,
             "cached": getattr(details, "cached_tokens", None) or 0}
    return text.rpartition("</think>")[2].strip(), r.choices[0].finish_reason, usage


def judge(item, t, text, finish, usage):
    scored = kk.score(item, text)
    reason = ("empty" if not text else "truncated" if finish == "length"
              else "no answer" if scored["pred"] is None else "wrong answer" if not scored["correct"]
              else "mentions the reference or attempts" if MENTIONS.search(text) else None)
    return {"id": item["id"], "try": t, "response": text, "finish_reason": finish, **scored, "rejected": reason,
            "usage": usage}


tries = {item["id"]: [] for item in items}
for row in read(out_dir / "tries.jsonl"):
    if row["id"] in tries:
        tries[row["id"]].append(row)
for rows in tries.values():
    rows.sort(key=lambda r: r["try"])


def accepted(i):
    return next((r for r in tries[i] if not r["rejected"]), None)


def finished(i):
    return accepted(i) is not None or len(tries[i]) >= args.max_tries


todo = [item for item in items if not finished(item["id"])]
print(f"{len(items) - len(todo)} items already finished, {len(todo)} to go")
path = out_dir / "tries.jsonl"  # rewrite without a cut-short last line, atomically, then append
tmp = path.with_suffix(".tmp")
tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for rows in tries.values() for r in rows))
tmp.replace(path)

pool = ThreadPoolExecutor(args.concurrency)
pending = {}  # future -> (item, try)
bar = tqdm(total=len(todo), desc="items")
interrupted = False
try:
    with open(path, "a") as f:
        def step(item):
            if finished(item["id"]):
                bar.update(1)
            else:
                t = len(tries[item["id"]]) + 1
                pending[pool.submit(complete, item, t)] = (item, t)

        for item in todo:
            step(item)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                item, t = pending.pop(fut)
                try:
                    row = judge(item, t, *fut.result())
                except litellm.BadRequestError as e:  # e.g. the prompt does not fit the teacher's context
                    row = {"id": item["id"], "try": t, "response": "", "finish_reason": None, "pred": None,
                           "correct": False, "rejected": f"request rejected: {type(e).__name__}", "error": str(e)[:500]}
                tries[item["id"]].append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                step(item)
except KeyboardInterrupt:
    interrupted = True
finally:
    pool.shutdown(wait=False, cancel_futures=True)
    bar.close()

kept = [item for item in items if accepted(item["id"])]
outputs = {  # the consolidated outputs, rebuilt from tries.jsonl on every run
    "rewrites.jsonl": [{**{k: v for k, v in item.items() if k != "attempts"},
                        "response": (accepted(item["id"]) or {}).get("response", ""), "accepted": bool(accepted(item["id"])),
                        "tries": tries[item["id"]]} for item in items],
    "sft.jsonl": [{"messages": [{"role": "user", "content": kk.prompt(item)},
                                {"role": "assistant", "content": accepted(item["id"])["response"]}]} for item in kept],
    "sft_gold.jsonl": [{"messages": [{"role": "user", "content": kk.prompt(item)},
                                     {"role": "assistant", "content": item["solution"]}]} for item in kept],
}
for name, rows in outputs.items():
    (out_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
all_tries = [r for rows in tries.values() for r in rows]
summary = {"items": len(items), "accepted": len(kept), "acceptance_rate": len(kept) / len(items) if items else None,
           "accepted_at_try": dict(sorted(Counter(accepted(i["id"])["try"] for i in kept).items())),
           "unfinished": sum(not finished(i["id"]) for i in items),
           "rejections": dict(Counter(r["rejected"] for r in all_tries if r["rejected"])),
           "tokens": {k: sum(r.get("usage", {}).get(k, 0) for r in all_tries) for k in ["prompt", "cached", "completion"]},
           "teacher": args.teacher, "teacher_args": teacher_args, "max_tries": args.max_tries,
           "examples": [(ex["id"], ex["run"]) for ex in examples], "input": str(args.input), "limit": args.limit}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
if interrupted:
    print(f"stopped: every try so far is in {path}; rerun the same command to resume", flush=True)
    os._exit(130)  # everything is on disk, so don't wait for the requests still in flight
