"""Teacher feedback: the student answers a training question, a teacher writes a step-by-step review of everything
wrong with the attempt (without giving away the answer), and the student writes a new solution with its attempt and
the review in front of it. One round per item; the teacher writes no solution tokens, so every accepted answer is the
student's own.

The first answer is read from evaluate.py --train's r0.jsonl for the same student, --sample and --limit when that file
exists (--initial to point elsewhere); only items missing from it are generated. An answer is accepted when it is not
empty, not truncated and correct, and (for the revision) does not mention the earlier attempt or the review. Wrong,
unanswered and truncated first answers go to the teacher, who sees the problem, the reference solution, the attempt and
what is wrong with it. A review that states the final answer in the bench's answer format (\\boxed{} for math, a full
CONCLUSION for kk) goes back to the teacher to rewrite, up to 3 tries.

Every response is written the moment it arrives: first answers to <out-dir>/first.jsonl, each teacher try to
feedback.jsonl, revisions to revised.jsonl. Stop the run at any point (Ctrl-C, a crash, a killed job) and rerun the same
command: each item's next request is worked out from those files, so only the requests in flight are lost.
dialogues.jsonl, sft.jsonl, sft_gold.jsonl and summary.json are rebuilt from them at the end of every run, a stopped
one included. --force starts over.

    uv run src/teacher_feedback.py --bench math --sample 1000 \\
        --student hosted_vllm/Qwen/Qwen2.5-3B-Instruct --teacher anthropic/claude-haiku-5-5
"""

import argparse
import importlib
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import httpx
import litellm
from tqdm import tqdm

litellm.suppress_debug_info = True
# a revision has to stand on its own
MENTIONS = re.compile(r"\bfeedback\b|\b(?:the|teacher's) review\b|\bteacher\b|\b(?:previous|earlier|prior|first|original) "
                      r"(?:attempt|solution|answer)s?\b", re.IGNORECASE)
TEACHER_REQUEST = {"max_tokens": 4096}
FEEDBACK_TRIES = 3  # teacher calls per item before one whose review keeps revealing the answer is dropped

TEACHER_PROMPT = """You are a teacher reviewing a student's attempt at a problem. Below are the problem exactly as the student was given it, a reference solution that is known to be correct, and the student's attempt.

Problem (as given to the student):
<problem>
{problem}
</problem>

Reference solution (for your eyes only):
<reference>
{solution}
</reference>

Student's attempt:
<attempt>
{attempt}
</attempt>

{status}

Write a review that walks through the attempt step by step and points out everything that is wrong with it. This is the only review the student gets before trying again, so cover every mistake now, not just the first one.
- Go through the attempt in order, as a numbered list with one mistake per item.
- For each mistake, say where it is (quote or name the step), what is wrong, and why: a misread statement, a wrong rule or formula, a calculation slip, an unjustified step, a case that was skipped, a conclusion that does not follow from the work.
- Then point out anything a complete solution needs that the attempt leaves out, and whether the final answer is in the required format.
- You may say which idea or check the student should use instead, but do not state the final answer, do not give the correct value of anything the problem asks for, and do not write out a corrected solution.
- Speak to the student directly, keep each item short, and do not mention the reference solution.

Now write your review."""

TEACHER_REVEALED = ("That review gives away the final answer. Rewrite it so that it still points out every mistake, "
                    "without stating the answer.")

STATUS = {  # what the teacher is told about a first answer, by rejection reason; other first answers get no review
    "wrong answer": "The final answer of this attempt is wrong.",
    "no answer": "This attempt does not give a final answer in the format the problem asks for.",
    "truncated": "This attempt was cut off for length before it finished, so it has no final answer.",
}

STUDENT_PROMPT = """{problem}

Below is your earlier attempt at this problem, followed by a teacher's review of its mistakes.

--- Your attempt ---
{attempt}

--- Teacher's review ---
{feedback}

Using the review, write a new solution from scratch and fix every mistake it points out. The solution must be complete and stand on its own: do not refer to your earlier attempt or to the review. Give your final answer in the format asked for above."""

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bench", required=True, choices=["math", "kk"], help="training set: data/<bench>_train.jsonl")
ap.add_argument("--student", required=True, help="LiteLLM model string of the model that answers")
ap.add_argument("--student-args", type=json.loads, default={}, help="JSON passed to litellm.completion; overrides bench defaults")
ap.add_argument("--teacher", required=True, help="LiteLLM model string of the model that writes the review")
ap.add_argument("--teacher-args", type=json.loads, default={}, help=f"JSON passed to litellm.completion; default {TEACHER_REQUEST}")
ap.add_argument("--initial", type=Path, help="first answers to reuse; default: evaluate.py --train's "
                                             "outputs/train/<student>[_sample<N>][_limit<N>]/<bench>/r0.jsonl")
ap.add_argument("--sample", type=int, help="random subset of N items (seed 0, same items as evaluate.py --train --sample N)")
ap.add_argument("--limit", type=int, help="first N items only")
ap.add_argument("--concurrency", type=int, default=32)
ap.add_argument("--out-dir", type=Path, help="default: outputs/teacher_feedback/<bench>_<student>_by_<teacher>")
ap.add_argument("--force", action="store_true", help="delete this run's files and outputs and start over")
args = ap.parse_args()
RESPONSES = ["first.jsonl", "feedback.jsonl", "revised.jsonl"]
OUTPUTS = ["dialogues.jsonl", "sft.jsonl", "sft_gold.jsonl", "summary.json"]
# LiteLLM's HTTP client keeps httpx's default pool of 100 connections, so more threads than that just queue
pool_limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
litellm.sync_transport = httpx.HTTPTransport(limits=pool_limits)


def short(model):
    return model.split("/")[-1]


bench = importlib.import_module(f"benchmarks.{args.bench}")
with open(Path(__file__).resolve().parents[1] / f"data/{args.bench}_train.jsonl") as f:
    items = [json.loads(line) for line in f]
if args.sample:  # the same draw as evaluate.py, so the ids of its --train run line up
    items = [items[i] for i in sorted(random.Random(0).sample(range(len(items)), args.sample))]
items = items[: args.limit]
out_dir = args.out_dir or Path("outputs/teacher_feedback") / f"{args.bench}_{short(args.student)}_by_{short(args.teacher)}"
out_dir.mkdir(parents=True, exist_ok=True)
student_args = {"num_retries": 3, **bench.REQUEST, **args.student_args}
teacher_args = {"num_retries": 3, **TEACHER_REQUEST, **args.teacher_args}
initial = args.initial or (Path("outputs/train") / (short(args.student) + (f"_sample{args.sample}" if args.sample else "")
                                                     + (f"_limit{args.limit}" if args.limit else "")) / args.bench / "r0.jsonl")

if args.force:
    for name in RESPONSES + OUTPUTS:
        (out_dir / name).unlink(missing_ok=True)


def read(path):
    """Records already written to a file; a last line cut short by an interruption is dropped."""
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def complete(model, kwargs, messages):
    """The answer part of a completion (anything up to </think> dropped) and its finish reason. Runs on a pool thread."""
    choice = litellm.completion(model=model, messages=messages, **kwargs).choices[0]
    text = choice.message.content or ""
    text = "" if text.lstrip().startswith("<think>") and "</think>" not in text else text  # cut off while thinking
    return text.rpartition("</think>")[2].strip(), choice.finish_reason


def judge(item, text, finish, revised):
    """Score on the main thread only: the math grader times out with SIGALRM."""
    scored = bench.score(item, text)
    reason = ("empty" if not text else "truncated" if finish == "length"
              else "no answer" if scored["pred"] is None else "wrong answer" if not scored["correct"]
              else "mentions the earlier attempt" if revised and MENTIONS.search(text) else None)
    return {"response": text, "finish_reason": finish, **scored, "rejected": reason}


def teacher_messages(item, tries):
    """The review request, then each earlier try that gave the answer away and the request to rewrite it."""
    first = firsts[item["id"]]
    msgs = [{"role": "user", "content": TEACHER_PROMPT.format(problem=bench.prompt(item), solution=item["solution"],
                                                              attempt=first["response"], status=STATUS[first["rejected"]])}]
    for fb in tries:
        if fb["feedback"]:
            msgs += [{"role": "assistant", "content": fb["feedback"]}, {"role": "user", "content": TEACHER_REVEALED}]
    return msgs


def student_prompt(item, feedback):
    return STUDENT_PROMPT.format(problem=bench.prompt(item), attempt=firsts[item["id"]]["response"], feedback=feedback)


def finished(item_id):
    """Accepted at once, not worth a review, or revised (or dropped because every review gave the answer away)."""
    first = firsts.get(item_id)
    return first is not None and (first["rejected"] not in STATUS or item_id in revised)


# resume: everything already written, by id
firsts = {row["id"]: row for row in read(out_dir / "first.jsonl")}
feedback = {}  # id -> that item's teacher tries, in order
for row in read(out_dir / "feedback.jsonl"):
    feedback.setdefault(row["id"], {})[row["try"]] = row
feedback = {i: [by_try[k] for k in sorted(by_try)] for i, by_try in feedback.items()}
revised = {row["id"]: row for row in read(out_dir / "revised.jsonl")}

# first answers from evaluate.py --train, for the items it has
if initial.exists():
    train = {row["id"]: row for row in read(initial)}
    reused = [item for item in items if item["id"] not in firsts and item["id"] in train]
    for item in reused:
        r = train[item["id"]]
        firsts[item["id"]] = {"id": item["id"], "source": str(initial), **judge(item, r["response"], r["finish_reason"], False)}
    summary_path = initial.with_suffix(".summary.json")
    note = f"model_args {json.loads(summary_path.read_text())['model_args']}" if summary_path.exists() else "run not finished"
    print(f"{len(reused)} first answers not yet in first.jsonl read from {initial} ({note})")
else:
    print(f"{initial} not found, so first answers are generated here")

todo = [item for item in items if not finished(item["id"])]
print(f"{len(items) - len(todo)} items already finished, {len(todo)} to go")
files = {}
for name, rows in zip(RESPONSES, [firsts.values(), [r for tries in feedback.values() for r in tries], revised.values()]):
    path = out_dir / name  # rewrite each file without a cut-short last line, atomically, then append
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)
    files[name] = open(path, "a")

# Each item is a small state machine: first answer -> review (rewritten while it reveals the answer) -> revision.
# Every response is written to its file the moment it arrives, and the next request is worked out from what has been
# written, so a resumed run picks up exactly where the stopped one was. The pool only makes requests; scoring, the next
# request and the file writes happen here, on the main thread.
pool = ThreadPoolExecutor(args.concurrency)
pending = {}  # future -> (item, "first" | "teacher" | "revised")
bar = tqdm(total=len(todo), desc="items")


def write(name, row):
    files[name].write(json.dumps(row, ensure_ascii=False) + "\n")
    files[name].flush()


def step(item):
    """Send the item's next request, or count it as finished."""
    i = item["id"]
    if i not in firsts:
        msgs = [{"role": "user", "content": bench.prompt(item)}]
        pending[pool.submit(complete, args.student, student_args, msgs)] = (item, "first")
        return
    if finished(i):
        bar.update(1)
        return
    tries = feedback.get(i, [])
    if tries and tries[-1]["feedback"] and not tries[-1]["reveals"]:  # a usable review: the student's turn
        msgs = [{"role": "user", "content": student_prompt(item, tries[-1]["feedback"])}]
        pending[pool.submit(complete, args.student, student_args, msgs)] = (item, "revised")
    elif len(tries) < FEEDBACK_TRIES:
        pending[pool.submit(complete, args.teacher, teacher_args, teacher_messages(item, tries))] = (item, "teacher")
    else:
        revised[i] = {"id": i, "response": None,
                      "rejected": "review reveals the answer" if tries[-1]["feedback"] else "empty review"}
        write("revised.jsonl", revised[i])
        step(item)


interrupted = False
try:
    for item in todo:
        step(item)
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for fut in done:
            item, kind = pending.pop(fut)
            i = item["id"]
            if kind == "teacher":
                text, finish = fut.result()
                tries = feedback.setdefault(i, [])
                row = {"id": i, "try": len(tries) + 1, "feedback": text, "finish_reason": finish,
                       "reveals": bool(text) and bench.score(item, text)["correct"]}
                write("feedback.jsonl", row)
                tries.append(row)
            elif kind == "first":
                text, finish = fut.result()
                firsts[i] = {"id": i, "source": "generated", **judge(item, text, finish, False)}
                write("first.jsonl", firsts[i])
            else:
                try:
                    text, finish = fut.result()
                except litellm.BadRequestError as e:  # most often the attempt and review don't fit the student's context
                    revised[i] = {"id": i, "response": None, "rejected": f"request rejected: {type(e).__name__}",
                                  "error": str(e)[:500]}
                else:
                    revised[i] = {"id": i, "feedback_tries": len(feedback[i]), **judge(item, text, finish, True)}
                write("revised.jsonl", revised[i])
            step(item)
except KeyboardInterrupt:
    interrupted = True
finally:
    pool.shutdown(wait=False, cancel_futures=True)
    bar.close()
    for f in files.values():
        f.close()


def accepted(i):
    """The item's accepted response and where it came from, if any."""
    if i in firsts and not firsts[i]["rejected"]:
        return firsts[i]["response"], "first"
    if i in revised and not revised[i]["rejected"]:
        return revised[i]["response"], "revised"
    return None, None


kept = [item for item in items if accepted(item["id"])[0]]
outputs = {  # the consolidated outputs, rebuilt from the response files on every run
    "dialogues.jsonl": [{**item, "accepted_from": accepted(item["id"])[1], "response": accepted(item["id"])[0] or "",
                         "first": firsts.get(item["id"]), "feedback": feedback.get(item["id"], []),
                         "revised": revised.get(item["id"])} for item in items],
    "sft.jsonl": [{"messages": [{"role": "user", "content": bench.prompt(item)},
                                {"role": "assistant", "content": accepted(item["id"])[0]}]} for item in kept],
    "sft_gold.jsonl": [{"messages": [{"role": "user", "content": bench.prompt(item)},
                                     {"role": "assistant", "content": item["solution"]}]} for item in kept],
}
for name, rows in outputs.items():
    (out_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
ids = [item["id"] for item in items]
reviewed = [i for i in ids if i in firsts and firsts[i]["rejected"] in STATUS]
fixed = sum(accepted(i)[1] == "revised" for i in ids)
summary = {"items": len(items), "accepted": len(kept), "acceptance_rate": len(kept) / len(items),
           "accepted_first": sum(accepted(i)[1] == "first" for i in ids),
           "reviewed": len(reviewed), "fixed_by_review": fixed, "fix_rate": fixed / len(reviewed) if reviewed else None,
           "unfinished": sum(not finished(i) for i in ids),
           "first_rejections": dict(Counter(firsts[i]["rejected"] for i in ids if i in firsts and firsts[i]["rejected"])),
           "revised_rejections": dict(Counter(revised[i]["rejected"] for i in ids if i in revised and revised[i]["rejected"])),
           "review_rewrites": sum(len(feedback[i]) - 1 for i in ids if i in feedback),
           "first_from_train": sum(firsts[i]["source"] != "generated" for i in ids if i in firsts),
           "student": args.student, "student_args": student_args, "teacher": args.teacher, "teacher_args": teacher_args,
           "initial": str(initial), "sample": args.sample, "limit": args.limit}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
if interrupted:
    print(f"stopped: every response so far is in {out_dir}/{{{','.join(RESPONSES)}}}; rerun the same command to resume",
          flush=True)  # os._exit skips the usual flush
    os._exit(130)  # everything is on disk, so don't wait for the requests still in flight
