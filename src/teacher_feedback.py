"""Teacher feedback: the student answers a training question, a teacher reviews the attempt and writes feedback that
points out the mistake without giving away the answer, and the student tries again with its attempts and the feedback
in context. Up to --max-turns rounds of feedback per item; the teacher writes no solution tokens, so every accepted
answer is the student's own.

Turn 0 is the student's plain answer to the eval prompt. It is read from evaluate.py --train's r0.jsonl for the same
student, --sample and --limit when that file exists (--initial to point elsewhere); only items missing from it are
generated. An answer is accepted when it is not empty, not truncated and correct, and (after turn 0) does not mention
the earlier attempts or the feedback. The teacher sees the problem, the reference solution, every attempt and what was
wrong with it, and keeps its own conversation across turns. Feedback that states the final answer in the bench's answer
format (\\boxed{} for math, a full CONCLUSION for kk) goes back to the teacher to rewrite, up to 3 tries.

The student sees one user message: the eval prompt, then each attempt followed by the feedback on it.

Items move through their turns independently. Every response is written the moment it arrives: each teacher try to
<out-dir>/feedback<t>.jsonl, each student answer (with the feedback it answered) to <out-dir>/turn<t>.jsonl. Stop the
run at any point (Ctrl-C, a crash, a killed job) and rerun the same command: it works out each item's next request
from those files, so only the requests in flight are lost. dialogues.jsonl, sft.jsonl, sft_gold.jsonl and summary.json
are rebuilt from them at the end of every run, a stopped one included. --force starts over.

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
# an answer written after feedback has to stand on its own
MENTIONS = re.compile(r"\bfeedback\b|\bteacher\b|\battempt \d|\b(?:previous|earlier|prior|last|first|original) "
                      r"(?:attempt|solution|answer)s?\b", re.IGNORECASE)
TEACHER_REQUEST = {"max_tokens": 4096}
FEEDBACK_TRIES = 3  # teacher calls per turn before an item whose feedback keeps revealing the answer is dropped

TEACHER_FIRST = """You are a teacher reviewing a student's attempt at a problem. Below are the problem exactly as the student was given it, a reference solution that is known to be correct, and the student's attempt.

Write feedback that lets the student find and fix the mistake on their own:
- Point to the first place where the reasoning goes wrong, or the step that is missing, and say why it is wrong.
- You may hint at the right idea or the next step, but do not state the final answer, do not give the correct value of anything the problem asks for, and do not write out a corrected solution.
- Keep it short: a few sentences, or a few bullet points for several separate mistakes.
- Speak to the student directly and do not mention the reference solution.

Problem (as given to the student):
<problem>
{problem}
</problem>

Reference solution (for your eyes only):
<reference>
{solution}
</reference>

Student's attempt 1:
<attempt>
{attempt}
</attempt>

{status}

Now write your feedback."""

TEACHER_NEXT = """The student read your feedback and tried again.

Student's attempt {n}:
<attempt>
{attempt}
</attempt>

{status}

Write feedback on this attempt, following the same rules. Do not repeat points the student has already fixed."""

TEACHER_REVEALED = ("That feedback gives away the final answer. Rewrite it so that it points out the mistake and the "
                    "direction to take, without stating the answer.")

STATUS = {  # what the teacher is told about an attempt, by rejection reason; any other reason ends the item
    "wrong answer": "This attempt's final answer is wrong.",
    "no answer": "This attempt does not give a final answer in the format the problem asks for.",
    "truncated": "This attempt was cut off for length before it finished. Help the student reach the answer more directly.",
    "empty": "The student wrote nothing.",
    "mentions earlier attempts": ("This attempt's final answer is right, but the solution refers to earlier attempts or "
                                  "to feedback. Tell the student to rewrite it as a standalone solution."),
}

STUDENT_HISTORY = """

You have already attempted this problem. Below are your earlier attempts, each followed by a teacher's feedback on it.
{history}
Taking the feedback into account, write a new solution from scratch. It must be complete and stand on its own: do not refer to your earlier attempts or to the feedback. Give your final answer in the format asked for above."""

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--bench", required=True, choices=["math", "kk"], help="training set: data/<bench>_train.jsonl")
ap.add_argument("--student", required=True, help="LiteLLM model string of the model that answers")
ap.add_argument("--student-args", type=json.loads, default={}, help="JSON passed to litellm.completion; overrides bench defaults")
ap.add_argument("--teacher", required=True, help="LiteLLM model string of the model that writes feedback")
ap.add_argument("--teacher-args", type=json.loads, default={}, help=f"JSON passed to litellm.completion; default {TEACHER_REQUEST}")
ap.add_argument("--max-turns", type=int, default=3, help="rounds of feedback after the first answer")
ap.add_argument("--initial", type=Path, help="turn-0 answers to reuse; default: evaluate.py --train's "
                                             "outputs/train/<student>[_sample<N>][_limit<N>]/<bench>/r0.jsonl")
ap.add_argument("--sample", type=int, help="random subset of N items (seed 0, same items as evaluate.py --train --sample N)")
ap.add_argument("--limit", type=int, help="first N items only")
ap.add_argument("--concurrency", type=int, default=32)
ap.add_argument("--out-dir", type=Path, help="default: outputs/teacher_feedback/<bench>_<student>_by_<teacher>")
ap.add_argument("--force", action="store_true", help="delete this run's turn files and outputs and start over")
args = ap.parse_args()
OUTPUTS = ["dialogues.jsonl", "sft.jsonl", "sft_gold.jsonl", "summary.json"]
# LiteLLM's HTTP client keeps httpx's default pool of 100 connections, so more threads than that just queue
pool_limits = httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)
litellm.sync_transport = httpx.HTTPTransport(limits=pool_limits)


def seedable(model):  # e.g. Anthropic rejects seed; its requests differ only by sampling
    model_name, provider, *_ = litellm.get_llm_provider(model)
    supported = litellm.get_supported_openai_params(model=model_name, custom_llm_provider=provider)
    return supported is None or "seed" in supported


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
SEEDABLE = {args.student: seedable(args.student), args.teacher: seedable(args.teacher)}
initial = args.initial or (Path("outputs/train") / (short(args.student) + (f"_sample{args.sample}" if args.sample else "")
                                                     + (f"_limit{args.limit}" if args.limit else "")) / args.bench / "r0.jsonl")

if args.force:
    for path in [*out_dir.glob("turn*.jsonl"), *out_dir.glob("feedback*.jsonl"), *(out_dir / n for n in OUTPUTS)]:
        path.unlink(missing_ok=True)


def read(path):
    """Records already written to a file; a last line cut short by an interruption is dropped."""
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load(path):
    """A turn file (or evaluate.py's r<k>.jsonl), by id."""
    return {row["id"]: row for row in read(path)}


def load_tries(path):
    """A feedback file, by id: that item's teacher tries in order."""
    tries = {}
    for row in read(path):
        tries.setdefault(row["id"], {})[row["try"]] = row
    return {i: [by_try[k] for k in sorted(by_try)] for i, by_try in tries.items()}


def complete(model, kwargs, messages, seed):
    """The answer part of a completion (anything up to </think> dropped) and its finish reason. Runs on a pool thread."""
    kwargs = {**kwargs, "seed": kwargs.get("seed", 0) + seed}
    if not SEEDABLE[model]:
        del kwargs["seed"]
    choice = litellm.completion(model=model, messages=messages, **kwargs).choices[0]
    text = choice.message.content or ""
    text = "" if text.lstrip().startswith("<think>") and "</think>" not in text else text  # cut off while thinking
    return text.rpartition("</think>")[2].strip(), choice.finish_reason


def judge(item, text, finish, turn):
    """Score on the main thread only: the math grader times out with SIGALRM."""
    scored = bench.score(item, text)
    reason = ("empty" if not text else "truncated" if finish == "length"
              else "no answer" if scored["pred"] is None else "wrong answer" if not scored["correct"]
              else "mentions earlier attempts" if turn and MENTIONS.search(text) else None)
    return {"response": text, "finish_reason": finish, **scored, "rejected": reason}


def teacher_messages(item, rows):
    """The teacher's conversation so far: its first review, then each revision and the feedback it got."""
    msgs = [{"role": "user", "content": TEACHER_FIRST.format(problem=bench.prompt(item), solution=item["solution"],
                                                             attempt=rows[0]["response"], status=STATUS[rows[0]["rejected"]])}]
    for n, row in enumerate(rows[1:], 2):
        msgs += [{"role": "assistant", "content": row["feedback"]},
                 {"role": "user", "content": TEACHER_NEXT.format(n=n, attempt=row["response"], status=STATUS[row["rejected"]])}]
    return msgs


def student_prompt(item, rows, feedback):
    """The eval prompt, then each attempt followed by the feedback on it. rows[k] holds attempt k+1; the feedback on
    it is in rows[k+1], or is `feedback` for the latest attempt."""
    feedbacks = [row["feedback"] for row in rows[1:]] + [feedback]
    history = "".join(f"\n--- Attempt {n} ---\n{row['response']}\n\n--- Feedback on attempt {n} ---\n{fb}\n"
                      for n, (row, fb) in enumerate(zip(rows, feedbacks), 1))
    return bench.prompt(item) + STUDENT_HISTORY.format(history=history)


def finished(rows):
    """Accepted, out of turns, or ended by a reason the teacher can't act on (revealing feedback, a rejected request)."""
    return bool(rows) and (not rows[-1]["rejected"] or len(rows) > args.max_turns or rows[-1]["rejected"] not in STATUS)


# resume: every turn already written, per item, in order
by_turn = [load(out_dir / f"turn{t}.jsonl") for t in range(args.max_turns + 1)]
history = {}
for item in items:
    rows = history[item["id"]] = []
    for done in by_turn:
        if item["id"] not in done:
            break
        rows.append(done[item["id"]])

# turn 0 from evaluate.py --train, for the items it has
if initial.exists():
    train, reused = load(initial), 0
    for item in items:
        if not history[item["id"]] and item["id"] in train:
            r = train[item["id"]]
            row = {"id": item["id"], "turn": 0, "source": str(initial), **judge(item, r["response"], r["finish_reason"], 0)}
            history[item["id"]].append(row)
            by_turn[0][item["id"]] = row
            reused += 1
    summary_path = initial.with_suffix(".summary.json")
    note = f"model_args {json.loads(summary_path.read_text())['model_args']}" if summary_path.exists() else "run not finished"
    print(f"turn 0: {reused} answers not yet in turn0.jsonl read from {initial} ({note})")
else:
    print(f"turn 0: {initial} not found, so first answers are generated here")

# feedback written so far, per round and id: one record per teacher try, in order
feedback = {t: load_tries(out_dir / f"feedback{t}.jsonl") for t in range(1, args.max_turns + 1)}

todo = [item for item in items if not finished(history[item["id"]])]
print(f"{len(items) - len(todo)} items already finished, {len(todo)} to go")
files = {}
for name, rows in [*((f"turn{t}", list(done.values())) for t, done in enumerate(by_turn)),
                   *((f"feedback{t}", [r for tries in fb.values() for r in tries]) for t, fb in feedback.items())]:
    path = out_dir / f"{name}.jsonl"  # rewrite each file without a cut-short last line, atomically, then append
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)
    files[name] = open(path, "a")

# Each item is a small state machine: student answer -> teacher feedback (rewritten while it reveals the answer) ->
# student answer -> ... Every response is written to its file the moment it arrives, and the next request is worked
# out from what has been written, so a resumed run picks up exactly where the stopped one was. The pool only makes
# requests; scoring, the next request and the file writes happen here, on the main thread.
pool = ThreadPoolExecutor(args.concurrency)
pending = {}  # future -> (item, "student" | "teacher")
bar = tqdm(total=len(todo), desc="items")
n_accepted = sum(bool(rows) and not rows[-1]["rejected"] for rows in history.values())


def write(name, row):
    files[name].write(json.dumps(row, ensure_ascii=False) + "\n")
    files[name].flush()


def emit(item, fields):
    rows = history[item["id"]]
    row = {"id": item["id"], "turn": len(rows), **fields}
    write(f"turn{row['turn']}", row)
    rows.append(row)


def ask_student(item, prompt, seed):
    pending[pool.submit(complete, args.student, student_args, [{"role": "user", "content": prompt}], seed)] = (item, "student")


def step(item):
    """Send the item's next request, or count it as finished."""
    global n_accepted
    rows = history[item["id"]]
    if not rows:
        ask_student(item, bench.prompt(item), 0)
        return
    if finished(rows):
        n_accepted += not rows[-1]["rejected"]
        bar.update(1)
        bar.set_postfix(accepted=n_accepted)
        return
    t = len(rows)
    tries = feedback[t].get(item["id"], [])
    if tries and tries[-1]["feedback"] and not tries[-1]["reveals"]:  # usable feedback: the student's turn
        ask_student(item, student_prompt(item, rows, tries[-1]["feedback"]), t)
    elif len(tries) < FEEDBACK_TRIES:  # the teacher's turn, told off for every earlier try that gave the answer away
        msgs = teacher_messages(item, rows)
        for fb in tries:
            if fb["feedback"]:
                msgs += [{"role": "assistant", "content": fb["feedback"]}, {"role": "user", "content": TEACHER_REVEALED}]
        seed = t - 1 + 100 * len(tries)
        pending[pool.submit(complete, args.teacher, teacher_args, msgs, seed)] = (item, "teacher")
    else:
        emit(item, {**feedback_fields(tries), "response": None,
                    "rejected": "feedback reveals the answer" if tries[-1]["feedback"] else "empty feedback"})
        step(item)


def feedback_fields(tries):
    return {"feedback": tries[-1]["feedback"], "feedback_finish_reason": tries[-1]["finish_reason"],
            "feedback_tries": len(tries)} if tries else {"source": "generated"}


interrupted = False
try:
    for item in todo:
        step(item)
    while pending:
        done, _ = wait(pending, return_when=FIRST_COMPLETED)
        for fut in done:
            item, kind = pending.pop(fut)
            rows = history[item["id"]]
            t = len(rows)
            if kind == "teacher":
                text, finish = fut.result()
                tries = feedback[t].setdefault(item["id"], [])
                row = {"id": item["id"], "turn": t, "try": len(tries) + 1, "feedback": text, "finish_reason": finish,
                       "reveals": bool(text) and bench.score(item, text)["correct"]}
                write(f"feedback{t}", row)
                tries.append(row)
            else:
                fields = feedback_fields(feedback[t].get(item["id"], []) if t else [])
                try:
                    text, finish = fut.result()
                except litellm.BadRequestError as e:  # most often the attempts no longer fit the student's context
                    if not t:
                        raise
                    emit(item, {**fields, "response": None, "rejected": f"request rejected: {type(e).__name__}",
                                "error": str(e)[:500]})
                else:
                    emit(item, {**fields, **judge(item, text, finish, t)})
            step(item)
except KeyboardInterrupt:
    interrupted = True
finally:
    pool.shutdown(wait=False, cancel_futures=True)
    bar.close()
    for f in files.values():
        f.close()

accepted = {i: rows[-1] for i, rows in history.items() if rows and not rows[-1]["rejected"]}
kept = [item for item in items if item["id"] in accepted]
outputs = {  # the consolidated outputs, rebuilt from the turn files on every run
    "dialogues.jsonl": [{**item, "accepted": item["id"] in accepted,
                         "accepted_turn": accepted[item["id"]]["turn"] if item["id"] in accepted else None,
                         "response": accepted[item["id"]]["response"] if item["id"] in accepted else "",
                         "turns": [{k: v for k, v in r.items() if k != "id"} for r in history[item["id"]]]}
                        for item in items],
    "sft.jsonl": [{"messages": [{"role": "user", "content": bench.prompt(item)},
                                {"role": "assistant", "content": accepted[item["id"]]["response"]}]} for item in kept],
    "sft_gold.jsonl": [{"messages": [{"role": "user", "content": bench.prompt(item)},
                                     {"role": "assistant", "content": item["solution"]}]} for item in kept],
}
for name, rows in outputs.items():
    (out_dir / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
all_rows = [r for rows in history.values() for r in rows]
summary = {"items": len(items), "accepted": len(kept), "acceptance_rate": len(kept) / len(items),
           "accepted_at_turn": dict(sorted(Counter(r["turn"] for r in accepted.values()).items())),
           "unfinished": sum(not finished(rows) for rows in history.values()),
           "final_rejections": dict(Counter(rows[-1]["rejected"] for rows in history.values()
                                            if rows and rows[-1]["rejected"])),
           "rejections_by_turn": {t: dict(Counter(r["rejected"] for r in all_rows if r["turn"] == t and r["rejected"]))
                                  for t in range(args.max_turns + 1)},
           "feedback_rewrites": sum(r.get("feedback_tries", 1) - 1 for r in all_rows),
           "turn0_from_train": sum(r["turn"] == 0 and r["source"] != "generated" for r in all_rows),
           "student": args.student, "student_args": student_args, "teacher": args.teacher, "teacher_args": teacher_args,
           "max_turns": args.max_turns, "initial": str(initial),
           "sample": args.sample, "limit": args.limit}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
if interrupted:
    print(f"stopped: every response so far is in {out_dir}/turn*.jsonl and feedback*.jsonl; rerun the same command to resume",
          flush=True)  # os._exit skips the usual flush
    os._exit(130)  # everything is on disk, so don't wait for the requests still in flight
