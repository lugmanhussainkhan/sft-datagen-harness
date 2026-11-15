"""Builds the inputs of student_style_rewrite.py for kk.

data/kk_student_style.jsonl: one row per puzzle of --train-set (the self-rewrite SFT subset, user/assistant pairs
without ids, matched to kk_train_haiku.jsonl by quiz), with Haiku's reference solution and the student's four attempts
from evaluate.py --train --all-runs (r0..r3.jsonl in --trials; without --all-runs, solved puzzles get no
further attempts).

data/kk_student_style_examples.jsonl: the few-shot examples, the student's own correct answers to puzzles outside
--train-set. They were picked by hand (EXAMPLES): the base model's correct answers on 3+ people almost always reach the
right identities through a misread statement or a wrong step somewhere, so these are the soundest found, not flawless.

    uv run src/prepare_student_style.py
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [("2ppl_207", 0), ("3ppl_707", 1), ("4ppl_338", 1), ("4ppl_856", 2)]  # (id, run)
QUESTION = "### Question: "
FIELDS = ["id", "people", "quiz", "names", "answer"]

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--train-set", type=Path, default=ROOT.parent / "kk_Qwen2.5-1.5B-Instruct/train.jsonl")
ap.add_argument("--trials", type=Path, default=ROOT.parent / "qwen2.5-1.5b-kk-train-r4",
                help="evaluate.py --train output for kk: r0.jsonl .. r3.jsonl")
ap.add_argument("--runs", type=int, default=4)
args = ap.parse_args()


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


haiku = {row["quiz"]: row for row in read(ROOT / "data/kk_train_haiku.jsonl")}
kk_train = {row["id"]: row for row in read(ROOT / "data/kk_train.jsonl")}
runs = [{row["id"]: row for row in read(args.trials / f"r{k}.jsonl")} for k in range(args.runs)]

rows = []
for pair in read(args.train_set):
    user = pair["messages"][0]["content"]
    quiz = user.split(QUESTION, 1)[1].rsplit("\n### Answer:", 1)[0]
    item = haiku[quiz]  # KeyError: a puzzle of --train-set that Haiku did not solve
    attempts = [{"run": k, **{key: runs[k][item["id"]][key] for key in ["response", "finish_reason", "pred", "correct"]}}
                for k in range(args.runs)]
    rows.append({**{key: item[key] for key in FIELDS}, "solution": item["solution"], "attempts": attempts})
assert len({r["id"] for r in rows}) == len(rows), "duplicate puzzles in --train-set"

train_ids = {r["id"] for r in rows}
examples = []
for i, k in EXAMPLES:
    assert i not in train_ids, f"example {i} is one of the puzzles to rewrite"
    run = runs[k][i]
    assert run["correct"] and run["finish_reason"] == "stop", f"example {i} r{k} is not a finished correct answer"
    examples.append({**{key: kk_train[i][key] for key in FIELDS}, "run": k, "response": run["response"]})

for name, data in [("kk_student_style.jsonl", rows), ("kk_student_style_examples.jsonl", examples)]:
    (ROOT / "data" / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data))
    print(f"data/{name}: {len(data)} rows")
