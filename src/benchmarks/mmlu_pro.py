"""MMLU-Pro (12,032 questions): official 5-shot CoT protocol and answer extraction (no random guess on a miss)."""

import json
import re
from pathlib import Path

from . import accuracy

REQUEST = {"max_tokens": 2048, "stop": ["Question:"]}  # stop as the official script does
RUNS = 1
METRIC = "accuracy"
summarize = accuracy

SHOTS = {}
with open(Path(__file__).resolve().parents[2] / "data/mmlu_pro_validation.jsonl") as f:
    for row in map(json.loads, f):
        SHOTS.setdefault(row["category"], []).append(row)


def example(row, cot="Let's think step by step."):
    options = "".join(f"{'ABCDEFGHIJ'[i]}. {o}\n" for i, o in enumerate(row["options"]))
    return f"Question: {row['question']}\nOptions: {options}Answer: {cot.removeprefix('A: ')}\n\n"


def prompt(item):
    head = (f"The following are multiple choice questions (with answers) about {item['category']}. Think step by "
            "step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n")
    return head + "".join(example(s, s["cot_content"]) for s in SHOTS[item["category"]][:5]) + example(item)


def score(item, response):
    text = response.replace("**", "")
    m = (re.search(r"answer is \(?([A-J])\)?", text) or re.search(r".*[aA]nswer:\s*([A-J])", text)
         or re.search(r"\b([A-J])\b(?!.*\b[A-J]\b)", text, re.DOTALL))
    pred = m.group(1) if m else None
    return {"pred": pred, "correct": pred == item["answer"]}
