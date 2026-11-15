"""Knights and Knaves (700 puzzles, 100 each for 2-8 people): zero-shot CoT prompt of the K&K paper
(AlphaPav/mem-kk-logic), system instruction folded into the user turn. A puzzle counts as correct only if every
person's identity is right. The answer is read from the text after the last "CONCLUSION:"; each person takes the
role stated for them there ("<name> is a knight", "<name>: knave", ...), and a person with no role or with both
counts as wrong. No "CONCLUSION:" means no answer.
"""

import re

from . import accuracy

REQUEST = {"max_tokens": 16384}  # small models reason at length, even on 2-person puzzles
RUNS = 1
METRIC = "accuracy"
# self_rewrite.py reads this instead of kk_train: the 6,018 train puzzles Claude Haiku 5.5 solved correctly, with its
# solution as the reference (kk_train's own solutions follow one fixed template)
REWRITE_DATA = "kk_train_haiku"
PROMPT = """Your task is to solve a logical reasoning problem. You are given set of statements from which you must logically deduce the identity of a set of characters.

You must infer the identity of each character. First, explain your reasoning. At the end of your answer, you must clearly state the identity of each character by following the format:

CONCLUSION:
(1) ...
(2) ...
(3) ...

### Question: {quiz}
### Answer:
"""


REWRITE_PROMPT = """Below is a logic puzzle about knights and knaves, followed by a reference solution that is known to be correct.

Solve the puzzle yourself and write the solution in your own words, the way you would normally explain it. Use the reference solution only to make sure your reasoning and your final answer are right. Do not copy its sentences or layout, and do not mention that you were given a reference.

Puzzle:
{quiz}

Reference solution (for your eyes only):
{solution}

Now write your own solution. First explain your reasoning. At the end of your answer, you must clearly state the identity of each character by following the format:

CONCLUSION:
(1) ...
(2) ...
(3) ..."""


def prompt(item):
    return PROMPT.format(quiz=item["quiz"])


def rewrite_prompt(item):
    return REWRITE_PROMPT.format(quiz=item["quiz"], solution=item["solution"])


def score(item, response):
    text = response.split("### Question")[0].replace("**", "")  # stop where a model starts a new question
    marks = list(re.finditer(r"(?i)conclusion\s*:", text))
    if not marks:
        return {"pred": None, "correct": False}
    tail = text[marks[-1].end():].split("###")[0]
    pred = []
    for name in item["names"]:
        roles = {m.lower() for m in re.findall(
            rf"\b{re.escape(name)}\b[\s:,\-–—]*(?:is\s+)?(?:an?\s+)?(knight|knave)s?\b", tail, re.IGNORECASE)}
        pred.append(roles.pop() if len(roles) == 1 else None)
    return {"pred": pred, "correct": pred == item["answer"]}


def summarize(records):
    by_people = {}
    for r in records:
        by_people.setdefault(int(r["id"].partition("ppl")[0]), []).append(r["correct"])
    return {**accuracy(records), "by_people": {n: sum(c) / len(c) for n, c in sorted(by_people.items())}}
