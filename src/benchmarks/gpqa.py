"""GPQA Diamond (198 questions): zero-shot CoT as in OpenAI simple-evals; option order is fixed in the data file."""

import re

from . import accuracy

REQUEST = {"max_tokens": 2048}
RUNS = 3
METRIC = "accuracy"
summarize = accuracy
PROMPT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""


def prompt(item):
    return PROMPT.format(question=item["question"], **dict(zip("ABCD", item["options"])))


def score(item, response):
    text = response.replace("**", "")
    hits = re.findall(r"(?i)answer\s*:\s*\$?\(?([A-D])\)?", text) or re.findall(r"\\boxed\{\(?([A-D])\)?\}", text)
    pred = hits[-1].upper() if hits else None
    return {"pred": pred, "correct": pred == item["answer"]}
