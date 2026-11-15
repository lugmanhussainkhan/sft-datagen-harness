"""MATH levels 3-5: final answer in \\boxed{}, graded with the finetuning-with-sampling grader."""

import signal
import warnings

from .. import accuracy

warnings.filterwarnings("ignore", category=SyntaxWarning)  # old-style regex strings in the grader
from .math_grader import grade_answer
from .math_parse_utils import parse_answer

REQUEST = {"max_tokens": 1856}
RUNS = 2
METRIC = "accuracy"

# Eval prompt, also the user turn of the SFT data so training and evaluation match.
PROMPT = ("You are an expert mathematician. Can you solve the following math problem? {question}\n\n"
          "Explain your reasoning, thinking step by step, and return your final answer within \\boxed{{}}.")

REWRITE_PROMPT = """You are an expert mathematician. Below is a math problem, followed by a reference solution that is known to be correct.

Solve the problem yourself and write the solution in your own words, the way you would normally explain it. Use the reference solution only to make sure your steps and your final answer are right. Do not copy its sentences or layout, and do not mention that you were given a reference.

Problem:
{question}

Reference solution (for your eyes only):
{solution}

Now write your own solution to the problem. Explain your reasoning, thinking step by step, and return your final answer within \\boxed{{}}."""


def on_alarm(signum, frame):
    raise TimeoutError


signal.signal(signal.SIGALRM, on_alarm)  # a 10 s limit per answer: sympy can hang on pathological expressions
summarize = accuracy


def prompt(item):
    return PROMPT.format(question=item["question"])


def rewrite_prompt(item):
    return REWRITE_PROMPT.format(question=item["question"], solution=item["solution"])


def score(item, response):
    try:
        pred = parse_answer(response)
        signal.alarm(10)
        correct = pred is not None and bool(grade_answer(pred, item["answer"]))
    except Exception:  # noqa: BLE001 - unparseable or too slow counts as wrong
        pred, correct = None, False
    finally:
        signal.alarm(0)
    return {"pred": pred, "correct": correct}
