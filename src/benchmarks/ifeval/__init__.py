"""IFEval (541 prompts): official Google checker, strict and loose scoring as in its evaluation_lib.py."""

import random

import langdetect
import nltk
from absl import logging

from .instructions_registry import INSTRUCTION_DICT

REQUEST = {"max_tokens": 1280}
RUNS = 3
METRIC = "prompt_strict"

langdetect.DetectorFactory.seed = 0
logging.set_verbosity(logging.FATAL)  # silence "Unable to detect language" (the checker counts it as a pass)
for resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


def prompt(item):
    return item["prompt"]


def score(item, response, registry=INSTRUCTION_DICT):
    lines = response.split("\n")
    variants = [response, "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(), "\n".join(lines[1:-1]).strip()]
    variants += [v.replace("*", "") for v in variants]  # loose: also without first/last line and markdown stars
    random.seed(item["id"])  # some checkers draw random arguments; same draw for every run and model
    strict, loose = [], []
    for inst_id, kwargs in zip(item["instruction_id_list"], item["kwargs"]):
        inst = registry[inst_id](inst_id)
        inst.build_description(**{k: v for k, v in kwargs.items() if v is not None})
        if "prompt" in (inst.get_instruction_args() or {}):
            inst.build_description(prompt=item["prompt"])
        passed = []
        for text in variants:
            try:
                passed.append(bool(text.strip()) and bool(inst.check_following(text)))
            except Exception:  # noqa: BLE001 - a checker crash counts as a failure
                passed.append(False)
        strict.append(passed[0])
        loose.append(any(passed))
    return {"strict": strict, "loose": loose}


def summarize(records):
    n, inst = len(records), sum(len(r["strict"]) for r in records)
    return {"prompt_strict": sum(all(r["strict"]) for r in records) / n,
            "prompt_loose": sum(all(r["loose"]) for r in records) / n,
            "inst_strict": sum(sum(r["strict"]) for r in records) / inst,
            "inst_loose": sum(sum(r["loose"]) for r in records) / inst}
