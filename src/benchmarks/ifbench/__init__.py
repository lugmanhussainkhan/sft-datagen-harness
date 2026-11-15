"""IFBench (300 prompts, unseen constraint types): official AllenAI checker, scored like IFEval."""

from .. import ifeval
from .instructions_registry import INSTRUCTION_DICT

REQUEST = ifeval.REQUEST
RUNS = 1
METRIC = ifeval.METRIC
prompt = ifeval.prompt
summarize = ifeval.summarize


def score(item, response):
    return ifeval.score(item, response, INSTRUCTION_DICT)
