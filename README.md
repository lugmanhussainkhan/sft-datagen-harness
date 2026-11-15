# sft-data-generation

Evaluation and SFT-data generation for the learning-vs-forgetting experiments. Prompts, data, graders and
sampling match `~/Projects/lfm2.5-350m-data-forgetting`.

```
data/                  <bench>_<split>.jsonl
src/benchmarks/        one module per benchmark: prompt(), score(), summarize() (+ official checker code)
src/evaluate.py
src/report.py
src/self_rewrite.py
src/prepare_gpqa.py
```

## Setup

```bash
uv sync
HF_TOKEN=... uv run src/prepare_gpqa.py   # GPQA is gated: accept its terms on the Hub first
```

Data sources: IFEval (google-research@f8d6d14), IFBench (allenai/IFBench@cd189a9), MATH L3–5 test/train
(aakaran/finetuning-with-sampling@b3af646), MMLU-Pro (TIGER-Lab/MMLU-Pro@b189ec7), GPQA Diamond (Idavidrein/gpqa).

## Evaluate

```bash
uv run src/evaluate.py --bench math --model hosted_vllm/LiquidAI/LFM2.5-350M \
  --model-args '{"api_base": "http://localhost:8000/v1", "temperature": 0.1, "extra_body": {"top_k": 50, "repetition_penalty": 1.05}}'
uv run src/report.py
```

`--bench` is one of `ifeval`, `ifbench`, `math`, `gpqa`, `mmlu_pro`. `--model-args` is passed to `litellm.completion`
as-is and overrides the benchmark's defaults (e.g. `max_tokens`). Run *k* uses `seed + k`. Default runs: ifeval 3,
gpqa 3, math 2, others 1. Results go to `results/<model>/<bench>_r<k>.jsonl` and `.summary.json`.

## Self-rewrite

```bash
uv run src/self_rewrite.py --bench math --model hosted_vllm/LiquidAI/LFM2.5-350M --model-args '{...}' --sample 1000
```

The model gets the rewrite instruction, the question and the reference solution (`REWRITE_PROMPT` in
`src/benchmarks/math`). It has up to 4 attempts. An attempt is kept only if it is not empty, not truncated, has the right
answer (the benchmark's `score()`) and does not mention the reference. Output goes to
`outputs/self_rewrite/<bench>_<model>/`: `rewrites.jsonl`, `sft.jsonl`, `sft_gold.jsonl` (same items, gold
solutions) and `summary.json`.
