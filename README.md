# sft-data-generation

Benchmarks and SFT-data generation for studying how the source of training data affects learning and forgetting.
Models are called through [LiteLLM](https://docs.litellm.ai/docs/providers), so any vLLM server or hosted provider works.

```
data/              <bench>_<split>.jsonl
src/benchmarks/    one module per benchmark: prompt(), score(), summarize()
src/evaluate.py
src/self_rewrite.py
src/prepare_gpqa.py
```

## Setup

```bash
uv sync
HF_TOKEN=... uv run src/prepare_gpqa.py   # GPQA is gated: accept its terms on the Hub first
```

For a vLLM server, either pass `api_base` in `--model-args` or `export HOSTED_VLLM_API_BASE=http://localhost:8000/v1`.

## Evaluate

```bash
uv run src/evaluate.py --model hosted_vllm/Qwen/Qwen2.5-3B-Instruct --model-args '{"temperature": 0.1}'
uv run src/evaluate.py --bench math gpqa --model hosted_vllm/Qwen/Qwen2.5-3B-Instruct
```

| Bench | Items | Runs | Metric |
|---|---|---|---|
| `ifeval` | 541 | 3 | prompt-level strict accuracy |
| `ifbench` | 300 | 1 | prompt-level strict accuracy |
| `math` (MATH levels 3–5) | 1,024 | 2 | accuracy |
| `gpqa` (Diamond) | 198 | 3 | accuracy |
| `mmlu_pro` (5-shot) | 12,032 | 1 | accuracy |
| `kk` (Knights and Knaves, 2–8 people) | 700 | 1 | accuracy (every identity right; `by_people` in the summary) |

`--bench` defaults to all. `--model-args` is passed to `litellm.completion` and overrides the bench defaults
(e.g. `max_tokens`). Run *k* uses `seed + k`.

```
runs/<model>/report.json                 mean of each bench's metric over its runs
runs/<model>/<bench>/r<k>.jsonl          scored records, appended as answers arrive
runs/<model>/<bench>/r<k>.summary.json
```

Finished runs are skipped and interrupted ones resume. `--force` redoes the selected benches, `--limit N` writes to
`runs/<model>_limitN/`. Each invocation ends with a table of every model under `runs/`.

## Self-rewrite

```bash
uv run src/self_rewrite.py --bench math --model hosted_vllm/Qwen/Qwen2.5-3B-Instruct --sample 1000
```

The model is shown the question and its reference solution and writes its own solution. Each item gets up to 4
attempts; one is kept if it is not empty, not truncated, correct, and does not mention the reference. Output goes to
`outputs/self_rewrite/<bench>_<model>/`: `rewrites.jsonl`, `sft.jsonl`, `sft_gold.jsonl` (gold solutions for the same
items) and `summary.json`.

## Data sources

IFEval (google-research@f8d6d14), IFBench (allenai/IFBench@cd189a9), MATH (aakaran/finetuning-with-sampling@b3af646),
MMLU-Pro (TIGER-Lab/MMLU-Pro@b189ec7), GPQA Diamond (Idavidrein/gpqa),
Knights and Knaves (K-and-K/knights-and-knaves@2f68547, test split).
