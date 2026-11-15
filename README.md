# sft-data-generation

Benchmarks and SFT-data generation for studying how the source of training data affects learning and forgetting.
Models are called through [LiteLLM](https://docs.litellm.ai/docs/providers), so any vLLM server or hosted provider works.

```
data/              <bench>_<split>.jsonl
src/benchmarks/    one module per benchmark: prompt(), score(), summarize()
src/evaluate.py
src/self_rewrite.py
src/teacher_feedback.py
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

## Train-set generation

```bash
uv run src/evaluate.py --train --bench kk math --sample 1000 --runs 4 --model <strong model>
```

`--train` runs the same harness (prompt, request defaults, scoring) on `data/<bench>_train.jsonl` and writes to
`outputs/train/<model>[_sample<N>]/`, with the same layout and resume behavior as `runs/`. Benches without a train split
are skipped. Each bench also gets `sft.jsonl`: per item, the first run whose response is correct, not empty and not
truncated, paired with the eval prompt. `sft_gold.jsonl` holds the gold solutions for the same items. `--runs` sets how many
attempts each item gets (run *k* uses `seed + k`). `--sample N` picks the same items as `self_rewrite.py --sample N`.

## Self-rewrite

```bash
uv run src/self_rewrite.py --bench math --model hosted_vllm/Qwen/Qwen2.5-3B-Instruct --sample 1000
uv run src/self_rewrite.py --bench kk --model hosted_vllm/Qwen/Qwen2.5-3B-Instruct --sample 1000
```

The model is shown the question and its reference solution and writes its own solution. Each item gets up to 4
attempts (`--max-trials`); one is kept if it is not empty, not truncated, correct, and does not mention the reference.
Attempt *t* goes only to items no earlier attempt accepted. Output goes to `outputs/self_rewrite/<bench>_<model>/`:

```
attempt<t>.jsonl     one scored record per item sent in attempt t, appended as each response arrives
rewrites.jsonl       every item with its accepted response and all its attempts
sft.jsonl            accepted rewrites as user/assistant pairs with the eval prompt
sft_gold.jsonl       gold solutions for the same items
summary.json
```

A stopped run loses nothing: rerun the same command and it resumes from the attempt files. The last four files are
rebuilt from the attempt files at the end of every run. `--force` starts over.

## Teacher feedback

```bash
uv run src/teacher_feedback.py --bench math --sample 1000 \
    --student hosted_vllm/Qwen/Qwen2.5-3B-Instruct --teacher anthropic/claude-haiku-5-5
```

The student answers, a teacher writes a step-by-step review of everything wrong with the attempt without giving the
answer away, and the student writes a new full solution with that attempt and the review in front of it. There is one
round per item. The teacher writes no solution tokens, so every accepted answer comes from the student.

- **First answers** come from `evaluate.py --train`'s `r0.jsonl` for the same student, `--sample` and `--limit` when that
  file exists (`--initial` points elsewhere). Only the items it lacks are generated. The items are those of
  `data/<bench>_train.jsonl`, drawn the same way as `evaluate.py --train --sample N`.
- **Accepted** means not empty, not truncated and correct. A revision also must not mention the earlier attempt or the
  review.
- **The teacher** reviews wrong, unanswered and truncated first answers (empty ones get no review). It sees the problem,
  the reference solution, the attempt and what is wrong with it, and lists every mistake in order, each with where it
  is and why it is wrong. A review that states the answer in the bench's answer format (`\boxed{}`, or a full kk
  CONCLUSION) is sent back to be rewritten, up to 3 tries.
- **The student** gets one user message: the eval prompt, then `--- Your attempt ---` and `--- Teacher's review ---`. If
  the request is rejected (usually because it no longer fits the context window), the item has no revision.

Output goes to `outputs/teacher_feedback/<bench>_<student>_by_<teacher>/`:

```
first.jsonl          every first answer and its score
feedback.jsonl       every teacher try (review, whether it reveals the answer)
revised.jsonl        every revision and its score
dialogues.jsonl      every item with its accepted response, accepted_from (first/revised), first answer, reviews, revision
sft.jsonl            accepted responses (first or revised) as user/assistant pairs with the plain eval prompt
sft_gold.jsonl       gold solutions for the same items
summary.json         accepted, fixed_by_review, fix_rate, rejections, review rewrites, ...
```

Every response is appended to its file the moment it arrives, and each item's next request is worked out from those
files. So the run can be stopped at any point (Ctrl-C, a crash, a killed job) and only the requests in flight are lost.
Rerun the same command to resume. Ctrl-C exits at once and still rebuilds the last four files from what has been
written. `--force` starts over.

## Student-style rewrite (kk)

```bash
uv run src/prepare_student_style.py
uv run src/student_style_rewrite.py --teacher anthropic/claude-haiku-5-5
```

A teacher rewrites Haiku's reference solution in the student's own style. The system prompt is the instruction plus four
of the student's correct answers to other puzzles (fixed, so Anthropic caches it). The user prompt is the puzzle, the
reference solution and the student's four attempts at that puzzle, each marked with how its final answer scored.

- `prepare_student_style.py` writes `data/kk_student_style.jsonl`: the 2,674 puzzles of the self-rewrite SFT subset
  (`--train-set`) with Haiku's solution and the four `evaluate.py --train` attempts (`--trials`, `r0`–`r3.jsonl`). It also
  writes `data/kk_student_style_examples.jsonl`, the few-shot examples (`EXAMPLES`, picked by hand). The base model's
  correct answers on 3+ people almost always contain a wrong step somewhere, so the teacher is told to copy their style
  but not their logic.
- **Accepted** means not empty, not truncated, correct, and not mentioning the reference, the attempts or the student.
  Each item gets up to `--max-tries` (3) teacher calls.

Output goes to `outputs/student_style/kk_<teacher>/`: `tries.jsonl` (every try, appended as it arrives; resume by
rerunning), `rewrites.jsonl`, `sft.jsonl`, `sft_gold.jsonl` and `summary.json` (including token use and cache reads).

## Data sources

IFEval (google-research@f8d6d14), IFBench (allenai/IFBench@cd189a9), MATH (aakaran/finetuning-with-sampling@b3af646),
MMLU-Pro (TIGER-Lab/MMLU-Pro@b189ec7), GPQA Diamond (Idavidrein/gpqa),
Knights and Knaves (K-and-K/knights-and-knaves@2f68547; `solution` in `kk_train.jsonl` is the dataset's step-by-step reasoning plus the CONCLUSION block).
`kk_train_haiku.jsonl` is the same with Claude Haiku 5.5's solution instead (`evaluate.py --train`, `reasoning_effort: medium`), kept
for the 6,018 puzzles it solved correctly with a single conclusion. `self_rewrite.py --bench kk` uses it as the reference, so
its `sft_gold.jsonl` holds Haiku's solutions and `--sample N` draws from these 6,018.
