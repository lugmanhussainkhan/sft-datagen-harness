"""Build data/gpqa_test.jsonl from GPQA Diamond. The dataset is gated, so it isn't shipped in data/:
accept its terms on huggingface.co/datasets/Idavidrein/gpqa, then  HF_TOKEN=... uv run src/prepare_gpqa.py
"""

import csv
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

path = hf_hub_download("Idavidrein/gpqa", "gpqa_diamond.csv", repo_type="dataset")
out = Path(__file__).resolve().parents[1] / "data/gpqa_test.jsonl"
with open(path, newline="", encoding="utf-8") as f, open(out, "w") as fo:
    for r in csv.DictReader(f):
        options = [r["Correct Answer"]] + [r[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
        order = list(range(4))
        random.Random(r["Record ID"]).shuffle(order)  # fixed option order per question, same for every model
        fo.write(json.dumps({"id": r["Record ID"], "question": r["Question"].strip(),
                             "options": [options[i].strip() for i in order], "answer": "ABCD"[order.index(0)],
                             "domain": r["High-level domain"]}, ensure_ascii=False) + "\n")
print(f"wrote {out}")
