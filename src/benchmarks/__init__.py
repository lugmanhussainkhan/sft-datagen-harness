def accuracy(records):
    """Summary for the answer-matching benchmarks (math, gpqa, mmlu_pro)."""
    return {"accuracy": sum(r["correct"] for r in records) / len(records),
            "no_answer": sum(r["pred"] is None for r in records)}
