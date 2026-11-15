"""Mean headline metric per model and benchmark, over runs:  uv run src/report.py [results-dir]"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

table = defaultdict(lambda: defaultdict(list))
for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "results").glob("*/*_r*.summary.json")):
    s = json.loads(f.read_text())
    table[f.parent.name][f.name.rsplit("_r", 1)[0]].append(s[s["metric"]])

benches = sorted({b for row in table.values() for b in row})
print(f"{'model':40}" + "".join(f"{b:>14}" for b in benches) + "   (runs)")
for model, row in table.items():
    cells = [f"{100 * statistics.mean(row[b]):.2f} ({len(row[b])})" if b in row else "-" for b in benches]
    print(f"{model:40}" + "".join(f"{c:>14}" for c in cells))
