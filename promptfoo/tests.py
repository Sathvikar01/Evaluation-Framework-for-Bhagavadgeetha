"""Load Promptfoo test cases from the versioned gold dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation_v2.benchmark import load_benchmark  # noqa: E402


def generate_tests():
    dataset = os.environ.get(
        "GITA_BENCHMARK_DATASET",
        str(Path(__file__).resolve().parents[1] / "data/evaluation_v2/universal/starter_development.jsonl"),
    )
    split = os.environ.get("GITA_BENCHMARK_SPLIT", "development")
    return [
        {
            "description": row.query_id,
            "vars": {
                "query": row.query,
                "query_id": row.query_id,
                "gold_refs": list(row.gold_refs),
                "top_k": int(os.environ.get("GITA_BENCHMARK_TOP_K", "10")),
            },
        }
        for row in load_benchmark(dataset, split=split)
    ]
