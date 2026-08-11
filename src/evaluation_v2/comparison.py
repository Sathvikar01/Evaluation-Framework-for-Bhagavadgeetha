"""CLI-facing paired comparison of V2 result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics.statistics import compare_paired


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(path)
    if source.suffix.lower() == ".json":
        value = json.loads(source.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict) and isinstance(value.get("per_query"), list):
            return value["per_query"]
        if isinstance(value, dict) and isinstance(value.get("results"), list):
            return value["results"]
        if isinstance(value, list):
            return value
        raise ValueError(f"{source} is not a row file; compare per_query.jsonl or a JSON object containing per_query/results")
    rows = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"comparison row {source}:{line_no} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"comparison row file is empty: {source}")
    return rows


def compare_files(a: str | Path, b: str | Path, *, seed: int = 20260803, repetitions: int = 2000) -> dict[str, Any]:
    return compare_paired(load_rows(a), load_rows(b), seed=seed, repetitions=repetitions)
