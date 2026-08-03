"""CLI-facing paired comparison of V2 result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics.statistics import compare_paired


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig")) if Path(path).suffix == ".json" else None
    if value and isinstance(value, dict) and isinstance(value.get("per_query"), list): return value["per_query"]
    if value and isinstance(value, dict) and isinstance(value.get("results"), list): return value["results"]
    if Path(path).exists():
        rows = []
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            if line.strip(): rows.append(json.loads(line))
        return rows
    raise FileNotFoundError(path)


def compare_files(a: str | Path, b: str | Path, *, seed: int = 20260803, repetitions: int = 2000) -> dict[str, Any]:
    return compare_paired(load_rows(a), load_rows(b), seed=seed, repetitions=repetitions)
