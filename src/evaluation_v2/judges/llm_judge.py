"""Guarded placeholder for optional LLM judging.

The evaluator intentionally requires explicit paid/API enablement at the CLI;
this module only validates cached structured judgments and does not make calls
implicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .rubric import FIELDS, RUBRIC_VERSION


def validate_judgment(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("rubric_version") != RUBRIC_VERSION:
        raise ValueError("unexpected judge rubric version")
    for field in FIELDS:
        score = value.get(field)
        if score is not None and (not isinstance(score, (int, float)) or not 0 <= score <= 4):
            raise ValueError(f"judge score {field} must be in [0,4]")
    return value


def load_cached_judgments(path: str | Path) -> list[dict[str, Any]]:
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip(): result.append(validate_judgment(json.loads(line)))
    return result
