"""Latency summaries and component timing helpers."""

from __future__ import annotations

from typing import Iterable


def summarize(values: Iterable[float]) -> dict:
    values = sorted(float(value) for value in values)
    if not values: return {"n": 0, "p50": None, "p95": None, "mean": None, "max": None}
    def percentile(p: float) -> float: return values[min(len(values) - 1, int((len(values) - 1) * p))]
    return {"n": len(values), "p50": percentile(.5), "p95": percentile(.95), "mean": sum(values) / len(values), "max": values[-1]}
