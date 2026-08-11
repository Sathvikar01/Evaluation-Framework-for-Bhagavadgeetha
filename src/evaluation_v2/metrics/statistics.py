"""Paired statistical comparisons for Evaluation V2."""

from __future__ import annotations

import math
import random
from typing import Iterable


def bootstrap_ci(
    values: Iterable[float], *, seed: int = 20260803, repetitions: int = 2000,
    confidence: float = .95,
) -> dict[str, float | int]:
    """Percentile bootstrap confidence interval over evaluation units.

    Queries, not individual relevance judgments or retrieved documents, are
    the resampling unit.  This preserves the dependence among metrics from a
    single query and matches test-collection IR practice.
    """
    sample = [float(value) for value in values]
    if not sample:
        raise ValueError("bootstrap requires a non-empty sample")
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    n = len(sample)
    estimates = []
    for _ in range(repetitions):
        estimates.append(sum(sample[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    low_index = max(0, min(repetitions - 1, math.floor(alpha * repetitions)))
    high_index = max(0, min(repetitions - 1, math.ceil((1.0 - alpha) * repetitions) - 1))
    return {
        "low": estimates[low_index], "high": estimates[high_index],
        "confidence": confidence, "n": n, "repetitions": repetitions,
        "seed": seed,
    }


def paired_bootstrap_ci(a: Iterable[float], b: Iterable[float], *, seed: int = 20260803, repetitions: int = 2000, confidence: float = .95) -> dict[str, float | int]:
    a, b = list(a), list(b)
    if len(a) != len(b) or not a:
        raise ValueError("paired bootstrap requires equal, non-empty samples")
    rng = random.Random(seed)
    deltas = []
    n = len(a)
    for _ in range(repetitions):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(b[i] - a[i] for i in indices) / n)
    deltas.sort()
    alpha = (1 - confidence) / 2
    low = deltas[min(len(deltas) - 1, int(alpha * len(deltas)))]
    high = deltas[min(len(deltas) - 1, int((1 - alpha) * len(deltas)))]
    return {"delta": sum(b) / n - sum(a) / n, "low": low, "high": high, "confidence": confidence, "n": n, "repetitions": repetitions, "seed": seed}


def _binomial_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    p = .5
    observed = math.comb(n, k) * p ** n
    return min(1.0, sum(math.comb(n, i) * p ** n for i in range(n + 1) if math.comb(n, i) * p ** n <= observed + 1e-15))


def mcnemar(a_hits: Iterable[bool], b_hits: Iterable[bool]) -> dict[str, float | int]:
    a_hits, b_hits = list(a_hits), list(b_hits)
    if len(a_hits) != len(b_hits):
        raise ValueError("McNemar requires aligned samples")
    b01 = sum((not a and b) for a, b in zip(a_hits, b_hits))
    b10 = sum((a and not b) for a, b in zip(a_hits, b_hits))
    p_value = _binomial_two_sided(b01, b01 + b10)
    return {"a_wins": b10, "b_wins": b01, "ties": len(a_hits) - b01 - b10, "p_value": p_value}


def _top1_hit(metrics: dict) -> bool:
    """Binary top-1 relevance for McNemar (not fractional multi-gold recall)."""
    if "top1_hit" in metrics:
        return bool(metrics["top1_hit"])
    if metrics.get("first_relevant_rank") == 1:
        return True
    recall = metrics.get("recall@1")
    # Only treat exact 1.0 as a hit; fractional multi-gold recall is not binary success.
    return recall == 1 or recall == 1.0


def compare_paired(a_rows: list[dict], b_rows: list[dict], *, seed: int = 20260803, repetitions: int = 2000) -> dict:
    def index_rows(rows: list[dict], label: str) -> dict[str, dict]:
        indexed = {}
        for row in rows:
            example_id = row.get("example_id")
            if not example_id:
                raise ValueError(f"{label} comparison row is missing example_id")
            if example_id in indexed:
                raise ValueError(f"{label} comparison contains duplicate example_id: {example_id}")
            indexed[example_id] = row
        return indexed

    def scored(row: dict) -> bool:
        metrics = row.get("metrics", {})
        return bool(metrics) and not metrics.get("excluded") and "error" not in row

    a_by_id = index_rows(a_rows, "a")
    b_by_id = index_rows(b_rows, "b")
    raw_common = sorted(set(a_by_id) & set(b_by_id))
    common = [key for key in raw_common if scored(a_by_id[key]) and scored(b_by_id[key])]
    excluded_common = [key for key in raw_common if key not in common]
    a_r1, b_r1, a_mrr, b_mrr, a_ndcg, b_ndcg = [], [], [], [], [], []
    for key in common:
        am = a_by_id[key].get("metrics", {})
        bm = b_by_id[key].get("metrics", {})
        a_r1.append(_top1_hit(am))
        b_r1.append(_top1_hit(bm))
        a_mrr.append(float(am.get("mrr", 0) or 0))
        b_mrr.append(float(bm.get("mrr", 0) or 0))
        a_ndcg.append(float(am.get("ndcg@10", 0) or 0))
        b_ndcg.append(float(bm.get("ndcg@10", 0) or 0))
    mrr_deltas = [b - a for a, b in zip(a_mrr, b_mrr)]
    delta_mean = sum(mrr_deltas) / len(mrr_deltas) if mrr_deltas else None
    delta_variance = (
        sum((value - delta_mean) ** 2 for value in mrr_deltas) / (len(mrr_deltas) - 1)
        if len(mrr_deltas) > 1 and delta_mean is not None else 0.0
    )
    return {
        "aligned_n": len(common),
        "excluded_common": excluded_common,
        "a_only": sorted(set(a_by_id) - set(b_by_id)),
        "b_only": sorted(set(b_by_id) - set(a_by_id)),
        "mcnemar_r1": mcnemar(a_r1, b_r1) if common else None,
        "mrr_bootstrap": paired_bootstrap_ci(a_mrr, b_mrr, seed=seed, repetitions=repetitions) if common else None,
        "ndcg_bootstrap": paired_bootstrap_ci(a_ndcg, b_ndcg, seed=seed, repetitions=repetitions) if common else None,
        "mrr_effect_size_cohens_dz": (
            delta_mean / math.sqrt(delta_variance)
            if delta_mean is not None and delta_variance > 0 else 0.0 if delta_mean == 0 else None
        ),
        "changed_rankings": [
            key for key in common
            if a_by_id[key].get("retrieved_refs", []) != b_by_id[key].get("retrieved_refs", [])
        ],
    }
