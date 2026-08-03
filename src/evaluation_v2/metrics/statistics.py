"""Paired statistical comparisons for Evaluation V2."""

from __future__ import annotations

import math
import random
from typing import Iterable


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
    a_by_id = {row["example_id"]: row for row in a_rows}
    b_by_id = {row["example_id"]: row for row in b_rows}
    common = sorted(set(a_by_id) & set(b_by_id))
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
    return {
        "aligned_n": len(common),
        "a_only": sorted(set(a_by_id) - set(b_by_id)),
        "b_only": sorted(set(b_by_id) - set(a_by_id)),
        "mcnemar_r1": mcnemar(a_r1, b_r1) if common else None,
        "mrr_bootstrap": paired_bootstrap_ci(a_mrr, b_mrr, seed=seed, repetitions=repetitions) if common else None,
        "ndcg_bootstrap": paired_bootstrap_ci(a_ndcg, b_ndcg, seed=seed, repetitions=repetitions) if common else None,
        "changed_rankings": [
            key for key in common
            if a_by_id[key].get("retrieved_refs", []) != b_by_id[key].get("retrieved_refs", [])
        ],
    }
