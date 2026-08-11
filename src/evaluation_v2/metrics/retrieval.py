"""Retrieval metrics with explicit denominator and exclusion accounting."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


# Rate / containment fields that are safe to macro-average. Counts and rank
# positions are reported separately so they are not mistaken for percentages.
_SUMMARY_RATE_KEYS = {
    "mrr",
    "average_precision",
    "r_precision",
    "candidate_pool_containment",
    "hard_negative_accuracy",
}
_SUMMARY_RATE_PREFIXES = (
    "recall@", "precision@", "graded_recall@", "ndcg@", "hit_rate@",
    "success@", "average_precision@",
)


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _labels(gold: Iterable[str], graded: dict[str, int] | None) -> dict[str, int]:
    if graded:
        return {key: int(value) for key, value in graded.items() if value > 0}
    return {key: 1 for key in gold if key}


def _is_rate_metric(name: str) -> bool:
    if name in _SUMMARY_RATE_KEYS:
        return True
    return any(name.startswith(prefix) for prefix in _SUMMARY_RATE_PREFIXES)


def retrieval_metrics(
    ranked_refs: list[str], gold_refs: Iterable[str], *, graded_relevance: dict[str, int] | None = None,
    cutoffs: Iterable[int] = (1, 3, 5, 10, 20, 50), candidate_pool_refs: Iterable[str] | None = None,
    hard_negative_refs: Iterable[str] | None = None,
) -> dict[str, Any]:
    ranked = _unique(ranked_refs)
    labels = _labels(gold_refs, graded_relevance)
    relevant = set(labels)
    requested_cutoffs = tuple(dict.fromkeys(int(k) for k in cutoffs if int(k) > 0))
    result: dict[str, Any] = {"n_gold": len(relevant), "n_retrieved": len(ranked), "excluded": not bool(relevant)}
    for k in requested_cutoffs:
        top = ranked[:k]
        hits = [item for item in top if item in relevant]
        result[f"recall@{k}"] = len(set(hits)) / len(relevant) if relevant else None
        result[f"precision@{k}"] = len(set(hits)) / k if k > 0 else 0.0
        result[f"graded_recall@{k}"] = sum(labels[item] for item in set(hits)) / sum(labels.values()) if labels else None
        success = bool(hits)
        result[f"hit_rate@{k}"] = float(success) if relevant else None
        result[f"success@{k}"] = float(success) if relevant else None
        precision_sum = 0.0
        hit_count = 0
        for index, item in enumerate(top, 1):
            if item in relevant:
                hit_count += 1
                precision_sum += hit_count / index
        result[f"average_precision@{k}"] = (
            precision_sum / min(len(relevant), k) if relevant else None
        )
    first_rank = next((index for index, item in enumerate(ranked, 1) if item in relevant), None)
    result["mrr"] = 1.0 / first_rank if first_rank else 0.0
    result["first_relevant_rank"] = first_rank
    result["top1_hit"] = bool(first_rank == 1)
    ranks = [index for index, item in enumerate(ranked, 1) if item in relevant]
    result["mean_rank"] = sum(ranks) / len(ranks) if ranks else None
    if ranks:
        ordered_ranks = sorted(ranks)
        middle = len(ordered_ranks) // 2
        result["median_rank"] = (
            ordered_ranks[middle]
            if len(ordered_ranks) % 2
            else (ordered_ranks[middle - 1] + ordered_ranks[middle]) / 2
        )
    else:
        result["median_rank"] = None
    precision_sum = 0.0
    hit_count = 0
    for index, item in enumerate(ranked, 1):
        if item in relevant:
            hit_count += 1
            precision_sum += hit_count / index
    result["average_precision"] = precision_sum / len(relevant) if relevant else None
    r = len(relevant)
    result["r_precision"] = (
        len(set(ranked[:r]) & relevant) / r if relevant else None
    )
    for k in requested_cutoffs:
        dcg = sum((2 ** labels[item] - 1) / math.log2(index + 2) for index, item in enumerate(ranked[:k]) if item in labels)
        ideal = sorted(labels.values(), reverse=True)[:k]
        idcg = sum((2 ** label - 1) / math.log2(index + 2) for index, label in enumerate(ideal))
        result[f"ndcg@{k}"] = dcg / idcg if idcg else None
    negatives = set(_unique(hard_negative_refs or [])) - relevant
    if negatives:
        rank_by_ref = {ref: index for index, ref in enumerate(ranked, 1)}
        relevant_ranks = [rank_by_ref[ref] for ref in relevant if ref in rank_by_ref]
        best_relevant = min(relevant_ranks) if relevant_ranks else math.inf
        comparisons = [
            1.0 if best_relevant < rank_by_ref.get(ref, math.inf) else 0.0
            for ref in negatives
            if ref in rank_by_ref or best_relevant < math.inf
        ]
        result["hard_negative_accuracy"] = sum(comparisons) / len(comparisons) if comparisons else None
        result["n_hard_negatives"] = len(negatives)
    else:
        result["hard_negative_accuracy"] = None
        result["n_hard_negatives"] = 0
    result["final_rank_containment"] = bool(relevant.intersection(ranked))
    if candidate_pool_refs is None:
        result["candidate_pool_containment"] = None
    else:
        pool = set(_unique(candidate_pool_refs))
        result["candidate_pool_size"] = len(pool)
        result["candidate_pool_containment"] = bool(relevant.intersection(pool))
    result["retrieval_depth"] = len(ranked)
    result["cutoff_complete"] = {str(k): len(ranked) >= k for k in requested_cutoffs}
    result["failure"] = "missing_gold" if not relevant else "not_retrieved" if not relevant.intersection(ranked) else "success"
    return result


def summarize_retrieval(
    rows: list[dict[str, Any]], *, group_fields: tuple[str, ...] = (),
    bootstrap_seed: int = 20260803, bootstrap_repetitions: int = 2000,
    confidence: float = .95,
) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("metrics", {}).get("excluded") and row.get("metrics")]
    summary: dict[str, Any] = {"n": len(rows), "n_scored": len(valid), "n_excluded": len(rows) - len(valid), "metrics": {}, "rank_stats": {}}
    for name in sorted({key for row in valid for key in row["metrics"] if _is_rate_metric(key)}):
        if name == "candidate_pool_containment":
            values = [1.0 if row["metrics"].get(name) else 0.0 for row in valid if isinstance(row["metrics"].get(name), bool)]
        else:
            values = [
                float(row["metrics"][name])
                for row in valid
                if isinstance(row["metrics"].get(name), (int, float)) and not isinstance(row["metrics"].get(name), bool)
            ]
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
            from .statistics import bootstrap_ci
            interval = bootstrap_ci(
                values, seed=bootstrap_seed, repetitions=bootstrap_repetitions,
                confidence=confidence,
            )
            summary["metrics"][name] = {
                "value": mean, "mean": mean, "std": math.sqrt(variance),
                "denominator": len(values), "confidence_interval": interval,
            }
        else:
            summary["metrics"][name] = {"value": None, "mean": None, "std": None, "denominator": 0, "confidence_interval": None}
    for name in ("first_relevant_rank", "mean_rank", "median_rank", "n_gold", "n_retrieved"):
        values = [
            float(row["metrics"][name])
            for row in valid
            if isinstance(row["metrics"].get(name), (int, float)) and not isinstance(row["metrics"].get(name), bool)
        ]
        summary["rank_stats"][name] = {"value": sum(values) / len(values) if values else None, "denominator": len(values)}
    top1 = [1.0 if row["metrics"].get("top1_hit") else 0.0 for row in valid if "top1_hit" in row["metrics"]]
    if top1:
        summary["metrics"]["top1_hit_rate"] = {"value": sum(top1) / len(top1), "denominator": len(top1)}
    if group_fields:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(str(row.get(field, "")) for field in group_fields)].append(row)
        summary["groups"] = {
            "|".join(key): summarize_retrieval(
                value, bootstrap_seed=bootstrap_seed,
                bootstrap_repetitions=bootstrap_repetitions, confidence=confidence,
            )
            for key, value in grouped.items()
        }
    return summary
