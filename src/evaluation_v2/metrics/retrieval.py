"""Retrieval metrics with explicit denominator and exclusion accounting."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _labels(gold: Iterable[str], graded: dict[str, int] | None) -> dict[str, int]:
    if graded:
        return {key: int(value) for key, value in graded.items() if value > 0}
    return {key: 1 for key in gold if key}


def retrieval_metrics(
    ranked_refs: list[str], gold_refs: Iterable[str], *, graded_relevance: dict[str, int] | None = None,
    cutoffs: Iterable[int] = (1, 3, 5, 10, 50),
) -> dict[str, Any]:
    ranked = _unique(ranked_refs)
    labels = _labels(gold_refs, graded_relevance)
    relevant = set(labels)
    result: dict[str, Any] = {"n_gold": len(relevant), "n_retrieved": len(ranked), "excluded": not bool(relevant)}
    for k in cutoffs:
        top = ranked[:k]
        hits = [item for item in top if item in relevant]
        result[f"recall@{k}"] = len(set(hits)) / len(relevant) if relevant else None
        result[f"precision@{k}"] = len(set(hits)) / len(set(top)) if top else 0.0
        result[f"graded_recall@{k}"] = sum(labels[item] for item in set(hits)) / sum(labels.values()) if labels else None
    first_rank = next((index for index, item in enumerate(ranked, 1) if item in relevant), None)
    result["mrr"] = 1.0 / first_rank if first_rank else 0.0
    result["first_relevant_rank"] = first_rank
    ranks = [index for index, item in enumerate(ranked, 1) if item in relevant]
    result["mean_rank"] = sum(ranks) / len(ranks) if ranks else None
    result["median_rank"] = sorted(ranks)[len(ranks) // 2] if ranks else None
    for k in (5, 10):
        dcg = sum((2 ** labels[item] - 1) / math.log2(index + 2) for index, item in enumerate(ranked[:k]) if item in labels)
        ideal = sorted(labels.values(), reverse=True)[:k]
        idcg = sum((2 ** label - 1) / math.log2(index + 2) for index, label in enumerate(ideal))
        result[f"ndcg@{k}"] = dcg / idcg if idcg else None
    result["candidate_pool_containment"] = bool(relevant.intersection(ranked))
    result["failure"] = "missing_gold" if not relevant else "not_retrieved" if not relevant.intersection(ranked) else "success"
    return result


def summarize_retrieval(rows: list[dict[str, Any]], *, group_fields: tuple[str, ...] = ()) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("metrics", {}).get("excluded") and row.get("metrics")]
    summary: dict[str, Any] = {"n": len(rows), "n_scored": len(valid), "n_excluded": len(rows) - len(valid), "metrics": {}}
    names = sorted({key for row in valid for key in row["metrics"] if isinstance(row["metrics"].get(key), (int, float))})
    for name in names:
        values = [row["metrics"][name] for row in valid if isinstance(row["metrics"].get(name), (int, float))]
        summary["metrics"][name] = {"value": sum(values) / len(values) if values else None, "denominator": len(values)}
    if group_fields:
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(str(row.get(field, "")) for field in group_fields)].append(row)
        summary["groups"] = {"|".join(key): summarize_retrieval(value) for key, value in grouped.items()}
    return summary
