"""Metrics for deterministic exact-reference routing."""

from __future__ import annotations

from typing import Any

from ..datasets.with_id import expand_reference, parse_reference_query


def routing_metrics(rows: list[dict[str, Any]], inventory: set[str]) -> dict[str, Any]:
    valid = invalid = valid_correct = lookup_correct = normalization_correct = 0
    range_total = range_correct = 0
    false_positive = incorrect_route = 0
    latencies = []
    for row in rows:
        expected_valid = bool(row.get("expected_valid", row.get("example", {}).get("metadata", {}).get("expected_valid", False)))
        parsed = parse_reference_query(row.get("query", row.get("example", {}).get("query", "")))
        predicted = list(dict.fromkeys(row.get("predicted_refs", row.get("retrieved_refs", []))))
        if row.get("routing_latency_seconds") is not None:
            latencies.append(float(row["routing_latency_seconds"]))
        if expected_valid:
            valid += 1
            if parsed:
                valid_correct += 1
            expected = list(row.get("expected_refs", row.get("example", {}).get("gold_verse_refs", [])))
            if set(predicted) == set(expected): lookup_correct += 1
            if predicted and set(predicted) == set(expected): normalization_correct += 1
            if parsed and parsed[0][2] > parsed[0][1]:
                range_total += 1
                expanded = set(expand_reference(parsed[0], inventory))
                if set(predicted) == expanded: range_correct += 1
        else:
            invalid += 1
            if parsed or predicted:
                false_positive += 1
        if row.get("incorrect_chapter_or_verse"):
            incorrect_route += 1
    return {
        "sample_count": len(rows), "valid_count": valid, "invalid_count": invalid,
        "valid_reference_routing_accuracy": valid_correct / valid if valid else None,
        "exact_verse_lookup_accuracy": lookup_correct / valid if valid else None,
        "reference_normalization_accuracy": normalization_correct / valid if valid else None,
        "range_expansion_exact_match": range_correct / range_total if range_total else None,
        "range_expansion_count": range_total,
        "invalid_reference_rejection_rate": 1 - false_positive / invalid if invalid else None,
        "false_positive_short_circuit_rate": false_positive / invalid if invalid else None,
        "incorrect_chapter_verse_routing_rate": incorrect_route / max(valid, 1),
        "latency_seconds": _latency_summary(latencies),
    }


def _latency_summary(values: list[float]) -> dict[str, Any]:
    if not values: return {"n": 0, "p50": None, "p95": None, "mean": None, "max": None}
    ordered = sorted(values)
    percentile = lambda p: ordered[min(len(ordered) - 1, max(0, int((len(ordered) - 1) * p)))]
    return {"n": len(values), "p50": percentile(.50), "p95": percentile(.95), "mean": sum(values) / len(values), "max": max(values)}
