"""Transparent cross-track score for comparing complete V2 runs."""

from __future__ import annotations

from typing import Any


TRACK_PRIMARY_METRICS = {
    "with_id": ("exact_verse_lookup_accuracy", "summary"),
    "without_id_gita_qa": ("recall@1", "retrieval"),
    "cross_lingual_gita": ("recall@1", "retrieval"),
    "external_generalization": ("recall@1", "retrieval"),
    "generation": ("deterministic_score", "generation"),
}


def _primary_score(track: str, result: dict[str, Any]) -> tuple[float | None, int | None, str | None]:
    metric, kind = TRACK_PRIMARY_METRICS.get(track, ("", ""))
    if not metric or result.get("status") in {"blocked", "not_run"}:
        return None, None, None
    if kind == "summary":
        value = result.get("summary", {}).get(metric)
        denominator = result.get("summary", {}).get("valid_count")
    elif kind == "retrieval":
        entry = result.get("summary", {}).get("metrics", {}).get(metric, {})
        value = entry.get("value")
        denominator = result.get("summary", {}).get("n_scored") or entry.get("denominator")
    else:
        value = result.get(metric)
        denominator = result.get("n_scored")
    if not isinstance(value, (int, float)):
        return None, None, None
    return float(value), int(denominator) if isinstance(denominator, (int, float)) else None, metric


def aggregate_overall(track_results: dict[str, Any]) -> dict[str, Any]:
    """Return an equal-track-weighted percentage, never hiding missing tracks.

    Each track contributes one primary score in [0, 1]. Equal track weighting
    prevents the largest question set from dominating the composite. Missing,
    blocked, and zero-denominator tracks are listed separately and do not get
    silently treated as zero.
    """
    scores: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for track, result in track_results.items():
        if track not in TRACK_PRIMARY_METRICS:
            continue
        value, denominator, metric = _primary_score(track, result if isinstance(result, dict) else {})
        if value is None:
            missing.append(track)
            continue
        scores[track] = {"metric": metric, "value": value, "value_pct": value * 100.0, "denominator": denominator}

    values = [item["value"] for item in scores.values()]
    macro = sum(values) / len(values) if values else None
    weighted_values = [(item["value"], item["denominator"]) for item in scores.values() if item["denominator"]]
    total_n = sum(denominator for _, denominator in weighted_values)
    micro = sum(value * denominator for value, denominator in weighted_values) / total_n if total_n else None
    return {
        "metric": "overall_performance_pct",
        "value_pct": macro * 100.0 if macro is not None else None,
        "micro_value_pct": micro * 100.0 if micro is not None else None,
        "aggregation": "macro_mean_of_equal_track_primary_scores",
        "scored_tracks": sorted(scores),
        "missing_or_unscored_tracks": sorted(set(missing)),
        "track_scores": scores,
        "scored_track_count": len(scores),
        "available_question_count": total_n,
        "status": "complete" if not missing and scores else "partial" if scores else "not_available",
    }
