"""Side-by-side controlled ablation and cross-representation comparison."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Mapping

from .comparison import load_rows
from .metrics.statistics import compare_paired
from .runner import write_json


def _load_run(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = Path(path)
    if source.is_dir():
        summary_path = source / "summary.json"
        rows_path = source / "per_query.jsonl"
    elif source.name == "summary.json":
        summary_path = source
        rows_path = source.with_name("per_query.jsonl")
    else:
        raise ValueError("ablation inputs must be run directories or summary.json files")
    if not summary_path.exists() or not rows_path.exists():
        raise FileNotFoundError(f"run requires summary.json and per_query.jsonl: {source}")
    return json.loads(summary_path.read_text(encoding="utf-8-sig")), load_rows(rows_path)


def _holm_adjust(p_values: list[tuple[str, float]]) -> dict[str, float]:
    """Holm step-down family-wise error correction."""
    ordered = sorted(p_values, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = running
    return adjusted


def compare_ablation_runs(
    baseline: str | Path, variants: Mapping[str, str | Path], *,
    seed: int = 20260803, repetitions: int = 2000,
) -> dict[str, Any]:
    baseline_summary, baseline_rows = _load_run(baseline)
    baseline_name = str(baseline_summary.get("system_name", "baseline"))
    summaries = {baseline_name: baseline_summary}
    rows_by_run = {baseline_name: baseline_rows}
    comparisons = {}
    p_values = []
    for name, path in variants.items():
        summary, rows = _load_run(path)
        summaries[name] = summary
        rows_by_run[name] = rows
        comparison = compare_paired(baseline_rows, rows, seed=seed, repetitions=repetitions)
        comparisons[name] = comparison
        mcnemar = comparison.get("mcnemar_r1")
        if mcnemar is not None:
            p_values.append((name, float(mcnemar["p_value"])))
    adjusted = _holm_adjust(p_values)
    for name, value in adjusted.items():
        comparisons[name]["mcnemar_r1"]["holm_adjusted_p_value"] = value
    metric_names = ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "map", "ndcg@10", "robustness_score", "cross_script_score", "hard_negative_accuracy")
    table = {
        name: {metric: summary.get("leaderboard", {}).get(metric) for metric in metric_names}
        for name, summary in summaries.items()
    }
    representations = {
        name: str(summary.get("system_metadata", {}).get("corpus_representation", "unknown"))
        for name, summary in summaries.items()
    }
    cross_script = {"cross_script_score": None, "aligned_n": 0, "representations": representations}
    if len(set(representations.values()) - {"unknown", ""}) >= 2:
        indexed = {
            name: {str(row.get("example_id", row.get("query_id"))): row for row in rows}
            for name, rows in rows_by_run.items()
        }
        common = set.intersection(*(set(value) for value in indexed.values())) if indexed else set()
        invariance = []
        for query_id in sorted(common):
            values = []
            for name in indexed:
                metrics = indexed[name][query_id].get("metrics", {})
                value = metrics.get("success@10")
                if value is None:
                    value = metrics.get("top1_hit", metrics.get("recall@1", 0))
                values.append(float(value or 0))
            invariance.append(1.0 - (max(values) - min(values)))
        cross_script = {
            "cross_script_score": statistics.fmean(invariance) if invariance else None,
            "aligned_n": len(invariance), "representations": representations,
        }
    return {
        "schema_version": "gita_rag_ablation_v1.0",
        "baseline": baseline_name,
        "table": table,
        "paired_comparisons": comparisons,
        "cross_script_comparison": cross_script,
        "multiple_comparison_correction": "Holm family-wise correction over McNemar p-values",
        "interpretation_rule": "Treat a variant as supported only when the paired effect, interval, corrected test, and regression-sensitive subgroup metrics agree; do not infer causality if multiple components changed.",
    }


def write_ablation_report(path: str | Path, result: dict[str, Any]) -> None:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "ablation.json", result)
    metrics = sorted({metric for row in result["table"].values() for metric in row})
    lines = ["# Controlled ablation comparison", "", f"Baseline: `{result['baseline']}`", "", "| Run | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)]
    for name, values in result["table"].items():
        formatted = ["—" if values.get(metric) is None else f"{100 * float(values[metric]):.2f}%" for metric in metrics]
        lines.append("| " + name + " | " + " | ".join(formatted) + " |")
    (output / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
