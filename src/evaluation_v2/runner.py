"""Execution and stage-wise analysis for Evaluation V2."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import SCHEMA_VERSION
from .datasets.with_id import discover_canonical_inventory, parse_reference_query
from .latency import summarize as summarize_latency
from .metrics.generation import generation_checks
from .metrics.retrieval import retrieval_metrics, summarize_retrieval
from .metrics.routing import routing_metrics
from .pipeline_adapter import LivePipelineAdapter
from .schemas import BenchmarkExample, ExampleResult


def _refs(records: Iterable[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(record.get("verse_ref", "") for record in records if record.get("verse_ref")))


def stage_analysis(example: BenchmarkExample, result: dict[str, Any], final_refs: list[str]) -> dict[str, Any]:
    intermediate = result.get("intermediate", {})
    stage_refs = {name: _refs(intermediate.get(name, [])) for name in ("vector_results", "graph_results", "bm25_results", "fused_results", "reranked_results")}
    final_refs = list(dict.fromkeys(final_refs))
    gold = set(example.gold_verse_refs)
    containment = {name: bool(gold.intersection(refs)) for name, refs in stage_refs.items()}
    exact_short_circuit = bool(result.get("intermediate", {}).get("verse_ref_detected", False))
    if exact_short_circuit:
        failure = "exact_reference_short_circuit"
    elif not gold:
        failure = "not_applicable"
    elif not any(containment.get(name, False) for name in ("vector_results", "graph_results", "bm25_results")):
        failure = "source_recall_failure"
    elif not containment.get("fused_results", False):
        failure = "fusion_drop"
    elif not containment.get("reranked_results", False):
        failure = "reranker_miss"
    elif not gold.intersection(final_refs):
        failure = "final_formatting_or_mapping_failure"
    else:
        failure = "success"
    return {"stage_refs": stage_refs, "containment": containment, "union_pool_containment": bool(gold.intersection(set().union(*(set(value) for value in stage_refs.values())))), "failure_class": failure, "exact_reference_short_circuit": exact_short_circuit}


def run_retrieval(
    examples: list[BenchmarkExample], adapter: LivePipelineAdapter, *, config: dict[str, Any], with_id: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    health: list[dict[str, Any]] = []
    for example in examples:
        start = time.perf_counter()
        try:
            observation = adapter.query_retrieval(example.query, answer=example.reference_answer, use_api=bool(config.get("retrieval", {}).get("use_api", False)))
            result = observation.result
            retrieved_records = result.get("reranked_results", [])
            retrieved_refs = _refs(retrieved_records)
            metrics = retrieval_metrics(retrieved_refs, example.gold_verse_refs, graded_relevance=example.graded_relevance, cutoffs=config.get("retrieval", {}).get("cutoffs", [1, 3, 5, 10, 50]))
            stage = stage_analysis(example, result, retrieved_refs)
            row = {"example_id": example.example_id, "dataset_name": example.dataset_name, "track": example.track, "query": example.query, "query_language": example.query_language, "query_type": example.query_type, "gold_verse_refs": list(example.gold_verse_refs), "retrieved_refs": retrieved_refs, "retrieved_records": retrieved_records[:50], "metrics": metrics, "stages": stage, "latency": {"total": observation.elapsed_seconds, "retrieval": observation.elapsed_seconds}, "health": observation.health, "expected_valid": example.metadata.get("expected_valid", True), "parsed_refs": parse_reference_query(example.query), "predicted_refs": retrieved_refs}
            if with_id:
                row["routing_latency_seconds"] = observation.elapsed_seconds
                row["expected_refs"] = list(example.gold_verse_refs)
            health.append(observation.health)
        except Exception as exc:  # per-example errors are retained, never hidden
            row = {"example_id": example.example_id, "dataset_name": example.dataset_name, "track": example.track, "query": example.query, "gold_verse_refs": list(example.gold_verse_refs), "retrieved_refs": [], "metrics": {"excluded": True, "failure": "execution_error"}, "stages": {"failure_class": "execution_error"}, "latency": {"total": time.perf_counter() - start}, "error": f"{type(exc).__name__}: {exc}", "degraded": True, "expected_valid": example.metadata.get("expected_valid", True), "predicted_refs": []}
        rows.append(row)
    if with_id:
        inventory = discover_canonical_inventory(config.get("paths", {}).get("chunks", "data/processed/chunks.jsonl"))
        summary = routing_metrics(rows, inventory)
    else:
        summary = summarize_retrieval(rows, group_fields=("dataset_name", "query_language", "query_type"))
    return rows, {"summary": summary, "component_health": component_health_summary(health), "stage_analysis": summarize_stage(rows), "latency": summarize_latency(row.get("latency", {}).get("total", 0) for row in rows)}


def run_generation(examples: list[BenchmarkExample], adapter: LivePipelineAdapter, *, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for example in examples:
        try:
            observation = adapter.query_generation(example.query, answer=example.reference_answer, use_api=bool(config.get("generation", {}).get("allow_api", False)))
            result = observation.result
            retrieved = result.get("reranked_results", [])
            retrieved_refs = set(_refs(retrieved))
            answer = result.get("answer", "")
            checks = generation_checks(answer, retrieved_refs=retrieved_refs, gold_refs=set(example.gold_verse_refs), expect_citation=bool(example.gold_verse_refs))
            rows.append({"example_id": example.example_id, "dataset_name": example.dataset_name, "query": example.query, "reference_answer": example.reference_answer, "retrieved_verses": retrieved[:10], "generated_answer": answer, "citations": checks["citations"], "deterministic_scores": checks, "human_score_fields": {"answer_relevance": None, "claim_support": None, "citation_completeness": None, "philosophical_coverage": None, "translation_faithfulness": None, "unsupported_elaboration": None, "commentary_attribution": None, "answer_clarity": None, "abstention_quality": None}, "notes": "", "latency": {"generation": observation.elapsed_seconds}, "health": observation.health})
        except Exception as exc:
            rows.append({"example_id": example.example_id, "dataset_name": example.dataset_name, "query": example.query, "reference_answer": example.reference_answer, "generated_answer": "", "deterministic_scores": {"error": str(exc)}, "human_score_fields": {}, "notes": f"execution error: {exc}", "latency": {}, "error": str(exc)})
    valid = [row for row in rows if "error" not in row]
    score_values = [row["deterministic_scores"].get("deterministic_score", 0) for row in valid]
    return rows, {"n": len(rows), "n_scored": len(valid), "deterministic_score": sum(score_values) / len(score_values) if score_values else None, "latency": summarize_latency(row.get("latency", {}).get("generation", 0) for row in rows)}


def component_health_summary(health: list[dict[str, Any]]) -> dict[str, Any]:
    if not health: return {"n": 0, "operational": False}
    keys = sorted({key for item in health for key in item})
    return {"n": len(health), "operational": all(item.get("operational", True) and not item.get("degraded", False) for item in health), "components": {key: sum(bool(item.get(key)) for item in health) / len(health) for key in keys}}


def summarize_stage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    source_hits: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row.get("stages", {}).get("failure_class", "unknown")] += 1
        for stage, hit in row.get("stages", {}).get("containment", {}).items():
            source_hits[stage] += int(bool(hit))
    return {"failure_counts": dict(counts), "stage_containment_counts": dict(source_hits), "sample_count": len(rows)}


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
