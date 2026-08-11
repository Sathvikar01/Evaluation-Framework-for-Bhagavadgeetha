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


def _refs(records: Iterable[dict[str, Any]], *, verse_only: bool = True) -> list[str]:
    """Extract ranked verse refs; optionally keep only chunk_type=verse.

    Records without ``chunk_type`` are kept when ``verse_only`` is True so
    short-circuit / partially enriched candidates still score. Explicit non-
    verse types (commentary, pada, combined) are dropped.
    """
    refs: list[str] = []
    for record in records:
        ref = record.get("verse_ref", "")
        if not ref:
            continue
        chunk_type = record.get("chunk_type")
        if verse_only and chunk_type not in (None, "", "verse"):
            continue
        if ref not in refs:
            refs.append(ref)
    return refs


def stage_analysis(example: BenchmarkExample, result: dict[str, Any], final_refs: list[str]) -> dict[str, Any]:
    intermediate = result.get("intermediate", {})
    stage_names = ("vector_results", "graph_results", "bm25_results", "interpretation_results", "fused_results", "reranked_results")
    pool_stage_names = ("vector_results", "graph_results", "bm25_results", "interpretation_results", "fused_results")
    stage_refs = {name: _refs(intermediate.get(name, [])) for name in stage_names}
    candidate_pool_refs = list(dict.fromkeys(ref for name in pool_stage_names for ref in stage_refs[name]))
    final_refs = list(dict.fromkeys(final_refs))
    gold = set(example.gold_verse_refs)
    containment = {name: bool(gold.intersection(refs)) for name, refs in stage_refs.items()}
    final_hit = bool(gold.intersection(final_refs))
    exact_short_circuit = bool(result.get("intermediate", {}).get("verse_ref_detected", False))
    if exact_short_circuit:
        if gold and final_hit:
            failure = "success"
        elif not gold:
            # Invalid-reference probes: short-circuit without gold is a false positive.
            failure = "exact_reference_false_positive"
        else:
            failure = "exact_reference_short_circuit_miss"
    elif not gold:
        failure = "not_applicable"
    # The final output is the scored contract. A late injection can be absent
    # from an earlier trace stage and is still a successful retrieval.
    elif final_hit:
        failure = "success"
    elif not any(containment.get(name, False) for name in pool_stage_names):
        failure = "source_recall_failure"
    elif not containment.get("fused_results", False):
        failure = "fusion_drop"
    elif not containment.get("reranked_results", False):
        failure = "reranker_miss"
    elif not final_hit:
        failure = "final_formatting_or_mapping_failure"
    else:
        failure = "success"
    return {
        "stage_refs": stage_refs,
        "containment": containment,
        "candidate_pool_refs": candidate_pool_refs,
        "union_pool_containment": bool(gold.intersection(set(candidate_pool_refs))),
        "final_rank_containment": final_hit,
        "failure_class": failure,
        "exact_reference_short_circuit": exact_short_circuit,
    }


def run_retrieval(
    examples: list[BenchmarkExample], adapter: LivePipelineAdapter, *, config: dict[str, Any], with_id: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    health: list[dict[str, Any]] = []
    for example in examples:
        start = time.perf_counter()
        try:
            answer_aware = bool(config.get("retrieval", {}).get("answer_aware", False))
            observation = adapter.query_retrieval(
                example.query,
                answer=example.reference_answer if answer_aware else "",
                use_api=bool(config.get("retrieval", {}).get("use_api", False)),
            )
            result = dict(observation.result)
            intermediate = dict(result.get("intermediate") or {})
            retrieved_records = result.get("reranked_results") or intermediate.get("reranked_results") or []
            intermediate.setdefault("reranked_results", retrieved_records)
            result["intermediate"] = intermediate
            retrieved_refs = _refs(retrieved_records)
            stage = stage_analysis(example, result, retrieved_refs)
            metrics = retrieval_metrics(
                retrieved_refs,
                example.gold_verse_refs,
                graded_relevance=example.graded_relevance,
                cutoffs=config.get("retrieval", {}).get("cutoffs", [1, 3, 5, 10, 50]),
                candidate_pool_refs=stage["candidate_pool_refs"],
            )
            row = {
                "example_id": example.example_id,
                "dataset_name": example.dataset_name,
                "track": example.track,
                "query": example.query,
                "query_language": example.query_language,
                "query_type": example.query_type,
                "gold_verse_refs": list(example.gold_verse_refs),
                "retrieved_refs": retrieved_refs,
                "retrieved_records": retrieved_records[:50],
                "metrics": metrics,
                "stages": stage,
                "latency": {"total": observation.elapsed_seconds, "retrieval": observation.elapsed_seconds},
                "health": observation.health,
                "expected_valid": example.metadata.get("expected_valid", True),
                "parsed_refs": parse_reference_query(example.query),
                "predicted_refs": retrieved_refs,
                "exact_reference_short_circuit": stage["exact_reference_short_circuit"],
                "answer_aware": answer_aware,
            }
            if with_id:
                row["routing_latency_seconds"] = observation.elapsed_seconds
                row["expected_refs"] = list(example.gold_verse_refs)
                if retrieved_refs and set(retrieved_refs) != set(example.gold_verse_refs):
                    row["incorrect_chapter_or_verse"] = True
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
    health: list[dict[str, Any]] = []
    for example in examples:
        try:
            answer_aware = bool(config.get("generation", {}).get("answer_aware", False))
            observation = adapter.query_generation(
                example.query,
                answer=example.reference_answer if answer_aware else "",
                use_api=bool(config.get("generation", {}).get("allow_api", False)),
            )
            result = dict(observation.result)
            intermediate = dict(result.get("intermediate") or {})
            retrieved = result.get("reranked_results") or intermediate.get("reranked_results") or []
            retrieved_refs = set(_refs(retrieved))
            answer = result.get("answer", "")
            checks = generation_checks(answer, retrieved_refs=retrieved_refs, gold_refs=set(example.gold_verse_refs), expect_citation=bool(example.gold_verse_refs))
            rows.append({"example_id": example.example_id, "dataset_name": example.dataset_name, "query": example.query, "reference_answer": example.reference_answer, "retrieved_verses": retrieved[:10], "generated_answer": answer, "citations": checks["citations"], "deterministic_scores": checks, "human_score_fields": {"answer_relevance": None, "claim_support": None, "citation_completeness": None, "philosophical_coverage": None, "translation_faithfulness": None, "unsupported_elaboration": None, "commentary_attribution": None, "answer_clarity": None, "abstention_quality": None}, "notes": "", "latency": {"generation": observation.elapsed_seconds}, "health": observation.health, "answer_aware": answer_aware})
            health.append(observation.health)
        except Exception as exc:
            rows.append({"example_id": example.example_id, "dataset_name": example.dataset_name, "query": example.query, "reference_answer": example.reference_answer, "generated_answer": "", "deterministic_scores": {"error": str(exc)}, "human_score_fields": {}, "notes": f"execution error: {exc}", "latency": {}, "error": str(exc)})
            health.append({"operational": False, "degraded": True})
    valid = [row for row in rows if "error" not in row]
    score_values = [row["deterministic_scores"].get("deterministic_score", 0) for row in valid]
    return rows, {"n": len(rows), "n_scored": len(valid), "deterministic_score": sum(score_values) / len(score_values) if score_values else None, "latency": summarize_latency(row.get("latency", {}).get("generation", 0) for row in rows), "component_health": component_health_summary(health)}


def component_health_summary(health: list[dict[str, Any]]) -> dict[str, Any]:
    if not health: return {"n": 0, "operational": False, "degraded_count": 0, "stage_observability_required_count": 0, "stage_observability_complete_count": 0}
    keys = sorted({key for item in health for key in item})
    degraded_count = sum(bool(item.get("degraded", False)) for item in health)
    stage_required_count = sum("stage_observability" in item for item in health)
    stage_complete_count = sum(
        bool(item.get("exact_reference_short_circuit"))
        or (
            bool(item.get("stage_observability"))
            and all((item.get("stage_observability") or {}).values())
        )
        for item in health
    )
    components = {}
    for key in keys:
        if key == "stage_observability":
            components[key] = stage_complete_count / len(health)
        elif key == "degraded":
            components[key] = 1.0 - degraded_count / len(health)
        else:
            components[key] = sum(bool(item.get(key)) for item in health) / len(health)
    return {"n": len(health), "operational": all(item.get("operational", True) for item in health), "degraded_count": degraded_count, "stage_observability_required_count": stage_required_count, "stage_observability_complete_count": stage_complete_count, "components": components}


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
