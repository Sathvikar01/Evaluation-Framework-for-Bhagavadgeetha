"""Model-agnostic Bhagavad Gita RAG benchmark runner.

This module owns data, normalization, metrics, statistics, robustness, quality
audits, and reporting.  The evaluated system owns only ``retrieve(query, k)``.
"""

from __future__ import annotations

import json
import hashlib
import platform
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonicalize_many, canonical_verse_ref
from .metrics.generation import generation_checks
from .metrics.retrieval import retrieval_metrics, summarize_retrieval
from .runner import write_json, write_jsonl
from .universal_adapter import AdapterResponse, retrieve


QUERY_CATEGORIES = {
    "exact_verse", "verse_meaning", "conceptual", "philosophical",
    "chapter_level", "entity_person", "sanskrit_terminology", "paraphrase",
    "indirect_semantic", "multi_hop", "multi_verse", "context_dependent",
    "commentary_oriented", "confusable_verses", "cross_chapter", "very_short",
    "long_natural_language",
}
DIFFICULTIES = {"easy", "medium", "hard", "adversarial"}
REPRESENTATIONS = {
    "devanagari", "iast", "normalized_transliteration", "english_translation",
    "sanskrit_english_mixed", "commentary", "multilingual_combined", "unknown",
}


@dataclass(frozen=True)
class GoldExample:
    query_id: str
    query: str
    split: str
    category: str
    difficulty: str
    relevance: dict[str, int]
    dataset_version: str = "unknown"
    expected_chapters: tuple[int, ...] = ()
    concepts: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    hard_negatives: tuple[str, ...] = ()
    reference_answer: str = ""
    devanagari_reference: str = ""
    iast_reference: str = ""
    english_translation_reference: str = ""
    source: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    annotation_confidence: float | None = None
    annotations: tuple[dict[str, Any], ...] = ()
    corpus_representation: str = "unknown"
    retrieval_strategy: str = "unspecified"
    representation_group: str = ""
    variant_of: str = ""
    perturbation_type: str = ""
    ambiguous: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gold_refs(self) -> list[str]:
        return [ref for ref, grade in self.relevance.items() if grade > 0]

    @classmethod
    def from_dict(cls, row: Mapping[str, Any], *, dataset_version: str = "unknown") -> "GoldExample":
        query_id = str(row.get("query_id", row.get("example_id", ""))).strip()
        query = str(row.get("query", row.get("english_query", ""))).strip()
        if not query_id or not query:
            raise ValueError("benchmark rows require query_id and a non-empty query")
        split = str(row.get("split", "test")).lower()
        if split not in {"development", "dev", "validation", "test", "diagnostic"}:
            raise ValueError(f"unsupported benchmark split: {split}")
        category = str(row.get("query_category", row.get("category", row.get("query_type", "conceptual")))).lower()
        if category not in QUERY_CATEGORIES:
            raise ValueError(f"unsupported query category: {category}")
        difficulty = str(row.get("difficulty", "medium")).lower()
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"unsupported difficulty: {difficulty}")
        relevance: dict[str, int] = {}
        raw_relevance = row.get("relevance", row.get("graded_relevance", {}))
        if isinstance(raw_relevance, Mapping):
            for raw_ref, raw_grade in raw_relevance.items():
                for ref in canonicalize_many([str(raw_ref)]):
                    grade = int(raw_grade)
                    if not 0 <= grade <= 3:
                        raise ValueError("relevance grades must be integers in [0, 3]")
                    relevance[ref] = max(relevance.get(ref, 0), grade)
        gold_values = row.get("acceptable_relevant_verses", row.get("gold_verse_refs", row.get("expected_verses", [])))
        if isinstance(gold_values, str):
            gold_values = [gold_values]
        if isinstance(gold_values, Sequence):
            for ref in canonicalize_many([str(value) for value in gold_values]):
                relevance.setdefault(ref, 1)
        passages = row.get("gold_passages", [])
        if isinstance(passages, Sequence) and not isinstance(passages, (str, bytes)):
            for passage in passages:
                if not isinstance(passage, Mapping):
                    continue
                raw_ref = passage.get("passage_id", passage.get("verse_ref", passage.get("reference")))
                if raw_ref:
                    grade = int(passage.get("relevance_grade", passage.get("grade", 1)))
                    if not 0 <= grade <= 3:
                        raise ValueError("relevance grades must be integers in [0, 3]")
                    for ref in canonicalize_many([str(raw_ref)]):
                        relevance[ref] = max(relevance.get(ref, 0), grade)
        if not any(grade > 0 for grade in relevance.values()):
            raise ValueError(f"{query_id}: at least one positively relevant passage is required")
        hard_values = row.get("hard_negatives", [])
        if isinstance(hard_values, str):
            hard_values = [hard_values]
        hard_refs = canonicalize_many([
            str(value.get("passage_id", value.get("verse_ref", ""))) if isinstance(value, Mapping) else str(value)
            for value in hard_values
            if value
        ]) if hard_values else []
        chapters = row.get("expected_chapters", row.get("expected_chapter", []))
        if isinstance(chapters, (int, str)):
            chapters = [chapters]
        expected_chapters = tuple(sorted({int(value) for value in chapters})) if chapters else tuple(sorted({int(ref.split()[1].split(".")[0]) for ref, grade in relevance.items() if grade > 0}))
        if any(chapter < 1 or chapter > 18 for chapter in expected_chapters):
            raise ValueError("expected chapters must be in [1, 18]")
        confidence = row.get("annotation_confidence")
        confidence = float(confidence) if confidence is not None else None
        if confidence is not None and not 0 <= confidence <= 1:
            raise ValueError("annotation_confidence must be in [0, 1]")
        representation = str(row.get("corpus_representation", "unknown")).lower()
        if representation not in REPRESENTATIONS:
            raise ValueError(f"unsupported corpus representation: {representation}")
        annotations = row.get("annotations", [])
        normalized_annotations = []
        for annotation in annotations if isinstance(annotations, Sequence) and not isinstance(annotations, (str, bytes)) else []:
            if not isinstance(annotation, Mapping):
                raise ValueError("annotations must be objects")
            labels = annotation.get("relevance", {})
            if not isinstance(labels, Mapping):
                raise ValueError("annotation relevance must be a mapping")
            normalized_labels = {}
            for raw_ref, raw_grade in labels.items():
                grade = int(raw_grade)
                if not 0 <= grade <= 3:
                    raise ValueError("annotation relevance grades must be in [0, 3]")
                normalized_labels[canonical_verse_ref(str(raw_ref))] = grade
            normalized_annotations.append({**dict(annotation), "relevance": normalized_labels})
        return cls(
            query_id=query_id, query=query, split="development" if split == "dev" else split,
            category=category, difficulty=difficulty, relevance=relevance,
            dataset_version=str(row.get("dataset_version", dataset_version)),
            expected_chapters=expected_chapters,
            concepts=tuple(str(value) for value in row.get("concepts", [])),
            entities=tuple(str(value) for value in row.get("entities", [])),
            hard_negatives=tuple(ref for ref in hard_refs if ref not in relevance),
            reference_answer=str(row.get("reference_answer", row.get("english_translation_reference", "")) or ""),
            devanagari_reference=str(row.get("sanskrit_devanagari_reference", row.get("devanagari_reference", "")) or ""),
            iast_reference=str(row.get("iast_reference", "") or ""),
            english_translation_reference=str(row.get("english_translation_reference", "") or ""),
            source=str(row.get("source", "") or ""),
            provenance=dict(row.get("provenance") or {}),
            annotation_confidence=confidence,
            annotations=tuple(normalized_annotations),
            corpus_representation=representation,
            retrieval_strategy=str(row.get("retrieval_strategy", "unspecified")),
            representation_group=str(row.get("representation_group", "")),
            variant_of=str(row.get("variant_of", "")),
            perturbation_type=str(row.get("perturbation_type", "")),
            ambiguous=bool(row.get("ambiguous", False)),
            metadata=dict(row.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id, "query": self.query, "split": self.split,
            "query_category": self.category, "difficulty": self.difficulty,
            "relevance": self.relevance, "expected_chapters": list(self.expected_chapters),
            "concepts": list(self.concepts), "entities": list(self.entities),
            "hard_negatives": list(self.hard_negatives), "reference_answer": self.reference_answer,
            "sanskrit_devanagari_reference": self.devanagari_reference,
            "iast_reference": self.iast_reference,
            "english_translation_reference": self.english_translation_reference,
            "source": self.source, "provenance": self.provenance,
            "annotation_confidence": self.annotation_confidence,
            "annotations": list(self.annotations),
            "corpus_representation": self.corpus_representation,
            "retrieval_strategy": self.retrieval_strategy,
            "representation_group": self.representation_group,
            "variant_of": self.variant_of, "perturbation_type": self.perturbation_type,
            "ambiguous": self.ambiguous, "metadata": self.metadata,
            "dataset_version": self.dataset_version,
        }


def load_benchmark(path: str | Path, *, split: str | None = "test") -> list[GoldExample]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        dataset_version = "unknown"
    else:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        if isinstance(payload, Mapping):
            records = payload.get("examples", payload.get("data", []))
            dataset_version = str(payload.get("dataset_version", "unknown"))
        else:
            records, dataset_version = payload, "unknown"
    if not isinstance(records, list):
        raise ValueError("benchmark file must contain a list of examples")
    rows = [GoldExample.from_dict(record, dataset_version=dataset_version) for record in records]
    ids = [row.query_id for row in rows]
    duplicates = [value for value, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate query_id values: {duplicates[:10]}")
    if split is not None:
        normalized_split = "development" if split == "dev" else split
        rows = [row for row in rows if row.split == normalized_split]
    return rows


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-zA-Zāīūṛṝḷṃḥśṣñṅṭḍṇ]+", value.lower()))


def _cohen_kappa(a: list[int], b: list[int], *, weighted: bool = True) -> float | None:
    if len(a) != len(b) or not a:
        return None
    labels = sorted(set(a) | set(b))
    if len(labels) == 1:
        return 1.0
    # Qrels use a fixed 0..3 ordinal scale.  Keeping that scale fixed makes
    # agreement comparable across items whose observed labels span different
    # subsets of the rubric.
    max_distance = 3
    def weight(x: int, y: int) -> float:
        return ((x - y) / max_distance) ** 2 if weighted and max_distance else float(x != y)
    observed = sum(weight(x, y) for x, y in zip(a, b)) / len(a)
    pa = Counter(a)
    pb = Counter(b)
    expected = sum(weight(x, y) * pa[x] * pb[y] for x in labels for y in labels) / (len(a) ** 2)
    return 1.0 - observed / expected if expected else 1.0


def annotation_agreement(examples: Iterable[GoldExample]) -> dict[str, Any]:
    pair_scores: list[float] = []
    raw_agreements: list[float] = []
    pair_count = 0
    for example in examples:
        judgments: dict[str, dict[str, int]] = {}
        for annotation in example.annotations:
            annotator = str(annotation.get("annotator_id", ""))
            labels = annotation.get("relevance", {})
            if annotator and isinstance(labels, Mapping):
                judgments[annotator] = {canonical_verse_ref(str(ref)): int(grade) for ref, grade in labels.items()}
        annotators = sorted(judgments)
        for i, first in enumerate(annotators):
            for second in annotators[i + 1:]:
                refs = sorted(set(judgments[first]) | set(judgments[second]))
                a = [judgments[first].get(ref, 0) for ref in refs]
                b = [judgments[second].get(ref, 0) for ref in refs]
                score = _cohen_kappa(a, b, weighted=True)
                if score is not None:
                    pair_scores.append(score)
                    raw_agreements.append(sum(x == y for x, y in zip(a, b)) / len(a))
                    pair_count += 1
    return {
        "pair_count": pair_count,
        "weighted_cohen_kappa_mean": statistics.fmean(pair_scores) if pair_scores else None,
        "raw_agreement_mean": statistics.fmean(raw_agreements) if raw_agreements else None,
        "note": "Report per-item adjudication and confidence intervals externally when more than two annotators are used.",
    }


def audit_benchmark(examples: Sequence[GoldExample]) -> dict[str, Any]:
    normalized_queries = [" ".join(re.findall(r"\w+", row.query.lower())) for row in examples]
    duplicate_queries = [query for query, count in Counter(normalized_queries).items() if count > 1]
    lexical_overlap = []
    for row in examples:
        if not row.reference_answer:
            continue
        query_tokens = _tokens(row.query)
        answer_tokens = _tokens(row.reference_answer)
        lexical_overlap.append(len(query_tokens & answer_tokens) / len(query_tokens | answer_tokens) if query_tokens | answer_tokens else 0.0)
    chapter_counts = Counter(chapter for row in examples for chapter in row.expected_chapters)
    concept_counts = Counter(concept for row in examples for concept in row.concepts)
    confidence_values = [row.annotation_confidence for row in examples if row.annotation_confidence is not None]
    split_refs: dict[str, set[str]] = defaultdict(set)
    for row in examples:
        split_refs[row.split].update(row.gold_refs)
    split_overlap = {
        f"{a}|{b}": sorted(split_refs[a] & split_refs[b])
        for i, a in enumerate(sorted(split_refs)) for b in sorted(split_refs)[i + 1:]
        if split_refs[a] & split_refs[b]
    }
    return {
        "n_examples": len(examples),
        "duplicate_query_count": len(duplicate_queries),
        "duplicate_queries": duplicate_queries[:100],
        "ambiguous_count": sum(row.ambiguous for row in examples),
        "category_distribution": dict(sorted(Counter(row.category for row in examples).items())),
        "difficulty_distribution": dict(sorted(Counter(row.difficulty for row in examples).items())),
        "chapter_distribution": {str(key): value for key, value in sorted(chapter_counts.items())},
        "concept_distribution": dict(concept_counts.most_common()),
        "representation_distribution": dict(sorted(Counter(row.corpus_representation for row in examples).items())),
        "relevance_grade_distribution": dict(sorted(Counter(grade for row in examples for grade in row.relevance.values()).items())),
        "lexical_overlap": {
            "mean_jaccard": statistics.fmean(lexical_overlap) if lexical_overlap else None,
            "median_jaccard": statistics.median(lexical_overlap) if lexical_overlap else None,
            "denominator": len(lexical_overlap),
        },
        "annotation_confidence": {
            "mean": statistics.fmean(confidence_values) if confidence_values else None,
            "denominator": len(confidence_values),
        },
        "inter_annotator_agreement": annotation_agreement(examples),
        "cross_split_verse_overlap": split_overlap,
        "taxonomy_coverage": {
            "covered": sorted({row.category for row in examples}),
            "missing": sorted(QUERY_CATEGORIES - {row.category for row in examples}),
        },
    }


def _failure_class(example: GoldExample, ranked: list[str], metrics: Mapping[str, Any], response: AdapterResponse) -> str:
    if metrics.get("success@1"):
        return "success"
    retrieved_chapters = {int(ref.split()[1].split(".")[0]) for ref in ranked}
    if ranked and retrieved_chapters & set(example.expected_chapters):
        return "correct_chapter_wrong_verse"
    if set(ranked[:1]) & set(example.hard_negatives):
        return "hard_negative_confusion"
    if example.category == "multi_hop":
        return "multi_hop_retrieval_failure"
    if example.ambiguous:
        return "ambiguous_query_failure"
    if example.corpus_representation in {"devanagari", "iast", "normalized_transliteration"}:
        return "cross_lingual_or_transliteration_failure"
    if response.stages:
        return "pipeline_stage_failure_unlocalized"
    return "relevant_passage_absent_from_top_k"


def robustness_summary(rows: Sequence[dict[str, Any]], *, metric: str = "success@10") -> dict[str, Any]:
    by_id = {row["query_id"]: row for row in rows}
    pairs = []
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        original_id = row.get("variant_of")
        if not original_id or original_id not in by_id:
            continue
        original = float(by_id[original_id].get("metrics", {}).get(metric, 0) or 0)
        variant = float(row.get("metrics", {}).get(metric, 0) or 0)
        retained = variant / original if original > 0 else None
        pair = {
            "original_query_id": original_id, "variant_query_id": row["query_id"],
            "perturbation_type": row.get("perturbation_type", "unspecified"),
            "original": original, "variant": variant, "delta": variant - original,
            "retained_fraction": retained,
        }
        pairs.append(pair)
        if retained is not None:
            by_type[pair["perturbation_type"]].append(min(1.0, retained))
    retained_values = [pair["retained_fraction"] for pair in pairs if pair["retained_fraction"] is not None]
    return {
        "metric": metric, "paired_n": len(pairs),
        "robustness_score": statistics.fmean(min(1.0, value) for value in retained_values) if retained_values else None,
        "mean_absolute_degradation": statistics.fmean(pair["original"] - pair["variant"] for pair in pairs) if pairs else None,
        "by_perturbation": {key: {"score": statistics.fmean(values), "n": len(values)} for key, values in sorted(by_type.items())},
        "pairs": pairs,
    }


def script_invariance_summary(rows: Sequence[dict[str, Any]], *, metric: str = "success@10") -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("representation_group"):
            groups[row["representation_group"]].append(row)
    scores = []
    details = {}
    for key, values in groups.items():
        representations = {row.get("corpus_representation") for row in values}
        if len(representations) < 2:
            continue
        metric_values = [float(row.get("metrics", {}).get(metric, 0) or 0) for row in values]
        score = 1.0 - (max(metric_values) - min(metric_values))
        scores.append(score)
        details[key] = {"score": score, "representations": sorted(representations), "n": len(values)}
    return {"metric": metric, "cross_script_score": statistics.fmean(scores) if scores else None, "group_count": len(scores), "groups": details}


def evaluate(
    system: Any, benchmark: str | Path | Sequence[GoldExample], *,
    system_name: str = "unnamed-system", split: str = "test", top_k: int = 50,
    cutoffs: Iterable[int] = (1, 3, 5, 10, 20, 50), seed: int = 20260803,
    bootstrap_repetitions: int = 2000, confidence: float = .95,
    output_dir: str | Path | None = None, include_generation: bool = False,
    benchmark_version: str | None = None, system_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an arbitrary system and optionally persist a complete run."""
    examples = load_benchmark(benchmark, split=split) if isinstance(benchmark, (str, Path)) else [row for row in benchmark if row.split == split]
    if not examples:
        raise ValueError(f"benchmark contains no {split!r} examples")
    observed_versions = {row.dataset_version for row in examples if row.dataset_version != "unknown"}
    if benchmark_version is None and len(observed_versions) > 1:
        raise ValueError(f"mixed dataset versions in one run: {sorted(observed_versions)}")
    requested_cutoffs = tuple(dict.fromkeys(int(value) for value in cutoffs if int(value) > 0))
    if not requested_cutoffs or top_k < max(requested_cutoffs):
        raise ValueError("top_k must be at least the largest positive metric cutoff")
    rows: list[dict[str, Any]] = []
    run_metadata = dict(system_metadata or {})
    for example in examples:
        try:
            # Never expose qrels, hard negatives, answers, or annotator data to
            # the evaluated system.  A context-aware adapter receives only
            # non-label routing metadata needed for controlled representation
            # experiments.
            run_representation = str(run_metadata.get("corpus_representation") or example.corpus_representation)
            run_strategy = str(run_metadata.get("retrieval_architecture") or example.retrieval_strategy)
            response = retrieve(system, example.query, top_k, context={
                "query_id": example.query_id,
                "corpus_representation": run_representation,
                "representation_group": example.representation_group,
            })
            ranked = [item.passage_id for item in response.results]
            metrics = retrieval_metrics(
                ranked, example.gold_refs, graded_relevance=example.relevance,
                cutoffs=requested_cutoffs, hard_negative_refs=example.hard_negatives,
            )
            generation = generation_checks(
                response.answer, retrieved_refs=set(ranked), gold_refs=set(example.gold_refs),
                expect_citation=True,
            ) if include_generation else None
            failure_class = _failure_class(example, ranked, metrics, response)
            if include_generation and failure_class == "success" and (
                not response.answer or float((generation or {}).get("deterministic_score", 0) or 0) < .5
            ):
                failure_class = "retrieval_success_generation_failure"
            rows.append({
                "query_id": example.query_id, "example_id": example.query_id,
                "query": example.query, "query_category": example.category,
                "difficulty": example.difficulty, "expected_chapters": list(example.expected_chapters),
                "gold_refs": example.gold_refs, "graded_relevance": example.relevance,
                "hard_negatives": list(example.hard_negatives), "retrieved_refs": ranked,
                "retrieved_passages": [item.to_dict() for item in response.results],
                "metrics": metrics, "generation": generation,
                "latency_seconds": response.latency_seconds,
                "corpus_representation": run_representation,
                "retrieval_strategy": run_strategy,
                "chapter": str(example.expected_chapters[0]) if len(example.expected_chapters) == 1 else "multiple",
                "representation_group": example.representation_group,
                "variant_of": example.variant_of,
                "perturbation_type": example.perturbation_type,
                "failure_class": failure_class,
                "generation_review": {
                    "answer_correctness": None, "faithfulness": None,
                    "citation_completeness": None, "context_utilization": None,
                    "verse_attribution_accuracy": None, "chapter_verse_reference_accuracy": None,
                    "unsupported_claims": None, "reviewer_id": None, "notes": "",
                } if include_generation else None,
                "adapter_metadata": response.metadata,
                "stage_outputs": response.stages,
            })
        except Exception as exc:
            rows.append({
                "query_id": example.query_id, "example_id": example.query_id,
                "query": example.query, "query_category": example.category,
                "difficulty": example.difficulty, "expected_chapters": list(example.expected_chapters),
                "gold_refs": example.gold_refs, "metrics": {"excluded": True},
                "retrieved_refs": [], "latency_seconds": None,
                "corpus_representation": str(run_metadata.get("corpus_representation") or example.corpus_representation),
                "retrieval_strategy": str(run_metadata.get("retrieval_architecture") or example.retrieval_strategy),
                "chapter": str(example.expected_chapters[0]) if len(example.expected_chapters) == 1 else "multiple",
                "representation_group": example.representation_group,
                "variant_of": example.variant_of, "perturbation_type": example.perturbation_type,
                "failure_class": "adapter_error", "error": f"{type(exc).__name__}: {exc}",
            })
    summary = summarize_retrieval(
        rows, group_fields=("query_category", "difficulty", "chapter", "corpus_representation", "retrieval_strategy"),
        bootstrap_seed=seed, bootstrap_repetitions=bootstrap_repetitions, confidence=confidence,
    )
    summary["breakdowns"] = {}
    for field_name in ("query_category", "difficulty", "chapter", "corpus_representation", "retrieval_strategy"):
        grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped_rows[str(row.get(field_name, "unknown"))].append(row)
        summary["breakdowns"][field_name] = {
            key: summarize_retrieval(
                value, bootstrap_seed=seed, bootstrap_repetitions=bootstrap_repetitions,
                confidence=confidence,
            )
            for key, value in sorted(grouped_rows.items())
        }
    failure_counts = Counter(row["failure_class"] for row in rows)
    generation_rows = [row["generation"] for row in rows if row.get("generation")]
    generation_summary = None
    if generation_rows:
        fields = ("deterministic_score", "citation_precision", "citation_recall", "unsupported_citation_rate")
        generation_summary = {
            field: statistics.fmean(float(row[field]) for row in generation_rows if row.get(field) is not None)
            for field in fields if any(row.get(field) is not None for row in generation_rows)
        }
    metrics = summary.get("metrics", {})
    robustness_cutoff = 10 if 10 in requested_cutoffs else max(requested_cutoffs)
    multi_hop_values = [
        float(row.get("metrics", {}).get(f"recall@{robustness_cutoff}", 0) or 0)
        for row in rows if row.get("query_category") == "multi_hop" and not row.get("metrics", {}).get("excluded")
    ]
    leaderboard = {
        "recall@1": metrics.get("recall@1", {}).get("value"),
        "recall@3": metrics.get("recall@3", {}).get("value"),
        "recall@5": metrics.get("recall@5", {}).get("value"),
        "recall@10": metrics.get("recall@10", {}).get("value"),
        "mrr": metrics.get("mrr", {}).get("value"),
        "map": metrics.get("average_precision", {}).get("value"),
        "ndcg@10": metrics.get("ndcg@10", {}).get("value"),
        "robustness_score": None,
        "cross_script_score": None,
        "hard_negative_accuracy": metrics.get("hard_negative_accuracy", {}).get("value"),
        f"multi_hop_recall@{robustness_cutoff}": statistics.fmean(multi_hop_values) if multi_hop_values else None,
        "citation_accuracy": generation_summary.get("citation_precision") if generation_summary else None,
        "faithfulness": None,
    }
    robustness = robustness_summary(rows, metric=f"success@{robustness_cutoff}")
    cross_script = script_invariance_summary(rows, metric=f"success@{robustness_cutoff}")
    leaderboard["robustness_score"] = robustness["robustness_score"]
    leaderboard["cross_script_score"] = cross_script["cross_script_score"]
    dataset_version = benchmark_version or next(iter(observed_versions), "unknown")
    official = (
        split == "test"
        and all(row.provenance.get("human_verified") is True for row in examples)
        and all(row.annotation_confidence is not None for row in examples)
        and all(row.provenance.get("test_labels_locked") is True for row in examples)
        and all(row.provenance.get("contamination_audit") == "clean" for row in examples)
        and all(str(row.provenance.get("license", "unknown")).lower() not in {"", "unknown"} for row in examples)
        and not any("error" in row for row in rows)
    )
    benchmark_artifact = None
    if isinstance(benchmark, (str, Path)):
        source = Path(benchmark)
        benchmark_artifact = {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "size": source.stat().st_size,
        }
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        git_sha, git_dirty = "", None
    result = {
        "schema_version": "gita_rag_benchmark_v1.0",
        "benchmark_version": dataset_version,
        "system_name": system_name,
        "official": official,
        "officiality_reason": "human-verified, locked, contamination-clean held-out test labels" if official else "diagnostic: official scoring requires human-verified/confidence-labelled/locked test qrels, a clean contamination audit, known licensing, and zero adapter errors",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "split": split, "top_k": top_k, "cutoffs": list(requested_cutoffs),
            "random_seed": seed, "bootstrap_repetitions": bootstrap_repetitions,
            "confidence": confidence, "generation_enabled": include_generation,
        },
        "system_metadata": run_metadata,
        "reproducibility": {
            "benchmark_artifact": benchmark_artifact,
            "git": {"sha": git_sha, "dirty": git_dirty},
            "runtime": {"python": sys.version, "platform": platform.platform()},
            "random_seed": seed,
        },
        "retrieval": summary,
        "generation": generation_summary,
        "robustness": robustness,
        "cross_script": cross_script,
        "error_analysis": {"failure_counts": dict(sorted(failure_counts.items()))},
        "leaderboard": leaderboard,
        "benchmark_quality": audit_benchmark(examples),
        "per_query": rows,
    }
    if output_dir is not None:
        write_benchmark_report(output_dir, result)
    return result


def write_benchmark_report(output_dir: str | Path, result: Mapping[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = list(result.get("per_query", []))
    summary = {key: value for key, value in result.items() if key != "per_query"}
    write_json(output / "summary.json", summary)
    write_json(output / "manifest.json", {
        "schema_version": result.get("schema_version"),
        "benchmark_version": result.get("benchmark_version"),
        "system_name": result.get("system_name"),
        "created_utc": result.get("created_utc"),
        "configuration": result.get("configuration"),
        "system_metadata": result.get("system_metadata"),
        "reproducibility": result.get("reproducibility"),
        "official": result.get("official"),
    })
    write_json(output / "leaderboard.json", result.get("leaderboard", {}))
    write_json(output / "benchmark_quality.json", result.get("benchmark_quality", {}))
    write_json(output / "robustness.json", result.get("robustness", {}))
    write_json(output / "cross_script.json", result.get("cross_script", {}))
    write_json(output / "error_analysis.json", result.get("error_analysis", {}))
    write_jsonl(output / "per_query.jsonl", rows)
    write_jsonl(output / "raw_retrieval_results.jsonl", [
        {"query_id": row.get("query_id"), "query": row.get("query"), "retrieved_passages": row.get("retrieved_passages", [])}
        for row in rows
    ])
    write_jsonl(output / "failures.jsonl", [row for row in rows if row.get("failure_class") != "success"])
    leaderboard = result.get("leaderboard", {})
    lines = [
        "# Bhagavad Gita RAG Benchmark", "",
        f"- System: `{result.get('system_name')}`",
        f"- Benchmark: `{result.get('benchmark_version')}`",
        f"- Official: `{result.get('official')}`",
        "", "## Leaderboard", "",
        "| Metric | Score |", "|---|---:|",
    ]
    for name, value in leaderboard.items():
        lines.append(f"| {name} | {'—' if value is None else f'{100 * float(value):.2f}%'} |")
    lines.extend(["", "## Failure analysis", "", "```json", json.dumps(result.get("error_analysis", {}), indent=2), "```", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
