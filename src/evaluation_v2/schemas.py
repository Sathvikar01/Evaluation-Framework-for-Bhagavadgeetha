"""Typed, JSON-serialisable schemas used by Evaluation V2."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

_REF_RE = re.compile(r"^BhG\s+(\d+)\.(\d+)$")


def normalize_verse_ref(value: str) -> str:
    """Normalize a canonical reference, rejecting malformed values."""
    if not isinstance(value, str):
        raise ValueError(f"verse reference must be a string, got {type(value).__name__}")
    text = re.sub(r"\s+", " ", value.strip())
    match = _REF_RE.fullmatch(text)
    if not match:
        raise ValueError(f"malformed canonical verse reference: {value!r}")
    return f"BhG {int(match.group(1))}.{int(match.group(2))}"


@dataclass(frozen=True)
class BenchmarkExample:
    example_id: str
    dataset_name: str
    dataset_version: str
    split: str
    track: str
    query: str
    query_language: str
    query_type: str
    gold_verse_refs: tuple[str, ...] = ()
    graded_relevance: dict[str, int] = field(default_factory=dict)
    reference_answer: str = ""
    explicit_reference: bool = False
    source_identifier: str = ""
    source_url: str = ""
    license: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_record: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.example_id or not self.dataset_name or not self.query.strip():
            raise ValueError("example_id, dataset_name and non-empty query are required")
        if self.split not in {"train", "validation", "test", "all", "diagnostic"}:
            raise ValueError(f"unsupported split: {self.split}")
        if self.track not in {
            "with_id", "without_id_gita_qa", "cross_lingual_gita",
            "external_generalization", "generation", "legacy_compatibility",
        }:
            raise ValueError(f"unsupported evaluation track: {self.track}")
        refs = tuple(dict.fromkeys(normalize_verse_ref(r) for r in self.gold_verse_refs))
        object.__setattr__(self, "gold_verse_refs", refs)
        relevance = {}
        for ref, label in self.graded_relevance.items():
            normalized = normalize_verse_ref(ref)
            if not isinstance(label, int) or label < 0 or label > 3:
                raise ValueError("graded relevance labels must be integers in [0, 3]")
            relevance[normalized] = label
        object.__setattr__(self, "graded_relevance", relevance)

    def to_dict(self, include_raw: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data["gold_verse_refs"] = list(self.gold_verse_refs)
        if not include_raw:
            data.pop("raw_record", None)
        return data

    def json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass
class PipelineObservation:
    query: str
    result: dict[str, Any]
    elapsed_seconds: float
    health: dict[str, Any] = field(default_factory=dict)

    @property
    def reranked(self) -> list[dict[str, Any]]:
        return list(self.result.get("reranked_results", []))

    @property
    def intermediate(self) -> dict[str, Any]:
        return dict(self.result.get("intermediate", {}))


@dataclass
class ExampleResult:
    example: BenchmarkExample
    retrieved_refs: list[str] = field(default_factory=list)
    retrieved_records: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    error: str = ""
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "example": self.example.to_dict(),
            "example_id": self.example.example_id,
            "dataset_name": self.example.dataset_name,
            "track": self.example.track,
            "retrieved_refs": self.retrieved_refs,
            "retrieved_records": self.retrieved_records,
            "metrics": self.metrics,
            "stages": self.stages,
            "generation": self.generation,
            "latency": self.latency,
            "error": self.error,
            "degraded": self.degraded,
        }


def stable_example_id(dataset: str, split: str, query: str, source_id: str = "") -> str:
    import hashlib
    payload = "\x1f".join((dataset, split, source_id, query.strip()))
    return f"{dataset}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"
