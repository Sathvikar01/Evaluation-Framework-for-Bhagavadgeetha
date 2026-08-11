"""Universal adapters for arbitrary Bhagavad Gita RAG systems."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from .canonical import passage_ref_from_result


@dataclass
class RankedPassage:
    passage_id: str
    score: float | None = None
    text: str = ""
    rank: int | None = None
    document_type: str = "verse"
    corpus_representation: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage_id, "verse_ref": self.passage_id,
            "score": self.score, "text": self.text, "rank": self.rank,
            "document_type": self.document_type,
            "corpus_representation": self.corpus_representation,
            "metadata": self.metadata,
        }


@dataclass
class AdapterResponse:
    results: list[RankedPassage]
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    stages: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0


@runtime_checkable
class UniversalRAGAdapter(Protocol):
    """The only required system contract.

    ``retrieve`` may return a ranked list of mappings or an object containing
    a ``results``/``passages``/``retrieved`` list.  Each passage must expose a
    canonicalizable ID; scores and text are optional.
    """

    def retrieve(self, query: str, k: int) -> Any: ...


def _call_retrieve(system: Any, query: str, k: int, context: Mapping[str, Any] | None = None) -> Any:
    function = system.retrieve if hasattr(system, "retrieve") else system
    if not callable(function):
        raise TypeError("adapter must be callable or implement retrieve(query, k)")
    parameters = inspect.signature(function).parameters
    if "context" in parameters:
        return function(query, k, context=context or {})
    return function(query, k)


def normalize_response(value: Any, *, elapsed_seconds: float = 0.0) -> AdapterResponse:
    if isinstance(value, AdapterResponse):
        if not value.latency_seconds:
            value.latency_seconds = elapsed_seconds
        return value
    envelope: Mapping[str, Any]
    if isinstance(value, Mapping):
        envelope = value
        raw_results = value.get("results", value.get("passages", value.get("retrieved", value.get("reranked_results", []))))
    elif isinstance(value, (list, tuple)):
        envelope = {}
        raw_results = value
    else:
        raise TypeError("retrieve() must return a ranked list or response mapping")
    if not isinstance(raw_results, (list, tuple)):
        raise TypeError("adapter response results must be a list")
    normalized: list[RankedPassage] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_results, 1):
        if isinstance(item, str):
            record: Mapping[str, Any] = {"passage_id": item}
        elif isinstance(item, Mapping):
            record = item
        else:
            raise TypeError(f"ranked result {index} is neither a string nor a mapping")
        refs = passage_ref_from_result(record)
        if not refs:
            raise ValueError(f"ranked result {index} has no canonicalizable Bhagavad Gita passage ID")
        for ref in refs:
            if ref in seen:
                continue
            seen.add(ref)
            metadata = dict(record.get("metadata") or {}) if isinstance(record.get("metadata"), Mapping) else {}
            score = record.get("score")
            normalized.append(RankedPassage(
                passage_id=ref,
                score=float(score) if isinstance(score, (int, float)) else None,
                text=str(record.get("text", record.get("content", "")) or ""),
                rank=len(normalized) + 1,
                document_type=str(record.get("document_type", record.get("chunk_type", "verse")) or "verse"),
                corpus_representation=str(record.get("corpus_representation", metadata.get("corpus_representation", "unknown")) or "unknown"),
                metadata=metadata,
                raw=dict(record),
            ))
    stages = envelope.get("stages", envelope.get("intermediate", {}))
    return AdapterResponse(
        results=normalized,
        answer=str(envelope.get("answer", "") or ""),
        citations=list(envelope.get("citations", []) or []),
        stages=dict(stages) if isinstance(stages, Mapping) else {},
        metadata=dict(envelope.get("metadata") or {}) if isinstance(envelope.get("metadata"), Mapping) else {},
        latency_seconds=float(envelope.get("latency_seconds", elapsed_seconds) or elapsed_seconds),
    )


def retrieve(system: Any, query: str, k: int, *, context: Mapping[str, Any] | None = None) -> AdapterResponse:
    start = time.perf_counter()
    value = _call_retrieve(system, query, k, context)
    return normalize_response(value, elapsed_seconds=time.perf_counter() - start)


class HTTPAdapter:
    """POST ``{"query": ..., "k": ...}`` to any JSON endpoint."""

    def __init__(self, url: str, *, headers: Mapping[str, str] | None = None, timeout_seconds: float = 60.0):
        self.url = url
        self.headers = {"Content-Type": "application/json", **dict(headers or {})}
        self.timeout_seconds = timeout_seconds

    def retrieve(self, query: str, k: int) -> Any:
        request = urllib.request.Request(
            self.url, data=json.dumps({"query": query, "k": k}).encode("utf-8"),
            headers=self.headers, method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310 - explicit user endpoint
            return json.loads(response.read().decode("utf-8"))


class CommandAdapter:
    """Run a command that accepts one JSON request on stdin and returns JSON."""

    def __init__(self, command: list[str], *, timeout_seconds: float = 60.0, cwd: str | Path | None = None):
        self.command = list(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = str(cwd) if cwd else None

    def retrieve(self, query: str, k: int) -> Any:
        completed = subprocess.run(
            self.command, input=json.dumps({"query": query, "k": k}), text=True,
            capture_output=True, timeout=self.timeout_seconds, cwd=self.cwd, check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"adapter command failed ({completed.returncode}): {completed.stderr.strip()}")
        return json.loads(completed.stdout)


class ReplayAdapter:
    """Deterministic adapter for previously captured system outputs."""

    def __init__(self, path: str | Path):
        rows = []
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        self.by_query = {str(row["query"]): row for row in rows}

    def retrieve(self, query: str, k: int) -> Any:
        if query not in self.by_query:
            raise KeyError(f"query absent from replay: {query}")
        row = dict(self.by_query[query])
        for key in ("results", "passages", "retrieved", "reranked_results"):
            if isinstance(row.get(key), list):
                row[key] = row[key][:k]
                break
        return row


def load_python_adapter(target: str, *, kwargs: Mapping[str, Any] | None = None) -> Any:
    """Load ``module:object`` or ``path/to/file.py:object``."""
    if ":" not in target:
        raise ValueError("Python adapter target must be module:object or file.py:object")
    location, object_name = target.rsplit(":", 1)
    if location.endswith(".py") or Path(location).exists():
        path = Path(location).resolve()
        spec = importlib.util.spec_from_file_location(f"gita_benchmark_adapter_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import adapter file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(location)
    value = getattr(module, object_name)
    if inspect.isclass(value):
        return value(**dict(kwargs or {}))
    if callable(value) and not hasattr(value, "retrieve") and kwargs is not None:
        candidate = value(**dict(kwargs))
        if callable(candidate) or hasattr(candidate, "retrieve"):
            return candidate
    return value


def build_universal_adapter(config: Mapping[str, Any]) -> Any:
    kind = str(config.get("kind", "python"))
    if kind == "python":
        return load_python_adapter(str(config["target"]), kwargs=config.get("kwargs"))
    if kind == "http":
        return HTTPAdapter(str(config["url"]), headers=config.get("headers"), timeout_seconds=float(config.get("timeout_seconds", 60)))
    if kind == "command":
        return CommandAdapter(list(config["command"]), timeout_seconds=float(config.get("timeout_seconds", 60)), cwd=config.get("cwd"))
    if kind == "replay":
        return ReplayAdapter(str(config["path"]))
    raise ValueError(f"unsupported universal adapter kind: {kind}")
