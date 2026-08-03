"""Adapter for the clean, verse-grouped Bhagavad-Gita-QA benchmark."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..schemas import BenchmarkExample, normalize_verse_ref, stable_example_id
from .base import DatasetAdapter, load_json_records, sha256_file


def _first(record: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _refs(record: dict[str, Any]) -> list[str]:
    values = record.get("gold_verse_refs", record.get("verse_refs", []))
    if isinstance(values, str):
        values = [values]
    if not values:
        chapter = record.get("chapter", record.get("chapter_no"))
        verse = record.get("verse", record.get("verse_no"))
        if chapter is not None and verse is not None:
            values = [f"BhG {chapter}.{verse}"]
    if not isinstance(values, list) or not values:
        raise ValueError("Bhagavad-Gita-QA record has no explicit verse label")
    return [normalize_verse_ref(value if str(value).startswith("BhG") else f"BhG {value}") for value in values]


def _language(record: dict[str, Any]) -> str:
    value = _first(record, "language", "lang", "query_language", default="en").lower()
    return {"english": "en", "hindi": "hi", "gujarati": "gu"}.get(value, value)


def split_by_verse_group(
    examples: list[BenchmarkExample], *, seed: int, ratios: dict[str, float]
) -> dict[str, list[BenchmarkExample]]:
    """Split whole normalized verse groups, never individual questions."""
    groups: dict[tuple[str, ...], list[BenchmarkExample]] = defaultdict(list)
    for example in examples:
        key = tuple(sorted(example.gold_verse_refs))
        if not key:
            raise ValueError(f"cannot split unlabeled example {example.example_id}")
        groups[key].append(example)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    train_end = int(n * ratios.get("train", 0.7))
    valid_end = train_end + int(n * ratios.get("validation", 0.1))
    assignments = {key: "train" if i < train_end else "validation" if i < valid_end else "test" for i, key in enumerate(keys)}
    output = {"train": [], "validation": [], "test": []}
    for key, rows in groups.items():
        output[assignments[key]].extend(rows)
    for split, rows in output.items():
        output[split] = [
            BenchmarkExample(**{**row.to_dict(), "split": split}) for row in rows
        ]
    return output


class BhagavadGitaQAAdapter(DatasetAdapter):
    name = "bhagavad_gita_qa"
    track = "without_id_gita_qa"

    def __init__(self, path: str | Path, *, version: str = "local", seed: int = 20260803, ratios: dict[str, float] | None = None) -> None:
        super().__init__(path, version=version)
        self.seed = seed
        self.ratios = ratios or {"train": 0.7, "validation": 0.1, "test": 0.2}
        self._split_cache: dict[str, list[BenchmarkExample]] | None = None

    def _normalize(self) -> list[BenchmarkExample]:
        assert self.path is not None
        rows: list[BenchmarkExample] = []
        for index, record in enumerate(load_json_records(self.path)):
            query = _first(record, "question", "query", "prompt")
            answer = _first(record, "answer", "ground_truth", "reference_answer", "response")
            if not query:
                raise ValueError(f"record {index} in {self.path} has no question")
            refs = _refs(record)
            source_id = _first(record, "id", "question_id", "uid", default=str(index))
            rows.append(BenchmarkExample(
                example_id=stable_example_id(self.name, "all", query, source_id),
                dataset_name=self.name, dataset_version=self.version, split="all",
                track=self.track, query=query, query_language=_language(record),
                query_type=_first(record, "category", "question_type", "type", default="qa"),
                gold_verse_refs=tuple(refs),
                graded_relevance={ref: 3 for ref in refs}, reference_answer=answer,
                explicit_reference=bool(record.get("explicit_reference", False)),
                source_identifier=source_id, source_url=_first(record, "source_url", "url"),
                license=_first(record, "license", default="unknown"), metadata={
                    "original_index": index, "source_language": _language(record),
                }, raw_record=record,
            ))
        return rows

    def split_manifest(self) -> dict[str, Any]:
        if self._split_cache is None:
            self._split_cache = split_by_verse_group(self._normalize(), seed=self.seed, ratios=self.ratios)
        manifest = {"seed": self.seed, "dataset": self.name, "version": self.version,
                    "checksum": sha256_file(self.path), "splits": {}}
        for split, rows in self._split_cache.items():
            refs = sorted({ref for row in rows for ref in row.gold_verse_refs})
            manifest["splits"][split] = {"verse_refs": refs, "count": len(rows),
                                          "by_language": _counts(rows, "query_language"),
                                          "by_query_type": _counts(rows, "query_type")}
        return manifest

    def prepare(self, output_dir: str | Path) -> dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._split_cache = split_by_verse_group(self._normalize(), seed=self.seed, ratios=self.ratios)
        for split, rows in self._split_cache.items():
            path = output / f"{split}.jsonl"
            temp = path.with_suffix(".tmp")
            with temp.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(row.json() + "\n")
            temp.replace(path)
        manifest = self.split_manifest()
        (output / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"dataset": self.name, "normalized": True, "manifest": manifest}

    def load(self, split: str = "test", max_examples: int | None = None) -> list[BenchmarkExample]:
        if self._split_cache is None:
            self._split_cache = split_by_verse_group(self._normalize(), seed=self.seed, ratios=self.ratios)
        if split not in self._split_cache:
            raise ValueError(f"unknown split {split}")
        return self._limit(self._split_cache[split], max_examples)


def _counts(rows: list[BenchmarkExample], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(getattr(row, field))
        counts[value] = counts.get(value, 0) + 1
    return counts
