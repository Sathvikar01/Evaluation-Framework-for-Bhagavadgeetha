"""GitaDB adapter with explicit source-to-canonical mapping artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..schemas import BenchmarkExample, normalize_verse_ref, stable_example_id
from .base import DatasetAdapter, load_json_records, sha256_file


class GitaDBAdapter(DatasetAdapter):
    name = "gitadb"
    track = "cross_lingual_gita"

    def __init__(self, path: str | Path, *, mapping_path: str | Path | None = None, version: str = "manual") -> None:
        super().__init__(path, version=version)
        self.mapping_path = Path(mapping_path) if mapping_path else (Path(path).parent / "mapping.json")
        self.mapping: dict[str, list[str]] = {}
        self.unmapped: list[dict[str, Any]] = []
        self.ambiguous: list[dict[str, Any]] = []

    def _load_mapping(self) -> None:
        if not self.mapping_path.exists():
            raise FileNotFoundError(f"GitaDB mapping artifact required: {self.mapping_path}")
        raw = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        rows = raw.get("mappings", raw) if isinstance(raw, dict) else raw
        if not isinstance(rows, (dict, list)):
            raise ValueError("GitaDB mapping must be an object or list")
        if isinstance(rows, dict):
            rows = [{"source_id": key, "gold_verse_refs": value} for key, value in rows.items()]
        self.mapping = {}
        for row in rows:
            source_id = str(row.get("source_id", row.get("id", "")))
            values = row.get("gold_verse_refs", row.get("verse_refs", []))
            if isinstance(values, str): values = [values]
            refs = [normalize_verse_ref(v if str(v).startswith("BhG") else f"BhG {v}") for v in values]
            if not source_id or not refs:
                continue
            self.mapping[source_id] = list(dict.fromkeys(refs))

    def _records(self) -> list[BenchmarkExample]:
        assert self.path is not None
        self._load_mapping()
        output = []
        for index, record in enumerate(load_json_records(self.path)):
            source_id = str(record.get("source_id", record.get("id", index)))
            refs = self.mapping.get(source_id, [])
            if not refs:
                self.unmapped.append({"source_id": source_id, "record": record})
                continue
            if len(refs) > 1 or record.get("ambiguous"):
                self.ambiguous.append({"source_id": source_id, "refs": refs})
            query = str(record.get("translation", record.get("query", record.get("text", "")))).strip()
            if not query:
                raise ValueError(f"GitaDB record {source_id} has no translation text")
            language = str(record.get("language", record.get("lang", "en"))).lower()
            output.append(BenchmarkExample(
                example_id=stable_example_id(self.name, "test", query, source_id),
                dataset_name=self.name, dataset_version=self.version, split="test",
                track=self.track, query=query, query_language=language, query_type="translation",
                gold_verse_refs=tuple(refs), graded_relevance={ref: 3 for ref in refs},
                source_identifier=source_id, source_url=str(record.get("source_url", "")),
                license=str(record.get("license", "unknown")), metadata={"mapping_status": "mapped"}, raw_record=record,
            ))
        return output

    def load(self, split: str = "test", max_examples: int | None = None) -> list[BenchmarkExample]:
        return self._limit(self._records(), max_examples)

    def mapping_report(self) -> dict[str, Any]:
        # ``self.mapping`` contains only source IDs with usable canonical refs;
        # ``self.unmapped`` contains source records absent from that mapping.
        # Subtracting the latter from the former understated coverage (and could
        # even produce a negative mapped count) when the source was incomplete.
        mapped = len(self.mapping)
        total = mapped + len(self.unmapped)
        reverse: dict[str, list[str]] = defaultdict(list)
        for source_id, refs in self.mapping.items():
            for ref in refs: reverse[ref].append(source_id)
        return {"mapping_coverage": {"total_source_ids": total, "mapped": mapped, "unmapped": len(self.unmapped), "fraction": mapped / max(total, 1)},
                "mapped_examples": mapped, "unmapped_examples": self.unmapped,
                "ambiguous_examples": self.ambiguous,
                "one_to_many": {k: v for k, v in self.mapping.items() if len(v) > 1},
                "many_to_one": {k: v for k, v in reverse.items() if len(v) > 1}}
