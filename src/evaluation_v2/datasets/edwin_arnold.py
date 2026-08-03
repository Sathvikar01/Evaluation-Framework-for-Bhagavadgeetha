"""Edwin Arnold QA adapter; verse labels are optional and never fabricated."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schemas import BenchmarkExample, normalize_verse_ref, stable_example_id
from .base import DatasetAdapter, load_json_records


class EdwinArnoldAdapter(DatasetAdapter):
    name = "edwin_arnold"
    track = "generation"

    def load(self, split: str = "test", max_examples: int | None = None) -> list[BenchmarkExample]:
        assert self.path is not None
        rows = []
        for index, record in enumerate(load_json_records(self.path)):
            query = str(record.get("question", record.get("query", ""))).strip()
            if not query:
                raise ValueError(f"Edwin Arnold record {index} has no question")
            raw_refs = record.get("gold_verse_refs", record.get("verse_refs", []))
            if isinstance(raw_refs, str): raw_refs = [raw_refs]
            refs = []
            for raw_ref in raw_refs or []:
                try:
                    refs.append(normalize_verse_ref(raw_ref if str(raw_ref).startswith("BhG") else f"BhG {raw_ref}"))
                except ValueError:
                    raise ValueError(f"invalid explicit Edwin Arnold verse label at record {index}: {raw_ref}")
            source_id = str(record.get("id", record.get("source_id", index)))
            category = str(record.get("category", record.get("type", "verse_qa")))
            rows.append(BenchmarkExample(
                example_id=stable_example_id(self.name, split, query, source_id), dataset_name=self.name,
                dataset_version=self.version, split=split, track=self.track, query=query,
                query_language=str(record.get("language", "en")), query_type=category,
                gold_verse_refs=tuple(refs), graded_relevance={ref: 3 for ref in refs},
                reference_answer=str(record.get("answer", record.get("reference_answer", ""))),
                source_identifier=source_id, source_url=str(record.get("source_url", "")),
                license=str(record.get("license", "unknown")), metadata={"mapping_status": "source_label" if refs else "unmapped"}, raw_record=record,
            ))
        return self._limit(rows, max_examples)

    def mapping_review_export(self, output_path: str | Path) -> Path:
        rows = self.load(split="test")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                if not row.gold_verse_refs:
                    handle.write(f"{row.example_id}\t{row.query}\t{row.reference_answer}\n")
        return path
