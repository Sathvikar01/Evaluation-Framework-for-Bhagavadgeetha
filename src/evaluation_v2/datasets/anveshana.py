"""Out-of-domain Anveshana retrieval adapter."""

from __future__ import annotations

from pathlib import Path

from ..schemas import BenchmarkExample, normalize_verse_ref, stable_example_id
from .base import DatasetAdapter, load_json_records


class AnveshanaAdapter(DatasetAdapter):
    name = "anveshana"
    track = "external_generalization"

    def load(self, split: str = "test", max_examples: int | None = None) -> list[BenchmarkExample]:
        assert self.path is not None
        rows = []
        for index, record in enumerate(load_json_records(self.path)):
            query = str(record.get("query", record.get("question", record.get("translation", "")))).strip()
            if not query:
                raise ValueError(f"Anveshana record {index} has no query")
            refs = record.get("gold_verse_refs", record.get("passage_ids", []))
            if isinstance(refs, str): refs = [refs]
            # Only repository-compatible BhG labels are usable by the live
            # Gita pipeline. Other passage IDs remain explicitly unmapped.
            normalized = []
            for ref in refs or []:
                try:
                    normalized.append(normalize_verse_ref(ref if str(ref).startswith("BhG") else f"BhG {ref}"))
                except ValueError:
                    continue
            source_id = str(record.get("id", record.get("source_id", record.get("Name", index))))
            rows.append(BenchmarkExample(
                example_id=stable_example_id(self.name, split, query, source_id), dataset_name=self.name,
                dataset_version=self.version, split=split, track=self.track, query=query,
                query_language=str(record.get("language", "en")), query_type="external_scripture",
                gold_verse_refs=tuple(normalized), graded_relevance={ref: 3 for ref in normalized},
                reference_answer=str(record.get("answer", "")), source_identifier=source_id,
                source_url=str(record.get("source_url", "")), license=str(record.get("license", "unknown")),
                metadata={"external_passage_id": record.get("passage_id", record.get("Name", source_id)), "source_relevance_label": record.get("label"), "mapping_status": "mapped" if normalized else "unmapped"}, raw_record=record,
            ))
        return self._limit(rows, max_examples)
