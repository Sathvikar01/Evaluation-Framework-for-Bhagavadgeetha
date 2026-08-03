"""Deterministic exact-reference benchmark generation and routing helpers."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from ..schemas import BenchmarkExample, normalize_verse_ref, stable_example_id
from .base import DatasetAdapter

DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_REF_BODY = r"(\d{1,2})\s*[.।:]\s*(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?"
_CUE = r"(?:bhg|bg|bhagavad\s*[- ]?g[iī]t[aā]|gita|गीता|भगवद्गीता|भगवद्\s*गीता)"


def parse_reference_query(text: str) -> list[tuple[int, int, int]]:
    """Extract reference expressions as ``(chapter, start, end)``.

    The parser is intentionally stricter than a numeric regex: ordinary
    decimals, years and quantities are not references without a Gita cue or
    an explicit Chapter/Verse/Shloka cue.
    """
    if not isinstance(text, str):
        return []
    text = text.translate(DEV_DIGITS)
    if re.search(r"\d{1,2}\s*[.।:]\s*\d{1,3}\s*[-–]\s*\d{1,2}\s*[.।:]\s*\d{1,3}", text):
        return []
    matches: list[tuple[int, int, int]] = []
    patterns = [
        rf"{_CUE}\s*{_REF_BODY}",
        rf"(?:chapter|adhyaya|अध्याय)\s*(\d{{1,2}})\s*[,;:]?\s*(?:verse|shloka|śloka|ślोक|श्लोक|श्लोका|श्लोकः)\s*(\d{{1,3}})(?:\s*[-–]\s*(\d{{1,3}}))?",
        rf"(?:verse|shloka|śloka|श्लोक)\s*{_REF_BODY}",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            groups = match.groups()
            try:
                chapter, start = int(groups[0]), int(groups[1])
                end = int(groups[2]) if len(groups) > 2 and groups[2] else start
            except (ValueError, IndexError):
                continue
            matches.append((chapter, start, end))
    # A standalone dotted reference is accepted only if a Gita cue occurs.
    if re.search(_CUE, text, flags=re.IGNORECASE):
        for match in re.finditer(rf"(?<!\d){_REF_BODY}(?!\d)", text):
            chapter, start = int(match.group(1)), int(match.group(2))
            end = int(match.group(3)) if match.group(3) else start
            matches.append((chapter, start, end))
    unique = []
    seen = set()
    for item in matches:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def expand_reference(parsed: tuple[int, int, int], inventory: set[str]) -> list[str]:
    chapter, start, end = parsed
    return [f"BhG {chapter}.{verse}" for verse in range(start, end + 1)
            if f"BhG {chapter}.{verse}" in inventory]


def discover_canonical_inventory(chunks_path: str | Path = "data/processed/chunks.jsonl") -> set[str]:
    """Derive unique verse refs from canonical verse chunks, never a constant."""
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"canonical corpus metadata not found: {path}")
    refs: set[str] = set()
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed canonical metadata at line {line_no}") from exc
            if record.get("chunk_type") != "verse":
                continue
            try:
                refs.add(normalize_verse_ref(record["verse_ref"]))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"malformed canonical verse record at line {line_no}") from exc
    if not refs:
        raise ValueError(f"no verse references discovered in {path}")
    return refs


def _variant_templates(ref: str) -> list[str]:
    chapter, verse = ref.removeprefix("BhG ").split(".")
    return [
        ref, f"BG {chapter}:{verse}", f"Bhagavad Gita {chapter}.{verse}",
        f"Bhagavad-gītā {chapter}.{verse}", f"Chapter {chapter} verse {verse}",
        f"chapter {chapter}, shloka {verse}", f"Gita {chapter}.{verse}",
        f"bhg  {int(chapter):02d} . {int(verse):02d}",
        f"अध्याय {chapter} श्लोक {verse}",
        f"भगवद्गीता {chapter}।{verse}",
    ]


def generate_with_id_records(
    inventory: Iterable[str], *, seed: int = 20260803, include_invalid: bool = True,
) -> list[BenchmarkExample]:
    refs = sorted({normalize_verse_ref(ref) for ref in inventory})
    records: list[BenchmarkExample] = []
    for ref in refs:
        chapter, verse = ref.removeprefix("BhG ").split(".")
        for index, query in enumerate(_variant_templates(ref)):
            records.append(BenchmarkExample(
                example_id=f"with_id:{chapter}.{verse}:valid:{index}",
                dataset_name="with_id_canonical",
                dataset_version="inventory-derived-v1",
                split="test", track="with_id", query=query, query_language="mixed",
                query_type="exact_reference", gold_verse_refs=(ref,),
                explicit_reference=True, source_identifier=ref,
                metadata={"variant_index": index, "expected_valid": True},
                raw_record={"query": query},
            ))
    if include_invalid:
        rng = random.Random(seed)
        max_chapter = max(int(r.removeprefix("BhG ").split(".")[0]) for r in refs)
        invalid_queries = [
            ("Chapter 0 verse 1", "invalid_bounds"),
            (f"BhG {max_chapter + 1}.1", "invalid_chapter"),
            ("BhG 2.999", "invalid_verse"),
            ("BhG 2.47-1", "reverse_range"),
            ("BhG 2.47-3.1", "cross_chapter_range"),
            ("Bhagavad Gita", "incomplete"),
            ("The number 2.47 is a decimal, not a verse reference", "false_positive"),
            ("What happened in 1947?", "false_positive"),
        ]
        rng.shuffle(invalid_queries)
        for index, (query, kind) in enumerate(invalid_queries):
            records.append(BenchmarkExample(
                example_id=f"with_id:invalid:{index}", dataset_name="with_id_canonical",
                dataset_version="inventory-derived-v1", split="test", track="with_id",
                query=query, query_language="en", query_type="invalid_reference",
                metadata={"variant_kind": kind, "expected_valid": False},
                raw_record={"query": query},
            ))
    return records


class WithIDAdapter(DatasetAdapter):
    name = "with_id_canonical"
    track = "with_id"

    def __init__(self, chunks_path: str | Path = "data/processed/chunks.jsonl", **kwargs: Any) -> None:
        super().__init__(None, version="inventory-derived-v1")
        self.chunks_path = Path(chunks_path)

    def load(self, split: str = "test", max_examples: int | None = None) -> list[BenchmarkExample]:
        rows = generate_with_id_records(discover_canonical_inventory(self.chunks_path))
        return self._limit([row for row in rows if row.split == split], max_examples)

    def prepare(self, output_dir: str | Path) -> dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        inventory = discover_canonical_inventory(self.chunks_path)
        rows = generate_with_id_records(inventory)
        data_path = output / "with_id.jsonl"
        tmp_path = data_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row.json() + "\n")
        tmp_path.replace(data_path)
        (output / "inventory.json").write_text(json.dumps({"count": len(inventory), "refs": sorted(inventory)}, indent=2), encoding="utf-8")
        return {"dataset": self.name, "examples": len(rows), "inventory_count": len(inventory), "path": str(data_path)}
