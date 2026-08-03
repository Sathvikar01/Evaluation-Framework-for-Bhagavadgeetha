"""Deterministic audit of question coverage and surface-form diversity."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .datasets.base import load_json_records
from .datasets.with_id import parse_reference_query


_QUALITY_PATTERNS = {
    # The public English source contains templating artefacts such as
    # "in this teaching on King". Keep this deterministic and transparent so
    # flagged rows can be repaired or excluded without an LLM judge.
    "teaching_on_placeholder": re.compile(r"\bthis teaching on\b", re.IGNORECASE),
}


def question_quality_flags(question: str) -> list[str]:
    """Return deterministic quality flags for a benchmark question."""
    return [name for name, pattern in _QUALITY_PATTERNS.items() if pattern.search(question)]


def _stem(question: str) -> str:
    text = question.strip().casefold()
    for prefix in ("what ", "how ", "why ", "which ", "who ", "when ", "can ", "should ", "does ", "do ", "is ", "are ", "in ", "according to "):
        if text.startswith(prefix):
            return prefix.strip()
    return text.split(maxsplit=1)[0] if text else ""


def audit_questions(path: str | Path) -> dict[str, Any]:
    rows = load_json_records(path)
    by_ref: dict[str, list[str]] = defaultdict(list)
    all_questions: list[str] = []
    stems = Counter()
    quality_flags = Counter()
    flagged_questions = []
    explicit_reference_count = 0
    for row in rows:
        question = str(row.get("question", row.get("query", ""))).strip()
        if not question:
            continue
        chapter = row.get("chapter", row.get("chapter_no"))
        verse = row.get("verse", row.get("verse_no"))
        refs = row.get("gold_verse_refs", [])
        if not refs and chapter is not None and verse is not None:
            refs = [f"BhG {chapter}.{verse}"]
        if isinstance(refs, str):
            refs = [refs]
        for ref in refs:
            by_ref[str(ref)].append(question)
        all_questions.append(question)
        stems[_stem(question)] += 1
        row_flags = question_quality_flags(question)
        for flag in row_flags:
            quality_flags[flag] += 1
        if row_flags:
            flagged_questions.append({"question": question, "flags": row_flags, "verse": f"BhG {chapter}.{verse}" if chapter is not None and verse is not None else ""})
        explicit_reference_count += bool(parse_reference_query(question))
    duplicate_questions = sum(count - 1 for count in Counter(all_questions).values() if count > 1)
    per_verse_counts = Counter(len(set(questions)) for questions in by_ref.values())
    return {
        "path": str(path),
        "rows": len(rows),
        "questions": len(all_questions),
        "verses": len(by_ref),
        "questions_per_verse": dict(Counter(len(questions) for questions in by_ref.values())),
        "unique_questions_per_verse": dict(per_verse_counts),
        "duplicate_question_count": duplicate_questions,
        "explicit_reference_question_count": explicit_reference_count,
        "explicit_reference_fraction": explicit_reference_count / len(all_questions) if all_questions else 0.0,
        "surface_stems": dict(stems),
        "surface_stem_fraction": {key: value / len(all_questions) for key, value in stems.items()} if all_questions else {},
        "quality_flags": dict(quality_flags),
        "quality_flagged_question_count": len(flagged_questions),
        "quality_flagged_fraction": len(flagged_questions) / len(all_questions) if all_questions else 0.0,
        "quality_flagged_examples": flagged_questions[:25],
    }
