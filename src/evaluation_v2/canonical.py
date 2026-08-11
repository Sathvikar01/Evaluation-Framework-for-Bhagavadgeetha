"""Canonical Bhagavad Gita passage identities and result normalization.

The benchmark scores canonical passages, never surface text.  A translation,
commentary chunk, Devanagari verse, IAST verse, or mixed document for verse
2.47 therefore shares the identity ``BhG 2.47``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


CHAPTER_VERSE_COUNTS = {
    chapter: count for chapter, count in enumerate(
        (47, 72, 43, 42, 29, 47, 30, 28, 34, 42, 55, 20, 35, 27, 20, 24, 28, 78),
        1,
    )
}

_PREFIX = r"(?:BhG|BG|Bhagavad\s+Gita|Gita)?"
_SINGLE_RE = re.compile(
    rf"^\s*{_PREFIX}\s*(?:chapter\s*)?(\d{{1,2}})(?:\s*[.:]\s*|\s+verse\s+)(\d{{1,3}})\s*$",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    rf"^\s*{_PREFIX}\s*(\d{{1,2}})\s*[.:]\s*(\d{{1,3}})\s*(?:-|–|—|to)\s*(?:(\d{{1,2}})\s*[.:]\s*)?(\d{{1,3}})\s*$",
    re.IGNORECASE,
)


def canonical_verse_ref(value: str, *, validate: bool = True) -> str:
    """Normalize a common Bhagavad Gita reference spelling to ``BhG C.V``."""
    if not isinstance(value, str):
        raise ValueError("passage reference must be a string")
    match = _SINGLE_RE.fullmatch(value)
    if not match:
        raise ValueError(f"unrecognized Bhagavad Gita verse reference: {value!r}")
    chapter, verse = (int(part) for part in match.groups())
    if validate and (chapter not in CHAPTER_VERSE_COUNTS or not 1 <= verse <= CHAPTER_VERSE_COUNTS[chapter]):
        raise ValueError(f"verse outside canonical Bhagavad Gita inventory: {value!r}")
    return f"BhG {chapter}.{verse}"


def expand_passage_ref(value: str, *, validate: bool = True) -> list[str]:
    """Expand a single verse or same-chapter inclusive range."""
    try:
        return [canonical_verse_ref(value, validate=validate)]
    except ValueError:
        pass
    match = _RANGE_RE.fullmatch(value) if isinstance(value, str) else None
    if not match:
        raise ValueError(f"unrecognized Bhagavad Gita passage reference: {value!r}")
    start_chapter, start_verse, end_chapter, end_verse = match.groups()
    start_chapter, start_verse, end_verse = int(start_chapter), int(start_verse), int(end_verse)
    end_chapter = int(end_chapter) if end_chapter else start_chapter
    if end_chapter != start_chapter:
        raise ValueError("cross-chapter ranges must be annotated as separate passages")
    if end_verse < start_verse:
        raise ValueError("passage range end precedes start")
    refs = [f"BhG {start_chapter}.{verse}" for verse in range(start_verse, end_verse + 1)]
    if validate:
        for ref in refs:
            canonical_verse_ref(ref, validate=True)
    return refs


def canonicalize_many(values: Iterable[str], *, validate: bool = True) -> list[str]:
    refs: list[str] = []
    for value in values:
        refs.extend(expand_passage_ref(value, validate=validate))
    return list(dict.fromkeys(refs))


def passage_ref_from_result(result: Mapping[str, Any]) -> list[str]:
    """Extract canonical passage IDs from an architecture-agnostic result.

    Adapters may return a direct ID or put it under metadata.  Text is never
    guessed for official scoring because number extraction from translations
    or commentary creates silent false positives.
    """
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    candidates: list[Any] = []
    for key in ("passage_id", "verse_ref", "canonical_id", "reference", "id"):
        if result.get(key) is not None:
            candidates.append(result[key])
        if metadata.get(key) is not None:
            candidates.append(metadata[key])
    for candidate in candidates:
        raw_values = candidate if isinstance(candidate, (list, tuple)) else [candidate]
        refs: list[str] = []
        try:
            for raw in raw_values:
                refs.extend(expand_passage_ref(str(raw)))
        except ValueError:
            continue
        if refs:
            return list(dict.fromkeys(refs))
    return []
