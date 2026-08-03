"""Deterministic checks for generated answers and citations."""

from __future__ import annotations

import re
from typing import Any


_CITATION = re.compile(
    r"\b(?:BhG|BG|Bhagavad(?:\s*[- ]?Gita)?)\s+(\d+)\s*[.:]\s*(\d+)\b",
    re.IGNORECASE,
)


def generation_checks(
    answer: str, *, retrieved_refs: set[str], gold_refs: set[str] | None = None,
    expect_citation: bool = True, canonical_text_by_ref: dict[str, str] | None = None,
) -> dict[str, Any]:
    answer = answer or ""
    raw_matches = list(_CITATION.finditer(answer))
    citations = list(dict.fromkeys(f"BhG {m.group(1)}.{m.group(2)}" for m in raw_matches))
    valid = [ref for ref in citations if ref in retrieved_refs]
    unsupported = [ref for ref in citations if ref not in retrieved_refs]
    gold = gold_refs or set()
    exact_matches = []
    if canonical_text_by_ref:
        lowered = answer.lower()
        for ref, text in canonical_text_by_ref.items():
            if text and text.lower() in lowered:
                exact_matches.append(ref)
    citation_precision = len(set(valid)) / len(set(citations)) if citations else (1.0 if not expect_citation else 0.0)
    citation_recall = len(set(citations) & gold) / len(gold) if gold else None
    unsupported_penalty = 1.0 - (len(set(unsupported)) / len(set(citations)) if citations else 0.0)
    empty_or_refusal = not answer.strip() or bool(re.search(r"\b(I don't have|I cannot|unable to answer|no relevant verses)\b", answer, re.I))
    valid_answer = 0.0 if empty_or_refusal else 1.0
    deterministic_score = (citation_precision + unsupported_penalty + valid_answer) / 3.0 if not (empty_or_refusal and expect_citation) else 0.0
    return {
        "citations": citations,
        "valid_citations": valid,
        "unsupported_references": unsupported,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "retrieved_citation_rate": len(set(valid)) / len(set(retrieved_refs)) if retrieved_refs else 0.0,
        "duplicate_citation_count": len(raw_matches) - len(citations),
        "has_expected_citation": bool(citations) if expect_citation else True,
        "empty_or_refusal": empty_or_refusal,
        "sanskrit_or_iast_quote_matches": exact_matches,
        "chapter_verse_consistent": not unsupported,
        "unsupported_penalty": unsupported_penalty,
        "deterministic_score": deterministic_score,
    }
