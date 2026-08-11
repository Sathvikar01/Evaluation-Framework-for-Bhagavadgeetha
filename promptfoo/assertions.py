"""Deterministic Promptfoo assertions; LLM judges remain supplementary."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation_v2.canonical import canonical_verse_ref  # noqa: E402


_CITATION_RE = re.compile(r"(?:BhG|BG|Bhagavad\s+Gita)\s+(\d{1,2})[.:](\d{1,3})", re.IGNORECASE)


def _metadata(context):
    response = context.get("providerResponse", {}) or {}
    return response.get("metadata", context.get("metadata", {})) or {}


def _retrieved(output, context):
    metadata = _metadata(context)
    refs = metadata.get("retrieved_refs", [])
    if refs:
        return {canonical_verse_ref(str(ref)) for ref in refs}
    try:
        payload = json.loads(output)
        return {canonical_verse_ref(str(ref)) for ref in payload.get("retrieved_refs", [])}
    except (ValueError, TypeError, json.JSONDecodeError):
        return set()


def _citations(output):
    return {f"BhG {int(chapter)}.{int(verse)}" for chapter, verse in _CITATION_RE.findall(output)}


def _gold(context):
    values = context["vars"].get("gold_refs", [])
    if isinstance(values, str):
        values = [values]
    return {canonical_verse_ref(str(value)) for value in values}


def evidence_retrieved(output, context):
    gold = _gold(context)
    retrieved = _retrieved(output, context)
    score = len(gold & retrieved) / len(gold) if gold else 0.0
    return {"pass": score > 0, "score": score, "reason": f"retrieved {sorted(gold & retrieved)} of {sorted(gold)}"}


def citation_correctness(output, context):
    citations = _citations(output)
    if not citations:
        # Retrieval-only Promptfoo runs have no generated citations; they are
        # neutral here and are scored by EvidenceRetrieved.
        return {"pass": True, "score": 1.0, "reason": "retrieval-only output"}
    retrieved = _retrieved(output, context)
    score = len(citations & retrieved) / len(citations)
    return {"pass": score == 1.0, "score": score, "reason": f"unsupported citations: {sorted(citations - retrieved)}"}


def citation_completeness(output, context):
    citations = _citations(output)
    if not citations:
        return {"pass": True, "score": 1.0, "reason": "retrieval-only output"}
    gold = _gold(context)
    score = len(citations & gold) / len(gold) if gold else 1.0
    return {"pass": score > 0, "score": score, "reason": f"uncited gold passages: {sorted(gold - citations)}"}
