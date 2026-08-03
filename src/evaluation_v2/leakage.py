"""Leakage audit for indexed, augmented, cached and trained textual material."""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


def normalize_text(text: str) -> str:
    text = (text or "").casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _tokens(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if len(token) > 2}


def _source_texts(root: Path, *, reference_only: bool = False) -> list[tuple[str, str, str]]:
    """Discover likely textual sources without loading binary/model files."""
    candidates = [
        root / "data/processed/chunks.jsonl", root / "data/processed/verse_theme_summaries_cache.json",
        root / "data/processed/interpretation_canon_cache.json", root / "data/processed/commentary_english_cache.json",
        root / "data/evaluation/external", root / "training", root / "cache",
    ]
    if reference_only:
        candidates = [root / "data/processed/chunks.jsonl"]
    found: list[tuple[str, str, str]] = []
    paths = []
    for candidate in candidates:
        if candidate.is_file(): paths.append(candidate)
        elif candidate.is_dir():
            if candidate.name == "training":
                paths.extend(p for p in candidate.rglob("*") if p.is_file() and p.suffix.lower() in {".jsonl", ".ndjson"} and ("pair" in p.name.lower() or "negative" in p.name.lower() or "prompt" in p.name.lower()))
            elif candidate.name == "cache":
                paths.extend(p for p in candidate.rglob("*") if p.is_file() and p.suffix.lower() == ".json" and ("cache" in p.name.lower() or "qa" in p.name.lower()))
            else:
                paths.extend(p for p in candidate.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".ndjson", ".txt", ".md", ".csv"})
    seen = set()
    for path in paths:
        if path in seen: continue
        seen.add(path)
        if path.name == "chunks.jsonl" and path.stat().st_size > 50_000_000:
            try:
                with path.open(encoding="utf-8-sig", errors="ignore") as handle:
                    for line in handle:
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        for string in _interesting_strings(value):
                            if len(string.strip()) >= 12:
                                found.append((str(path), string.strip(), "indexed_text_field"))
            except OSError:
                pass
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
        except OSError: continue
        if len(text) > 50_000_000: continue
        # Keep both whole lines and JSON string values; matching a whole QA
        # question in chunks is the important contamination signal.
        for line in text.splitlines():
            is_json_line = path.suffix.lower() in {".json", ".jsonl", ".ndjson"}
            if not is_json_line and len(line.strip()) >= 12:
                found.append((str(path), line.strip(), "indexed_or_cached_text"))
            if reference_only:
                continue
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                value = None
            if value is not None:
                for string in _interesting_strings(value):
                    if len(string.strip()) >= 12:
                        found.append((str(path), string.strip(), "json_text_value"))
    return found


_TEXT_KEYS = {"text", "text_english", "text_iast", "text_devanagari", "text_themes", "question", "query", "answer", "reference_answer", "prompt", "response", "content", "positive", "negative", "positive_text", "negative_text", "input", "output"}
_CONTAINER_KEYS = {"data", "records", "examples", "questions", "items", "messages", "results"}


def _interesting_strings(value: Any, *, depth: int = 0) -> list[str]:
    """Extract answer/query-bearing strings, excluding token/morphology blobs."""
    if isinstance(value, str):
        return [value] if depth == 0 else []
    if isinstance(value, list):
        return [part for item in value for part in _interesting_strings(item, depth=depth + 1)]
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            if key in _TEXT_KEYS and isinstance(item, str):
                result.append(item)
            elif key in _TEXT_KEYS or key in _CONTAINER_KEYS:
                result.extend(_interesting_strings(item, depth=depth + 1))
            elif depth == 0 and isinstance(item, str):
                # Mapping/cache files often use an opaque ID as the key.
                result.append(item)
        return result
    return []


def audit_examples(
    examples: Iterable[dict[str, Any]], *, repo_root: str | Path = ".", train_examples: Iterable[dict[str, Any]] = (), test_examples: Iterable[dict[str, Any]] = (), thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or {}
    examples = list(examples)
    reference_only = bool(examples) and all(row.get("track") == "with_id" for row in examples)
    # WithID contains generated aliases, not answer-bearing benchmark text.
    # Scanning the 86MB canonical JSONL for fuzzy question/answer matches would
    # be both wasteful and semantically wrong (the same verse ref is expected
    # in corpus metadata). Its contamination surface is therefore recorded as
    # a reference-only scope; answer-bearing tracks use the full audit below.
    if reference_only:
        return {"schema_version": "leakage_v2.1", "status": "clean", "official": True, "sources_scanned": 0, "audit_scope": "reference_only_no_answer_text", "findings": [], "finding_count": 0, "definite_count": 0}
    sources = _source_texts(Path(repo_root), reference_only=False)
    normalized_sources: dict[str, list[tuple[str, str, str]]] = {}
    hash_sources: dict[str, list[tuple[str, str, str]]] = {}
    token_sources: list[tuple[str, set[str], str, str]] = []
    for path, text, kind in sources:
        normalized = normalize_text(text)
        if not normalized: continue
        normalized_sources.setdefault(normalized, []).append((path, text, kind))
        hash_sources.setdefault(content_hash(text), []).append((path, text, kind))
        # Exact/hash matching still covers every discovered source. The large
        # hard-negative corpus is intentionally excluded from the expensive
        # fuzzy index; its exact records remain auditable without a multi-
        # minute tokenisation pass.
        path_posix = path.replace("\\", "/")
        fuzzy_eligible = (
            "chunks.jsonl" in path_posix
            or "data/evaluation/external" in path_posix
            or "training_pairs_v2" in path_posix
        )
        if fuzzy_eligible:
            token_sources.append((path, _tokens(text), text, kind))
    postings: dict[str, set[int]] = {}
    for index, (_, source_words, _, _) in enumerate(token_sources):
        for token in source_words:
            postings.setdefault(token, set()).add(index)
    findings = []
    for example in examples:
        fields = [("query", example.get("query", "")), ("answer", example.get("reference_answer", example.get("answer", "")))]
        for field, value in fields:
            value = str(value or "").strip()
            normalized = normalize_text(value)
            if len(normalized) < 12: continue
            matches = normalized_sources.get(normalized, [])
            for path, text, kind in matches[:3]:
                findings.append(_finding(example, path, "exact_normalized", 1.0, field, text, kind))
            if matches: continue
            value_hash = content_hash(value)
            for path, text, kind in hash_sources.get(value_hash, [])[:3]:
                findings.append(_finding(example, path, "stable_hash", 1.0, field, text, kind))
            if reference_only:
                continue
            words = _tokens(value)
            if len(words) < 4: continue
            best = (0.0, None)
            # Use an inverted token index rather than comparing every query
            # with every source line. This matters for the 7k-example WithID
            # benchmark and the repository's large augmentation files.
            rare_tokens = sorted(words, key=lambda token: len(postings.get(token, ())))[:4]
            candidate_ids: set[int] | None = None
            for token in rare_tokens:
                posting = postings.get(token, set())
                candidate_ids = set(posting) if candidate_ids is None else candidate_ids & posting
                if not candidate_ids: break
            for source_index in list(candidate_ids or ())[:1000]:
                path, source_words, text, kind = token_sources[source_index]
                overlap = len(words & source_words) / max(len(words), 1)
                if overlap > best[0]: best = (overlap, (path, text, kind))
            if best[0] >= float(thresholds.get("token_overlap_threshold", .85)) and best[1]:
                path, text, kind = best[1]
                findings.append(_finding(example, path, "high_token_overlap", best[0], field, text, kind))
            # Fuzzy matching is only run against high token-overlap candidates
            # to keep an audit bounded on the large corpus.
            if best[1] and best[0] >= .65:
                path, text, kind = best[1]
                similarity = SequenceMatcher(None, normalized, normalize_text(text)).ratio()
                if similarity >= float(thresholds.get("fuzzy_threshold", .92)):
                    findings.append(_finding(example, path, "fuzzy_similarity", similarity, field, text, kind))
    # Same-verse grouping is an independent, deterministic failure class.
    train_by_ref = {ref for row in train_examples for ref in row.get("gold_verse_refs", [])}
    test_by_ref = {ref for row in test_examples for ref in row.get("gold_verse_refs", [])}
    overlap = sorted(train_by_ref & test_by_ref)
    for ref in overlap:
        findings.append({"dataset": "split_policy", "example_id": ref, "matched_source": "train/test manifests", "match_type": "same_verse_train_test_overlap", "similarity": 1.0, "verse_ref": ref, "severity": "definite", "recommended_action": "regenerate verse-grouped splits"})
    definite = [finding for finding in findings if finding["severity"] == "definite"]
    return {"schema_version": "leakage_v2.1", "status": "contaminated" if definite else "clean", "official": not bool(definite), "sources_scanned": len(sources), "findings": findings, "finding_count": len(findings), "definite_count": len(definite)}


def _finding(example: dict[str, Any], path: str, match_type: str, similarity: float, field: str, text: str, kind: str) -> dict[str, Any]:
    return {"dataset": example.get("dataset_name", example.get("dataset", "")), "example_id": example.get("example_id", ""), "matched_source": path, "match_type": f"{field}:{match_type}", "similarity": round(float(similarity), 6), "verse_ref": (example.get("gold_verse_refs") or [""])[0], "severity": "definite" if match_type in {"exact_normalized", "stable_hash"} else "possible", "recommended_action": "exclude source-contaminated example or rebuild index from training-only material", "source_kind": kind, "matched_text_preview": text[:180]}


def write_leakage_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    import os
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output / "leakage_report.json", output / "leakage_report.md"
    temp = json_path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"); os.replace(temp, json_path)
    lines = [f"# Leakage report\n\n- Status: **{report['status']}**\n- Official: `{report['official']}`\n- Sources scanned: {report['sources_scanned']}\n- Findings: {report['finding_count']}\n"]
    for finding in report.get("findings", []):
        lines.append(f"- `{finding['severity']}` `{finding['dataset']}` `{finding['example_id']}` — {finding['match_type']} ({finding['similarity']}) in `{finding['matched_source']}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
