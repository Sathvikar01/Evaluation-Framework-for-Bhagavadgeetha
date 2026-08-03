"""Balanced, small diagnostic benchmarks for fast model checks.

The quick QA set deliberately samples one source question per verse from the
active public source. It is a coverage diagnostic, not a replacement for the
held-out release test split: the source has QA for 700 of the 701 canonical
verses, so the missing verse is recorded rather than filled with fabricated
semantic text. Leakage status is reported by the normal audit gate.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .datasets.base import DatasetAdapter, load_json_records, sha256_file
from .datasets.bhagavad_gita_qa import BhagavadGitaQAAdapter
from .datasets.with_id import discover_canonical_inventory, generate_with_id_records, parse_reference_query
from .question_audit import question_quality_flags
from .schemas import BenchmarkExample, stable_example_id


QUICK_VERSION = "balanced-v1"


# A small number of verses have no clean, reference-free question in the
# public source. These deterministic manual rewrites keep the balanced quick
# benchmark free of exact-reference routing and templating artefacts without
# changing the archived source dataset.
_QUICK_QUALITY_OVERRIDES = {
    "BhG 11.28": "What does the image of rivers entering the ocean reveal about the warriors entering Krishna's blazing mouths?",
    "BhG 12.14": "How do contentment and firm determination shape a devotee's spiritual practice?",
    "BhG 13.11": "How does distinguishing true knowledge from ignorance guide spiritual growth?",
    "BhG 13.33": "What does the sun analogy teach about the Self's illumination of the field?",
    "BhG 13.5": "How do the ten organs and five sense objects contribute to the description of the field?",
    "BhG 15.17": "How is the Supreme Person distinguished from ordinary beings in this verse?",
    "BhG 18.38": "Why is sense-based pleasure compared with nectar first and poison later?",
    "BhG 18.73": "What change in Arjuna's state of mind is expressed after Krishna's guidance?",
    "BhG 6.12": "How do controlling the mind and senses support self-purification in meditation?",
    "BhG 6.18": "What does a disciplined mind look like when one is established in Yoga?",
    "BhG 6.23": "How is Yoga described as freedom from sorrow, and what effort is required?",
    "BhG 6.47": "Why is the devoted yogi who focuses the mind on Krishna considered the highest yogi?",
}


def _apply_quick_override(example: BenchmarkExample) -> BenchmarkExample:
    ref = example.gold_verse_refs[0] if len(example.gold_verse_refs) == 1 else ""
    query = _QUICK_QUALITY_OVERRIDES.get(ref)
    if not query:
        return example
    metadata = {**example.metadata, "quick_quality_override": True}
    return BenchmarkExample(**{
        **example.to_dict(),
        "example_id": stable_example_id(example.dataset_name, "quick", query, example.source_identifier),
        "query": query,
        "metadata": metadata,
        "explicit_reference": False,
    })


class PreparedQuickAdapter(DatasetAdapter):
    """Load a generated quick JSONL while preserving the original track."""

    def __init__(self, path: str | Path, *, name: str, track: str, version: str = QUICK_VERSION) -> None:
        super().__init__(path, version=version)
        self.name = name
        self.track = track

    def load(self, split: str = "test", max_examples: int | None = None) -> list[BenchmarkExample]:
        assert self.path is not None
        rows = [BenchmarkExample(**record) for record in load_json_records(self.path)]
        return self._limit([row for row in rows if row.split == split], max_examples)


def _write_examples(path: Path, rows: list[BenchmarkExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.json() + "\n")
    temp.replace(path)


def _qa_quick_rows(adapter: BhagavadGitaQAAdapter, inventory: set[str], preferred_languages: tuple[str, ...] = ("en",)) -> tuple[list[BenchmarkExample], dict[str, Any]]:
    # Use the full clean source only to build a balanced diagnostic set. The
    # official held-out split remains unchanged and is still the release score.
    all_rows = adapter._normalize()  # noqa: SLF001 - adapter owns normalization
    by_ref_language: dict[str, dict[str, list[BenchmarkExample]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        for ref in row.gold_verse_refs:
            by_ref_language[ref][row.query_language].append(row)

    selected: list[BenchmarkExample] = []
    refs = sorted(by_ref_language)
    for index, ref in enumerate(refs):
        languages = by_ref_language[ref]
        preferred = preferred_languages[index % len(preferred_languages)] if preferred_languages else ""
        language = preferred if languages.get(preferred) else sorted(languages)[0]
        # Stable lexical selection keeps the quick set identical across runs.
        candidates = [
            item for item in languages[language]
            if not parse_reference_query(item.query)
            and not question_quality_flags(item.query)
        ]
        # If a future verse has no clean reference-free question, preserve
        # coverage with the least constrained deterministic fallback.
        if not candidates:
            candidates = [item for item in languages[language] if not parse_reference_query(item.query)]
        if not candidates:
            candidates = languages[language]
        row = sorted(candidates, key=lambda item: (item.query, item.example_id))[0]
        selected.append(_apply_quick_override(BenchmarkExample(
            **{**row.to_dict(), "split": "test", "dataset_version": QUICK_VERSION,
               "metadata": {**row.metadata, "quick_balanced": True, "quick_source_split": "all"}}
        )))

    missing = sorted(inventory - set(by_ref_language))
    language_counts = Counter(row.query_language for row in selected)
    counts = Counter(ref for row in selected for ref in row.gold_verse_refs)
    return selected, {
        "source_records": len(all_rows),
        "selected_examples": len(selected),
        "covered_verses": len(counts),
        "canonical_verses": len(inventory),
        "missing_from_source": missing,
        "questions_per_covered_verse": sorted(set(counts.values())),
        "language_counts": dict(language_counts),
        "equal_questions_per_verse": len(set(counts.values())) <= 1,
    }


def prepare_quick(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"].get("evaluation_root", "data/evaluation_v2"))
    quick_root = root / "quick"
    chunks_path = config["paths"].get("chunks", "data/processed/chunks.jsonl")
    inventory = discover_canonical_inventory(chunks_path)

    qa_config = config["datasets"]["bhagavad_gita_qa"]
    qa_adapter = BhagavadGitaQAAdapter(
        qa_config["path"], version=qa_config.get("version", "unknown"),
        seed=config.get("split", {}).get("seed", config.get("seed", 20260803)),
        ratios=config.get("split", {}),
    )
    qa_rows, qa_stats = _qa_quick_rows(qa_adapter, inventory, tuple(qa_config.get("languages", ["en"])))

    with_id_rows = [row for row in generate_with_id_records(inventory, include_invalid=False)
                    if row.metadata.get("variant_index") == 0]
    with_id_rows = [BenchmarkExample(**{**row.to_dict(), "dataset_version": QUICK_VERSION,
                                        "metadata": {**row.metadata, "quick_balanced": True}})
                    for row in with_id_rows]
    with_id_counts = Counter(ref for row in with_id_rows for ref in row.gold_verse_refs)

    qa_path = quick_root / "bhagavad_gita_qa.jsonl"
    with_id_path = quick_root / "with_id.jsonl"
    _write_examples(qa_path, qa_rows)
    _write_examples(with_id_path, with_id_rows)
    manifest = {
        "schema_version": "evaluation_v2.1",
        "quick_version": QUICK_VERSION,
        "seed": config.get("seed", 20260803),
        "status": "balanced_diagnostic",
        "official_release_score": False,
        "source_paths": {"qa": str(qa_adapter.path), "chunks": str(chunks_path)},
        "source_sha256": {"qa": sha256_file(qa_adapter.path), "chunks": sha256_file(chunks_path)},
        "tracks": {
            "without_id_gita_qa": {"path": str(qa_path), **qa_stats},
            "with_id": {"path": str(with_id_path), "selected_examples": len(with_id_rows),
                         "covered_verses": len(with_id_counts), "canonical_verses": len(inventory),
                         "questions_per_verse": sorted(set(with_id_counts.values())),
                         "missing_from_inventory": sorted(inventory - set(with_id_counts)),
                         "equal_questions_per_verse": len(set(with_id_counts.values())) <= 1},
        },
    }
    (quick_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def quick_adapter(config: dict[str, Any], name: str) -> PreparedQuickAdapter:
    root = Path(config["paths"].get("evaluation_root", "data/evaluation_v2")) / "quick"
    if name == "quick_bhagavad_gita_qa":
        return PreparedQuickAdapter(root / "bhagavad_gita_qa.jsonl", name=name, track="without_id_gita_qa")
    if name == "quick_with_id":
        return PreparedQuickAdapter(root / "with_id.jsonl", name=name, track="with_id")
    raise KeyError(f"unknown quick dataset: {name}")
