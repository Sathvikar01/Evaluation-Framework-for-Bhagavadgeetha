"""Configuration loading for Evaluation V2."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = {
    "schema_version": "evaluation_v2.1",
    "mode": "official",
    "seed": 20260803,
    "enabled_tracks": ["with_id", "without_id_gita_qa", "cross_lingual_gita", "external_generalization", "generation", "legacy_compatibility"],
    "paths": {
        "evaluation_root": "data/evaluation_v2",
        "results_root": "results/evaluation_v2",
        "chunks": "data/processed/chunks.jsonl",
        "legacy_external": "data/evaluation/external",
        "quick_root": "data/evaluation_v2/quick",
    },
    "datasets": {
        "bhagavad_gita_qa": {"path": "data/evaluation_v2/bhagavad_gita_qa/english_source.jsonl", "version": "JDhruv14-2025-en", "languages": ["en"], "source_url": "https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita-QA", "license": "MIT"},
        "gitadb": {"path": "data/evaluation_v2/gitadb", "version": "manual", "languages": ["en", "hi", "gu", "or", "ta", "te"]},
        "edwin_arnold": {"path": "data/evaluation/external/edwin_arnold_qa.jsonl", "version": "local", "languages": ["en"]},
        "anveshana": {"path": "data/evaluation_v2/anveshana/test_data.csv", "version": "manojbalaji1-2025", "languages": ["en"], "source_url": "https://huggingface.co/datasets/manojbalaji1/anveshana", "license": "unknown"},
    },
    "split": {"seed": 20260803, "train": 0.70, "validation": 0.10, "test": 0.20},
    "retrieval": {"cutoffs": [1, 3, 5, 10, 20, 50], "max_examples": None, "use_api": False, "answer_aware": False},
    "generation": {"enabled": False, "max_examples": 50, "answer_aware": False},
    "judge": {"enabled": False, "allow_paid": False, "max_examples": 0, "temperature": 0.0, "rubric_version": "v1"},
    "bootstrap": {"seed": 20260803, "repetitions": 2000, "confidence": 0.95},
    "leakage": {"exact_threshold": 1.0, "token_overlap_threshold": 0.85, "fuzzy_threshold": 0.92, "ngram_size": 5},
    "strict_component_health": True,
    "warmup_runs": 0,
}


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path = "configs/evaluation_v2.yaml") -> dict[str, Any]:
    config_path = Path(path)
    extra = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if not isinstance(extra, dict):
        raise ValueError(f"evaluation config must be a mapping: {config_path}")
    config = _merge(DEFAULT_CONFIG, extra)
    config["config_path"] = str(config_path)
    return config
