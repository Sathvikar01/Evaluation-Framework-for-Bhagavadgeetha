"""Official V2 dataset registry.

Kaggle is intentionally absent. It remains a legacy diagnostic in the old
evaluator only and cannot enter an official V2 run through this registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .datasets import AnveshanaAdapter, BhagavadGitaQAAdapter, EdwinArnoldAdapter, GitaDBAdapter, WithIDAdapter


OFFICIAL_DATASETS = {
    "with_id": "with_id_canonical",
    "bhagavad_gita_qa": "bhagavad_gita_qa",
    "gitadb": "gitadb",
    "edwin_arnold": "edwin_arnold",
    "anveshana": "anveshana",
}


def build_adapter(name: str, config: dict[str, Any]):
    if name not in OFFICIAL_DATASETS:
        raise KeyError(f"dataset {name!r} is not in the official Evaluation V2 registry")
    datasets = config.get("datasets", {})
    if name == "with_id":
        return WithIDAdapter(config.get("paths", {}).get("chunks", "data/processed/chunks.jsonl"))
    entry = datasets.get(name, {})
    path = entry.get("path")
    if name == "bhagavad_gita_qa":
        prepared_dir = Path(config.get("paths", {}).get("evaluation_root", "data/evaluation_v2")) / "bhagavad_gita_qa"
        return BhagavadGitaQAAdapter(
            path,
            version=entry.get("version", "unknown"),
            seed=config.get("split", {}).get("seed", config.get("seed")),
            ratios=config.get("split", {}),
            prepared_dir=prepared_dir if prepared_dir.exists() else None,
        )
    if name == "gitadb":
        if not path or not Path(path).exists():
            raise FileNotFoundError("GitaDB source is unavailable. The paper-linked tickloop/gitadb repository is no longer reachable; provide a licensed export and mapping.json.")
        return GitaDBAdapter(path, mapping_path=entry.get("mapping_path"), version=entry.get("version", "unknown"))
    if name == "edwin_arnold":
        return EdwinArnoldAdapter(path, version=entry.get("version", "unknown"))
    return AnveshanaAdapter(path, version=entry.get("version", "unknown"))


def official_dataset_names() -> list[str]:
    return list(OFFICIAL_DATASETS)
