"""Complete V2 reproducibility manifests."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): _json_safe(item) for key, item in value.items() if key != "func"}
    if isinstance(value, (list, tuple)): return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return repr(value)


def _git(args: list[str], root: Path) -> str:
    try: return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return ""


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(root: Path, path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.exists() or not candidate.is_file():
        return None
    try:
        display = str(candidate.relative_to(root))
    except ValueError:
        display = str(candidate)
    return {"path": display, "size": candidate.stat().st_size, "sha256": _sha256_file(candidate)}


def _configured_artifacts(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Capture hashes for split, mapping, and index inputs used by a run."""
    split_hashes: dict[str, Any] = {}
    evaluation_root = Path(config.get("paths", {}).get("evaluation_root", "data/evaluation_v2"))
    if not evaluation_root.is_absolute():
        evaluation_root = root / evaluation_root
    if evaluation_root.exists():
        for path in sorted(evaluation_root.rglob("split_manifest.json")):
            record = _artifact_record(root, path)
            if record:
                split_hashes[str(path.relative_to(root))] = record

    mapping_hashes: dict[str, Any] = {}
    candidates = []
    for entry in config.get("datasets", {}).values():
        if isinstance(entry, dict) and entry.get("mapping_path"):
            candidates.append(entry["mapping_path"])
    candidates.append(evaluation_root / "gitadb" / "mapping.json")
    for path in candidates:
        record = _artifact_record(root, path)
        if record:
            mapping_hashes[record["path"]] = record

    production_config = root / "configs/config.yaml"
    production: dict[str, Any] = {}
    if production_config.exists():
        try:
            import yaml
            value = yaml.safe_load(production_config.read_text(encoding="utf-8"))
            production = value if isinstance(value, dict) else {}
        except Exception:
            production = {}
    data_cfg = production.get("data", {}) if isinstance(production.get("data"), dict) else {}
    embedding_cfg = production.get("embedding", {}) if isinstance(production.get("embedding"), dict) else {}
    dual_cfg = embedding_cfg.get("dual_index", {}) if isinstance(embedding_cfg.get("dual_index"), dict) else {}
    index_paths = {
        "chunks": config.get("paths", {}).get("chunks") or data_cfg.get("chunks_file"),
        "faiss_index": data_cfg.get("faiss_index"),
        "faiss_metadata": data_cfg.get("faiss_metadata"),
        "dual_faiss_index": dual_cfg.get("faiss_index"),
        "dual_faiss_metadata": dual_cfg.get("faiss_metadata"),
        "interpretation_index": "data/processed/interpretation_index.json",
    }
    index_metadata: dict[str, Any] = {}
    for name, path in index_paths.items():
        if path:
            record = _artifact_record(root, path)
            if record:
                index_metadata[name] = record
    return split_hashes, mapping_hashes, index_metadata


def capture_manifest(*, config: dict[str, Any], args: dict[str, Any], repo_root: str | Path = ".", health: dict[str, Any] | None = None, leakage_status: str = "not_run") -> dict[str, Any]:
    root = Path(repo_root)
    production_config = root / "configs/config.yaml"
    config_snapshot = production_config.read_text(encoding="utf-8") if production_config.exists() else ""
    packages = {}
    for name in ("numpy", "scipy", "pytest", "pyyaml", "sentence-transformers", "faiss-cpu", "neo4j"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    split_hashes, mapping_hashes, index_metadata = _configured_artifacts(root, config)
    return {"schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "git": {"sha": _git(["rev-parse", "HEAD"], root), "branch": _git(["branch", "--show-current"], root), "dirty": bool(_git(["status", "--porcelain"], root))}, "cli_args": _json_safe(args), "evaluation_config": _json_safe(config), "production_config_snapshot": config_snapshot, "dataset_versions": {name: value.get("version") for name, value in config.get("datasets", {}).items()}, "split_manifest_hashes": split_hashes, "mapping_artifact_hashes": mapping_hashes, "index_metadata": index_metadata, "models": {"embedding": "live pipeline config", "reranker": "live pipeline config", "generation": "live pipeline config", "query_expansion": "live pipeline config"}, "runtime": {"python": sys.version, "platform": platform.platform(), "packages": packages}, "random_seeds": {"evaluation": config.get("seed"), "split": config.get("split", {}).get("seed"), "bootstrap": config.get("bootstrap", {}).get("seed")}, "component_health": _json_safe(health or {}), "api_dependent_components_used": {"retrieval_api": bool(config.get("retrieval", {}).get("use_api")), "generation": bool(config.get("generation", {}).get("enabled")), "llm_judge": bool(config.get("judge", {}).get("enabled"))}, "cost": {"estimated": None, "currency": None}, "leakage_audit": {"status": leakage_status, "official": leakage_status == "clean"}}


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
