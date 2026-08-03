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


def capture_manifest(*, config: dict[str, Any], args: dict[str, Any], repo_root: str | Path = ".", health: dict[str, Any] | None = None, leakage_status: str = "not_run") -> dict[str, Any]:
    root = Path(repo_root)
    production_config = root / "configs/config.yaml"
    config_snapshot = production_config.read_text(encoding="utf-8") if production_config.exists() else ""
    packages = {}
    for name in ("numpy", "scipy", "pytest", "pyyaml", "sentence-transformers", "faiss-cpu", "neo4j"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return {"schema_version": SCHEMA_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(), "git": {"sha": _git(["rev-parse", "HEAD"], root), "branch": _git(["branch", "--show-current"], root), "dirty": bool(_git(["status", "--porcelain"], root))}, "cli_args": _json_safe(args), "evaluation_config": _json_safe(config), "production_config_snapshot": config_snapshot, "dataset_versions": {name: value.get("version") for name, value in config.get("datasets", {}).items()}, "split_manifest_hashes": {}, "mapping_artifact_hashes": {}, "index_metadata": {}, "models": {"embedding": "live pipeline config", "reranker": "live pipeline config", "generation": "live pipeline config", "query_expansion": "live pipeline config"}, "runtime": {"python": sys.version, "platform": platform.platform(), "packages": packages}, "random_seeds": {"evaluation": config.get("seed"), "split": config.get("split", {}).get("seed"), "bootstrap": config.get("bootstrap", {}).get("seed")}, "component_health": _json_safe(health or {}), "api_dependent_components_used": {"retrieval_api": bool(config.get("retrieval", {}).get("use_api")), "generation": bool(config.get("generation", {}).get("enabled")), "llm_judge": bool(config.get("judge", {}).get("enabled"))}, "cost": {"estimated": None, "currency": None}, "leakage_audit": {"status": leakage_status, "official": leakage_status == "clean"}}


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
