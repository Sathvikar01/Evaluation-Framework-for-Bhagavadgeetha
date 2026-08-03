"""Common dataset adapter utilities."""

from __future__ import annotations

import hashlib
import json
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from ..schemas import BenchmarkExample


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"record at {path}:{line_no} is not an object")
                records.append(value)
        return records
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(value, dict):
        for key in ("data", "records", "questions", "examples"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"expected a list of JSON objects in {path}")
    return value


class DatasetAdapter(ABC):
    name: str
    version: str
    track: str

    def __init__(self, path: str | Path | None = None, *, version: str = "unknown") -> None:
        self.path = Path(path) if path else None
        self.version = version

    @abstractmethod
    def load(self, split: str = "test", max_examples: int | None = None) -> list[BenchmarkExample]:
        raise NotImplementedError

    def prepare(self, output_dir: str | Path) -> dict[str, Any]:
        if self.path is None or not self.path.exists():
            raise FileNotFoundError(f"{self.name} data is unavailable; provide a local source path")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rows = self.load(split="test")
        normalized_path = output / "normalized.jsonl"
        temp = normalized_path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(row.json() + "\n")
        temp.replace(normalized_path)
        result = {"dataset": self.name, "path": str(self.path), "sha256": sha256_file(self.path), "normalized_path": str(normalized_path), "normalized": True, "examples": len(rows)}
        if hasattr(self, "mapping_report"):
            result["mapping_report"] = self.mapping_report()
        return result

    @staticmethod
    def _limit(items: list[BenchmarkExample], max_examples: int | None) -> list[BenchmarkExample]:
        return items if not max_examples else items[:max_examples]
