"""Machine-readable and Markdown report generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .runner import write_json, write_jsonl
from .overall import aggregate_overall


def build_summary(track_results: dict[str, Any], *, official: bool, leakage_status: str, health: dict[str, Any]) -> dict[str, Any]:
    tracks = dict(track_results)
    for required in ("with_id", "without_id_gita_qa", "cross_lingual_gita", "external_generalization", "generation", "legacy_compatibility"):
        tracks.setdefault(required, {"status": "not_run"})
    return {"schema_version": "evaluation_v2.1", "official": bool(official), "leakage_status": leakage_status, "overall": aggregate_overall(tracks), "tracks": tracks, "component_health": health}


def write_run_report(output_dir: str | Path, summary: dict[str, Any], rows: list[dict[str, Any]], *, generation_rows: list[dict[str, Any]] | None = None) -> None:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_jsonl(output / "per_query.jsonl", rows)
    write_jsonl(output / "failures.jsonl", [row for row in rows if row.get("error") or row.get("stages", {}).get("failure_class") not in {None, "success", "not_applicable"}])
    write_json(output / "stage_analysis.json", {"tracks": summary.get("tracks", {})})
    write_json(output / "latency.json", {name: value.get("latency", {}) for name, value in summary.get("tracks", {}).items() if isinstance(value, dict)})
    if generation_rows is not None:
        write_jsonl(output / "generation_review.jsonl", generation_rows)
        if generation_rows:
            with (output / "generation_review.csv").open("w", newline="", encoding="utf-8") as handle:
                fields = sorted({key for row in generation_rows for key in row})
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({key: row.get(key, "") for key in fields} for row in generation_rows)
    lines = ["# SansRAG Evaluation V2", "", f"- Official: `{summary.get('official')}`", f"- Leakage: **{summary.get('leakage_status')}**", "", "## Tracks", ""]
    for track, result in summary.get("tracks", {}).items():
        lines.append(f"### {track}")
        lines.append("")
        lines.append("```json")
        import json
        lines.append(json.dumps(result, ensure_ascii=False, indent=2)[:8000])
        lines.append("```")
    lines += ["", "## Component health", "", "```json", __import__("json").dumps(summary.get("component_health", {}), indent=2), "```"]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
