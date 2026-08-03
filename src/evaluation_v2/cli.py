"""Command line interface for Evaluation V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from .comparison import compare_files
from .config import load_config
from .acquisition import prepare_public_dataset
from .datasets.with_id import discover_canonical_inventory
from .leakage import audit_examples, write_leakage_report
from .manifest import capture_manifest, write_manifest
from .pipeline_adapter import LivePipelineAdapter
from .quick import prepare_quick, quick_adapter
from .question_audit import audit_questions
from .registry import build_adapter, official_dataset_names
from .reporting import build_summary, write_run_report
from .runner import run_generation, run_retrieval, write_json, write_jsonl


def _config(args: argparse.Namespace) -> dict:
    return load_config(args.config)


def _dataset_names(args: argparse.Namespace, default: list[str]) -> list[str]:
    values = args.datasets or default
    if "kaggle_gita_qa" in values:
        raise SystemExit("Kaggle is excluded from the official Evaluation V2 registry")
    return values


def cmd_prepare(args: argparse.Namespace) -> int:
    config = _config(args); names = official_dataset_names() if args.all else [args.dataset]
    output = Path(config["paths"]["evaluation_root"])
    statuses = []
    for name in names:
        try:
            acquisition = None
            if name in {"bhagavad_gita_qa", "anveshana"}:
                source_path = Path(config.get("datasets", {}).get(name, {}).get("path", ""))
                if not source_path.exists():
                    acquisition = prepare_public_dataset(name, config["paths"]["evaluation_root"])
            adapter = build_adapter(name, config)
            target = output / ("with_id" if name == "with_id" else name)
            status = adapter.prepare(target)
            if acquisition is not None:
                status["acquisition"] = acquisition
            metadata_path = Path(config["paths"]["evaluation_root"]) / name / "acquisition_metadata.json"
            if metadata_path.exists():
                status["acquisition_metadata_path"] = str(metadata_path)
            statuses.append(status)
        except (FileNotFoundError, ValueError) as exc:
            if not args.skip_missing_datasets: raise
            statuses.append({"dataset": name, "status": "blocked", "error": str(exc)})
    print(json.dumps(statuses, ensure_ascii=False, indent=2)); return 0


def cmd_prepare_quick(args: argparse.Namespace) -> int:
    manifest = prepare_quick(_config(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2)); return 0


def cmd_audit_questions(args: argparse.Namespace) -> int:
    report = audit_questions(args.path)
    output = Path(args.output) if args.output else Path(args.path).with_name("question_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)); return 0


def _load_examples(config: dict, names: list[str], split: str, max_examples: int | None) -> list:
    examples = []
    for name in names:
        adapter = quick_adapter(config, name) if name.startswith("quick_") else build_adapter(name, config)
        examples.extend(adapter.load(split=split, max_examples=max_examples))
    return examples


def cmd_leakage(args: argparse.Namespace) -> int:
    config = _config(args)
    examples = []
    train_examples = []
    names = args.datasets or ["bhagavad_gita_qa"]
    for name in names:
        try:
            examples.extend(_load_examples(config, [name], "test", None))
            if name not in {"with_id"} and not str(name).startswith("quick_"):
                train_examples.extend(_load_examples(config, [name], "train", None))
        except FileNotFoundError:
            continue
    report = audit_examples(
        [row.to_dict(include_raw=False) for row in examples],
        repo_root=".",
        thresholds=config.get("leakage"),
        train_examples=[row.to_dict(include_raw=False) for row in train_examples],
        test_examples=[row.to_dict(include_raw=False) for row in examples],
    )
    write_leakage_report(report, Path(config["paths"]["results_root"]))
    print(json.dumps({key: report.get(key) for key in ("status", "official", "sources_scanned", "finding_count", "definite_count")}, ensure_ascii=False, indent=2)); return 0 if report["official"] or args.allow_contaminated else 2


def _default_datasets_for_track(track: str) -> list[str]:
    return {
        "with_id": ["with_id"],
        "without_id_gita_qa": ["bhagavad_gita_qa"],
        "cross_lingual_gita": ["gitadb"],
        "external_generalization": ["anveshana"],
        "generation": ["edwin_arnold"],
    }.get(track, ["bhagavad_gita_qa"])


def _audit_with_split_policy(config: dict, names: list[str], test_examples: list) -> dict:
    """Audit test examples and enforce same-verse train/test overlap when available."""
    train_rows: list[dict] = []
    for name in names:
        if name.startswith("quick_") or name == "with_id":
            continue
        try:
            adapter = build_adapter(name, config)
            if hasattr(adapter, "load"):
                train_rows.extend(row.to_dict(include_raw=False) for row in adapter.load(split="train", max_examples=None))
        except (FileNotFoundError, ValueError, KeyError):
            continue
    return audit_examples(
        [row.to_dict(include_raw=False) for row in test_examples],
        repo_root=".",
        thresholds=config.get("leakage"),
        train_examples=train_rows,
        test_examples=[row.to_dict(include_raw=False) for row in test_examples],
    )


def _run(args: argparse.Namespace, track: str) -> int:
    config = _config(args)
    names = _dataset_names(args, _default_datasets_for_track(track))
    examples = _load_examples(config, names, "test", args.max_examples)
    if not examples: raise SystemExit("no examples available; run prepare-data or configure a local dataset")
    audit = _audit_with_split_policy(config, names, examples)
    if audit["definite_count"] and not args.allow_contaminated:
        raise SystemExit("official run blocked by definite leakage; use --allow-contaminated only for a visibly diagnostic run")
    adapter = LivePipelineAdapter(rebuild_indices=bool(getattr(args, "rebuild_indices", False))).build()
    try:
        if track == "generation":
            rows, details = run_generation(examples, adapter, config={**config, "generation": {**config.get("generation", {}), "allow_api": args.allow_api}})
            tracks = {"generation": details}
            retrieval_rows = []
        else:
            rows, details = run_retrieval(examples, adapter, config=config, with_id=track == "with_id")
            tracks = {track: details}
            retrieval_rows = rows
    finally:
        adapter.close()
    out = Path(args.output) if args.output else Path(config["paths"]["results_root"]) / track
    summary = build_summary(tracks, official=not audit["definite_count"] and not args.allow_contaminated, leakage_status=audit["status"], health=details.get("component_health", {}))
    write_run_report(out, summary, retrieval_rows, generation_rows=rows if track == "generation" else None)
    write_leakage_report(audit, out)
    write_manifest(out / "manifest.json", capture_manifest(config=config, args=vars(args), health=details.get("component_health", {}), leakage_status=audit["status"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


def cmd_quick(args: argparse.Namespace) -> int:
    config = _config(args)
    quick_root = Path(config["paths"].get("quick_root", Path(config["paths"]["evaluation_root"]) / "quick"))
    if not (quick_root / "manifest.json").exists():
        prepare_quick(config)
    track = "with_id" if args.track == "with_id" else "without_id_gita_qa"
    names = ["quick_with_id"] if track == "with_id" else ["quick_bhagavad_gita_qa"]
    examples = _load_examples(config, names, "test", args.max_examples)
    if not examples:
        raise SystemExit("quick dataset is empty; run prepare-quick or check the configured source")
    audit = audit_examples([row.to_dict(include_raw=False) for row in examples], repo_root=".", thresholds=config.get("leakage"))
    if audit["definite_count"] and not args.allow_contaminated:
        raise SystemExit("quick diagnostic blocked by definite leakage; use --allow-contaminated only for diagnosis")
    adapter = LivePipelineAdapter(rebuild_indices=bool(getattr(args, "rebuild_indices", False))).build()
    try:
        rows, details = run_retrieval(examples, adapter, config=config, with_id=track == "with_id")
    finally:
        adapter.close()
    out = Path(args.output) if args.output else Path(config["paths"]["results_root"]) / "quick" / track
    # Quick sets are deliberately a balanced diagnostic, not the held-out
    # release score. They still expose the same track metrics and overall
    # percentage for fast model-to-model checks.
    summary = build_summary({track: details}, official=False, leakage_status=audit["status"], health=details.get("component_health", {}))
    summary["quick"] = {"status": "balanced_diagnostic", "official_release_score": False}
    write_run_report(out, summary, rows)
    write_leakage_report(audit, out)
    write_manifest(out / "manifest.json", capture_manifest(config=config, args=vars(args), health=details.get("component_health", {}), leakage_status=audit["status"]))
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


def cmd_all(args: argparse.Namespace) -> int:
    config = _config(args); output = Path(args.output_dir or Path(config["paths"]["results_root"]) / "run")
    all_rows = []; tracks = {}; health = {}; audits = []
    for track, names in (("with_id", ["with_id"]), ("without_id_gita_qa", ["bhagavad_gita_qa"]), ("cross_lingual_gita", ["gitadb"]), ("external_generalization", ["anveshana"])):
        try:
            examples = _load_examples(config, names, "test", args.max_examples)
            if not examples: continue
            audit = _audit_with_split_policy(config, names, examples)
            audits.append(audit)
            if audit["definite_count"] and not args.allow_contaminated: raise RuntimeError(f"{track}: leakage gate failed")
            adapter = LivePipelineAdapter(rebuild_indices=bool(getattr(args, "rebuild_indices", False))).build()
            try: rows, details = run_retrieval(examples, adapter, config=config, with_id=track == "with_id")
            finally: adapter.close()
            tracks[track] = details; all_rows.extend(rows); health[track] = details.get("component_health", {})
        except (FileNotFoundError, RuntimeError) as exc:
            if not args.skip_missing_datasets: raise
            tracks[track] = {"status": "blocked", "error": str(exc)}
    if audits and any(audit["definite_count"] for audit in audits):
        leakage_status = "contaminated"
    elif audits and all(audit["status"] == "clean" for audit in audits):
        leakage_status = "clean"
    else:
        leakage_status = "not_run"
    summary = build_summary(tracks, official=leakage_status == "clean" and not args.allow_contaminated and not args.skip_missing_datasets, leakage_status=leakage_status, health=health)
    write_run_report(output, summary, all_rows)
    combined_audit = {"schema_version": "leakage_v2.1", "status": leakage_status, "official": leakage_status == "clean", "sources_scanned": sum(a.get("sources_scanned", 0) for a in audits), "findings": [finding for audit in audits for finding in audit.get("findings", [])]}
    combined_audit["finding_count"] = len(combined_audit["findings"]); combined_audit["definite_count"] = sum(1 for finding in combined_audit["findings"] if finding.get("severity") == "definite")
    write_leakage_report(combined_audit, output)
    write_manifest(output / "manifest.json", capture_manifest(config=config, args=vars(args), health=health, leakage_status=leakage_status))
    print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0


def cmd_compare(args: argparse.Namespace) -> int:
    print(json.dumps(compare_files(args.a, args.b, seed=args.seed, repetitions=args.repetitions), ensure_ascii=False, indent=2)); return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.evaluation_v2")
    parser.add_argument("--config", default="configs/evaluation_v2.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-data"); prep.add_argument("--config", default=argparse.SUPPRESS); prep.add_argument("--dataset", choices=official_dataset_names()); prep.add_argument("--all", action="store_true"); prep.add_argument("--skip-missing-datasets", action="store_true"); prep.set_defaults(func=cmd_prepare)
    quick_prep = sub.add_parser("prepare-quick"); quick_prep.add_argument("--config", default=argparse.SUPPRESS); quick_prep.set_defaults(func=cmd_prepare_quick)
    q_audit = sub.add_parser("audit-questions"); q_audit.add_argument("path", default="data/evaluation_v2/bhagavad_gita_qa/english_source.jsonl", nargs="?"); q_audit.add_argument("--output"); q_audit.set_defaults(func=cmd_audit_questions)
    leak = sub.add_parser("leakage-audit"); leak.add_argument("--config", default=argparse.SUPPRESS); leak.add_argument("--datasets", nargs="*"); leak.add_argument("--allow-contaminated", action="store_true"); leak.set_defaults(func=cmd_leakage)
    for name, track in (("with-id", "with_id"), ("without-id", "without_id_gita_qa"), ("external", "external_generalization"), ("generation", "generation")):
        cmd = sub.add_parser(name); cmd.add_argument("--config", default=argparse.SUPPRESS); cmd.add_argument("--datasets", nargs="*"); cmd.add_argument("--max-examples", type=int); cmd.add_argument("--output"); cmd.add_argument("--allow-contaminated", action="store_true"); cmd.add_argument("--allow-api", action="store_true"); cmd.add_argument("--rebuild-indices", action="store_true", help="Rebuild FAISS/BM25 during adapter build (default: load existing only)"); cmd.set_defaults(func=lambda args, track=track: _run(args, track))
    all_cmd = sub.add_parser("all"); all_cmd.add_argument("--config", default=argparse.SUPPRESS); all_cmd.add_argument("--output-dir"); all_cmd.add_argument("--max-examples", type=int); all_cmd.add_argument("--allow-contaminated", action="store_true"); all_cmd.add_argument("--skip-missing-datasets", action="store_true"); all_cmd.add_argument("--rebuild-indices", action="store_true"); all_cmd.set_defaults(func=cmd_all)
    quick_cmd = sub.add_parser("quick"); quick_cmd.add_argument("--config", default=argparse.SUPPRESS); quick_cmd.add_argument("--track", choices=("with_id", "without_id_gita_qa"), default="without_id_gita_qa"); quick_cmd.add_argument("--max-examples", type=int); quick_cmd.add_argument("--output"); quick_cmd.add_argument("--allow-contaminated", action="store_true"); quick_cmd.add_argument("--rebuild-indices", action="store_true"); quick_cmd.set_defaults(func=cmd_quick)
    comp = sub.add_parser("compare"); comp.add_argument("--config", default=argparse.SUPPRESS); comp.add_argument("a"); comp.add_argument("b"); comp.add_argument("--seed", type=int, default=20260803); comp.add_argument("--repetitions", type=int, default=2000); comp.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
