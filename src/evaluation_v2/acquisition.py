"""Reproducible acquisition of public V2 benchmark sources.

Raw downloads are kept untouched under ``data/evaluation_v2/raw``. The
acquisition metadata records URL, checksum, retrieval time and the upstream
license claim; it never upgrades an unspecified license to a permissive one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


PUBLIC_SOURCES = {
    "bhagavad_gita_qa": {
        "license": "MIT (upstream dataset card)",
        "source_url": "https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita-QA",
        "files": {
            "english.csv": "https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita-QA/resolve/main/English/english.csv?download=true",
            "hindi.csv": "https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita-QA/resolve/main/Hindi/hindi.csv?download=true",
            "gujarati.csv": "https://huggingface.co/datasets/JDhruv14/Bhagavad-Gita-QA/resolve/main/Gujarati/gujarati.csv?download=true",
        },
    },
    "anveshana": {
        "license": "unknown (upstream dataset card does not declare one)",
        "source_url": "https://huggingface.co/datasets/manojbalaji1/anveshana",
        "files": {
            "train_data.csv": "https://huggingface.co/datasets/manojbalaji1/anveshana/resolve/main/train_data.csv?download=true",
            "val_data.csv": "https://huggingface.co/datasets/manojbalaji1/anveshana/resolve/main/val_data.csv?download=true",
            "test_data.csv": "https://huggingface.co/datasets/manojbalaji1/anveshana/resolve/main/test_data.csv?download=true",
        },
    },
}


# Deterministic coverage correction for the English active track. This is a
# reviewed, hand-written question/answer pair; it is not LLM-generated and is
# kept in the acquisition code so regeneration remains reproducible.
ENGLISH_COVERAGE_OVERRIDES = [
    {
        "id": "manual-en-BhG-13.35",
        "chapter_no": 13,
        "verse_no": 35,
        "language": "english",
        "question": "How does understanding the difference between the body and the knower of the body lead to liberation?",
        "answer": "Bhagavad Gita 13.35 teaches that seeing the distinction between the field, the body and material nature, and the knower of the field, together with understanding release from material nature, leads to the supreme goal.",
        "source": "manual_coverage_override",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".download")
    request = Request(url, headers={"User-Agent": "SansRAG-Evaluation-V2/1.0"})
    with urlopen(request, timeout=120) as response, temp.open("wb") as handle:
        while True:
            block = response.read(1024 * 1024)
            if not block: break
            handle.write(block)
    os.replace(temp, path)
    return sha256(path)


def prepare_public_dataset(name: str, output_root: str | Path = "data/evaluation_v2") -> dict:
    if name not in PUBLIC_SOURCES:
        raise KeyError(f"no verified public acquisition spec for {name}")
    spec = PUBLIC_SOURCES[name]
    root = Path(output_root)
    raw_dir = root / "raw" / name
    normalized_dir = root / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for filename, url in spec["files"].items():
        target = raw_dir / filename
        checksums[filename] = download(url, target) if not target.exists() else sha256(target)
    if name == "bhagavad_gita_qa":
        output = normalized_dir / "source.jsonl"
        english_output = normalized_dir / "english_source.jsonl"
        clean_output = normalized_dir / "clean_source.jsonl"
        temp = output.with_suffix(".tmp")
        english_temp = english_output.with_suffix(".tmp")
        clean_temp = clean_output.with_suffix(".tmp")
        normalized_dir.mkdir(parents=True, exist_ok=True)
        with temp.open("w", encoding="utf-8") as out, english_temp.open("w", encoding="utf-8") as english_out, clean_temp.open("w", encoding="utf-8") as clean_out:
            for filename in ("english.csv", "hindi.csv", "gujarati.csv"):
                language = filename.split(".")[0]
                with (raw_dir / filename).open(encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        row["language"] = language
                        serialized = json.dumps(row, ensure_ascii=False) + "\n"
                        out.write(serialized)
                        if language == "english":
                            english_out.write(serialized)
                        else:
                            clean_out.write(serialized)
            for row in ENGLISH_COVERAGE_OVERRIDES:
                english_out.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temp, output)
        os.replace(english_temp, english_output)
        os.replace(clean_temp, clean_output)
        normalized_path = english_output
    else:
        # Keep Anveshana's official test split as the evaluation input; all
        # raw splits remain available for provenance and future preparation.
        normalized_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = normalized_dir / "test_data.csv"
        if normalized_path.exists(): normalized_path.unlink()
        import shutil
        shutil.copyfile(raw_dir / "test_data.csv", normalized_path)
    metadata = {
        "dataset": name, "source_url": spec["source_url"], "license": spec["license"],
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "raw_dir": str(raw_dir),
        "raw_sha256": checksums, "normalized_path": str(normalized_path),
        "all_languages_normalized_path": str(output) if name == "bhagavad_gita_qa" else None,
        "english_normalized_path": str(normalized_dir / "english_source.jsonl") if name == "bhagavad_gita_qa" else None,
        "clean_normalized_path": str(normalized_dir / "clean_source.jsonl") if name == "bhagavad_gita_qa" else None,
        "english_leakage_status": {"status": "known_contaminated", "reason": "exact contamination with legacy hf_gita_qa material embedded in the production index"} if name == "bhagavad_gita_qa" else None,
        "official_license_clearance": spec["license"].lower().startswith("mit"),
    }
    (normalized_dir / "acquisition_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
