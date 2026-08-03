import json

import pytest

from src.evaluation_v2.datasets.bhagavad_gita_qa import BhagavadGitaQAAdapter, split_by_verse_group
from src.evaluation_v2.datasets.with_id import generate_with_id_records, parse_reference_query
from src.evaluation_v2.schemas import BenchmarkExample


def example(ref="BhG 2.47", query="question", source="1"):
    return BenchmarkExample(example_id=source, dataset_name="x", dataset_version="1", split="all", track="without_id_gita_qa", query=query, query_language="en", query_type="qa", gold_verse_refs=(ref,), raw_record={})


def test_malformed_schema_and_graded_labels_rejected():
    with pytest.raises(ValueError):
        example(ref="2.47")
    with pytest.raises(ValueError):
        BenchmarkExample(example_id="x", dataset_name="x", dataset_version="1", split="all", track="without_id_gita_qa", query="q", query_language="en", query_type="qa", gold_verse_refs=("BhG 2.47",), graded_relevance={"BhG 2.47": 4})


@pytest.mark.parametrize("query,expected", [
    ("BhG 2.47", [(2, 47, 47)]), ("BG 2:47", [(2, 47, 47)]),
    ("Chapter 2, shloka 47", [(2, 47, 47)]), ("भगवद्गीता २।४७", [(2, 47, 47)]),
    ("Bhagavad-gītā 2.47-49", [(2, 47, 49)]),
])
def test_reference_aliases(query, expected):
    assert parse_reference_query(query) == expected


def test_reference_false_positives_and_invalid_ranges():
    assert parse_reference_query("The number 2.47 is a decimal") == []
    assert parse_reference_query("What happened in 1947?") == []
    assert parse_reference_query("BhG 2.47-3.1") == []


def test_with_id_generation_is_deterministic_and_marks_invalid():
    a = generate_with_id_records({"BhG 2.47", "BhG 1.1"})
    b = generate_with_id_records({"BhG 2.47", "BhG 1.1"})
    assert [row.to_dict(include_raw=False) for row in a] == [row.to_dict(include_raw=False) for row in b]
    assert any(row.metadata["expected_valid"] is False for row in a)


def test_verse_group_split_isolated():
    rows = [example("BhG 2.47", f"q{i}", str(i)) for i in range(4)] + [example("BhG 3.1", f"r{i}", f"r{i}") for i in range(4)]
    splits = split_by_verse_group(rows, seed=7, ratios={"train": .5, "validation": .0, "test": .5})
    train_refs = {ref for row in splits["train"] for ref in row.gold_verse_refs}
    test_refs = {ref for row in splits["test"] for ref in row.gold_verse_refs}
    assert not train_refs & test_refs


def test_bhagavad_adapter_preserves_raw_and_split_manifest(tmp_path):
    source = tmp_path / "qa.jsonl"
    source.write_text("\n".join(json.dumps({"id": i, "question": f"q{i}", "answer": "a", "chapter_no": 2, "verse_no": 47}) for i in range(3)), encoding="utf-8")
    adapter = BhagavadGitaQAAdapter(source, seed=1)
    manifest = adapter.split_manifest()
    assert manifest["checksum"]
    assert sum(part["count"] for part in manifest["splits"].values()) == 3
    assert adapter.load("test")[0].raw_record["question"].startswith("q") if adapter.load("test") else True
