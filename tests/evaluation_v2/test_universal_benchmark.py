import json

import pytest

from src.evaluation_v2.benchmark import GoldExample, annotation_agreement, evaluate, load_benchmark
from src.evaluation_v2.canonical import canonical_verse_ref, expand_passage_ref
from src.evaluation_v2.metrics.retrieval import retrieval_metrics
from src.evaluation_v2.universal_adapter import normalize_response
from src.evaluation_v2.ablation import compare_ablation_runs


def test_canonical_aliases_ranges_and_inventory_validation():
    assert canonical_verse_ref("BG 2:47") == "BhG 2.47"
    assert canonical_verse_ref("Bhagavad Gita chapter 18 verse 66") == "BhG 18.66"
    assert expand_passage_ref("BhG 2.47-49") == ["BhG 2.47", "BhG 2.48", "BhG 2.49"]
    with pytest.raises(ValueError):
        canonical_verse_ref("BhG 2.999")
    with pytest.raises(ValueError):
        expand_passage_ref("BhG 3.1-2.72")


def test_universal_response_requires_and_deduplicates_canonical_ids():
    response = normalize_response({"results": [
        {"passage_id": "BG 2:47", "score": 1.0, "text": "first"},
        {"verse_ref": "BhG 2.47", "score": .9},
        {"metadata": {"canonical_id": "Bhagavad Gita 18.66"}},
    ]})
    assert [row.passage_id for row in response.results] == ["BhG 2.47", "BhG 18.66"]
    with pytest.raises(ValueError):
        normalize_response([{"text": "no identity"}])


def test_complete_ir_metrics_include_map_success_and_hard_negatives():
    result = retrieval_metrics(
        ["BhG 2.47", "BhG 3.35", "BhG 18.66"],
        ["BhG 2.47", "BhG 18.66"],
        graded_relevance={"BhG 2.47": 3, "BhG 18.66": 2},
        hard_negative_refs=["BhG 3.35"], cutoffs=(1, 3),
    )
    assert result["hit_rate@1"] == 1.0
    assert result["success@3"] == 1.0
    assert result["average_precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert result["ndcg@3"] < 1.0
    assert result["hard_negative_accuracy"] == 1.0


def _row(query_id, query, relevance, **extra):
    return {
        "query_id": query_id, "query": query, "split": "test",
        "query_category": extra.pop("query_category", "conceptual"),
        "difficulty": extra.pop("difficulty", "medium"),
        "relevance": relevance,
        "annotation_confidence": .9,
        "provenance": {"human_verified": True, "annotators": ["a", "b"],
                       "test_labels_locked": True, "contamination_audit": "clean", "license": "CC0-1.0"},
        **extra,
    }


def test_model_agnostic_evaluate_writes_full_report(tmp_path):
    records = [
        _row("base", "What is disciplined action?", {"BhG 2.47": 3},
             hard_negatives=["BhG 3.35"], corpus_representation="english_translation",
             representation_group="action"),
        _row("variant", "disciplned acton", {"BhG 2.47": 3}, variant_of="base",
             hard_negatives=["BhG 3.35"], perturbation_type="spelling_mistake", corpus_representation="iast",
             representation_group="action"),
    ]
    path = tmp_path / "benchmark.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")

    class Adapter:
        def retrieve(self, query, k):
            if query.startswith("What"):
                return {"results": [{"passage_id": "BG 2:47"}, {"passage_id": "BG 3:35"}]}
            return {"results": [{"passage_id": "BG 3:35"}]}

    output = tmp_path / "run"
    result = evaluate(
        Adapter(), path, system_name="mock", top_k=3, cutoffs=(1, 3),
        bootstrap_repetitions=30, output_dir=output,
    )
    assert result["leaderboard"]["map"] == .5
    assert result["leaderboard"]["hard_negative_accuracy"] == .5
    assert result["robustness"]["robustness_score"] == 0.0
    assert result["cross_script"]["cross_script_score"] == 0.0
    assert result["official"] is True
    assert (output / "summary.json").exists()
    assert (output / "raw_retrieval_results.jsonl").exists()
    assert len((output / "per_query.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_schema_rejects_unknown_taxonomy_and_supports_annotations(tmp_path):
    record = _row(
        "a", "Explain duty", {"BhG 2.47": 3},
        annotations=[
            {"annotator_id": "one", "relevance": {"BhG 2.47": 3, "BhG 3.35": 0}},
            {"annotator_id": "two", "relevance": {"BhG 2.47": 3, "BhG 3.35": 0}},
        ],
    )
    example = GoldExample.from_dict(record)
    assert annotation_agreement([example])["weighted_cohen_kappa_mean"] == 1.0
    record["query_category"] = "made_up"
    with pytest.raises(ValueError):
        GoldExample.from_dict(record)


def test_load_benchmark_detects_duplicate_ids(tmp_path):
    row = _row("duplicate", "A query", {"BhG 1.1": 1})
    path = tmp_path / "dup.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "query": "Another"}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate query_id"):
        load_benchmark(path)


def test_gold_labels_are_never_exposed_to_context_aware_adapter(tmp_path):
    path = tmp_path / "benchmark.jsonl"
    path.write_text(json.dumps(_row("heldout", "A held-out query", {"BhG 2.47": 3})), encoding="utf-8")

    class ContextAdapter:
        observed = None

        def retrieve(self, query, k, context=None):
            self.observed = context
            return ["BhG 2.47"]

    adapter = ContextAdapter()
    evaluate(adapter, path, top_k=1, cutoffs=(1,), bootstrap_repetitions=5)
    assert adapter.observed == {
        "query_id": "heldout", "corpus_representation": "unknown", "representation_group": ""
    }


def test_ablation_aligns_queries_and_corrects_multiple_tests(tmp_path):
    def make_run(name, hit):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "summary.json").write_text(json.dumps({
            "system_name": name,
            "leaderboard": {"recall@1": float(hit), "mrr": float(hit)},
        }), encoding="utf-8")
        (directory / "per_query.jsonl").write_text(json.dumps({
            "example_id": "q", "metrics": {"top1_hit": hit, "mrr": float(hit), "ndcg@10": float(hit)},
            "retrieved_refs": ["BhG 2.47"] if hit else ["BhG 3.35"],
        }), encoding="utf-8")
        return directory

    baseline = make_run("baseline", False)
    variant = make_run("variant", True)
    result = compare_ablation_runs(baseline, {"dense+bm25": variant}, repetitions=10)
    assert result["paired_comparisons"]["dense+bm25"]["aligned_n"] == 1
    assert "holm_adjusted_p_value" in result["paired_comparisons"]["dense+bm25"]["mcnemar_r1"]
