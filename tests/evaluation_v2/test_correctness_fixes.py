"""Regression tests for evaluation correctness fixes."""

from src.evaluation_v2.cli import _default_datasets_for_track, build_parser
from src.evaluation_v2.metrics.generation import generation_checks
from src.evaluation_v2.metrics.retrieval import retrieval_metrics, summarize_retrieval
from src.evaluation_v2.metrics.routing import routing_metrics
from src.evaluation_v2.metrics.statistics import compare_paired
from src.evaluation_v2.comparison import load_rows
from src.evaluation_v2.runner import _refs, stage_analysis
from src.evaluation_v2.schemas import BenchmarkExample


def test_invalid_rejection_uses_pipeline_not_parser():
    rows = [{
        "query": "BhG 2.999",
        "expected_valid": False,
        "predicted_refs": [],
        "retrieved_refs": [],
        "exact_reference_short_circuit": False,
    }]
    summary = routing_metrics(rows, {"BhG 2.47"})
    assert summary["invalid_reference_rejection_rate"] == 1.0
    assert summary["false_positive_short_circuit_rate"] == 0.0


def test_false_positive_when_pipeline_returns_refs_on_invalid():
    rows = [{
        "query": "BhG 2.999",
        "expected_valid": False,
        "predicted_refs": ["BhG 2.47"],
        "exact_reference_short_circuit": True,
    }]
    summary = routing_metrics(rows, {"BhG 2.47"})
    assert summary["false_positive_short_circuit_rate"] == 1.0
    assert summary["invalid_reference_rejection_rate"] == 0.0


def test_short_circuit_success_not_failure():
    example = BenchmarkExample(
        example_id="x", dataset_name="with_id_canonical", dataset_version="1", split="test",
        track="with_id", query="BhG 2.47", query_language="en", query_type="exact_reference",
        gold_verse_refs=("BhG 2.47",), metadata={"expected_valid": True},
    )
    result = {
        "intermediate": {
            "verse_ref_detected": True,
            "vector_results": [],
            "graph_results": [],
            "bm25_results": [],
            "fused_results": [],
            "reranked_results": [{"verse_ref": "BhG 2.47", "chunk_type": "verse"}],
        }
    }
    stage = stage_analysis(example, result, ["BhG 2.47"])
    assert stage["failure_class"] == "success"
    assert stage["exact_reference_short_circuit"] is True


def test_refs_skip_non_verse_chunks():
    refs = _refs([
        {"verse_ref": "BhG 1.1", "chunk_type": "commentary"},
        {"verse_ref": "BhG 2.47", "chunk_type": "verse"},
        {"verse_ref": "BhG 3.1"},  # missing type kept
    ])
    assert refs == ["BhG 2.47", "BhG 3.1"]


def test_summarize_excludes_rank_and_bool_noise():
    m1 = retrieval_metrics(["BhG 1.1"], ["BhG 1.1"])
    m2 = retrieval_metrics(["BhG 9.9", "BhG 1.1"], ["BhG 1.1"])
    summary = summarize_retrieval([{"metrics": m1}, {"metrics": m2}])
    assert "excluded" not in summary["metrics"]
    assert "n_gold" not in summary["metrics"]
    assert "first_relevant_rank" in summary["rank_stats"]
    assert summary["metrics"]["recall@1"]["value"] == 0.5


def test_mcnemar_ignores_fractional_multi_gold_recall():
    a = [{"example_id": "1", "metrics": {"recall@1": 0.5, "mrr": 1, "ndcg@10": 1, "top1_hit": False}, "retrieved_refs": ["a"]}]
    b = [{"example_id": "1", "metrics": {"recall@1": 0.0, "mrr": 0, "ndcg@10": 0, "top1_hit": False}, "retrieved_refs": ["b"]}]
    result = compare_paired(a, b, repetitions=10)
    assert result["mcnemar_r1"]["a_wins"] == 0
    assert result["mcnemar_r1"]["ties"] == 1


def test_cli_track_defaults():
    assert _default_datasets_for_track("generation") == ["edwin_arnold"]
    assert _default_datasets_for_track("external_generalization") == ["anveshana"]
    assert _default_datasets_for_track("with_id") == ["with_id"]
    parser = build_parser()
    args = parser.parse_args(["generation", "--max-examples", "1"])
    assert args.func  # wired


def test_generation_accepts_bg_alias_and_penalizes_unsupported():
    checks = generation_checks(
        "See BG 2.47 and Bhagavad Gita 3.1.",
        retrieved_refs={"BhG 2.47"},
        gold_refs={"BhG 2.47"},
    )
    assert "BhG 2.47" in checks["citations"]
    assert "BhG 3.1" in checks["unsupported_references"]
    assert checks["unsupported_penalty"] == 0.5
    assert 0.0 < checks["deterministic_score"] < 1.0


def test_precision_at_k_uses_cutoff_k_denominator():
    metrics = retrieval_metrics(["BhG 1.1"], ["BhG 1.1"], cutoffs=(5, 10))
    assert metrics["precision@5"] == 0.2
    assert metrics["precision@10"] == 0.1


def test_generation_refusal_zeroes_score():
    checks = generation_checks(
        "I don't have relevant verses to answer this question.",
        retrieved_refs={"BhG 2.47"},
        gold_refs={"BhG 2.47"},
        expect_citation=True,
    )
    assert checks["empty_or_refusal"] is True
    assert checks["deterministic_score"] == 0.0


def test_generation_score_uses_recall_once_and_penalty_is_badness_rate():
    checks = generation_checks(
        "See BhG 2.47 and BhG 3.1.",
        retrieved_refs={"BhG 2.47"},
        gold_refs={"BhG 2.47"},
    )
    assert checks["unsupported_penalty"] == 0.5
    assert checks["citation_recall"] == 1.0
    assert checks["deterministic_score"] == (0.5 + 1.0 + 1.0) / 3.0


def test_late_final_injection_is_success_and_pool_is_traced():
    example = BenchmarkExample(
        example_id="late", dataset_name="d", dataset_version="1", split="test",
        track="without_id_gita_qa", query="q", query_language="en", query_type="qa",
        gold_verse_refs=("BhG 2.47",),
    )
    result = {"intermediate": {
        "vector_results": [], "graph_results": [], "bm25_results": [],
        "interpretation_results": [{"verse_ref": "BhG 2.47", "chunk_type": "verse"}],
        "fused_results": [], "reranked_results": [],
    }}
    stage = stage_analysis(example, result, ["BhG 2.47"])
    assert stage["failure_class"] == "success"
    assert stage["containment"]["interpretation_results"] is True
    assert stage["union_pool_containment"] is True


def test_pool_containment_is_not_final_rank_containment():
    metrics = retrieval_metrics(
        ["BhG 1.1"], ["BhG 2.47"], candidate_pool_refs=["BhG 2.47"], cutoffs=(1,)
    )
    assert metrics["candidate_pool_containment"] is True
    assert metrics["final_rank_containment"] is False
    assert metrics["recall@1"] == 0.0


def test_even_rank_median_is_the_statistical_median():
    metrics = retrieval_metrics(
        ["BhG 2.47", "BhG 1.1", "BhG 18.66"],
        ["BhG 2.47", "BhG 18.66"],
        cutoffs=(1,),
    )
    assert metrics["median_rank"] == 2.0


def test_compare_rejects_summary_json(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text('{"tracks": {}}', encoding="utf-8")
    try:
        load_rows(path)
    except ValueError as exc:
        assert "per_query" in str(exc)
    else:
        raise AssertionError("summary.json must not be treated as per-query rows")


def test_routing_metrics_returns_none_for_zero_valid():
    summary = routing_metrics([], {"BhG 2.47"})
    assert summary["incorrect_chapter_verse_routing_rate"] is None

