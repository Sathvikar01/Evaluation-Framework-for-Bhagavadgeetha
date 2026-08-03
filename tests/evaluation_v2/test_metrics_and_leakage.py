from src.evaluation_v2.leakage import audit_examples
from src.evaluation_v2.metrics.generation import generation_checks
from src.evaluation_v2.metrics.retrieval import retrieval_metrics, summarize_retrieval
from src.evaluation_v2.metrics.statistics import compare_paired, paired_bootstrap_ci
from src.evaluation_v2.overall import aggregate_overall


def test_retrieval_metrics_multiple_gold_graded_and_duplicates():
    result = retrieval_metrics(["BhG 2.47", "BhG 2.47", "BhG 18.66"], ["BhG 2.47", "BhG 18.66"], graded_relevance={"BhG 2.47": 3, "BhG 18.66": 1})
    assert result["recall@1"] == .5
    assert result["mrr"] == 1.0
    assert result["ndcg@5"] > 0


def test_empty_and_macro_summary():
    empty = retrieval_metrics([], [], cutoffs=(1,))
    assert empty["excluded"] is True
    summary = summarize_retrieval([{"dataset_name": "a", "metrics": {"recall@1": 1.0}}, {"dataset_name": "b", "metrics": {"recall@1": 0.0}}])
    assert summary["metrics"]["recall@1"]["denominator"] == 2


def test_generation_checks_unsupported_and_duplicate_citations():
    checks = generation_checks("See BhG 2.47 and BhG 2.47. Also BhG 3.1.", retrieved_refs={"BhG 2.47"}, gold_refs={"BhG 2.47"})
    assert checks["unsupported_references"] == ["BhG 3.1"]
    assert checks["duplicate_citation_count"] == 1
    assert checks["citation_precision"] == .5


def test_leakage_exact_and_clean(tmp_path):
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    (data_dir / "chunks.jsonl").write_text('{"text_english":"A leaked unique question about liberation."}\n', encoding="utf-8")
    contaminated = {"example_id": "x", "dataset_name": "d", "query": "A leaked unique question about liberation.", "reference_answer": "", "gold_verse_refs": ["BhG 2.47"]}
    report = audit_examples([contaminated], repo_root=tmp_path)
    assert report["status"] == "contaminated"
    clean = audit_examples([{**contaminated, "query": "A never indexed query about samadhi 991."}], repo_root=tmp_path)
    assert clean["status"] == "clean"


def test_same_verse_split_leakage(tmp_path):
    report = audit_examples([], repo_root=tmp_path, train_examples=[{"gold_verse_refs": ["BhG 2.47"]}], test_examples=[{"gold_verse_refs": ["BhG 2.47"]}])
    assert report["definite_count"] == 1


def test_paired_statistics_and_alignment():
    ci = paired_bootstrap_ci([0.0, 1.0], [1.0, 1.0], seed=2, repetitions=20)
    assert ci["delta"] == .5
    result = compare_paired(
        [{"example_id": "1", "metrics": {"recall@1": 1, "mrr": 1, "ndcg@10": 1}, "retrieved_refs": ["a"]}],
        [{"example_id": "1", "metrics": {"recall@1": 0, "mrr": 0, "ndcg@10": 0}, "retrieved_refs": ["b"]}], repetitions=20,
    )
    assert result["aligned_n"] == 1
    assert result["mcnemar_r1"]["a_wins"] == 1


def test_overall_is_equal_track_macro_percentage_and_exposes_missing_tracks():
    result = aggregate_overall({
        "with_id": {"summary": {"exact_verse_lookup_accuracy": 1.0, "valid_count": 10}},
        "without_id_gita_qa": {"summary": {"n_scored": 10, "metrics": {"recall@1": {"value": 0.5, "denominator": 10}}}},
        "cross_lingual_gita": {"status": "blocked"},
    })
    assert result["value_pct"] == 75.0
    assert result["aggregation"] == "macro_mean_of_equal_track_primary_scores"
    assert result["status"] == "partial"
    assert "cross_lingual_gita" in result["missing_or_unscored_tracks"]
