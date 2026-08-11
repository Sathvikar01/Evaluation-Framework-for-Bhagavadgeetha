import json

from src.evaluation_v2.pipeline_adapter import LivePipelineAdapter
from src.evaluation_v2.runner import run_retrieval
from src.evaluation_v2.schemas import BenchmarkExample


class MockPipeline:
    def __init__(self):
        self.calls = []
        self.vector_store = type("V", (), {"index": object()})()
        self.reranker = object()

    def query(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return {"reranked_results": [{"chunk_type": "verse", "verse_ref": "BhG 2.47", "chunk_id": "x"}], "intermediate": {"vector_results": [{"verse_ref": "BhG 2.47"}], "graph_results": [], "bm25_results": [], "interpretation_results": [], "fused_results": [{"verse_ref": "BhG 2.47"}], "reranked_results": [{"verse_ref": "BhG 2.47"}]}}


def test_runner_calls_live_pipeline_retrieval_only():
    mock = MockPipeline()
    adapter = LivePipelineAdapter(mock)
    example = BenchmarkExample(example_id="x", dataset_name="d", dataset_version="1", split="test", track="without_id_gita_qa", query="q", query_language="en", query_type="qa", gold_verse_refs=("BhG 2.47",))
    rows, details = run_retrieval([example], adapter, config={"retrieval": {"cutoffs": [1]}})
    assert mock.calls[0][1]["retrieval_only"] is True
    assert mock.calls[0][1]["answer"] == ""
    assert details["component_health"]["components"]["llm_calls_disabled"] == 1.0
    assert rows[0]["metrics"]["recall@1"] == 1.0
    assert details["stage_analysis"]["failure_counts"]["success"] == 1
