"""Thin adapter around the live SansRAG production pipeline."""

from __future__ import annotations

import time
from typing import Any, Callable

from .schemas import PipelineObservation


class LivePipelineAdapter:
    """Use ``SRAGGraphPipeline.query`` without recreating retrieval logic."""

    def __init__(
        self,
        pipeline: Any | None = None,
        *,
        config: Any | None = None,
        factory: Callable[[], Any] | None = None,
        rebuild_indices: bool = False,
        disable_llm_side_effects: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.config = config
        self.factory = factory
        self.rebuild_indices = rebuild_indices
        self.disable_llm_side_effects = disable_llm_side_effects
        self._owned = False
        self._llm_disabled = False

    def build(self) -> "LivePipelineAdapter":
        if self.pipeline is not None:
            if self.disable_llm_side_effects:
                self._disable_llm_side_effects()
            return self
        if self.factory:
            self.pipeline = self.factory()
        else:
            from src.langchain_components.graph import SRAGGraphPipeline
            from src.utils.config import Config
            self.config = self.config or Config()
            self.pipeline = SRAGGraphPipeline(self.config)
            # Evaluation must not rebuild production indices unless explicitly requested.
            if self.rebuild_indices:
                if hasattr(self.pipeline, "preprocess"):
                    self.pipeline.preprocess()
                if hasattr(self.pipeline, "build_indices"):
                    self.pipeline.build_indices()
        self._owned = True
        if self.disable_llm_side_effects:
            self._disable_llm_side_effects()
        return self

    def _disable_llm_side_effects(self) -> None:
        """Best-effort disable of retrieval-time LLM expansion / interpretation."""
        if self.pipeline is None or self._llm_disabled:
            return
        cfg = getattr(self.pipeline, "config", None) or self.config
        raw = None
        if cfg is not None:
            raw = getattr(cfg, "_config", None)
            if raw is None and isinstance(cfg, dict):
                raw = cfg
            elif hasattr(cfg, "get") and not isinstance(cfg, dict):
                try:
                    raw = dict(cfg)  # type: ignore[arg-type]
                except Exception:
                    raw = None
        if isinstance(raw, dict):
            retrieval = raw.setdefault("retrieval", {})
            if isinstance(retrieval, dict):
                qe = retrieval.setdefault("query_expansion", {})
                if isinstance(qe, dict):
                    qe["enabled"] = False
                    llm = qe.setdefault("llm", {})
                    if isinstance(llm, dict):
                        llm["enabled"] = False
            reranking = raw.setdefault("reranking", {})
            if isinstance(reranking, dict):
                reranking["interpretation_score_weight"] = 0.0
                reranking["interpretation_smoothing_lambda"] = 0.0
        # Clear lazy expanders if already constructed.
        for attr in ("_llm_query_expander", "llm_query_expander", "_query_expander", "query_expander"):
            if hasattr(self.pipeline, attr):
                try:
                    setattr(self.pipeline, attr, None)
                except Exception:
                    pass
        self._llm_disabled = True

    def health(self) -> dict[str, Any]:
        if self.pipeline is None:
            return {"pipeline": "not_built", "operational": False}
        checks = {"pipeline": True}
        vector = getattr(self.pipeline, "vector_store", None)
        checks["vector_index"] = bool(vector is not None and getattr(vector, "index", None) is not None)
        checks["reranker"] = getattr(self.pipeline, "reranker", None) is not None or getattr(self.pipeline, "cross_encoder", None) is not None
        checks["graph_connector"] = bool(
            getattr(self.pipeline, "_graph_retriever", None) is not None
            and getattr(self.pipeline, "_graph_connected", False)
        )
        checks["generation_api_configured"] = getattr(getattr(self.pipeline, "generator", None), "api_key", None) is not None
        checks["operational"] = bool(checks["pipeline"] and checks["vector_index"])
        checks["llm_side_effects_disabled"] = bool(self._llm_disabled or self.disable_llm_side_effects)
        return checks

    def query_retrieval(self, query: str, *, answer: str = "", use_api: bool = False) -> PipelineObservation:
        self.build()
        start = time.perf_counter()
        result = self.pipeline.query(query, use_api=use_api, retrieval_only=True, answer=answer)
        elapsed = time.perf_counter() - start
        health = self.health()
        intermediate = result.get("intermediate", {})
        required = ("vector_results", "graph_results", "bm25_results", "fused_results", "reranked_results")
        health["stage_observability"] = {key: key in intermediate for key in required}
        health["retrieval_only"] = True
        health["llm_calls_disabled"] = bool(health.get("llm_side_effects_disabled", False))
        # Exact-reference routing intentionally bypasses semantic stages; that
        # is a declared path, not a degraded fallback.
        health["exact_reference_short_circuit"] = bool(intermediate.get("verse_ref_detected"))
        health["degraded"] = not health["exact_reference_short_circuit"] and not all(health["stage_observability"].values())
        return PipelineObservation(query=query, result=result, elapsed_seconds=elapsed, health=health)

    def query_generation(self, query: str, *, answer: str = "", use_api: bool = False) -> PipelineObservation:
        self.build()
        start = time.perf_counter()
        result = self.pipeline.query(query, use_api=use_api, retrieval_only=False, answer=answer)
        elapsed = time.perf_counter() - start
        health = self.health()
        health["generation_returned"] = "answer" in result
        health["degraded"] = not health["generation_returned"]
        return PipelineObservation(query=query, result=result, elapsed_seconds=elapsed, health=health)

    def close(self) -> None:
        if self._owned and self.pipeline is not None and hasattr(self.pipeline, "close"):
            self.pipeline.close()
        self.pipeline = None if self._owned else self.pipeline
