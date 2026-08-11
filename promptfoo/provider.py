"""Promptfoo provider for any universal Bhagavad Gita RAG adapter."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation_v2.universal_adapter import build_universal_adapter, retrieve  # noqa: E402


_ADAPTER = None


def _adapter():
    global _ADAPTER
    if _ADAPTER is not None:
        return _ADAPTER
    config_path = os.environ.get("GITA_BENCHMARK_ADAPTER_CONFIG")
    if not config_path:
        raise RuntimeError("set GITA_BENCHMARK_ADAPTER_CONFIG to an adapter YAML/JSON file")
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = payload.get("adapter", payload.get("system", {}).get("adapter", payload))
    _ADAPTER = build_universal_adapter(config)
    return _ADAPTER


def call_api(prompt, options, context):
    top_k = int(context.get("vars", {}).get("top_k", 10))
    variables = context.get("vars", {})
    response = retrieve(
        _adapter(), str(prompt), top_k,
        context={"query_id": variables.get("query_id", "")},
    )
    retrieved_refs = [row.passage_id for row in response.results]
    output = response.answer or json.dumps({"retrieved_refs": retrieved_refs}, ensure_ascii=False)
    return {
        "output": output,
        "latencyMs": round(response.latency_seconds * 1000),
        "metadata": {"retrieved_refs": retrieved_refs, "citations": response.citations},
    }
