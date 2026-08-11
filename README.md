# Bhagavad Gita RAG Benchmark

A model-agnostic, research-oriented benchmark for Bhagavad Gita retrieval and retrieval-augmented generation. It evaluates canonical evidence retrieval independently from answer generation and makes no assumptions about the retriever, embedding model, vector database, graph, reranker, LLM, or implementation language.

The required system contract is deliberately small:

```python
class MySystem:
    def retrieve(self, query: str, k: int):
        return [{"passage_id": "BhG 2.47", "score": 0.91}]
```

Passage identity is canonicalized across `BG 2:47`, `Bhagavad Gita 2.47`, Devanagari, IAST, translations, commentary, and mixed documents. Official scoring uses human qrels and never exposes relevance labels, reference answers, or hard negatives to the evaluated adapter.

## What is implemented

- A universal Python, HTTP, command, and replay adapter API
- A versioned gold-example JSON Schema with graded multi-passage relevance
- Canonical verse/range normalization against the 18-chapter inventory
- Recall, Precision, Hit Rate, Success, MRR, AP/MAP, R-Precision, graded recall, and nDCG at configurable cutoffs
- Query-level bootstrap confidence intervals, standard deviations, paired McNemar tests, paired bootstrap deltas, effect size, and Holm correction for ablations
- Breakdowns by query taxonomy, difficulty, chapter, corpus representation, and retrieval strategy
- Controlled robustness pairs, script-invariance groups, hard-negative accuracy, and a universal failure taxonomy
- Separate deterministic generation/citation checks and optional Promptfoo evaluation
- Dataset quality, duplicate, distribution, cross-split, lexical-overlap, annotation-confidence, and inter-annotator-agreement audits
- Complete per-query, raw retrieval, failure, leaderboard, quality, robustness, error, configuration, and reproducibility artifacts
- A compatibility adapter retaining the existing SansRAG Evaluation V2 workflows

## Quick start: any RAG system

```powershell
python -m src.evaluation_v2 audit-benchmark data/evaluation_v2/universal/starter_development.jsonl

python -m src.evaluation_v2 universal `
  --dataset data/evaluation_v2/universal/starter_development.jsonl `
  --adapter examples/universal_adapter.py:adapter `
  --system-name my-rag `
  --split development `
  --output results/universal/my-rag
```

For an external service, copy [examples/http_adapter.yaml](examples/http_adapter.yaml) and pass `--adapter-config`. For an in-process system, copy [examples/universal_adapter.py](examples/universal_adapter.py). Programmatic use is exactly:

```python
from src.evaluation_v2 import evaluate

report = evaluate(system, "benchmark.jsonl", system_name="my-system", output_dir="results/my-system")
```

The included starter set is diagnostic scaffolding, not an official leaderboard test set: its labels explicitly have `human_verified: false`. An official release requires independent domain-expert annotation, adjudication, licensing, a locked held-out test set, and contamination review. The existing 3,501-question English SansRAG asset is also diagnostic because it collides with QA text in the production index.

## Promptfoo

Promptfoo is used only for supplementary generation and citation evaluation; it is not allowed to replace human qrels.

```powershell
$env:GITA_BENCHMARK_ADAPTER_CONFIG = "examples/http_adapter.yaml"
$env:GITA_BENCHMARK_DATASET = "path/to/gold.jsonl"
promptfoo eval -c promptfoo/promptfooconfig.yaml
```

The offline integration check is:

```powershell
promptfoo eval -c promptfoo/smoke.yaml --no-cache
```

## Controlled comparisons

Run every arm on the identical benchmark version, split, query IDs, `top_k`, and seed. Then:

```powershell
python -m src.evaluation_v2 ablation `
  --baseline results/bm25 `
  --run dense=results/dense `
  --run hybrid=results/hybrid `
  --output results/ablation
```

See [benchmark methodology](docs/benchmark_methodology.md), [annotation protocol](docs/annotation_protocol.md), [requirements compliance](docs/compliance_matrix.md), and [SansRAG compatibility notes](docs/evaluation_v2.md).

## Verification

```powershell
python -m pytest tests/evaluation_v2 -q
python -m src.evaluation_v2 audit-benchmark data/evaluation_v2/universal/starter_development.jsonl
```
