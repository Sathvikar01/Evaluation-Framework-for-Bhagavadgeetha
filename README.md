# Evaluation Framework for Bhagavad Geetha

Standalone Evaluation V2 for SansRAG Bhagavad Gita retrieval and generation experiments.

The framework provides:

- English-only Bhagavad Gita QA evaluation
- balanced quick evaluation with one question per canonical verse
- exact-reference routing evaluation
- retrieval metrics: Recall@1/3/5/10/50, MRR, nDCG, rank, and pool containment
- overall performance percentage with explicit denominators
- leakage auditing, stage analysis, latency, and reproducibility manifests
- retrieval-only execution with generation and LLM calls disabled by default

## Quick start

```powershell
python -m src.evaluation_v2 audit-questions data/evaluation_v2/bhagavad_gita_qa/english_source.jsonl
python -m src.evaluation_v2 prepare-quick --config configs/evaluation_v2.yaml
python -m src.evaluation_v2 quick --config configs/evaluation_v2.yaml --track without_id_gita_qa --max-examples 700 --allow-contaminated
python -m pytest tests/evaluation_v2 -q
```

The English benchmark and archived Hindi/Gujarati artifacts are included as versioned evaluation assets. Official scoring remains blocked until production-index leakage and external dataset licensing/mapping gates are resolved.

This repository is an evaluation layer intended to run alongside the SansRAG production pipeline.
