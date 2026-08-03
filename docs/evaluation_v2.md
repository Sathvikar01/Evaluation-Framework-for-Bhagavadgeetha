# SansRAG Evaluation V2

Evaluation V2 is an isolated, versioned benchmark layer for SansRAG. The historical evaluator under `src/evaluation/` remains unchanged so its baselines and CLI stay comparable. V2 uses the live `SRAGGraphPipeline.query` path and records stage outputs, component health, latency, leakage status, and reproducibility metadata.

## Tracks

- **WithID** derives the canonical verse inventory from `chunks.jsonl` and tests exact routing, aliases, ranges, invalid references, and rejection.
- **WithoutID Gita QA** uses a clean public QA source and splits whole verse groups into train/validation/test. Only test is evaluated.
- **GitaDB** evaluates multilingual translations using an explicit versioned mapping artifact; unmapped and ambiguous rows are reported.
- **Edwin Arnold** is primarily a generation benchmark. Missing verse labels remain missing and are exported for review.
- **Anveshana** is an external, out-of-domain track and is never merged into a Gita aggregate.
- **Balanced quick evaluation** is a fast diagnostic: one deterministic QA question per covered verse and one exact-reference query for every canonical verse. It is not the held-out release score.
- **Legacy compatibility** is diagnostic only. Kaggle is not in the V2 registry.

## Data acquisition and leakage policy

The public Bhagavad-Gita-QA and Anveshana sources are downloaded reproducibly into `data/evaluation_v2/raw/`, checksummed, and recorded in `acquisition_metadata.json`. Dataset adapters reject malformed records and do not invent verse labels. The leakage auditor scans indexed chunks, QA augmentation, caches, theme material, training files, prompt examples, and discovered textual sources. Definite query/answer matches or train/test verse overlap block official runs. `--allow-contaminated` produces a visibly diagnostic, non-official run.

The active QA configuration is now English-only (`english_source.jsonl`). The Hindi and Gujarati raw files, normalized source, splits, and prior quick set are archived at `data/evaluation_v2/archive/non_english_qa_20260803.zip` and are not used by the active commands. The English source is currently diagnostic/non-official because all 3,500 English questions and answers exactly collide with legacy QA material embedded in the production index; a clean English source or a separately rebuilt English-free production index is required for official English scoring.

Anveshana is acquired and normalized, but its upstream card does not declare a license and its passage IDs have not been reviewed against the local 701-verse canonical inventory. It remains an external, unmapped diagnostic track and cannot contribute to the Gita aggregate. GitaDB remains blocked until a licensed export and reviewed `mapping.json` are supplied; the paper describes the 640-verse resource but does not provide a currently reachable data artifact.

## Setup and commands

```powershell
python -m src.evaluation_v2 prepare-data --dataset with_id
python -m src.evaluation_v2 prepare-data --dataset bhagavad_gita_qa
python -m src.evaluation_v2 prepare-data --dataset gitadb
python -m src.evaluation_v2 prepare-data --dataset edwin_arnold
python -m src.evaluation_v2 prepare-data --dataset anveshana
python -m src.evaluation_v2 prepare-quick --config configs/evaluation_v2.yaml
python -m src.evaluation_v2 audit-questions data/evaluation_v2/bhagavad_gita_qa/english_source.jsonl --output data/evaluation_v2/bhagavad_gita_qa/question_audit.json
python -m src.evaluation_v2 leakage-audit --config configs/evaluation_v2.yaml --datasets bhagavad_gita_qa
python -m src.evaluation_v2 with-id --max-examples 20 --output results/evaluation_v2/with_id.json
python -m src.evaluation_v2 without-id --datasets bhagavad_gita_qa --max-examples 50
python -m src.evaluation_v2 quick --track without_id_gita_qa --allow-contaminated --output results/evaluation_v2/quick/qa
python -m src.evaluation_v2 quick --track with_id --output results/evaluation_v2/quick/with_id
python -m src.evaluation_v2 external --datasets anveshana --max-examples 50
python -m src.evaluation_v2 generation --datasets edwin_arnold --max-examples 20 --allow-api
python -m src.evaluation_v2 all --skip-missing-datasets --output-dir results/evaluation_v2/smoke
python -m src.evaluation_v2 compare results/evaluation_v2/a/per_query.jsonl results/evaluation_v2/b/per_query.jsonl
```

Retrieval-only evaluation disables answer generation, LLM query expansion, and LLM interpretation canonicalization; the health record explicitly reports `retrieval_only` and `llm_calls_disabled`. Optional generation and LLM judging are disabled by default. `--allow-api` is an explicit guard for generation calls; do not use it without credentials, a sample limit, and approval for any paid provider.

## Metrics, overall score, and reports

Retrieval reports Recall@1/3/5/10/50, MRR, mean/median rank, nDCG, graded recall, pool containment, exclusions, and denominators. Stage analysis records vector, BM25, graph, fused, reranked, and final outputs and classifies the first failure. Generation reports deterministic citation validity, precision/recall, unsupported references, quote matches, emptiness/refusal, and produces a human-review JSONL/CSV with versioned rubric fields.

Each summary also includes `overall.value_pct`. This is an equal-track macro average of primary scores: WithID exact lookup accuracy, semantic-track Recall@1, and generation deterministic score. The summary also reports `micro_value_pct`, scored tracks, missing tracks, denominators, and whether the aggregate is complete or partial. Missing or blocked tracks are never silently converted to zero. Values in JSON are percentages under `value_pct`; per-query metric values remain normalized fractions in `[0, 1]`.

The quick benchmark is stored under `data/evaluation_v2/quick/` with a manifest. The active English QA source contains 3,501 questions across all 701 canonical verses: the original 3,500 rows cover 700 verses, and a deterministic manual coverage question was added for BhG 13.35. The balanced English quick set contains 701 questions, one per verse; the recorded 700-row run is a diagnostic slice that includes BhG 13.35. The WithID quick set contains 701 questions, one per canonical verse. The full English QA split contains 1,410 held-out test questions, but remains diagnostic until leakage is resolved.

The question audit is stored at `data/evaluation_v2/bhagavad_gita_qa/question_audit.json`. It reports that questions are not all identical in form: there are eight surface stems, one duplicate question across the source, and five unique questions per verse except the newly covered BhG 13.35, which has one. It also flags 747 deterministic `this teaching on ...` template artefacts in the full source. The balanced quick set excludes these where possible and uses 12 documented manual, reference-free rewrites where the source had no clean alternative.

## Retrieval-fix phases

Phase 0 is complete: stage traces now preserve top-50 vector, graph, BM25, fused, and reranked candidates; BM25 candidates are enriched with verse references before scoring; and graph health reflects actual connection state. Phase 1 is complete: retrieval QA augmentation can be stripped for cross-encoder experiments without changing the first-stage index, and the quick benchmark filters deterministic question-template artefacts. The clean-reranker ablation improved deeper recall on its completed sample but did not pass the Recall@1 gate, so `reranking.clean_candidate_text` remains disabled by default. Phase 2 should fine-tune/calibrate the cross-encoder on train-only Gita QA pairs with hard negatives; Phase 3 should then optimize dense/BM25/graph candidate recall using the corrected stage traces. New retrieval methods should not be accepted unless they improve the clean quick benchmark without a Recall@1 regression.

Every run writes `summary.json`, `per_query.jsonl`, `manifest.json`, leakage reports, stage/latency files, failures, Markdown, and generation review files as applicable. Comparison aligns stable IDs and uses McNemar's test plus paired bootstrap confidence intervals; unpaired aggregate deltas are not treated as improvements.

## Extending V2

Add a `DatasetAdapter`, register it in `registry.py`, preserve raw records and source/license metadata, add mapping/split tests, and keep it in its own track. New metrics belong under `metrics/`, must specify denominators and exclusion behavior, and should have unit tests with multiple-gold and empty-result cases. Human review should edit only `human_score_fields`; deterministic and optional LLM scores remain separate.

## Remaining release gates

The active language limitation is resolved in configuration and data preparation: only English questions are selected, while Hindi/Gujarati artifacts are archived. The English QA track remains leakage-blocked for official scoring until a clean English source or rebuilt production index is available. An official aggregate also requires either (a) a licensed GitaDB export plus reviewed verse mapping, or (b) explicitly removing that track from the aggregate protocol. Anveshana remains out-of-domain and license-blocked. Neo4j, local model availability, and generation credentials remain environment-dependent runtime gates.
