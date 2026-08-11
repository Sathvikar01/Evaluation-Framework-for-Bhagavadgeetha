# Scientific methodology

## Evaluation object and independence

The benchmark is a test collection: English information needs, canonical Bhagavad Gita passage identities, and relevance judgments. A system receives only `query`, `k`, and optional non-label routing metadata. It never receives qrels, hard negatives, reference answers, annotator identities, or confidence labels. This enforces a strict boundary between the evaluator and the system under test.

The unit of retrieval is a canonical verse (`BhG chapter.verse`). A document in Devanagari, IAST, English, commentary, or a combined form maps to the same identity. Same-chapter ranges are expanded to individual canonical verses. Cross-chapter passages are annotated as separate IDs.

## Relevance judgments

Grades are:

- 3: primary/highly relevant; directly resolves the information need
- 2: relevant; materially supports the answer
- 1: marginal/contextual relevance
- 0: judged nonrelevant

Multiple relevant verses are normal. Binary measures treat grades 1–3 as relevant; nDCG uses gain `2^grade - 1`. LLM judgments are supplementary and may not create or overwrite official qrels.

## Retrieval measures

For each query and configured cutoff, the framework reports Recall, Precision, Hit Rate, Success, graded recall, AP, and nDCG. It also reports MRR, full-ranking AP (whose macro mean is MAP), R-Precision, first relevant rank, and hard-negative accuracy. Duplicate passage IDs are removed while preserving first occurrence. Precision@K uses K as its denominator, including when a system returns fewer than K results; `cutoff_complete` makes shallow outputs visible.

The selected family follows established test-collection practice: TREC CAR reports MAP, reciprocal rank, R-Precision, and nDCG, noting that nDCG uses graded judgments ([TREC CAR overview](https://trec.nist.gov/pubs/trec27/papers/Overview-CAR.pdf)).

## Uncertainty and comparison

Each aggregate rate includes its query denominator, arithmetic mean, sample standard deviation, and a seeded percentile-bootstrap confidence interval. Queries—not passages or individual labels—are resampled. NIST reports that standard and percentile-bootstrap intervals have strong empirical coverage for common IR measures outside extremely poor retrieval regimes ([Soboroff, 2014](https://www.nist.gov/publications/computing-confidence-intervals-common-ir-measures)).

Two systems are aligned only on identical scored query IDs. Recall@1 changes use an exact paired McNemar test. MRR and nDCG deltas use paired bootstrap intervals, and MRR includes Cohen's `d_z` when variance permits. Multi-arm ablations apply Holm family-wise correction to McNemar p-values. A p-value alone is never an acceptance criterion: report the paired effect, interval, adjusted test, subgroup regressions, and changed rankings.

## Robustness

Controlled variants reference an original with `variant_of` and identify one perturbation: paraphrase, synonym, spelling, grammar, informality, length, transliterated Sanskrit, alternate Sanskrit spelling, terminology substitution, or ambiguity. The Robustness Score is the mean retained Success@K among originals that succeeded. The report also preserves every paired delta and perturbation breakdown. Do not combine uncontrolled rewrites with this score.

## Cross-lingual and cross-script experiments

Use the same queries, qrels, corpus content, retrieval budget, and random seed. Change only representation: Devanagari, IAST, normalized transliteration, English translation, Sanskrit-English mixed, or multilingual combined. Record `representation_group` for paired items or compare separate run directories through the ablation command. The Cross-Script Score is `1 - range(Success@K)` per paired group, macro-averaged. Always report each representation's ordinary IR measures alongside this invariance score; invariance at uniformly poor performance is not quality.

## Hard negatives

Hard negatives are human-reviewed nonanswers selected for lexical overlap, same-chapter proximity, shared entities, shared Sanskrit terminology, or semantic closeness. Hard-negative accuracy asks whether the best relevant passage outranks each annotated negative. Negatives must be sampled before the evaluated run and cannot be mined from the final test outputs of a system being optimized.

## Retrieval versus generation

Retrieval metrics are computed first and remain valid without an LLM. Generation is a separate track covering deterministic citation validity, citation precision/recall, unsupported citations, empty/refusal behavior, and review fields for correctness, faithfulness, context utilization, attribution, and clarity. Automated RAG metrics can be useful—RAGAS explicitly separates context relevance, answer relevance, and faithfulness—but the paper validates them against human preferences, reinforcing their supplementary status ([RAGAS, EACL 2024](https://aclanthology.org/2024.eacl-demo.16/)).

Promptfoo's Python-provider and Python-assertion interfaces connect any Python/API system to a repeatable evaluation matrix ([Python provider](https://www.promptfoo.dev/docs/providers/python/), [Python assertions](https://www.promptfoo.dev/docs/configuration/expected-outputs/python/)). In this benchmark it supplements, rather than replaces, canonical retrieval scoring.

## Failure attribution

Architecture-independent classes include relevant passage absent from Top-K, correct chapter/wrong verse, hard-negative confusion, multi-hop failure, ambiguity, cross-lingual/transliteration failure, adapter failure, and success. If an adapter exposes stage outputs, those raw traces are preserved without imposing stage names. A system-specific adapter may add finer labels such as source recall, fusion drop, reranker failure, or generation failure.

## Officiality and reproducibility

An official result requires a held-out test split, human-verified provenance for every qrel, annotation confidence for every example, zero adapter errors, a frozen benchmark version, and external contamination review. Every run stores configuration, seed, timestamp, system metadata, raw ranked results, per-query measures, quality audit, and summary artifacts. The evaluator labels incomplete data as diagnostic instead of silently promoting it to a leaderboard.

