# Specification compliance matrix

| # | Requirement | Implementation | Status |
|---:|---|---|---|
| 1 | Gold dataset | JSON Schema, `GoldExample`, graded multi-passage qrels, provenance/confidence/translation fields | Implemented; official expert qrels remain a release gate |
| 2 | Query taxonomy | 17 enforced categories and four difficulties | Implemented |
| 3 | Cross-lingual/script | Representation metadata, paired groups, Cross-Script Score, controlled-run protocol | Implemented; corpus variants supplied by researcher |
| 4 | Retrieval metrics | Recall/Precision/Hit/Success at K, MRR, AP/MAP, R-Precision, nDCG, graded recall | Implemented |
| 5 | Verse correctness | Alias/range normalization and canonical 18-chapter inventory | Implemented |
| 6 | Semantic relevance | Multi-gold grades 0–3; human qrels primary; LLM judges supplementary | Implemented |
| 7 | Hard negatives | Schema, rationales, per-query and aggregate accuracy | Implemented |
| 8 | Robustness | Controlled variants, retained-performance score, perturbation breakdown | Implemented |
| 9 | Architecture comparison | Python/HTTP/command/replay adapters with only `retrieve(query,k)` required | Implemented |
| 10 | End-to-end RAG | Retrieval and generation separated; deterministic citations plus review/Promptfoo | Implemented |
| 11 | Ablations | Multi-run table, paired tests, confidence intervals, Holm correction | Implemented |
| 12 | Statistics | Query bootstrap CIs, SD, McNemar, paired bootstrap, effect size | Implemented |
| 13 | Leakage | Split-aware schema/audits; labels withheld from adapter; existing repository scanner | Implemented; hidden-server deployment is operational |
| 14 | Dataset validation | Duplicates, ambiguity, distributions, overlap, confidence, lexical overlap, agreement | Implemented |
| 15 | Error analysis | Universal failure taxonomy, stored failures, optional raw stage traces | Implemented |
| 16 | Reproducibility | Version/config/seed/time/system metadata/raw results/per-query artifacts | Implemented |
| 17 | Universal interface | Programmatic `evaluate(system, benchmark)` and CLI | Implemented |
| 18 | Leaderboard | Standard JSON and Markdown table plus detailed breakdowns | Implemented |
| 19 | Benchmark philosophy | Held-out policy, adversarial examples, contamination resistance, transparent methods | Documented and enforced where machine-checkable |

The diagnostic starter set demonstrates the schema and execution path but is intentionally not called a scientific gold release. Promoting it without expert annotation would violate requirements 1, 6, 13, and 19.

