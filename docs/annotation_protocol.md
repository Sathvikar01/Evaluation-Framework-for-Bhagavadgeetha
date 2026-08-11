# Annotation and release protocol

## Roles

Use at least two independent annotators with demonstrated Bhagavad Gita familiarity and one adjudicator. Record stable pseudonymous annotator IDs, source edition(s), translation(s), date, and license. Do not let annotators see system rankings when forming initial qrels.

## Procedure

1. Write the information need and assign one taxonomy category and difficulty.
2. Pool candidate passages from source study and diverse retrieval systems; pooling improves coverage but does not make unjudged passages relevant.
3. Independently grade every pooled passage 0–3 using the rubric in the methodology.
4. Mark near-miss grade-0 passages as hard negatives and record a rationale.
5. Adjudicate disagreements without erasing the original judgments.
6. Record confidence in `[0,1]`, ambiguity, concepts, entities, canonical chapters, source/provenance, and optional Devanagari/IAST/translation references.
7. Create robustness variants only after the original qrel is fixed; verify that each variant preserves the information need.
8. Assign whole semantic groups—including originals, variants, and multi-gold connected passages—to one split.

Report raw agreement and weighted Cohen's kappa for two-annotator graded labels. Cohen's kappa adjusts observed agreement for expected chance agreement; the framework uses quadratic weights for the ordinal 0–3 scale ([scikit-learn definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)). With more than two annotators, additionally report a suitable multi-rater statistic and its uncertainty outside the built-in pairwise summary.

## Split and contamination policy

- Development: labels visible; tuning allowed.
- Validation: labels visible only for scheduled model selection; report selection count.
- Test: labels evaluator-side/locked; no parameter tuning, hard-negative mining, prompt editing, or query rewriting based on results.
- Diagnostic: may be contaminated, synthetic, single-annotator, or incompletely licensed; never an official leaderboard score.

Keep all variants of one information need in the same split. Keep verse-connected multi-gold groups together when feasible. Hash and version the split manifest and qrels. Release public test queries separately from private test qrels if an independent evaluation server is available.

Before release, audit duplicates, template artifacts, chapter/concept/category/difficulty distributions, lexical overlap, confidence, agreement, source licenses, index/training/prompt contamination, and cross-split passage overlap. Any unresolved gate must appear in the benchmark card and set `official=false`.

