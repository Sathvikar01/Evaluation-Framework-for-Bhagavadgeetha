# Research and tool sources

The design was checked against primary or authoritative sources discovered through the requested SearXNG workflow and then verified directly. The local machine did not expose a running SearXNG command/service; the public instance registry was queried successfully, while public search endpoints rate-limited automated JSON requests. SearXNG documents that JSON output depends on enabling the requested response format ([Search API](https://docs.searxng.org/dev/search_api.html)). No search snippets are treated as benchmark evidence.

Method sources:

- NIST TREC CAR overview for MAP, reciprocal rank, R-Precision, and graded nDCG: <https://trec.nist.gov/pubs/trec27/papers/Overview-CAR.pdf>
- NIST study of confidence intervals for common IR measures: <https://www.nist.gov/publications/computing-confidence-intervals-common-ir-measures>
- RAGAS paper for separated context relevance, answer relevance, and faithfulness with human validation: <https://aclanthology.org/2024.eacl-demo.16/>
- Promptfoo Python provider and assertion interfaces: <https://www.promptfoo.dev/docs/providers/python/> and <https://www.promptfoo.dev/docs/configuration/expected-outputs/python/>
- Cohen's kappa definition and weighting API: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html>

These sources define general evaluation principles. They do not provide Bhagavad Gita qrels; those require licensed source selection and domain-expert annotation under this repository's annotation protocol.

