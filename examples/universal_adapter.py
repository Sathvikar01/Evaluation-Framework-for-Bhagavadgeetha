"""Minimal adapter example; replace the body with any RAG implementation."""


class MyBhagavadGitaRAG:
    def retrieve(self, query: str, k: int):
        # Call a vector DB, BM25, graph, HTTP service, or complete hybrid RAG.
        # The benchmark needs only ranked canonical passage identities.
        return {
            "results": [
                {
                    "passage_id": "BhG 2.47",
                    "score": 0.91,
                    "text": "Optional passage text",
                    "document_type": "verse",
                    "corpus_representation": "english_translation",
                    "metadata": {"retrieval_strategy": "example"},
                }
            ][:k]
        }


adapter = MyBhagavadGitaRAG()

