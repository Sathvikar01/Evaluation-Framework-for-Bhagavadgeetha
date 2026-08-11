"""Universal Bhagavad Gita RAG benchmark with a SansRAG compatibility adapter.

This package is deliberately independent from :mod:`src.evaluation`, whose
historical output and metric semantics remain frozen for compatibility.
"""

SCHEMA_VERSION = "evaluation_v2.1"

from .benchmark import GoldExample, evaluate, load_benchmark
from .universal_adapter import UniversalRAGAdapter

__all__ = ["SCHEMA_VERSION", "GoldExample", "UniversalRAGAdapter", "evaluate", "load_benchmark"]
