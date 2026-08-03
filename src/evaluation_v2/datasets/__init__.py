"""Dataset adapters registered for Evaluation V2."""

from .base import DatasetAdapter, load_json_records, sha256_file
from .bhagavad_gita_qa import BhagavadGitaQAAdapter
from .edwin_arnold import EdwinArnoldAdapter
from .gitadb import GitaDBAdapter
from .anveshana import AnveshanaAdapter
from .with_id import WithIDAdapter

__all__ = [
    "DatasetAdapter", "load_json_records", "sha256_file", "BhagavadGitaQAAdapter",
    "EdwinArnoldAdapter", "GitaDBAdapter", "AnveshanaAdapter", "WithIDAdapter",
]
