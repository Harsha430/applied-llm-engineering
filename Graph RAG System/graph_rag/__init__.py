"""
graph_rag/__init__.py
Package init — re-exports key components for convenient imports.
"""

from graph_rag.embeddings import get_embeddings
from graph_rag.schema import SCHEMA_STRING, NODE_LABELS, RELATIONSHIP_TYPES
from graph_rag.few_shots import FewShotSelector
from graph_rag.cypher_chain import build_cypher_chain

__all__ = [
    "get_embeddings",
    "SCHEMA_STRING",
    "NODE_LABELS",
    "RELATIONSHIP_TYPES",
    "FewShotSelector",
    "build_cypher_chain",
]
