"""
graph_rag/embeddings.py
─────────────────────────────────────────────────────────────────────────────
Singleton HuggingFace embeddings loader.

Why singleton?
  Loading sentence-transformers is expensive (~500 MB, 3-5 sec first run).
  We load once and reuse across few-shot selection and any vector operations.

Model: sentence-transformers/all-MiniLM-L6-v2
  • 384-dim dense vectors  • Fast inference  • Great semantic similarity
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFaceEmbeddings instance.
    First call downloads the model (if not cached locally).
    Subsequent calls return the same object instantly.
    """
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},       # change to "cuda" if GPU available
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity ready
    )
    logger.success(f"Embedding model loaded: {EMBEDDING_MODEL}")
    return embeddings
