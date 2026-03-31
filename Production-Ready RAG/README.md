# Production-Ready RAG

Retrieval-Augmented Generation application integrating FAISS vector stores with document loaders to query unstructured documents securely and efficiently.

## Overview

This repository demonstrates the architecture of a full-stack RAG pipeline. It handles everything from chunking and embedding, all the way to semantic search querying context to an LLM for final generation.

## Usage

1. Create a `.env` file containing required API keys.
2. Ensure you have installed necessary dependencies from `requirements.txt`.
3. First, build the FAISS index by running:
    ```bash
    python build_db.py
    ```
4. Start querying the application:
    ```bash
    python app.py
    ```
