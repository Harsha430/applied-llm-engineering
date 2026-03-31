# Hybrid Retrieval System

This project implements a hybrid document retrieval pipeline combining keyword-based (BM25) and semantic retrieval strategies using LangChain and PyMuPDF.

## Overview

A critical component of any capable Retrieval-Augmented Generation (RAG) system is accurate document retrieval. By leveraging text mining with dense vectors and lexical scores, this module retrieves context efficiently even in the presence of noise and structural edge cases.

## Usage

1. Create a `.env` file containing required API keys.
2. Ensure you have activated your Python environment and installed necessary dependencies.
3. Run the main processing script:
    ```bash
    python main.py
    ```

## Files include:
- `hybrid_retriever.py`: Contains the logic for the hybrid ensemble retriever.
- `main.py`: Main execution workflow integrating multiple steps.
- `test_corpus.py`: Used to validate and text chunking outputs.
