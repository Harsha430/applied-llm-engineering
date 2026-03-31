import re
import numpy as np
from typing import List, Dict, Tuple, Any
from langchain_core.documents import Document

def tokenize(text: str) -> List[str]:
    """Robust tokenization for BM25"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

class HybridRetriever:
    def __init__(self, vectorstore, bm25_corpus, documents: List[Document], alpha: float = 0.5, beta: float = 0.5):
        """
        vectorstore: FAISS vectorstore instance.
        bm25_corpus: the initialized rank_bm25 object.
        documents: list of all documents with 'doc_id' in metadata.
        alpha: weight for Dense (vectorstore) retriever.
        beta: weight for Sparse (BM25) retriever.
        """
        self.vectorstore = vectorstore
        self.bm25 = bm25_corpus
        self.documents = documents
        
        # We index documents by their unique doc_id for fast lookup
        self.doc_dict = {doc.metadata["doc_id"]: doc for doc in documents}
        self.alpha = alpha
        self.beta = beta
        self.rrf_k = 60

    def get_relevant_documents(self, query: str, k: int = 5) -> Tuple[List[Document], Dict[str, Any]]:
        # 1. Expand candidate pool to ensure effective hybrid intersection
        candidate_k = max(k * 5, 20)
        
        # 2. Get Dense Results
        dense_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=candidate_k)
        
        dense_rank = {}
        for rank, (doc, score) in enumerate(dense_results):
            doc_id = doc.metadata["doc_id"]
            dense_rank[doc_id] = rank + 1  # 1-indexed rank
        
        # 3. Get Sparse (BM25) Results
        tokenized_query = tokenize(query)
        bm25_all_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top candidate_k indices
        top_bm25_indices = np.argsort(bm25_all_scores)[::-1][:candidate_k]
        
        bm25_rank = {}
        for rank, idx in enumerate(top_bm25_indices):
            # Only consider docs that actually had a non-zero BM25 score 
            # (or just use rank, but rank for 0 score is still penalized in RRF)
            doc_id = self.documents[idx].metadata["doc_id"]
            if bm25_all_scores[idx] > 0:
                bm25_rank[doc_id] = rank + 1
            else:
                # If score is exactly 0, it doesn't match at all. Assign bad rank.
                bm25_rank[doc_id] = float('inf')
        
        # 4. RRF (Reciprocal Rank Fusion) Scoring
        all_unique_ids = set(list(dense_rank.keys()) + list(bm25_rank.keys()))
        
        final_scores = {}
        for doc_id in all_unique_ids:
            # 1 / (k + rank)
            d_rank = dense_rank.get(doc_id, float('inf'))
            b_rank = bm25_rank.get(doc_id, float('inf'))
            
            d_score = 1.0 / (self.rrf_k + d_rank) if d_rank != float('inf') else 0.0
            b_score = 1.0 / (self.rrf_k + b_rank) if b_rank != float('inf') else 0.0
            
            # Weighted RRF
            final_scores[doc_id] = (self.alpha * d_score) + (self.beta * b_score)
            
        # 5. Sort and get top-K combined
        sorted_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)[:k]
        
        final_docs = [self.doc_dict[doc_id] for doc_id in sorted_ids]
        
        # Metadata for Failure Analysis
        analysis_metadata = {
            "query": query,
            "results": []
        }
        
        for doc_id in sorted_ids:
            doc = self.doc_dict[doc_id]
            content_preview = doc.page_content.replace('\n', ' ')[:80] + "..."
            
            d_rank = dense_rank.get(doc_id, "N/A")
            b_rank = bm25_rank.get(doc_id, "N/A")
            
            analysis_metadata["results"].append({
                "content_preview": content_preview,
                "doc_id": doc_id,
                "dense_rank": d_rank,
                "bm25_rank": b_rank,
                "final_rrf_score": final_scores[doc_id]
            })
            
        return final_docs, analysis_metadata
