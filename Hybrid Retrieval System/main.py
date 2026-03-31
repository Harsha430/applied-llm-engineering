import os
import sys
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from rank_bm25 import BM25Okapi

from hybrid_retriever import HybridRetriever, tokenize

# Load env variables for Groq API Key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in .env")
    sys.exit(1)

def is_valid_chunk(text: str) -> bool:
    if len(text.strip()) < 50:
        return False
    if text.strip().isdigit():
        return False
    if "chapter" in text.lower() and len(text) < 100:
        return False
    return True

def main():
    print("🔥 STEP 1 — Setup Project (Completed)")
    print("Loading The Bhagavad Gita.pdf using PyMuPDF...")
    pdf_path = "The Bhagavad Gita.pdf"
    
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
        
    print(f"Extracted {len(full_text)} characters.")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    raw_chunks = text_splitter.split_text(full_text)
    print(f"Initial raw chunks: {len(raw_chunks)}")
    
    # FILTER AND NOISE REMOVAL
    documents = []
    current_doc_id = 0
    for chunk in raw_chunks:
        if is_valid_chunk(chunk):
            documents.append(Document(page_content=chunk, metadata={"doc_id": current_doc_id}))
            current_doc_id += 1
            
    print(f"Created {len(documents)} clean, valid chunks for hybrid indexing.")
    
    print("🔥 STEP 2 — Build Dense Index")
    # Set up HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        # normalize embeddings so FAISS returns distances cleanly
        encode_kwargs={'normalize_embeddings': True} 
    )
    # Using FAISS for Dense Retrieval
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print("🔥 STEP 3 — Build BM25 Index")
    # Tokenize corpus for BM25 using robust tokenizer
    tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("Initializing Custom Hybrid Retriever...")
    # Initialize our custom weights: Alpha=0.6, Beta=0.4 (slightly more balanced for mixed queries)
    hybrid_retriever = HybridRetriever(
        vectorstore=vectorstore,
        bm25_corpus=bm25,
        documents=documents,
        alpha=0.6,
        beta=0.4
    )

    # Set up the LLM
    print("Initializing System LLM (ChatGroq)")
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)
    
    queries = [
        {
            "type": "Case 1: Keyword query",
            "query": "Who speaks at the beginning of the Gita?",
            "expected": "Expected: BM25 contributes heavily to exact English terms"
        },
        {
            "type": "Case 2: Semantic query",
            "query": "What does Krishna teach about duty without attachment to results?",
            "expected": "Expected: Dense dominates due to conceptual phrase matching"
        },
        {
            "type": "Case 3: Mixed",
            "query": "How does Arjuna escape the cycle of rebirth according to Krishna?",
            "expected": "Expected: Both contribute"
        }
    ]
    
    print("\n" + "="*60)
    print("🔥 Failure Analysis Output")
    print("="*60)
    
    for case in queries:
        print(f"\n{case['type']}")
        print(f"Query: '{case['query']}'")
        print(f"{case['expected']}\n")
        
        # 🔥 STEP 4 to 7
        top_k = 3
        retrieved_docs, metadata = hybrid_retriever.get_relevant_documents(case['query'], k=top_k)
        
        print("--- Retriever Scores Breakdown ---")
        for i, res in enumerate(metadata["results"], 1):
            print(f"Top {i} Chunk [Doc ID: {res['doc_id']}]")
            print(f"  Final RRF Score: {res['final_rrf_score']:.6f}")
            print(f"  Dense Rank : {res['dense_rank']}  (weight: {hybrid_retriever.alpha})")
            print(f"  BM25 Rank  : {res['bm25_rank']}  (weight: {hybrid_retriever.beta})")
            print(f"  Content Preview  : {res['content_preview'].strip()}")
            print("-" * 30)
            
        print("Sending Context to LLM -> generating answer...\n")
        
        # Construct Context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"""Use the following context to answer the user's query clearly and concisely.

Context:
{context}

Query: {case['query']}

Answer:"""
        
        response = llm.invoke(prompt)
        print(f"💡 LLM Response:\n{response.content.strip()}")
        print("="*60)

if __name__ == "__main__":
    main()
