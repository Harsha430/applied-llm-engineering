import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "transformers.pdf"
DB_PATH = "faiss_index"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

MIN_CHUNK_LENGTH = 120

load_dotenv()

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("<EOS>", "")
    text = text.replace("<pad>", "")
    text = " ".join(text.split())
    return text.strip()


def is_valid_chunk(text: str) -> bool:
    t = text.lower()

    if len(text) < MIN_CHUNK_LENGTH:
        return False

    if "arxiv" in t and "[" in t:
        return False

    if "et al" in t and "2017" in t:
        return False

    return True

def score_chunk(text: str) -> int:
    t = text.lower()
    score = 0

    if "we propose" in t:
        score += 3
    if "transformer" in t:
        score += 3
    if "architecture" in t:
        score += 2
    if "attention mechanism" in t:
        score += 2
    if "abstract" in t:
        score += 3
    if "equation" in t:
        score -= 2
    if "softmax" in t:
        score -= 1
    if "figure" in t:
        score -= 1

    return score


def load_documents(path):
    loader = PyPDFLoader(path)
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def process_chunks(splits):
    cleaned = []

    for doc in splits:
        text = clean_text(doc.page_content)

        if is_valid_chunk(text):
            doc.page_content = text
            cleaned.append(doc)

    return cleaned


def sort_by_importance(docs):
    return sorted(
        docs,
        key=lambda d: score_chunk(d.page_content),
        reverse=True
    )


def build_and_save_db(docs):
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embedding)
    db.save_local(DB_PATH)

    print(f"\n✅ Saved FAISS DB at: {DB_PATH}")
    print(f"✅ Total vectors: {db.index.ntotal}")


def debug_samples(docs, n=5):
    print("\n[DEBUG] Top ranked chunks:\n")
    for i in range(min(n, len(docs))):
        print(f"\n--- Chunk {i+1} ---\n")
        print(docs[i].page_content[:400])

if __name__ == "__main__":
    print("\n🔹 Loading documents...")
    docs = load_documents(DATA_PATH)

    print(f"Loaded {len(docs)} pages")

    print("\n🔹 Splitting...")
    splits = split_documents(docs)
    print(f"Initial chunks: {len(splits)}")

    print("\n🔹 Cleaning + filtering...")
    cleaned = process_chunks(splits)
    print(f"Valid chunks: {len(cleaned)}")

    print("\n🔹 Ranking chunks...")
    ranked = sort_by_importance(cleaned)

    debug_samples(ranked)

    print("\n🔹 Building DB...")
    build_and_save_db(ranked)