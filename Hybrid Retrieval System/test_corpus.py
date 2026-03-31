import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import re

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()

def is_valid_chunk(text: str) -> bool:
    if len(text.strip()) < 50:
        return False
    if text.strip().isdigit():
        return False
    if "chapter" in text.lower() and len(text) < 100:
        return False
    return True

print("Loading The Bhagavad Gita.pdf using PyMuPDF...")
pdf_path = "The Bhagavad Gita.pdf"
doc = fitz.open(pdf_path)
full_text = ""
for page in doc:
    full_text += page.get_text() + "\n"

print(f"Extracted {len(full_text)} characters.")

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", " ", ""]
)
raw_chunks = text_splitter.split_text(full_text)

documents = []
current_doc_id = 0
for chunk in raw_chunks:
    if is_valid_chunk(chunk):
        documents.append(Document(page_content=chunk, metadata={"doc_id": current_doc_id}))
        current_doc_id += 1

print(f"Created {len(documents)} clean chunks.")

print("\n--- STEP 3: Inspect Corpus Manually (Top 10 chunks) ---")
for i in range(min(10, len(documents))):
    print(f"CHUNK {i}:\n{documents[i].page_content.strip()}\n{'-'*30}")

print("\n--- STEP 4: Validate BM25 Independently ---")
tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

query = "Dhritarashtra uvacha"
print(f"Query: {query}")
# getting top 5
tokenized_query = tokenize(query)
results = bm25.get_top_n(tokenized_query, documents, n=5)

for r in results:
    print(f"SCORE: {bm25.get_scores(tokenized_query)[r.metadata['doc_id']]:.4f}")
    print(r.page_content.strip()[:100] + "...")
    print("-" * 30)
