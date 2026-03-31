import os
import json
from dotenv import load_dotenv
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

DB_PATH = "faiss_index"
EVAL_FILE = "eval_data.json"

TOP_K = 8
MMR_LAMBDA = 0.4

MAX_CONTEXT_CHARS = 6000
MEMORY_MAX_CHARS = 1500

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY")


def load_db():
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_PATH,
        embedding,
        allow_dangerous_deserialization=True
    )

    print(f"[LOG] DB size: {db.index.ntotal}")
    return db


def get_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=GROQ_API_KEY,
        streaming=True
    )


class Memory:
    def __init__(self):
        self.summary = ""

    def update(self, llm, query, answer):
        prompt = f"""
Summarize conversation briefly.

Previous:
{self.summary}

New:
User: {query}
AI: {answer}

Summary:
"""
        self.summary = llm.invoke(prompt).content.strip()[:MEMORY_MAX_CHARS]

    def get(self):
        return self.summary


def rewrite_query(llm, query, memory):
    prompt = f"""
Rewrite the query to retrieve HIGH-LEVEL explanations from a research paper.

Focus on:
- definitions
- architecture overview
- key ideas

Avoid:
- equations
- formulas

Conversation:
{memory}

Query:
{query}

Rewritten query:
"""
    return llm.invoke(prompt).content.strip()


def retrieve_docs(db, query):
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "lambda_mult": MMR_LAMBDA
        }
    )
    return retriever.invoke(query)


def rerank_docs(query, docs):
    def score(doc):
        t = doc.page_content.lower()
        s = 0

        if "transformer" in t:
            s += 3
        if "attention" in t:
            s += 3
        if "architecture" in t:
            s += 2

        if "equation" in t:
            s -= 2
        if "figure" in t:
            s -= 1

        return s

    return sorted(docs, key=score, reverse=True)


def filter_docs(docs):
    clean = []

    for d in docs:
        t = d.page_content.lower()

        if "law will never be perfect" in t:
            continue

        if len(t) < 100:
            continue

        clean.append(d)

    return clean


def format_docs(docs: List):
    context = "\n\n".join(
        f"[Chunk {i+1}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )
    return context[:MAX_CONTEXT_CHARS]


def build_prompt(context, query, memory):
    return f"""
You are an expert AI assistant.

- Answer clearly and conceptually
- Use ONLY provided context
- If insufficient info → say "I don't know"

Conversation Summary:
{memory}

Context:
{context}

Question:
{query}

Answer:
"""


def evaluate_answer(llm, query, expected, actual):
    prompt = f"""
You are evaluating an AI system.

Question:
{query}

Expected Answer:
{expected}

Actual Answer:
{actual}

Score the answer from 1 to 10 based on correctness and relevance.

Return ONLY a number.
"""
    try:
        score = llm.invoke(prompt).content.strip()
        return float(score)
    except:
        return 0.0


class RAGSystem:
    def __init__(self, db):
        self.db = db
        self.llm = get_llm()
        self.memory = Memory()

    def answer(self, query, verbose=True):
        if verbose:
            print(f"\n[LOG] USER QUERY: {query}")

        rewritten = rewrite_query(
            self.llm,
            query,
            self.memory.get()
        )

        docs = retrieve_docs(self.db, rewritten)
        docs = rerank_docs(rewritten, docs)
        docs = filter_docs(docs)

        if not docs:
            return "No relevant information found."

        context = format_docs(docs)

        if len(context) < 200:
            return "Insufficient context to answer."

        prompt = build_prompt(
            context,
            query,
            self.memory.get()
        )

        response_chunks = []
        for chunk in self.llm.stream(prompt):
            if verbose:
                print(chunk.content, end="", flush=True)
            response_chunks.append(chunk.content)

        final = "".join(response_chunks)

        self.memory.update(self.llm, query, final)

        return final


def run_evaluation(rag):
    if not os.path.exists(EVAL_FILE):
        print("eval_data.json not found")
        return

    with open(EVAL_FILE, "r") as f:
        data = json.load(f)

    scores = []

    print("\n[LOG] RUNNING EVALUATION...\n")

    for item in data:
        query = item["query"]
        expected = item["expected"]

        print(f"\n[TEST] {query}")

        actual = rag.answer(query, verbose=False)

        score = evaluate_answer(rag.llm, query, expected, actual)

        print(f"Score: {score}")
        scores.append(score)

    avg_score = sum(scores) / len(scores)

    print("\n==========================")
    print(f"AVERAGE SCORE: {avg_score}")
    print("==========================\n")


if __name__ == "__main__":
    db = load_db()
    rag = RAGSystem(db)

    mode = input("Enter mode (chat/eval): ").strip()

    if mode == "eval":
        run_evaluation(rag)
    else:
        while True:
            q = input("\nAsk (or 'exit'): ")
            if q.lower() == "exit":
                break

            print("\nAnswer:\n")
            rag.answer(q)
            print("\n" + "-"*50)