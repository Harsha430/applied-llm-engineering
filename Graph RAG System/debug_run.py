import os
from loguru import logger
from langchain_groq import ChatGroq
from graph_rag.debugger import demonstrate_return_error, demonstrate_schema_hallucination
from graph_rag.cypher_chain import build_cypher_chain, run_query

def main():
    print("\n" + "="*60)
    print("      POKÉMON GRAPH RAG SYSTEM — DEBUG DEMO")
    print("="*60)
    print("In this demo, we will test the common failure modes and then run a final-answer query.")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")
    
    # ── Reproduce failures ───────────────────────────────────────────
    demonstrate_return_error(llm)
    print("-" * 30)
    demonstrate_schema_hallucination(llm)
    print("-" * 30)
    
    # ── Show finalized Graph RAG retrieval ──────────────────────────
    print("\nRunning Finalized Graph RAG Retrieval …\n")
    chain, graph = build_cypher_chain()
    
    question = "Which category does Pidgeot belong to?"
    response = run_query(chain, question)
    
    print(f"\n[Final Question]: {question}")
    print(f"[Generated Cypher]: {response['cypher']}")
    print(f"[Raw Results]: {response['raw_results']}")
    print(f"[Final Answer]: {response['answer']}")

if __name__ == "__main__":
    main()
