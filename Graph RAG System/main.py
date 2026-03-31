import os
import sys
from loguru import logger
from graph_rag.cypher_chain import build_cypher_chain, run_query

# Configure loguru to be less verbose for the main CLI experience
logger.remove()
logger.add(sys.stderr, level="INFO")

def main():
    print("\n" + "="*60)
    print("      POKÉMON GRAPH RAG SYSTEM — EXPLORER")
    print("="*60)
    print("Build on Neo4j, Groq, and LangChain.")
    print("Type 'exit' or 'quit' to stop.\n")

    try:
        # Build the Graph RAG chain
        # Uses k=3 few-shots and Groq LLM
        chain, graph = build_cypher_chain()
    except Exception as e:
        logger.error(f"Failed to initialize Graph RAG System: {e}")
        return

    while True:
        try:
            question = input("\n[Question]: ")
            if question.lower() in ["exit", "quit"]:
                print("Exiting. Keep exploring!")
                break
            
            if not question.strip():
                continue

            # Run the query through the chain
            response = run_query(chain, question)

            # Output results with high-end formatting
            print("\n" + "─"*60)
            print("  Generated Cypher:")
            print(f"  \033[94m{response['cypher']}\033[0m")
            print("─"*60)
            
            if response['raw_results']:
                print("  Graph Results (Raw):")
                print(f"  {response['raw_results']}")
                print("─"*60)
                
            print(f"\n  💡 \033[1mAnswer:\033[0m {response['answer']}")
            print("─"*60)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    main()
