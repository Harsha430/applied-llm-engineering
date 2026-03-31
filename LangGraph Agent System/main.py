import os
from dotenv import load_dotenv
from graph import build_graph

load_dotenv()

def main():
    """
    Main entry point for the LangGraph agent.
    """
    # Compile the graph
    app = build_graph()
    
    print("Welcome to the LangGraph Agentic Search System!")
    print("Flow: Plan -> Retrieve -> Evaluate -> Retry (if needed) -> Answer")
    print("-" * 50)
    
    while True:
        query = input("\nUser Query (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        # Initial State
        initial_state = {
            "query": query,
            "plan": "",
            "retrieved_docs": [],
            "evaluation": {},
            "retry_count": 0,
            "answer": ""
        }
        
        print("\n--- Agent is thinking... ---")
        
        # Run the graph
        # We'll stream the updates to see the state changes
        for output in app.stream(initial_state):
            for node, state in output.items():
                print(f"DEBUG: Current Node -> [{node}]")
                if node == "planner":
                    print(f"Plan: {state['plan']}")
                elif node == "evaluator":
                    print(f"Evaluation: {state['evaluation']['satisfactory']}")
                    if not state['evaluation']['satisfactory']:
                        print(f"Reason: {state['evaluation']['explanation']}")
                elif node == "responder":
                    print("\n--- Final Answer ---")
                    print(state['answer'])

if __name__ == "__main__":
    main()
