import os
from dotenv import load_dotenv
from agent.react_loop import run_agent

load_dotenv()

def main():
    print("Welcome to the Tool-Based ReAct Agent!")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ("exit", "quit"):
            break
        if not query.strip():
            continue
            
        print("\nRunning agent...\n")
        try:
            response = run_agent(query)
            print("\nFinal Answer:\n", response.get("output", "No output returned."))
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
