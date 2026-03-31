import sys
from memory.summarizer import chat_with_memory, get_session_messages

def main():
    print("Welcome to the LangChain Groq Memory System!")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'switch' to change to a different session.")
    print("Type 'history' to view the full un-trimmed session history.")
    
    session_id = input("Enter a session ID to start (e.g., 'user_123'): ").strip()
    if not session_id:
        session_id = "default_session"
        print(f"Using default session ID: {session_id}")

    while True:
        try:
            user_input = input(f"\n[{session_id}] You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break
            
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        elif user_input.lower() == 'switch':
            session_id = input("Enter a new session ID: ").strip()
            print(f"Switched to session: {session_id}")
            continue
        elif user_input.lower() == 'history':
            messages = get_session_messages(session_id)
            print(f"\n--- History for {session_id} ---")
            for msg in messages:
                print(f"{msg.type.capitalize()}: {msg.content}")
            print("-------------------------")
            continue
        elif not user_input.strip():
            continue
            
        # Get AI response
        print("AI is thinking...")
        try:
            response = chat_with_memory(session_id, user_input)
            print(f"AI: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
