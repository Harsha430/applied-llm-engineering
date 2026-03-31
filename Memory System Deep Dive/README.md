# Memory System Deep Dive

This project focuses on stateful conversational contexts for LLMs using LangChain's Memory components.

## Overview

The core feature being explored here is keeping conversation history and passing states between multiple LLM calls. We utilize `RunnableWithMessageHistory` and implement token limits to prevent context blowing out in extended chatbot interactions.

## Usage

1. Add your Groq (or other provider) API key to the `.env` file.
2. Install the necessary dependencies inside your virtual environment.
3. Run the entrypoint script:
    ```bash
    python main.py
    ```
