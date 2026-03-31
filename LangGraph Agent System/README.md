# Project 6 — LangGraph Agent System

A multi-step reasoning agent built with **LangGraph**, **Groq**, and **Tavily**.

## Features
- **Dynamic Planning**: Decomposes complex queries into manageable search steps.
- **Smart Retrieval**: Fetches real-time web data using Tavily.
- **Self-Evaluation**: Critically assesses the quality of retrieved information.
- **Auto-Retry Loop**: If information is insufficient, the agent re-plans and retries (up to 3 times).
- **State Persistence**: Uses LangGraph's state management to accumulate context across loops.

## Core Flow
`Plan` → `Retrieve` → `Evaluate` → `Retry (if needed)` → `Answer`

## Setup
1. Define environment variables in `.env`:
   - `GROQ_API_KEY`
   - `TAVILY_API_KEY`
2. Activate your virtual environment and install dependencies:
   ```bash
   pip install langgraph langchain-groq tavily-python python-dotenv
   ```
3. Run the agent:
   ```bash
   python main.py
   ```

## Files
- `state.py`: Defines the agent's memory (TypedDict).
- `nodes.py`: Implementation of reasoning nodes.
- `graph.py`: Orchestration of the workflow using `StateGraph`.
- `main.py`: CLI interface for interacting with the agent.
