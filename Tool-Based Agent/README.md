# Tool-Based Agent

This project explores an intelligent ReAct (Reasoning and Acting) loop, embedding LLMs with tool-calling capabilities. 

## Overview

We designed an LLM structure that iteratively plans an action, evaluates tools available (e.g., search tools, calculators, external APIs), acts via executing tool calls, and reflects on the results until the completion or goal of the task is met.

## Usage

1. Add your API keys to the `.env` file.
2. Install the necessary dependencies inside your virtual environment.
3. Start the agent:
    ```bash
    python main.py
    ```
