from duckduckgo_search import DDGS
from langchain_core.tools import tool
import json

@tool
def search_web(query: str):
    """Search the web for the given query"""
    try:
        with DDGS() as search:
            results = list(search.text(query, max_results=3))
            return json.dumps(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"