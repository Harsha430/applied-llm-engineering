import os
import httpx
import base64
from typing import Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("Developer Assistant")

# Initialize LLM (using Groq)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")

async def get_github_repo_info(repo_url: str):
    """Fetches README and file structure from GitHub repository."""
    # Extract owner and repo name from URL
    try:
        parts = repo_url.rstrip("/").split("/")
        owner, repo = parts[-2], parts[-1]
    except Exception:
        return None, "Invalid GitHub URL. Use format: https://github.com/owner/repo"

    headers = {}
    if GITHUB_API_KEY:
        headers["Authorization"] = f"token {GITHUB_API_KEY}"
    
    async with httpx.AsyncClient(headers=headers) as client:
        # Get README
        readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
        readme_content = ""
        r_readme = await client.get(readme_url)
        if r_readme.status_code == 200:
            readme_data = r_readme.json()
            readme_content = base64.b64decode(readme_data["content"]).decode("utf-8")
        
        # Get file tree (limit to top level for simplicity/rate limits)
        tree_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
        r_tree = await client.get(tree_url)
        files = []
        if r_tree.status_code == 200:
            files = [item["name"] for item in r_tree.json()]

    return {
        "owner": owner,
        "repo": repo,
        "readme": readme_content[:5000],  # Truncate if too long
        "files": files
    }, None

@mcp.tool()
async def analyze_github_repo(repo_url: str) -> str:
    """
    Analyzes a GitHub repository to extract tech stack, strengths, weaknesses, and suggestions.
    
    Args:
        repo_url: The full URL of the GitHub repository.
    """
    repo_info, error = await get_github_repo_info(repo_url)
    if error:
        return error
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert software architect. Analyze the following GitHub repository information:
    Repository: {owner}/{repo}
    Files: {files}
    README snippet: {readme}
    
    Provide a structured analysis including:
    1. Tech Stack (languages, frameworks, libraries used).
    2. Strengths of the project.
    3. Potential Weaknesses or areas for improvement.
    4. Actionable Suggestions for the developer.
    """)
    
    chain = prompt | llm | StrOutputParser()
    analysis = await chain.ainvoke({
        "owner": repo_info["owner"],
        "repo": repo_info["repo"],
        "files": ", ".join(repo_info["files"]),
        "readme": repo_info["readme"]
    })
    
    return analysis

@mcp.tool()
async def generate_readme(repo_summary: str, features: str, tech_stack: Optional[str] = None) -> str:
    """
    Generates a professional README.md content based on repository details.
    
    Args:
        repo_summary: A brief description of what the project does.
        features: Key features of the project.
        tech_stack: Optional list of technologies used.
    """
    prompt = ChatPromptTemplate.from_template("""
    Create a high-quality, professional README.md for a project with the following details:
    Summary: {repo_summary}
    Features: {features}
    Tech Stack: {tech_stack}
    
    The README should include sections for Introduction, Features, Getting Started, Usage, and Contributing. 
    Use clear, concise markdown.
    """)
    
    chain = prompt | llm | StrOutputParser()
    readme_md = await chain.ainvoke({
        "repo_summary": repo_summary,
        "features": features,
        "tech_stack": tech_stack or "Not specified"
    })
    
    return readme_md

@mcp.tool()
async def debug_code(code: str, error: str) -> str:
    """
    Analyzes code and an error message to provide a bug explanation and a suggested fix.
    
    Args:
        code: The source code snippet containing the bug.
        error: The error message or traceback.
    """
    prompt = ChatPromptTemplate.from_template("""
    You are a senior debugger. Analyze the code and the error provided below.
    Code:
    ```python
    {code}
    ```
    
    Error:
    {error}
    
    Provide:
    1. Bug Explanation: What is causing the error?
    2. Suggested Fix: The corrected code snippet.
    """)
    
    chain = prompt | llm | StrOutputParser()
    debug_result = await chain.ainvoke({
        "code": code,
        "error": error
    })
    
    return debug_result

@mcp.tool()
async def suggest_projects(skills: str) -> str:
    """
    Suggests advanced project ideas based on the user's skill set.
    
    Args:
        skills: A list of programming languages or technologies the user knows.
    """
    prompt = ChatPromptTemplate.from_template("""
    The user has the following skills: {skills}.
    Suggest 3 advanced, unique, and challenging project ideas that would help them level up.
    For each project, include:
    - Title
    - Concept
    - Why it's challenging
    - Suggested tech stack (integrating their skills and new ones)
    """)
    
    chain = prompt | llm | StrOutputParser()
    suggestions = await chain.ainvoke({"skills": skills})
    
    return suggestions

if __name__ == "__main__":
    # Start the server (defaults to stdio transport for local use with Claude Desktop)
    mcp.run()
