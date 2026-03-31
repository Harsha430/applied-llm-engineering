# 🛠️ Developer Assistant MCP Server

A powerful, Claude-compatible **Model Context Protocol (MCP)** server that provides advanced repository analysis, automated documentation, debugging, and project brainstorming tools.

Built with **Python**, **FastMCP**, and **Groq (Llama 3.3-70B)**.

---

## 🌟 Key Features

| Tool | Capability |
| :--- | :--- |
| **`analyze_github_repo`** | Fetches README & file tree to extract tech stack, strengths, and suggestions. |
| **`generate_readme`** | Creates high-quality `README.md` files based on summary and features. |
| **`debug_code`** | Analyzes code + tracebacks to provide root-cause explanations and fixes. |
| **`suggest_projects`** | Brainstorms unique, challenging project ideas tailored to your skill set. |

---

## 🚀 Deployment (Prefect Horizon)

This server is optimized for deployment on **[Prefect Horizon](https://horizon.prefect.io/)**, a managed hosting platform for FastMCP.

### Prerequisites

1.  A **Groq API Key** (for LLM logic).
2.  A **GitHub API Key** (optional, for higher rate limits on `analyze_github_repo`).

### Horizon Setup

1.  **Connect Repo**: Sign in to [Horizon](https://horizon.prefect.io/) with GitHub and select this repository.
2.  **Environment Variables**:
    -   `GROQ_API_KEY`: Your Groq API key.
    -   `GITHUB_API_KEY`: (Optional) Your GitHub personal access token.
3.  **Entrypoint**:
    -   Set the entrypoint to: **`main.py:mcp`**
4.  **Deploy**: Click **Deploy Server**. Horizon will automatically build and host your server.

---

## 💻 Local Usage

### 1. Installation & Environment

Create a virtual environment and install only the essential dependencies:

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
GITHUB_API_KEY=your_github_token
```

### 3. Run the Server

```bash
python main.py
```

The server will start listening on `stdio` by default, which is perfect for **Claude Desktop**.

---

## 🔌 Connecting to Claude Desktop

Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dev-assistant": {
      "command": "python",
      "args": ["c:/absolute/path/to/main.py"],
      "env": {
        "GROQ_API_KEY": "your-key-here",
        "GITHUB_API_KEY": "your-key-here"
      }
    }
  }
}
```

---

## 🛠️ Built With

- **[Model Context Protocol](https://modelcontextprotocol.io)** - Open standard for LLM tools.
- **[Groq](https://groq.com)** - Ultra-fast inference engine for Llama 3.3.
- **[FastMCP](https://github.com/jlowin/fastmcp)** - High-level Python SDK for MCP.
- **[LangChain](https://langchain.com)** - For LLM orchestration.
