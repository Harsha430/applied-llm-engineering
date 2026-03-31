# 🧬 Pokémon Graph RAG — Production-Ready Knowledge Retrieval

A production-grade **Graph Retrieval-Augmented Generation (Graph RAG)** system built with **Neo4j**, **LangChain**, and **Groq**. This project demonstrates how to extract structured data from a Pokémon Pokedex PDF and build a high-performance knowledge graph for natural language questioning.

---

## 🚀 Features

*   **PDF Extraction**: Automated parsing of `pokedex.pdf` using `PyMuPDF` and Regex to build a 300+ entry Knowledge Graph.
*   **Knowledge Representation**: Real-world modeling of Pokémon, Categories, and complex relationships (e.g., `PREYS_ON`).
*   **Semantic Alignment**: FAISS-powered semantic few-shot selection to "steer" the LLM toward accurate Cypher generation.
*   **Production Reliability**: Built-in Cypher validation, schema injection, and a dedicated failure-mode debugger.
*   **Interactive QA**: Seamless CLI interface powered by Groq's high-speed inference.

---

## 🏗️ Architecture

1.  **Ingestion**: `seed_db.py` parses raw PDF text into structured relationships and properties.
2.  **Seeding**: Neo4j Aura (Cloud) stores the resulting Pokémon nodes and predator-prey graph.
3.  **FAISS Vector Selection**: User queries are embedded via HuggingFace and used to retrieve the 3 most relevant Question-Cypher few-shots.
4.  **Chain Orchestration**: `GraphCypherQAChain` combines the schema, few-shots, and user query to generate executable Cypher.
5.  **LLM Resolution**: The graph results are passed back to the LLM for a natural, friendly response.

---

## 🛠️ Getting Started

### 📋 Prerequisites
*   Python 3.10+
*   Neo4j Aura Instance (Free Tier works great)
*   Groq API Key

### ⚙️ Installation
1.  **Clone the Repository**:
    ```bash
    git clone <your-repository-url>
    cd pokedex-graph-rag
    ```
2.  **Setup Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment**:
    Create a `.env` file from the provided context:
    ```env
    NEO4J_URI=neo4j+s://<your-instance-id>.databases.neo4j.io
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=<your-password>
    NEO4J_DATABASE=neo4j
    GROQ_API_KEY=<your-groq-api-key>
    GROQ_MODEL=llama-3.1-8b-instant
    ```

---

## 🔍 Usage

### Seeding the Knowledge Graph
```bash
python seed_db.py
```

### Interactive Explorer
```bash
python main.py
```

### Debug & Validation Demo
```bash
python debug_run.py
```

---

## 🗃️ Project Structure

*   `seed_db.py`: Extraction and seeding logic.
*   `main.py`: Interactive CLI entry point.
*   `debug_run.py`: Script to verify system robustness.
*   `graph_rag/`: Core library containing schema, few-shots, and chain logic.
*   `documentation.html`: Rich, premium design project documentation.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## 📄 License
This project is licensed under the MIT License.
