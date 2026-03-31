"""
graph_rag/cypher_chain.py
─────────────────────────────────────────────────────────────────────────────
GraphCypherQAChain — the core of the Graph RAG system
─────────────────────────────────────────────────────────────────────────────

FLOW:
  User Question
       │
       ▼
  [Cypher Generation Prompt]
      ┌─────────────────────────────────────┐
      │  System: Schema + Few-shot examples  │
      │  Human:  User question               │
      └─────────────────────────────────────┘
       │
       ▼
  Groq LLM (llama-3.1-8b-instant)
       │
       ▼
  Generated Cypher query
       │
       ▼
  validate_cypher=True → checks for RETURN clause
       │
       ▼
  Execute against Neo4j Aura
       │
       ▼
  Raw graph results
       │
       ▼
  [QA Prompt] → LLM formats natural-language answer
       │
       ▼
  Final Answer

KEY CONFIGURATION FLAGS:
  validate_cypher=True          → prevents RETURN-clause bugs
  return_intermediate_steps=True → exposes generated Cypher + raw results
  allow_dangerous_requests=True  → required by LangChain for write protection
  top_k=10                       → max results returned from Neo4j
  verbose=True                   → logs intermediate steps
"""

import os

from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from loguru import logger

from graph_rag.schema import SCHEMA_STRING
from graph_rag.few_shots import FewShotSelector

load_dotenv()

# ── Environment ────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# ── Prompt templates ───────────────────────────────────────────────────────

CYPHER_GENERATION_TEMPLATE = """\
You are an expert Neo4j Cypher query generator for a Pokémon knowledge graph.

{schema}

{few_shots}

STRICT RULES — follow every rule or your query will fail:
1. Generate ONLY a valid Cypher query. No explanations, no markdown, no code fences.
2. Every query MUST include a RETURN clause — this is mandatory.
3. Use ONLY the node labels, relationship types, and property names defined in the schema above.
4. For case-insensitive name searches, use: toLower(p.name) = toLower('value')
5. Prefer DISTINCT to avoid duplicate results.
6. Default LIMIT is 10 unless the user asks for all results.
7. Do NOT generate CREATE, DELETE, MERGE, SET, or REMOVE statements.

User Question: {question}

Cypher Query:"""

QA_TEMPLATE = """\
You are a helpful Pokémon expert who answers questions from graph database results.

Given the user's question and the graph database results below, write a clear,
friendly, and complete answer. If the results are empty, say so honestly.

Question: {question}
Graph Results: {context}

Answer:"""


def connect_to_neo4j() -> Neo4jGraph:
    """Establishes and returns a Neo4jGraph connection from env vars."""
    logger.info(f"Connecting to Neo4j …")
    graph = Neo4jGraph()
    # Refresh schema from live database (live schema injection)
    graph.refresh_schema()
    logger.success("Neo4j connected — schema refreshed ✓")
    return graph


def build_cypher_chain(
    k_few_shots: int = 3,
    top_k: int = 10,
    verbose: bool = True,
) -> tuple[GraphCypherQAChain, Neo4jGraph]:
    """
    Builds and returns the GraphCypherQAChain.

    Args:
        k_few_shots : number of semantically selected few-shot examples
        top_k       : max Neo4j results to return per query
        verbose     : whether to log intermediate steps

    Returns:
        (chain, graph) — the chain and the raw Neo4jGraph (for direct queries)
    """
    # 1. Connect to Neo4j
    graph = connect_to_neo4j()

    # 2. Initialize LLM
    logger.info(f"Loading Groq LLM: {GROQ_MODEL}")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,       # deterministic Cypher generation
        max_tokens=1024,
    )

    # 3. Few-shot selector
    few_shot_selector = FewShotSelector(k=k_few_shots)

    # 4. Build Cypher-generation prompt (backed by semantic selection)
    # Prefix includes the schema definition
    prefix = (
        "You are an expert Neo4j Cypher generator for a Pokémon knowledge graph.\n\n"
        "{schema}\n\n"
        "=== FEW-SHOT CYPHER EXAMPLES (most relevant) ===\n"
    )
    # Suffix includes rules and the actual user question
    suffix = (
        "\n=================================================\n\n"
        "STRICT RULES:\n"
        "1. Return ONLY a valid Cypher query. No markdown, no code fences.\n"
        "2. Every query MUST have a RETURN clause — mandatory.\n"
        "3. Use ONLY labels/properties defined in the schema.\n"
        "4. Use toLower() for case-insensitive name matching.\n"
        "5. Use DISTINCT to avoid duplicates.\n"
        "6. Default LIMIT 10 unless asked for all.\n"
        "7. Do NOT generate write operations (CREATE/DELETE/MERGE/SET).\n\n"
        "Question: {question}\n"
        "Cypher Query:"
    )

    # Build the FewShotPromptTemplate using semantic similarity selector
    cypher_prompt = few_shot_selector.build_few_shot_prompt(
        prefix=prefix,
        suffix=suffix,
        input_variables=["schema", "question"]
    )

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=QA_TEMPLATE,
    )

    # 5. Build GraphCypherQAChain
    logger.info("Building GraphCypherQAChain …")
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        validate_cypher=True,              # FIX: catches missing RETURN clause
        return_intermediate_steps=True,    # exposes generated Cypher + raw results
        allow_dangerous_requests=True,     # required flag for LangChain safety
        top_k=top_k,
        verbose=verbose,
    )
    logger.success("GraphCypherQAChain ready ✓")
    return chain, graph


def run_query(chain: GraphCypherQAChain, question: str) -> dict:
    """
    Runs a natural-language question through the chain.

    Returns a dict with:
      - question    : original question
      - cypher      : generated Cypher query
      - raw_results : list of records from Neo4j
      - answer      : final natural-language answer
    """
    logger.info(f"Question: {question}")
    result = chain.invoke({"query": question})

    # Extract intermediate steps
    intermediate = result.get("intermediate_steps", [])
    cypher_query = ""
    raw_results  = []

    if intermediate:
        # Step 0: generated Cypher
        if len(intermediate) > 0 and "query" in intermediate[0]:
            cypher_query = intermediate[0]["query"]
        # Step 1: raw Neo4j results
        if len(intermediate) > 1 and "context" in intermediate[1]:
            raw_results = intermediate[1]["context"]

    return {
        "question":    question,
        "cypher":      cypher_query,
        "raw_results": raw_results,
        "answer":      result.get("result", ""),
    }
