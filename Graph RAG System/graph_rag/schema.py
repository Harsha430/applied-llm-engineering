"""
graph_rag/schema.py
─────────────────────────────────────────────────────────────────────────────
Graph Schema Definition — Pokémon Knowledge Graph
─────────────────────────────────────────────────────────────────────────────

WHY SCHEMA INJECTION MATTERS
─────────────────────────────
Without schema context the LLM has no idea what labels, relationship types,
or property names exist in the graph.  It will hallucinate (e.g. MATCH (m:Movie)
when the label is actually :Pokemon).

We inject a hand-written schema string directly into the Cypher-generation
prompt so the model stays grounded to the actual graph structure.

GRAPH DOMAIN: Pokémon Universe
─────────────────────────────
Nodes:
  (:Pokemon)   — individual Pokémon species
  (:Type)      — elemental type (Fire, Water, …)
  (:Ability)   — special abilities (Blaze, Overgrow, …)
  (:Move)      — moves/attacks (Flamethrower, Tackle, …)
  (:Trainer)   — trainer characters (Ash, Misty, …)
  (:Region)    — geographic regions (Kanto, Johto, …)

Relationships:
  (:Pokemon)-[:HAS_TYPE]->(:Type)
  (:Pokemon)-[:HAS_ABILITY]->(:Ability)
  (:Pokemon)-[:KNOWS_MOVE]->(:Move)
  (:Pokemon)-[:EVOLVES_INTO]->(:Pokemon)
  (:Pokemon)-[:FOUND_IN]->(:Region)
  (:Trainer)-[:OWNS]->(:Pokemon)
  (:Trainer)-[:FROM]->(:Region)
  (:Move)-[:IS_TYPE]->(:Type)
"""

# ── Node labels with their key properties ─────────────────────────────────
NODE_LABELS: dict[str, list[str]] = {
    "Pokemon":  ["name", "num", "description", "height", "weight"],
    "Category": ["name"],
}

# ── Relationship types with direction info ─────────────────────────────────
RELATIONSHIP_TYPES: dict[str, dict] = {
    "BELONGS_TO_CATEGORY": {"from": "Pokemon", "to": "Category", "properties": []},
    "PREYS_ON":            {"from": "Pokemon", "to": "Pokemon",  "properties": []},
}

# ── Human-readable schema string injected into every LLM prompt ───────────
SCHEMA_STRING = """
=== POKÉMON KNOWLEDGE GRAPH — DATABASE SCHEMA ===

NODE LABELS AND PROPERTIES:
• (:Pokemon)  [ name: string, num: int, description: string, height: string, weight: string ]
• (:Category) [ name: string ]

RELATIONSHIP TYPES (direction matters!):
• (:Pokemon)-[:BELONGS_TO_CATEGORY]->(:Category)
• (:Pokemon)-[:PREYS_ON]->(:Pokemon)

IMPORTANT RULES FOR CYPHER GENERATION:
1. Always use EXACT label names above (case-sensitive). Use :Pokemon NOT :pokemon or :POKEMON.
2. Always use EXACT property names above.
3. Every query MUST end with a RETURN clause.
4. Use DISTINCT when listing multiple results to avoid duplicates.
5. For name searches, use toLower() for case-insensitive matching if needed.
6. Limit results with LIMIT 10 unless asked for all.
7. Do NOT use labels or properties not listed in this schema.
=================================================================
"""


def get_schema_string() -> str:
    """Returns the schema string for injection into prompts."""
    return SCHEMA_STRING
