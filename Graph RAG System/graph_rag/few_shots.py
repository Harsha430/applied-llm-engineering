"""
graph_rag/few_shots.py
─────────────────────────────────────────────────────────────────────────────
Semantic Few-Shot Example Selector for Cypher Generation
─────────────────────────────────────────────────────────────────────────────

WHY FEW-SHOT ALIGNMENT?
────────────────────────
Raw LLMs often generate syntactically broken or schema-mismatched Cypher.
Providing question → Cypher examples in the prompt "steers" the model toward
the correct pattern.

WHY SEMANTIC SELECTION (vs. injecting ALL examples)?
──────────────────────────────────────────────────────
• Injecting all 15+ examples wastes tokens and may confuse the model.
• Instead we embed each example question and the incoming user question,
  then pick the 3 most similar examples using FAISS vector search.
• Result: always relevant, token-efficient, and accurate.

FLOW:
  User question
       │
       ▼
  HuggingFace embed (all-MiniLM-L6-v2)
       │
       ▼
  FAISS similarity search over example questions
       │
       ▼
  Top-K most similar (question, cypher, answer) triples
       │
       ▼
  Formatted string injected into Cypher-gen prompt
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from loguru import logger

from graph_rag.embeddings import get_embeddings

# ── Ground-truth few-shot examples ────────────────────────────────────────
# Format: question  →  correct Cypher  →  expected answer shape
CYPHER_EXAMPLES: list[dict] = [
    # ── Simple attribute lookup ─────────────────────────────────────────
    {
        "question": "What is Pidgeot's height?",
        "cypher": (
            "MATCH (p:Pokemon) WHERE toLower(p.name) = 'pidgeot' "
            "RETURN p.name AS name, p.height AS height"
        ),
        "answer": "Pidgeot is 4’ 11\" tall.",
    },
    {
        "question": "What is the weight of Bulbasaur?",
        "cypher": (
            "MATCH (p:Pokemon) WHERE toLower(p.name) = 'bulbasaur' "
            "RETURN p.name AS name, p.weight AS weight"
        ),
        "answer": "Bulbasaur weighs 15 lbs.",
    },
    {
        "question": "Show me the description of Metapod.",
        "cypher": (
            "MATCH (p:Pokemon) WHERE toLower(p.name) = 'metapod' "
            "RETURN p.description AS description"
        ),
        "answer": "Metapod is a Cocoon Pokémon that is vulnerable to attack while its shell is soft.",
    },

    # ── Category queries ────────────────────────────────────────────────────
    {
        "question": "What category does Charizard belong to?",
        "cypher": (
            "MATCH (p:Pokemon {{name: 'Charizard'}})-[:BELONGS_TO_CATEGORY]->(c:Category) "
            "RETURN c.name AS category"
        ),
        "answer": "Charizard belongs to the Flame category.",
    },
    {
        "question": "List all Bird Pokémon.",
        "cypher": (
            "MATCH (p:Pokemon)-[:BELONGS_TO_CATEGORY]->(c:Category) WHERE toLower(c.name) = 'bird' "
            "RETURN p.name AS name ORDER BY p.name"
        ),
        "answer": "Bird Pokémon include Pidgey, Pidgeotto, and Pidgeot.",
    },

    # ── Predator/Prey queries ───────────────────────────────────────────────
    {
        "question": "Which Pokémon prey on Magikarp?",
        "cypher": (
            "MATCH (p:Pokemon)-[:PREYS_ON]->(prey:Pokemon) WHERE toLower(prey.name) = 'magikarp' "
            "RETURN p.name AS predator"
        ),
        "answer": "Pidgeot is known to prey on Magikarp.",
    },
    {
        "question": "What does Pidgeot hunt?",
        "cypher": (
            "MATCH (p:Pokemon {{name: 'Pidgeot'}})-[:PREYS_ON]->(prey:Pokemon) "
            "RETURN prey.name AS prey"
        ),
        "answer": "Pidgeot hunts Magikarp.",
    },

    # ── Multi-attribute / search ───────────────────────────────────────
    {
        "question": "Identify Pokémon with a height of 4' 11\".",
        "cypher": (
            "MATCH (p:Pokemon) WHERE p.height = \"4’ 11\\\"\" "
            "RETURN p.name AS name"
        ),
        "answer": "Pidgeot has a height of 4' 11\".",
    },
]

# ── PromptTemplate for each individual example ─────────────────────────────
EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["question", "cypher", "answer"],
    template=(
        "Question: {question}\n"
        "Cypher:   {cypher}\n"
        "Answer:   {answer}"
    ),
)


class FewShotSelector:
    """
    Wraps SemanticSimilarityExampleSelector for Cypher few-shot retrieval.

    Usage:
        selector = FewShotSelector(k=3)
        examples_str = selector.format_examples("What type is Gengar?")
        # → formatted string of 3 most relevant examples
    """

    def __init__(self, k: int = 3):
        self.k = k
        self._selector: SemanticSimilarityExampleSelector | None = None
        logger.info(f"FewShotSelector initialized (k={k})")

    def _build_selector(self) -> SemanticSimilarityExampleSelector:
        """Lazily builds FAISS index over example questions."""
        logger.info("Building FAISS index over few-shot examples …")
        selector = SemanticSimilarityExampleSelector.from_examples(
            examples=CYPHER_EXAMPLES,
            embeddings=get_embeddings(),
            vectorstore_cls=FAISS,
            k=self.k,
            input_keys=["question"],   # embed only the question field
        )
        logger.success(f"FAISS index built with {len(CYPHER_EXAMPLES)} examples")
        return selector

    @property
    def selector(self) -> SemanticSimilarityExampleSelector:
        if self._selector is None:
            self._selector = self._build_selector()
        return self._selector

    def get_examples(self, question: str) -> list[dict]:
        """Returns the top-k most relevant examples for the given question."""
        return self.selector.select_examples({"question": question})

    def format_examples(self, question: str) -> str:
        """Returns a formatted string of the top-k examples, ready for prompt injection."""
        examples = self.get_examples(question)
        lines = ["=== FEW-SHOT CYPHER EXAMPLES (most relevant) ==="]
        for i, ex in enumerate(examples, 1):
            lines.append(f"\n[Example {i}]")
            lines.append(f"  Question : {ex['question']}")
            lines.append(f"  Cypher   : {ex['cypher']}")
            lines.append(f"  Answer   : {ex['answer']}")
        lines.append("\n=================================================")
        return "\n".join(lines)

    def build_few_shot_prompt(
        self,
        prefix: str,
        suffix: str,
        input_variables: list[str],
    ) -> FewShotPromptTemplate:
        """
        Builds a LangChain FewShotPromptTemplate backed by semantic selection.
        Use this when constructing the Cypher-generation prompt dynamically.
        """
        return FewShotPromptTemplate(
            example_selector=self.selector,
            example_prompt=EXAMPLE_PROMPT,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
        )
