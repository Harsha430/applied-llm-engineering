from loguru import logger
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph
from graph_rag.schema import SCHEMA_STRING

def demonstrate_return_error(llm: ChatGroq):
    """
    Simulates a 'missing RETURN' error by providing a weak prompt
    that doesn't emphasize the mandatory RETURN clause.
    """
    logger.warning("DEMONSTRATION: Missing RETURN clause hallucination")
    
    # A prompt that might NOT force a RETURN clause
    weak_prompt = PromptTemplate(
        input_variables=["question"],
        template="Translate this question into Neo4j Cypher:\nQuestion: {question}\nCypher:"
    )
    
    question = "List all Pokemon."
    chain = weak_prompt | llm
    prediction = chain.invoke({"question": question})
    
    # LLM often returns "MATCH (p:Pokemon)" without the RETURN p.name
    logger.info(f"Question: {question}")
    logger.info(f"Generated Cypher (may be broken): {prediction.content}")
    
    if "RETURN" not in prediction.content.upper():
        logger.error("FAILURE REPRODUCED: The generated Cypher is missing a RETURN clause.")
    else:
        logger.success("The LLM included a RETURN clause despite the weak prompt.")

def demonstrate_schema_hallucination(llm: ChatGroq):
    """
    Simulates a 'schema mismatch' error where the LLM uses labels
    not present in the provided schema.
    """
    logger.warning("DEMONSTRATION: Schema Property Hallucination")
    
    # We provide the schema but ask a question about a property that DOESN'T exist
    # (e.g., 'attack' stat which we didn't extract from this PDF)
    question = "Which Pokemon has the highest attack?"
    
    prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=(
            "Use the schema below to write Cypher:\n{schema}\n\n"
            "Question: {question}\nCypher Query:"
        )
    )
    
    chain = prompt | llm
    prediction = chain.invoke({"schema": SCHEMA_STRING, "question": question})
    
    logger.info(f"Question: {question}")
    logger.info(f"Generated Cypher: {prediction.content}")
    
    # Check if LLM hallucinated 'attack' property
    if "attack" in prediction.content.lower():
        logger.error("FAILURE REPRODUCED: The LLM hallucinated an 'attack' property not in schema.")
    else:
        logger.success("The LLM correctly stayed within schema limits.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")
    demonstrate_return_error(llm)
    print("-" * 30)
    demonstrate_schema_hallucination(llm)
