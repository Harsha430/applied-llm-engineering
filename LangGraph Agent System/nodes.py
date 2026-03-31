import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from state import AgentState

load_dotenv()

# Initialize LLM and Tools
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
search_tool = TavilySearchResults(max_results=3)

def planner(state: AgentState):
    """
    Planner Node: Decomposes the initial query into a search strategy.
    Increments retry count for tracking.
    """
    query = state["query"]
    retry_count = state.get("retry_count", 0) + 1
    
    prompt = f"""
    You are an expert search planner. Your task is to decompose the following query into a concise search plan.
    Identify the key entities and the missing information needed to provide a comprehensive answer.
    
    Query: {query}
    
    Provide a search plan as a set of bullet points.
    """
    response = llm.invoke([SystemMessage(content="You are a strategic search analyst."), HumanMessage(content=prompt)])
    
    return {"plan": response.content, "retry_count": retry_count}

def retriever(state: AgentState):
    """
    Retriever Node: Executes search queries based on the plan.
    """
    query = state["query"]
    plan = state["plan"]
    
    # We use the original query enriched by the plan for actual search
    search_query = f"{query} {plan}"
    print(f"DEBUG: Running search for: {search_query}")
    
    # Ensure results is a list of dicts
    results = search_tool.invoke({"query": search_query})
    
    if isinstance(results, str):
        print("DEBUG: Tool returned a string, converting to list of one dict.")
        results = [{"content": results}]
    elif not isinstance(results, list):
        print(f"DEBUG: Tool returned unexpected type {type(results)}, wrapping in list.")
        results = [results]
    
    print(f"DEBUG: Retrieved {len(results)} docs.")
    return {"retrieved_docs": results}

def evaluator(state: AgentState):
    """
    Evaluator Node: Assesses whether the retrieved documents are sufficient to answer the query.
    """
    query = state["query"]
    retrieved_docs = state["retrieved_docs"]
    
    docs_text = "\n\n".join([f"Source {i+1}: {doc['content']}" for i, doc in enumerate(retrieved_docs)])
    
    prompt = f"""
    You are a quality evaluator. Your task is to evaluate if the retrieved information is sufficient to answer the user query.
    
    User Query: {query}
    
    Retrieved Information (Context):
    {docs_text}
    
    Respond in JSON format with two keys:
    - "satisfactory": (boolean) True if the info is sufficient, False otherwise.
    - "explanation": (string) Reasoning for your decision.
    """
    
    # Using JSON mode or parsing simple response (Groq supports JSON)
    response = llm.invoke([SystemMessage(content="Evaluate information quality. Return JSON only."), HumanMessage(content=prompt)])
    
    # Simple JSON extraction (improving robustness later if needed)
    import json
    try:
        eval_result = json.loads(response.content)
    except:
        # Fallback if LLM doesn't output perfect JSON
        eval_result = {"satisfactory": "True" in response.content, "explanation": response.content}
        
    return {"evaluation": eval_result}

def responder(state: AgentState):
    """
    Responder Node: Synthesizes the final answer using the retrieved documents.
    """
    query = state["query"]
    retrieved_docs = state["retrieved_docs"]
    
    docs_text = "\n\n".join([f"Source {i+1}: {doc['content']}" for i, doc in enumerate(retrieved_docs)])
    
    prompt = f"""
    You are a knowledgeable assistant. Using the context provided below, answer the user query comprehensively and accurately.
    
    User Query: {query}
    
    Context:
    {docs_text}
    
    Answer:
    """
    response = llm.invoke([SystemMessage(content="Draft a professional response based on context."), HumanMessage(content=prompt)])
    return {"answer": response.content}
