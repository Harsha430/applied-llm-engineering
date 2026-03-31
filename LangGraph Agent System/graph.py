from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import planner, retriever, evaluator, responder

def build_graph():
    """
    Build the LangGraph state machine.
    """
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("planner", planner)
    workflow.add_node("retriever", retriever)
    workflow.add_node("evaluator", evaluator)
    workflow.add_node("responder", responder)
    
    # Set Entry Point
    workflow.set_entry_point("planner")
    
    # Add Edges
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "evaluator")
    
    # Define Conditional Edge Logic
    def should_retry(state: AgentState):
        """
        Decision function for the conditional edge.
        """
        evaluation = state.get("evaluation", {})
        retry_count = state.get("retry_count", 0)
        
        # If satisfactory or we've retried too many times, go to responder
        if evaluation.get("satisfactory", False) or retry_count >= 2:
            return "responder"
        
        # Otherwise, retry the loop
        return "retry"

    # Add Conditional Edge from Evaluator
    workflow.add_conditional_edges(
        "evaluator",
        should_retry,
        {
            "responder": "responder",
            "retry": "planner"
        }
    )
    
    # Final edge
    workflow.add_edge("responder", END)
    
    return workflow.compile()
