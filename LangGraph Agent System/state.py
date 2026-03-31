from typing import Annotated, TypedDict, List, Dict
import operator

class AgentState(TypedDict):
    query: str
    plan: str
    retrieved_docs: Annotated[List[Dict], operator.add]
    evaluation: Dict
    retry_count: int
    answer: str
