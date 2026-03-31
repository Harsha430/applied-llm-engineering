import os
from operator import itemgetter
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, trim_messages, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory

# Load environment variables (like GROQ_API_KEY)
load_dotenv()

# Dictionary to store session histories in-memory
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def get_token_count(messages) -> int:
    return sum(len(m.content) // 4 for m in messages)

# Initialize ChatGroq LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# Initialize a message trimmer
# max_tokens will limit the number of tokens retained in the conversation history.
trimmer = trim_messages(
    max_tokens=2000, 
    strategy="last",
    token_counter=get_token_count,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Create the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer concisely."),
    MessagesPlaceholder(variable_name="messages"),
])

# Create the chain: format the messages through the trimmer, then prompt, then LLM
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | llm
)

# Wrap the chain with RunnableWithMessageHistory
memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

def chat_with_memory(session_id: str, human_input: str):
    """
    Sends a message to the LLM within a specific session, preserving history.
    """
    response = memory_chain.invoke(
        {"messages": [HumanMessage(content=human_input)]},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content

def get_session_messages(session_id: str):
    """
    Returns the current full message history for a given session.
    """
    history = get_session_history(session_id)
    return history.messages
