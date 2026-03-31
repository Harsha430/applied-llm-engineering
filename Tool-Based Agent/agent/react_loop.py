from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from tools.calculator import add, sub, mul, div
from tools.search import search_web
from tools.api_tool import get_weather

def run_agent(query: str):
    # Initialize the LLM (requires GROQ_API_KEY in .env)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    # List of all available tools
    tools = [add, sub, mul, div, search_web, get_weather]

    # ReAct prompt template
    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)

    # Use create_react_agent purely from langchain_classic
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    return agent_executor.invoke({"input": query})
