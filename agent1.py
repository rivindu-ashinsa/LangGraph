from langgraph.graph import StateGraph, END, START  
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

import os
open_ai_api = os.getenv("OPENAI_KEY")



llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=open_ai_api,
    openai_api_base="https://openrouter.ai/api/v1")


class AgentState(TypedDict):
    messages: List[HumanMessage]    


def process(state: AgentState) -> AgentState:
    """
    Process the messages in the state and return the updated state.
    """
    response = llm.invoke(state["messages"])
    print(response.content)
    # Here you can add any processing logic you need
    return state


graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


user_input = input("Enter your message: ")
agent.invoke({
    "messages": [HumanMessage(content=user_input)]})