from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
import os
from dotenv import load_dotenv
load_dotenv()


open_ai_api = os.getenv("OPENAI_KEY")

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=open_ai_api,
    openai_api_base="https://openrouter.ai/api/v1")


def process(state: AgentState) -> AgentState:
    """
    Process the messages in the state and return the updated state.
    """
    response = llm.invoke(state['messages'])
    print(f"AI : {response.content}")
    state['messages'].append(AIMessage(content=response.content))
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversational_history = []

user_input = input("Enter: ")

while user_input != "exit":
    conversational_history.append(HumanMessage(content=user_input))
    result = agent.invoke({
        "messages": conversational_history})
    result['messages'] = conversational_history
    

with open("logging.txt",'w') as file:
    file.write(f"Your Log !\n")
    for message in conversational_history:
        if isinstance(message, HumanMessage):
            file.write(f"you : {str(message.content)}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI : {str(message.content)}\n\n")
    file.write("End of Conversation")

print("History saved !")