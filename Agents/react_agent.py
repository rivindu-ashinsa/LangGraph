from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import os
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages # Reducer function 

load_dotenv()

open_ai_api = os.getenv("OPENAI_KEY")

@tool
def add(a: int, b: int) -> int:
    """
    Add two integers together.
    """
    return a + b    

tools = [add]

llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=open_ai_api,
    openai_api_base="https://openrouter.ai/api/v1"
).bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Annotated with the reducer function


def model_call(state: AgentState) -> AgentState:
    """
    Process the messages in the state and return the updated state.
    """
    system_message = SystemMessage(
        content="You are a helpful assistant. Respond to the user's messages."
    )
    response = llm.invoke([system_message] + state['messages'])
    return {"messages" : [response]}

def should_continue(state: AgentState) -> bool:
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.content:
        return "end"

    else:
        return "continue"
    
tool_node = ToolNode(tools=tools)

graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
graph.add_node("tool_node", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent", 
    should_continue,
    {  
        "continue": "tool_node",
        "end": END
    }

)

graph.add_edge("tool_node", "agent")
app = graph.compile()


# from IPython.display import Image, display

# display(Image(app.get_graph().draw_mermaid_png()))

inputs = {"messages" : [("user", "what is 3 + 8")]}
def print_stream(stream): 
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

print_stream(app.stream(inputs, stream_mode="values"))