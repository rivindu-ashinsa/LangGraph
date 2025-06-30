from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import os
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages # Reducer function 

load_dotenv()

open_ai_api = os.getenv("OPENAI_KEY")

document_content = ""

class Agentstate(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # Annotated with the reducer function


@tool
def update(content: str)  -> str:
    """updates the document with provided content"""
    global document_content
    document_content = content
    return f"Document updated with content: {content}"

@tool 
def save(filename: str) -> str:
    """saves the current document to a text file and finish the process
    Args: 
        filename : name of the text file
    
    """

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
            print("Document saved successfully.")
        return f"Document saved to {filename}"  
    except Exception as e:
        return f"Error saving document: {str(e)}"
    

tools = [update, save]

llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=open_ai_api,
    openai_api_base="https://openrouter.ai/api/v1"
).bind_tools(tools)


def model_call(state: Agentstate) -> Agentstate:
    system_prompt = SystemMessage(f""" 
        you are drafter, a helpful assistant that helps users to draft documents. you are going to help the user to update and modify documents. 
        - if the user wants to update the document, use the update tool to update the document with the provided content.
        - if the user wants to save the document, use the save tool to save the document to a text file.
        - make sure to always show the current document state after modification.

        the current document content is : {document_content}
        """)
    
    if not state['messages']:
        user_input = "I'm ready to help you update the ducument. what would you like to create"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("What would you like to do with the document? ")
        print(f"\nUser : {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state['messages']) + [user_message]

    response = llm.invoke(all_messages)
    print(f"\nAssistant : {response.content}")

    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_message = ToolMessage(
            content=response.content,
            tool_calls=response.tool_calls
        )

    return {"messages": list(state['messages']) + [user_message, response]}

def should_continue(state: Agentstate) -> str:
    """Determine if we should end the conversation or continue."""
    messages = state['messages']
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
        "saved" in message.content.lower() and
        "document" in message.content.lower()):
            return "end"
    return "continue"

def print_messages(messages):
    if not messages:
        return
    for message in messages[-3]:
        print(f"\nTOOL RESULT : {message.content}")

graph = StateGraph(Agentstate)
graph.add_node("agent",model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("agent")
graph.add_edge("agent", "tool_node")
graph.add_conditional_edges(
    "tool_node", 
    should_continue,
    {  
        "continue": "agent",
        "end": END
    }
)

graph.compile()