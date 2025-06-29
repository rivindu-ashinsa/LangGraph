from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

import os
open_ai_api = os.getenv("OPENAI_KEY")

import openai

client = openai.OpenAI(
    api_key=open_ai_api,
    base_url="https://openrouter.ai/api/v1"
)

response = client.chat.completions.create(
    model="mistralai/mistral-7b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a fun fact about space."}
    ]
)

print(response.choices[0].message.content)
