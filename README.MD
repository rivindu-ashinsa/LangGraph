# LangGraph Examples

This repository contains a collection of Jupyter notebooks and Python scripts demonstrating the use of [LangGraph](https://github.com/langchain-ai/langgraph) for building stateful, composable, and interactive agent workflows. The examples cover a variety of graph-based agent patterns, including simple greeting bots, conditional logic, looping, multi-step workflows, and integration with large language models (LLMs) via OpenAI and OpenRouter.

## Repository Structure

```
.
├── .env
├── .gitignore
├── email_to_bob.txt
├── LangGraph.ipynb
├── README.MD
├── requirements.txt
├── test.py
├── Agents/
│   ├── agent1.py
│   ├── agent2.py
│   ├── logging.txt
│   └── react_agent.py
├── Project/
│   └── drafter.py
└── .ipynb_checkpoints/
    ├── app-checkpoint.py
    ├── LangGraph-checkpoint.ipynb
    └── logging-checkpoint.txt
```

- **LangGraph.ipynb**: Main notebook with LangGraph examples.
- **Agents/**: Example agent scripts.
- **Project/drafter.py**: Document drafting agent example.
- **.env**: Environment variables (API keys).
- **requirements.txt**: Python dependencies.
- **test.py**: Utility or test script.
- **email_to_bob.txt**: Example output/data file.
- **.ipynb_checkpoints/**: Jupyter notebook/script checkpoints.

## Key Examples

### 1. Basic Graphs

- Simple greeting node using a `TypedDict` state.
- Defining nodes, entry/exit points, and invoking the graph.

### 2. Conditional and Multi-step Graphs

- Graphs with multiple nodes and edges.
- Conditional routing and multi-step workflows.

### 3. Looping Graphs

- Looping until a condition is met (e.g., number guessing).

### 4. LLM Integration

- Using `langchain_openai.ChatOpenAI` with LangGraph.
- Passing API keys via `.env` and using OpenRouter.
- Conversational memory in agent state.

### 5. Project Agents

- Document drafting and updating with tool integration.

## Setup

1. **Clone the repository**
   ```
   git clone <repo-url>
   cd <repo-directory>
   ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file:
   ```
   OPENAI_KEY=your-openai-or-openrouter-key
   ```

4. **Run the notebooks**

   Open `LangGraph.ipynb` in Jupyter or VS Code and run the cells.

## Example: Running a Graph

```python
from langgraph.graph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    message: str

def greeting_node(state: AgentState) -> AgentState:
    state['message'] = "Hello, " + state['message']
    return state

graph = StateGraph(AgentState)
graph.add_node("greeter", greeting_node)
graph.set_entry_point("greeter")
graph.set_finish_point("greeter")
app = graph.compile()

result = app.invoke({"message": "World"})
print(result['message'])  # Output: Hello, World
```

## LLM Agent Example

See the section in `LangGraph.ipynb` for a conversational agent using OpenAI/OpenRouter:

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
import os

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    openai_api_key=os.getenv("OPENAI_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(response.content)
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()
```

## License

This repository is for educational and demonstration purposes.

---
For more details and advanced usage, see the code and notebooks