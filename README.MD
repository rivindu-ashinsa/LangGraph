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

- **[`LangGraph.ipynb`](LangGraph.ipynb)**: Main notebook containing step-by-step LangGraph examples, from basic graphs to advanced LLM-powered agents and project agents.
- **[`Agents/`](Agents/)**: Example agent scripts, including [`agent1.py`](Agents/agent1.py), [`agent2.py`](Agents/agent2.py), and [`react_agent.py`](Agents/react_agent.py).
- **[`Project/drafter.py`](Project/drafter.py)**: Example of a document drafting agent using LangGraph and LLM tools.
- **.env**: Environment variables (e.g., API keys for OpenAI/OpenRouter).
- **requirements.txt**: Python dependencies.
- **.ipynb_checkpoints/**: Jupyter notebook and script checkpoints (auto-generated).
- **email_to_bob.txt**: Example output or data file.
- **test.py**: Additional test or utility script.

## Key Examples

### 1. Basic Graphs

- Simple greeting node using a `TypedDict` state.
- How to define nodes, set entry/exit points, and invoke the graph.

### 2. Conditional and Multi-step Graphs

- Graphs with multiple nodes and edges.
- Conditional routing based on state (e.g., sum vs. product, addition vs. subtraction).
- Multi-step workflows with state accumulation.

### 3. Looping Graphs

- Examples of looping until a condition is met (e.g., random number generation, number guessing game).

### 4. LLM Integration

- Using `langchain_openai.ChatOpenAI` with LangGraph for conversational agents.
- Passing API keys via [`.env`](.env) and using OpenRouter as a backend.
- Maintaining conversational memory in the agent state.

### 5. Project Agents

- Document drafting and updating with tool integration (see [`Project/drafter.py`](Project/drafter.py)).

## Setup

1. **Clone the repository**
   ```sh
   git clone <repo-url>
   cd <repo-directory>
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
   *(You may need to create `requirements.txt` with packages like `langgraph`, `langchain-core`, `langchain-openai`, `python-dotenv`, `jupyter`, etc.)*

3. **Set up environment variables**

   Create a [`.env`](.env) file with your API keys:
   ```
   OPENAI_KEY=your-openai-or-openrouter-key
   ```

4. **Run the notebooks**

   Open [`LangGraph.ipynb`](LangGraph.ipynb) in Jupyter or VS Code and run the cells.

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

See the section in [`LangGraph.ipynb`](LangGraph.ipynb) for a conversational agent using OpenAI/OpenRouter:

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

For more details and advanced usage,