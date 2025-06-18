import os
import langgraph
from langgraph.graph import StateGraph, END
from langchain.llms import OpenAI

# Set LM Studio endpoint
os.environ["OPENAI_API_KEY"] = "lm-studio"
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"

# Load your local LLM (Mistral or any model running in LM Studio)
llm = OpenAI(model_name="TheBloke/Mistral-7B-Instruct-v0.1-GGUF", temperature=0)

# Step 1: Define the state (memory structure)
def starter_node(state):
    print("🟢 Start Node")
    return {"question": "What is the capital of India?"}

# Step 2: Define a node that uses LLM to answer
def responder_node(state):
    print("🤖 Responder Node")
    question = state["question"]
    answer = llm(question)
    return {"question": question, "answer": answer}

# Step 3: Create the graph
graph = StateGraph()

# Add nodes
graph.add_node("start", starter_node)
graph.add_node("responder", responder_node)

# Add edges
graph.set_entry_point("start")
graph.add_edge("start", "responder")
graph.add_edge("responder", END)

# Compile
app = graph.compile()

# Step 4: Run the graph
final_output = app.invoke({})
print("✅ Final Output:\n", final_output)

