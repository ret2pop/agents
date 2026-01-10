import operator
from typing import Annotated, List, TypedDict, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# --- CONFIGURATION ---

# If you have multiple models installed (e.g., mistral, gemma, llama3),
# you can assign different models to different roles for true diversity.
# For simplicity, we default all to 'llama3'.
MODEL_CONFIG = {
    "drafter": "qwen3-vl:8b",
    "skeptic": "qwen3-vl:8b",
    "structuralist": "qwen3-vl:8b"
}

# --- STATE DEFINITION ---

class QuorumState(TypedDict):
    question: str
    current_answer: str
    critiques: List[str]  # Stores feedback from the quorum
    iteration: int        # Tracks how many refinement loops we've done

# --- NODES (The Agents) ---

def initial_drafter_node(state: QuorumState):
    """Generates the first draft of the answer."""
    print(f"--- DRAFTING INITIAL ANSWER (Model: {MODEL_CONFIG['drafter']}) ---")
    
    llm = ChatOllama(model=MODEL_CONFIG['drafter'], temperature=0.7)
    
    prompt = (
        "You are an expert assistant. Provide a detailed, preliminary answer "
        "to the following question. Be comprehensive but open to refinement.\n\n"
        f"Question: {state['question']}"
    )
    
    response = llm.invoke(prompt)
    return {"current_answer": response.content, "iteration": 0}

def quorum_critique_node(state: QuorumState):
    """
    Runs the 'Quorum': Multiple personas critique the current answer.
    """
    print(f"--- QUORUM DELIBERATION (Iteration {state['iteration'] + 1}) ---")
    
    current_answer = state["current_answer"]
    question = state["question"]
    critiques = []

    # --- Persona 1: The Skeptic ---
    # Looks for factual errors, logical gaps, or assumptions.
    print("   > The Skeptic is reviewing...")
    skeptic_llm = ChatOllama(model=MODEL_CONFIG['skeptic'], temperature=0.3)
    skeptic_prompt = (
        "You are 'The Skeptic'. Your job is to find flaws, logical fallacies, "
        "missing context, or weak arguments in the provided answer.\n"
        "Be harsh but fair. If the answer is good, acknowledge it but find at least one improvement.\n\n"
        f"Question: {question}\n"
        f"Draft Answer: {current_answer}\n\n"
        "Critique:"
    )
    res_skeptic = skeptic_llm.invoke(skeptic_prompt)
    critiques.append(f"Skeptic's Feedback: {res_skeptic.content}")

    # --- Persona 2: The Structuralist ---
    # Looks for formatting, flow, clarity, and tone.
    print("   > The Structuralist is reviewing...")
    struct_llm = ChatOllama(model=MODEL_CONFIG['structuralist'], temperature=0.3)
    struct_prompt = (
        "You are 'The Structuralist'. Focus ONLY on clarity, structure, formatting, "
        "and flow. Is the answer easy to read? Does it use headers effectively?\n\n"
        f"Draft Answer: {current_answer}\n\n"
        "Critique:"
    )
    res_struct = struct_llm.invoke(struct_prompt)
    critiques.append(f"Structuralist's Feedback: {res_struct.content}")

    return {"critiques": critiques}

def refiner_node(state: QuorumState):
    """
    Synthesizes the critiques and rewrites the answer.
    """
    print("--- REFINING ANSWER ---")
    
    question = state["question"]
    current_answer = state["current_answer"]
    critiques_text = "\n\n".join(state["critiques"])
    
    llm = ChatOllama(model=MODEL_CONFIG['drafter'], temperature=0.5)
    
    refine_prompt = (
        "You are the Lead Editor. You have an original draft and a set of critiques "
        "from a panel of experts. Your goal is to rewrite the draft to incorporate "
        "this feedback and create the 'Final Golden Answer'.\n\n"
        f"Original Question: {question}\n"
        f"Current Draft: {current_answer}\n\n"
        f"--- Panel Feedback ---\n{critiques_text}\n"
        "----------------------\n\n"
        "Please provide the rewritten, improved answer below. Do not include a preamble about the changes, just the answer."
    )
    
    response = llm.invoke(refine_prompt)
    
    # Increment iteration count
    return {
        "current_answer": response.content, 
        "iteration": state["iteration"] + 1,
        "critiques": [] # Clear critiques for next round if needed (though we usually stop)
    }

# --- EDGES & LOGIC ---

def should_continue(state: QuorumState):
    """
    Decides if we have reached the 'Chain of Thought' quorum limit.
    """
    MAX_ITERATIONS = 2  # How many times to refine. 1 or 2 is usually sufficient.
    
    if state["iteration"] < MAX_ITERATIONS:
        return "critique"
    return END

# --- GRAPH CONSTRUCTION ---

workflow = StateGraph(QuorumState)

workflow.add_node("drafter", initial_drafter_node)
workflow.add_node("quorum", quorum_critique_node)
workflow.add_node("refiner", refiner_node)

# Entry
workflow.set_entry_point("drafter")

# Flow
workflow.add_edge("drafter", "quorum")
workflow.add_edge("quorum", "refiner")

# Loop Decision
workflow.add_conditional_edges(
    "refiner",
    should_continue,
    {
        "critique": "quorum", # Loop back for another round of feedback
        END: END              # Finish
    }
)

app = workflow.compile()

# --- EXECUTION ---

if __name__ == "__main__":
    prompt = input("Enter your prompt for the Quorum: ")
    
    inputs = {
        "question": prompt,
        "current_answer": "",
        "critiques": [],
        "iteration": 0
    }
    
    print(f"\nSubmitting to the Quorum: {prompt}\n")
    
    # We invoke the graph to run to completion
    final_state = app.invoke(inputs)
    
    print("\n" + "="*40)
    print("FINAL REFINED ANSWER")
    print("="*40 + "\n")
    print(final_state["current_answer"])
