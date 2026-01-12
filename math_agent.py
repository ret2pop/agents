import sys
import subprocess
import os
import re
from typing import TypedDict, Literal, Optional

# --- UI & EDITING ---
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.styles import Style as PtStyle

# --- LANGCHAIN / AI ---
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# --- CONFIGURATION ---
MODEL_CONFIG = {
    "theorist": "qwen2.5-coder:14b",   # Good at reasoning/LaTeX
    "formalizer": "qwen2.5-coder:14b", # Good at syntax
    "arbiter": "qwen2.5-coder:14b"     # Good at classifying errors
}
MAX_RETRIES = 6
LEAN_FILE = "proof_attempt.lean"

console = Console()

class MathState(TypedDict):
    objective: str           # "Prove sqrt(2) is irrational"
    
    # Informal Layer
    informal_proof: str      # LaTeX/English sketch
    
    # Formal Layer
    lean_code: str           # The .lean file content
    compiler_output: str     # Stderr from Lean compiler
    
    # Meta Layer
    error_type: Optional[Literal["SYNTAX", "LOGIC"]] # Routing decision
    critique: str            # Arbiter's explanation
    success: bool
    iterations: int

# --- HELPER FUNCTIONS ---

def clean_text(text: str) -> str:
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()

def extract_code(text: str, lang="lean") -> str:
    """Extracts code from markdown blocks."""
    pattern = r"```" + lang + r"(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try generic block
    if "```" in text:
        return text.split("```")[1].strip()
    return text.strip()

def run_llm(role: str, user_prompt: str, system_prompt: str = None, temperature=0.2) -> str:
    model_name = MODEL_CONFIG[role]
    llm = ChatOllama(model=model_name, temperature=temperature)
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=user_prompt))
    try:
        response = llm.invoke(messages)
        return clean_text(response.content)
    except Exception as e:
        return f"LLM Error: {str(e)}"

# --- NODES ---

def theorist_node(state: MathState):
    """The Creative Mathematician: Sketches the proof."""
    objective = state["objective"]
    critique = state.get("critique", "")
    iterations = state["iterations"]
    
    console.print(Panel(f"Thinking (Iteration {iterations+1})", title="[bold blue]Theorist[/bold blue]", border_style="blue"))

    system_prompt = (
        "You are an expert Mathematician. Your goal is to provide a rigorous INFORMAL proof sketch.\n"
        "1. State necessary definitions.\n"
        "2. State the theorem clearly.\n"
        "3. Provide a step-by-step proof in natural language + LaTeX.\n"
        "4. DO NOT write Lean code yet."
    )

    if iterations == 0:
        prompt = f"Objective: {objective}"
    else:
        # We are looping back because the Logic was flawed
        prompt = (
            f"Objective: {objective}\n"
            f"Previous Attempt Failed.\n"
            f"Arbiter Critique: {critique}\n\n"
            "Please restructure your proof strategy to avoid this logical pitfall."
        )

    with console.status("[blue]Sketching proof strategy..."):
        informal_proof = run_llm("theorist", prompt, system_prompt)
    
    console.print(Markdown(informal_proof))
    return {"informal_proof": informal_proof, "iterations": iterations + 1}

def formalizer_node(state: MathState):
    """The Translator: Converts English/LaTeX to Lean4."""
    objective = state["objective"]
    informal_proof = state["informal_proof"]
    prev_code = state.get("lean_code", "")
    compiler_output = state.get("compiler_output", "")
    critique = state.get("critique", "")
    error_type = state.get("error_type", None)
    
    console.print(Panel("Formalizing...", title="[bold cyan]Formalizer[/bold cyan]", border_style="cyan"))

    system_prompt = (
        "You are a Lean4 Expert. Translate the informal proof into valid Lean4 code.\n"
        "1. Use `import Mathlib` if needed.\n"
        "2. Ensure all types and definitions are strictly declared.\n"
        "3. Output ONLY the Lean code inside ```lean ... ``` blocks."
    )

    prompt = ""
    if error_type == "SYNTAX":
        # Fix existing code
        prompt = (
            f"The previous Lean code had a syntax/tactic error.\n"
            f"Error Log:\n{compiler_output}\n\n"
            f"Arbiter Tip: {critique}\n\n"
            f"Original Code:\n```lean\n{prev_code}\n```\n"
            "Fix the code."
        )
    else:
        # Fresh translation from Theorist's new plan
        prompt = (
            f"Objective: {objective}\n"
            f"Informal Proof Strategy:\n{informal_proof}\n\n"
            "Translate this into a complete `.lean` file."
        )

    with console.status("[cyan]Writing Lean4 code..."):
        response = run_llm("formalizer", prompt, system_prompt)
        lean_code = extract_code(response, "lean")
        
    console.print(Syntax(lean_code, "lean", theme="monokai", line_numbers=True))
    return {"lean_code": lean_code}

def kernel_node(state: MathState):
    """The Execution Environment: Runs the Lean Compiler."""
    lean_code = state["lean_code"]
    
    # Write to file
    with open(LEAN_FILE, "w") as f:
        f.write(lean_code)
        
    console.print(Panel("Compiling...", title="[bold yellow]Kernel[/bold yellow]", border_style="yellow"))
    
    try:
        # Run Lean
        # NOTE: This assumes 'lean' is in PATH. 
        # If using 'lake', command might be different.
        result = subprocess.run(
            ["lean", LEAN_FILE],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode == 0:
            console.print("[bold green]✅ Proof Verified (Compiled Successfully)[/bold green]")
            return {"compiler_output": output, "success": True}
        else:
            console.print("[bold red]❌ Verification Failed[/bold red]")
            # Print last few lines of error for context
            error_preview = "\n".join(output.splitlines()[:10])
            console.print(f"[dim]{error_preview} ...[/dim]")
            return {"compiler_output": output, "success": False}
            
    except FileNotFoundError:
        return {"compiler_output": "Error: 'lean' executable not found. Please install Lean4.", "success": False}
    except Exception as e:
        return {"compiler_output": f"System Error: {str(e)}", "success": False}

def arbiter_node(state: MathState):
    """The Judge: Decides if it's a Syntax error or a Logic error."""
    if state["success"]:
        return {}
        
    compiler_output = state["compiler_output"]
    lean_code = state["lean_code"]
    informal_proof = state["informal_proof"]
    
    console.print(Panel("Analyzing Error...", title="[bold purple]Arbiter[/bold purple]", border_style="purple"))
    
    system_prompt = (
        "You are an expert Debugger for Lean4.\n"
        "Analyze the error log and decide if the failure is due to:\n"
        "1. SYNTAX: The math is likely correct, but the code/tactics are wrong (e.g., 'unknown identifier', 'type mismatch').\n"
        "2. LOGIC: The proof strategy itself is flawed or the goal is unprovable (e.g., 'goals not accomplished', 'contradiction').\n\n"
        "Output Format:\n"
        "TYPE: <SYNTAX or LOGIC>\n"
        "CRITIQUE: <Short explanation of what to fix>"
    )
    
    user_prompt = (
        f"Lean Code:\n```lean\n{lean_code}\n```\n"
        f"Compiler Output:\n{compiler_output}\n"
    )
    
    response = run_llm("arbiter", user_prompt, system_prompt)
    
    # Parse output
    error_type = "SYNTAX" # Default
    critique = response
    
    if "TYPE: LOGIC" in response or "TYPE: LOGIC" in response:
        error_type = "LOGIC"
    elif "TYPE: SYNTAX" in response:
        error_type = "SYNTAX"
        
    critique = response.replace("TYPE: SYNTAX", "").replace("TYPE: LOGIC", "").strip()
    
    console.print(f"[bold]{error_type} ERROR DETECTED[/bold]")
    console.print(f"[dim]{critique}[/dim]")
    
    return {"error_type": error_type, "critique": critique}

# --- GRAPH ---

def router(state: MathState):
    if state["success"]:
        return "end"
    if state["iterations"] >= MAX_RETRIES:
        console.print("[bold red]Max retries reached.[/bold red]")
        return "end"
        
    # The Arbiter decides where to loop back to
    if state["error_type"] == "SYNTAX":
        return "formalizer" # Go fix the code
    elif state["error_type"] == "LOGIC":
        return "theorist"   # Go fix the math
    
    return "end" # Fallback

workflow = StateGraph(MathState)

workflow.add_node("theorist", theorist_node)
workflow.add_node("formalizer", formalizer_node)
workflow.add_node("kernel", kernel_node)
workflow.add_node("arbiter", arbiter_node)

# Flow
workflow.set_entry_point("theorist")
workflow.add_edge("theorist", "formalizer")
workflow.add_edge("formalizer", "kernel")
workflow.add_edge("kernel", "arbiter")

workflow.add_conditional_edges(
    "arbiter",
    router,
    {
        "end": END,
        "formalizer": "formalizer",
        "theorist": "theorist"
    }
)

math_app = workflow.compile()

if __name__ == "__main__":
    console.print(Panel("TRIANGLE OF TRUTH\n[dim]Theorist ↔ Formalizer ↔ Kernel[/dim]", style="bold blue"))
    
    style = PtStyle.from_dict({'prompt': 'ansiblue bold'})
    goal = pt_prompt("Theorem to Prove > ", style=style)
    
    math_app.invoke({
        "objective": goal, 
        "informal_proof": "", 
        "lean_code": "", 
        "compiler_output": "", 
        "success": False, 
        "iterations": 0,
        "error_type": None
    })
