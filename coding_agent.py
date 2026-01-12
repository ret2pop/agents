import sys
import subprocess
import ast
import os
import base64
import re
from typing import TypedDict, List, Optional

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
    "coder": "qwen2.5-coder:14b",
    "tester": "qwen2.5-coder:14b", # Using coder model for writing tests too
    "verifier": "qwen3-vl:8b" 
}
MAX_RETRIES = 10 
SCRIPT_NAME = "temp_sandbox_script.py"
TEST_NAME = "temp_generated_tests.py"

# Initialize Rich Console
console = Console()

class SandboxState(TypedDict):
    objective: str        
    code: str             
    test_code: str        # NEW: Stores the generated test suite
    output: str           
    verification_error: Optional[str] 
    success: bool         
    iterations: int       

# --- HELPER FUNCTIONS ---

def clean_text(text: str) -> str:
    """Removes <think> tags from model output."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()

def extract_code(text: str) -> str:
    """Extracts code from markdown blocks."""
    if "```python" in text:
        text = text.split("```python")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return text.strip()

def run_llm(role: str, user_prompt: str, system_prompt: str = None, temperature=0.1) -> str:
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

def run_vision_llm(role: str, prompt_text: str, image_path: Optional[str], temperature=0.1) -> str:
    model_name = MODEL_CONFIG[role]
    llm = ChatOllama(model=model_name, temperature=temperature)
    
    content_parts = [{"type": "text", "text": prompt_text}]

    if image_path and os.path.exists(image_path):
        console.print(f"[dim][Vision Runner] 👁️  Loading image for analysis...[/dim]")
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        content_parts.append({
            "type": "image_url",
            "image_url": f"data:image/png;base64,{image_data}"
        })
    
    msg = HumanMessage(content=content_parts)
    try:
        response = llm.invoke([msg])
        return clean_text(response.content)
    except Exception as e:
        return f"Vision LLM Error: {str(e)}"

def get_third_party_imports(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [] 
    
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split('.')[0])

    std_lib = sys.stdlib_module_names if hasattr(sys, 'stdlib_module_names') else set(sys.builtin_module_names)
    third_party = [m for m in imported_modules if m not in std_lib]
    return list(third_party)

# --- NODES ---

def tester_node(state: SandboxState):
    """Generates a test suite BEFORE the code is written."""
    objective = state["objective"]
    
    # Only write tests once at the beginning
    if state["iterations"] > 0 and state.get("test_code"):
        return {}

    console.print(Panel(f"Generating Test Suite...", title="[bold cyan]Tester[/bold cyan]", border_style="cyan"))

    system_prompt = (
        "You are a QA Engineer specializing in TDD (Test Driven Development).\n"
        f"Write a `pytest` test file for a Python script named `{SCRIPT_NAME[:-3]}`.\n"
        "Requirements:\n"
        "1. Define the expected function signatures based on the user objective.\n"
        "2. Write comprehensive test cases (edge cases, happy paths).\n"
        f"3. Import the module using `import {SCRIPT_NAME[:-3]} as app`.\n"
        "4. Output ONLY the python code."
    )

    prompt = f"Objective: {objective}"
    
    with console.status("[cyan]Writing tests..."):
        response = run_llm("tester", prompt, system_prompt)
        test_code = extract_code(response)
    
    # Display the tests for the user
    console.print(Syntax(test_code, "python", theme="monokai", line_numbers=True))
    
    return {"test_code": test_code}

def coder_node(state: SandboxState):
    objective = state["objective"]
    test_code = state.get("test_code", "")
    iterations = state["iterations"]
    prev_output = state.get("output", "")
    verification_error = state.get("verification_error", None)
    
    system_instruction = (
        "Requirements:\n"
        "1. Output ONLY the code inside markdown blocks ```python ... ```\n"
        "2. Do not use 'input()'.\n"
        "3. ALWAYS set `matplotlib.use('Agg')` before importing pyplot.\n"
        "4. Save plots to 'output_plot.png'.\n"
        "5. IMPORTANT: Your code must be compatible with the provided Test Suite.\n"
        "6. Ensure you expose the functions/classes expected by the tests.\n"
        "7. Make sure there is a __name__ == \"__main__\" section that runs in a meaningfully useful way."
    )

    prompt = ""
    console.print(Panel(f"Coding (Attempt {iterations+1})", title="[bold green]Coder[/bold green]", border_style="green"))

    if iterations == 0:
        prompt = (
            f"Objective: {objective}\n\n"
            f"Here is the Test Suite you must pass:\n```python\n{test_code}\n```\n\n"
            f"Write the script `{SCRIPT_NAME}` to pass these tests and solve the objective.\n"
            f"{system_instruction}"
        )
        
    elif not verification_error:
        # Runtime/Test Failure
        prompt = (
            f"Goal: {objective}\n"
            f"The script failed during execution or testing:\n{prev_output}\n\n"
            f"Here are the tests:\n```python\n{test_code}\n```\n"
            "Fix the code to pass the tests and resolve the crash.\n"
            "Output ONLY the fixed code."
        )
        
    else:
        # Logic/Visual Failure
        prompt = (
            f"Goal: {objective}\n"
            f"The output was rejected by the Verifier:\n"
            f"Critique: {verification_error}\n\n"
            f"Previous Output: {prev_output}\n\n"
            "Modify the code to satisfy the critique.\n"
            "Output ONLY the fixed code."
        )
        
    with console.status("[green]Generating solution..."):
        response = run_llm("coder", prompt, temperature=0.2)
        clean_code = extract_code(response)
    
    return {"code": clean_code, "iterations": iterations + 1, "verification_error": None}

def dependency_manager_node(state: SandboxState):
    code = state["code"]
    # Also check tests for dependencies (e.g. pytest, numpy)
    test_code = state.get("test_code", "")
    
    required_packages = set(get_third_party_imports(code) + get_third_party_imports(test_code))
    # Ensure pytest is installed
    required_packages.add("pytest")
    
    if required_packages:
        console.print(f"[dim][DepManager] Checking packages: {required_packages}[/dim]")
        for pkg in required_packages:
            # Simple check to avoid spamming poetry add
            subprocess.run(["poetry", "add", pkg], capture_output=True, text=True)

    return {} 

def executor_node(state: SandboxState):
    code = state["code"]
    test_code = state.get("test_code", "")
    
    # 1. Write Source Code
    if os.path.exists("output_plot.png"):
        os.remove("output_plot.png")
    
    with open(SCRIPT_NAME, "w") as f:
        f.write(code)

    # 2. Write Test Code
    with open(TEST_NAME, "w") as f:
        f.write(test_code)

    console.print(Panel("Running Logic & Tests...", title="[bold yellow]Executor[/bold yellow]", border_style="yellow"))
    
    # 3. Run the Script (to generate plots/stdout)
    output_log = "--- SCRIPT EXECUTION ---\n"
    try:
        result = subprocess.run(
            [sys.executable, SCRIPT_NAME],
            capture_output=True, text=True, timeout=30
        )
        output_log += f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}\n"
        
        if result.returncode != 0:
            console.print("[red]❌ Script Crash[/red]")
            return {"output": output_log, "success": False}
            
    except Exception as e:
        return {"output": f"System Error: {str(e)}", "success": False}

    # 4. Run Tests (Pytest)
    output_log += "\n--- TEST EXECUTION ---\n"
    try:
        # Run pytest on the generated test file
        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", TEST_NAME],
            capture_output=True, text=True, timeout=30
        )
        output_log += test_result.stdout
        output_log += test_result.stderr
        
        if test_result.returncode != 0:
            console.print("[red]❌ Tests Failed[/red]")
            # Tests failed, so we treat this as a failure state for the Coder to fix
            return {"output": output_log, "success": False}
        else:
            console.print("[green]✅ Tests Passed[/green]")

    except Exception as e:
        output_log += f"\nTest Runner Error: {e}"
        return {"output": output_log, "success": False}

    # If we got here, Script ran AND Tests passed.
    if os.path.exists("output_plot.png"):
        output_log += "\n[System Note]: 'output_plot.png' was generated."
        
    return {"output": output_log, "success": True} 

def verifier_node(state: SandboxState):
    objective = state["objective"]
    code = state["code"]          # <--- NEW: Read the source code
    output = state["output"]
    image_path = "output_plot.png"
    
    if not state["success"]:
        return {}

    console.print(Panel("Verifying Output...", title="[bold purple]Verifier[/bold purple]", border_style="purple"))
    
    # Updated Prompt: Explicitly asks to check the main block for trivial inputs
    prompt_text = (
        f"User Objective: {objective}\n\n"
        f"--- SOURCE CODE ---\n{code}\n\n"
        f"--- EXECUTION LOGS ---\n{output}\n\n"
        "The automated tests PASSED. Now you must verify the LOGIC and RIGOR.\n\n"
        "CRITICAL CHECKS:\n"
        "1. Look at the `if __name__ == '__main__':` block at the bottom.\n"
        "2. Are the input parameters TRIVIAL? (e.g., angles set to 0.0, time set to 0, or mass set to 0).\n"
        "3. If the inputs are trivial/zeros, the simulation is meaningless even if it runs.\n"
        "4. Check the LOGIC of the program. Does it actually do what it is supposed to do?\n\n"
        "DECISION:\n"
        "- If inputs are trivial or the plot looks like a flat line: Reply 'FAILED: <explanation>'\n"
        "- If inputs look interesting and the plot looks valid: Reply 'PASSED'"
    )

    critique = run_vision_llm("verifier", prompt_text, image_path, temperature=0.1)
    
    if "PASSED" in critique:
        console.print("[bold green]✅ PASSED VERIFICATION[/bold green]")
        return {"verification_error": None, "success": True}
    else:
        error_msg = critique.replace("FAILED:", "").strip()
        console.print(f"[bold red]❌ REJECTED:[/bold red] {error_msg}")
        # Return success=False to force the Coder to retry
        return {"verification_error": error_msg, "success": False}

# --- GRAPH ---

def should_continue(state: SandboxState):
    if state["success"]:
        return "end"
    if state["iterations"] >= MAX_RETRIES:
        console.print("[bold red]Max retries reached.[/bold red]")
        return "end"
    return "retry"

workflow = StateGraph(SandboxState)

workflow.add_node("tester", tester_node)
workflow.add_node("coder", coder_node)
workflow.add_node("dependency_manager", dependency_manager_node)
workflow.add_node("executor", executor_node)
workflow.add_node("verifier", verifier_node)

# Flow: Tester -> Coder -> Dep -> Exec -> Verifier -> Loop
workflow.set_entry_point("tester")
workflow.add_edge("tester", "coder")
workflow.add_edge("coder", "dependency_manager")
workflow.add_edge("dependency_manager", "executor")
workflow.add_edge("executor", "verifier") 
workflow.add_conditional_edges("verifier", should_continue, {"end": END, "retry": "coder"})

coding_app = workflow.compile()

if __name__ == "__main__":
    console.print(Panel("VISUAL SCIENTIST AGENT\n[dim]Powered by TDD & Vision[/dim]", style="bold blue"))
    
    # Prompt Toolkit for nice input
    style = PtStyle.from_dict({'prompt': 'ansicyan bold'})
    goal = pt_prompt("Objective > ", style=style)
    
    coding_app.invoke({
        "objective": goal, 
        "code": "", 
        "test_code": "",
        "output": "", 
        "success": False, 
        "iterations": 0, 
        "verification_error": None
    })
