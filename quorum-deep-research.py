import re
import operator
import requests
import time
from bs4 import BeautifulSoup
from typing import Annotated, List, TypedDict

from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# --- RICH FORMATTING IMPORTS ---
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.style import Style

console = Console()

# --- CONFIGURATION ---
MODEL_CONFIG = {
    "planner": "qwen3:14b",      
    "researcher": "qwen3:14b",  
    "writer": "qwen3:14b",      
    "skeptic": "phi4-reasoning:plus", 
    "editor": "ministral-3:14b"           
}

# --- TOOLING ---
search_tool = DuckDuckGoSearchResults(num_results=3)

def scrape_text(url: str):
    """Fetches and cleans text from a URL for deep reading."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"Failed to load page (Status: {response.status_code})"
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer", "iframe"]):
            script.extract()
            
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:10000] 
    except Exception as e:
        return f"Error reading page: {str(e)}"

# --- HELPER FUNCTIONS ---

def clean_text(text: str) -> str:
    """Removes <think> tags."""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned.strip()

def run_llm(role: str, user_prompt: str, system_prompt: str = None, temperature=0.1) -> str:
    """Centralized LLM runner with auto-cleaning."""
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

# --- STATE ---
class AgentState(TypedDict):
    topic: str
    research_plan: List[str]
    research_notes: Annotated[List[str], operator.add] 
    current_draft: str
    critiques: List[str]
    loop_count: int

# --- NODES ---

def planner_node(state: AgentState):
    loop_count = state.get("loop_count", 0)
    topic = state["topic"]
    critiques = state.get("critiques", [])
    
    # Visual Header
    console.print(Panel(f"Iteration: {loop_count+1}\nTopic: {topic}", title="[bold cyan]PLANNER[/bold cyan]", border_style="cyan"))
    
    with console.status("[bold cyan]Generating Research Plan...", spinner="dots"):
        if loop_count == 0:
            prompt = (
                f"Topic: {topic}\n"
                "Generate 3 highly specific search queries to gather deep technical or historical context.\n"
                "Return ONLY the queries as a list, separated by newlines."
            )
        else:
            critique_text = "\n".join(critiques)
            prompt = (
                f"Topic: {topic}\n"
                f"Address these gaps identified by the Skeptic: {critique_text}\n"
                "Generate 2 NEW search queries to fill these specific gaps.\n"
                "Return ONLY the queries as a list, separated by newlines."
            )

        response = run_llm("planner", prompt)
        plan = [line.strip().replace("- ", "").replace("* ", "") for line in response.split('\n') if line.strip()]
    
    console.print(f"[bold cyan]Research Plan:[/bold cyan]")
    for i, query in enumerate(plan, 1):
        console.print(f"  [cyan]{i}.[/cyan] {query}")
    console.print("") # spacing

    return {"research_plan": plan}

def researcher_node(state: AgentState):
    plan = state["research_plan"]
    new_notes = []
    
    console.print(Panel(f"Executing {len(plan)} search queries", title="[bold yellow]RESEARCHER[/bold yellow]", border_style="yellow"))
    
    for query in plan:
        with console.status(f"[yellow]Searching: {query}...", spinner="earth"):
            search_results = search_tool.invoke(query)
            
            # Selector
            selector_prompt = (
                f"Query: {query}\n"
                f"Search Results: {search_results}\n\n"
                "Analyze the results. Return the single best URL that is likely to contain detailed, long-form information.\n"
                "Return ONLY the URL. Nothing else."
            )
            best_url = run_llm("planner", selector_prompt).strip()
        
        if "http" not in best_url:
            console.print(f"  [red]No valid URL found for:[/red] {query}")
            new_notes.append(f"### Findings for '{query}':\n{search_results}\n")
            continue

        console.print(f"  [green]Reading:[/green] {best_url}")
        
        with console.status("[yellow]Scraping & Extracting...", spinner="bouncingBall"):
            page_content = scrape_text(best_url)
            
            extraction_prompt = (
                f"Query: {query}\n"
                f"Source URL: {best_url}\n"
                f"Page Content (Truncated): {page_content}\n\n"
                "Extract comprehensive, detailed findings, statistics, and arguments from this text.\n"
                "Capture specific numbers and dates.\n"
                "Format: [Fact] (Source: URL)"
            )
            summary = run_llm("researcher", extraction_prompt)
            new_notes.append(f"### Deep Dive on '{query}':\n{summary}\n")

    return {"research_notes": new_notes}

def writer_node(state: AgentState):
    topic = state["topic"]
    flat_notes = "\n".join(state["research_notes"])
    current_draft = state.get("current_draft", "")
    loop_count = state.get("loop_count", 0)
    
    console.print(Panel("Synthesizing Report", title="[bold magenta]WRITER[/bold magenta]", border_style="magenta"))
    system = "You are a report generation engine. Output Markdown only. No conversational text."
    
    with console.status("[magenta]Writing Draft...", spinner="aesthetic"):
        if loop_count == 0:
            prompt = (
                f"Write a definitive, long-form report on: {topic}.\n"
                f"Research Notes:\n{flat_notes}\n\n"
                "Requirements:\n"
                "1. Deeply detailed, academic tone.\n"
                "2. Use H2 and H3 headers.\n"
                "3. Cite sources inline [1] matching the URLs in the notes.\n"
                "4. Output ONLY the report."
            )
        else:
            prompt = (
                f"Expand the report on {topic}.\n"
                f"Current Draft:\n{current_draft}\n\n"
                f"New Research Notes:\n{flat_notes}\n\n"
                "Integrate the new findings. Make the report longer and more comprehensive.\n"
                "Output the FULL updated report."
            )

        content = run_llm("writer", prompt, system_prompt=system, temperature=0.3)
    
    console.print(f"[magenta]Draft Generated[/magenta] ({len(content)} chars)")
    return {"current_draft": content}

def quorum_node(state: AgentState):
    console.print(Panel("Reviewing Draft", title="[bold red]QUORUM DELIBERATION[/bold red]", border_style="red"))
    draft = state["current_draft"]
    critiques = []
    
    # --- ACTIVE SKEPTIC ---
    with console.status("[red]Skeptic: Hunting for weak claims...", spinner="grenade"):
        identify_prompt = (
            f"Read this draft:\n{draft[:4000]}...\n\n"
            "Identify the single most questionable empirical claim that lacks citation or seems biased.\n"
            "Generate a search query to FIND COUNTER-EVIDENCE for this claim.\n"
            "Output ONLY the search query."
        )
        skeptic_query = run_llm("skeptic", identify_prompt, temperature=0.1).strip().replace('"', '')
    
    console.print(f"  [red bold]Skeptic Audit:[/red bold] Checking '{skeptic_query}'")
    skeptic_evidence = search_tool.invoke(skeptic_query)
    
    with console.status("[red]Skeptic: Formulating critique...", spinner="grenade"):
        critique_prompt = (
            f"Draft Text:\n{draft[:4000]}...\n\n"
            f"External Evidence Found via '{skeptic_query}':\n{skeptic_evidence}\n\n"
            "Write a critique of the draft explicitly referencing this external evidence.\n"
            "Point out where the draft contradicts reality or lacks nuance based on the search results."
        )
        critiques.append(run_llm("skeptic", critique_prompt, temperature=0.3))
    
    # --- EDITOR (Structure) ---
    with console.status("[blue]Editor: Reviewing structure...", spinner="bouncingBar"):
        prompt_editor = f"Critique the structure. Is the introduction strong? Are the headers logical?\n{draft[:6000]}..."
        critiques.append(run_llm("editor", prompt_editor, temperature=0.1))
    
    return {"critiques": critiques}

def refiner_node(state: AgentState):
    draft = state["current_draft"]
    critiques = "\n".join(state["critiques"])
    # Pass notes so the refiner can verify/rebuild the bibliography
    flat_notes = "\n".join(state["research_notes"])
    
    console.print(Panel("Applying Critiques", title="[bold green]REFINER[/bold green]", border_style="green"))
    
    system = "You are a specialized academic editor."
    
    # STRICTER PROMPT
    prompt = (
        f"Original Draft:\n{draft}\n\n"
        f"Critiques to Apply:\n{critiques}\n\n"
        f"Reference Notes (for citations):\n{flat_notes}\n\n"
        "Instructions:\n"
        "1. Rewrite the draft to address the critiques.\n"
        "2. CRITICAL: You must PRESERVE or RE-INSERT inline citations (e.g., [1], [2]) for every specific claim (numbers, dates, study results).\n"
        "3. DO NOT summarize the citations at the end. You must append a full 'References' section listing the actual URLs/Titles from the Reference Notes.\n"
        "4. If a claim is made without a citation, check the Reference Notes and add the matching citation.\n"
        "5. Output the full, final polished report."
    )
    
    with console.status("[green]Refining Text & Fixing Citations...", spinner="dots12"):
        # You might want to boost temperature slightly (0.2 -> 0.3) or use a smarter model here if possible
        content = run_llm("writer", prompt, system_prompt=system, temperature=0.25)
        
    return {"current_draft": content, "loop_count": state["loop_count"] + 1}

# --- GRAPH CONSTRUCTION ---

def should_continue(state: AgentState):
    if state["loop_count"] < 2: 
        console.print(Panel("Critique feedback loop triggered.", style="bold yellow"))
        return "loop"
    console.print(Panel("Research limit reached. Finalizing.", style="bold green"))
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("quorum", quorum_node)
workflow.add_node("refiner", refiner_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "quorum")
workflow.add_edge("quorum", "refiner")
workflow.add_conditional_edges("refiner", should_continue, {"loop": "planner", "end": END})

app = workflow.compile()

# --- RUN ---
if __name__ == "__main__":
    console.print(Panel.fit("[bold white]Deep Research Agent[/bold white]", style="blue"))
    topic = console.input("[bold yellow]Enter research topic:[/bold yellow] ")
    
    inputs = {"topic": topic, "research_plan": [], "research_notes": [], "current_draft": "", "critiques": [], "loop_count": 0}
    
    final_state = app.invoke(inputs)
    
    with open("deep_research_report.md", "w", encoding="utf-8") as f:
        f.write(final_state["current_draft"])
    
    console.print(f"\n[bold green]Done![/bold green] Report saved to 'deep_research_report.md'")
