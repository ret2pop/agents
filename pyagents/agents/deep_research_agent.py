import os
import sys
import uuid
from typing import Annotated, List, TypedDict

from langchain_community.tools import DuckDuckGoSearchResults
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# --- RICH FORMATTING IMPORTS ---
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.style import Style
from rich.box import ROUNDED

# --- PROMPT TOOLKIT ---
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style as PromptStyle
except ImportError:
    print("Error: This script requires 'prompt_toolkit' for line editing.")
    print("Please run: pip install prompt_toolkit")
    sys.exit(1)

from pyagents.config import DEEP_RESEARCH_MODEL_CONFIG
from pyagents.utils import run_llm
from pyagents.tools.search_tool import WebScout # Or use DuckDuckGoSearchResults directly as before?
# The original code used DuckDuckGoSearchResults and a custom `scrape_text` function.
# Let's try to reuse `WebScout` concepts or stick to the original implementation but refactored.
# The original implementation used DuckDuckGoSearchResults + requests/bs4 for scraping.
# I'll stick to the original logic for now but use the shared `run_llm`.

import requests
from bs4 import BeautifulSoup
import re

console = Console()

# --- TOOLING ---
search_tool = DuckDuckGoSearchResults(num_results=10)
max_loop = 2 # Max loops PER SECTION

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
def pretty_print_queries(queries: List[str], topic: str):
    """
    Renders a stylized panel for research queries to the console.
    """
    content = ""
    for i, query in enumerate(queries, 1):
        content += f"[bold cyan]{i}.[/bold cyan] [white]{query}[/white]\n"

    console.print(Panel(
        content.strip(),
        title=f"[bold yellow]Research Plan: {topic}[/bold yellow]",
        border_style="cyan",
        box=ROUNDED,
        padding=(1, 2),
        expand=False
    ))

# --- STATE ---
class AgentState(TypedDict):
    # GLOBAL STATE
    main_topic: str
    section_plan: List[str]       # The Todo List
    completed_sections: List[str] # The finished text blocks
    current_section_idx: int      # Where we are in the list
    final_report: str             # The output

    # LOCAL STATE (Per Section)
    topic: str                    # Current sub-topic
    research_plan: List[str]
    research_notes: List[str]
    current_draft: str
    critiques: List[str]
    loop_count: int

# --- NODES ---

def global_planner_node(state: AgentState):
    """Generates the Table of Contents."""
    main_topic = state["main_topic"]

    console.print(Panel(f"Mapping: {main_topic}", title="[bold white]GLOBAL PLANNER[/bold white]", style="white"))

    prompt = (
        f"Topic: {main_topic}\n"
        "Create a logical outline for a comprehensive report on this topic.\n"
        "Return a list of 4 to 6 distinct section headers (e.g., 'Historical Context', 'Technical Implementation').\n"
        "Do NOT include an Introduction or Conclusion in this list (I will add those automatically).\n"
        "CRITICAL RULE: Your outline must be NEUTRAL and INVESTIGATIVE.\n"
        " - BAD: 'The Benefits of X' (Assumes there are benefits)\n"
        " - GOOD: 'Analysis of Impact of X' (Allows for positive or negative findings)\n"
        " - BAD: 'How X Solves Y' (Assumes it solves it)\n"
        " - GOOD: 'Evaluation of X as a Solution for Y'\n"
        "Return ONLY the list of headers, separated by newlines."
    )

    response = run_llm(DEEP_RESEARCH_MODEL_CONFIG["global_planner"], prompt)
    sections = [line.strip().replace("- ", "").replace("* ", "") for line in response.split('\n') if line.strip()]

    console.print("[white]Global Plan:[/white]")
    for s in sections:
        console.print(f" - {s}")

    return {
        "section_plan": sections,
        "current_section_idx": 0,
        "completed_sections": [],
        "research_notes": [],
        "critiques": []
    }

def section_initiator_node(state: AgentState):
    """Prepares state for the next section."""
    idx = state["current_section_idx"]
    sections = state["section_plan"]

    current_topic = sections[idx]

    console.print(Panel(f"Starting Section {idx+1}/{len(sections)}: {current_topic}", style="bold blue"))

    return {
        "topic": current_topic,
        "loop_count": 0,
        "research_plan": [],
        "research_notes": [],  # Start fresh for this section
        "current_draft": "",
        "critiques": []
    }

def deep_researcher_node(state: AgentState):
    """(Formerly Planner) Generates queries for the CURRENT SECTION."""
    loop_count = state.get("loop_count", 0)
    topic = state["topic"]
    main_topic = state["main_topic"]
    critiques = state.get("critiques", [])

    with console.status(f"[cyan]Planning research for: {topic}...", spinner="dots"):
        if loop_count == 0:
            prompt = (
                f"Main Report Topic: {main_topic}\n"
                f"Current Section: {topic}\n"
                "Generate 3 highly specific search queries to gather information specifically for this section.\n"
                "Return ONLY the queries as a list, separated by newlines.\n"
                "Make the queries in few words and use KEYWORDS only to make the search."
            )
        else:
            critique_text = "\n".join(critiques)
            prompt = (
                f"Section: {topic}\n"
                f"Address these gaps: {critique_text}\n"
                "Generate 2 NEW search queries to fill these gaps.\n"
                "Return ONLY the queries as a list, separated by newlines.\n"
                "Make the queries in few words and use KEYWORDS only to make the search."
            )

        response = run_llm(DEEP_RESEARCH_MODEL_CONFIG["planner"], prompt)
        plan = [line.strip().replace("- ", "").replace("* ", "") for line in response.split('\n') if line.strip()]

    pretty_print_queries(plan, topic)
    return {"research_plan": plan}

def researcher_node(state: AgentState):
    plan = state["research_plan"]
    existing_notes = state["research_notes"]
    new_notes = []

    for query in plan:
        with console.status(f"[yellow]Searching: {query}...", spinner="earth"):
            search_results = search_tool.invoke(query)

            selector_prompt = (
                f"Query: {query}\nSearch Results: {search_results}\n\n"
                "Return the single best URL for deep reading. Return ONLY the URL."
            )
            best_url = run_llm(DEEP_RESEARCH_MODEL_CONFIG["planner"], selector_prompt).strip()

        if "http" not in best_url:
            new_notes.append(f"### Findings for '{query}':\n{search_results}\n")
            continue

        with console.status(f"[yellow]Reading: {best_url}...", spinner="bouncingBall"):
            page_content = scrape_text(best_url)
            extraction_prompt = (
                f"Query: {query}\nSource: {best_url}\nContent: {page_content}\n\n"
                "Extract comprehensive findings, statistics, and arguments.\nFormat: [Fact] (Source: URL)\n"
                "If the source is not relevant to the query, then use the format [no relevant facts found] (Source: URL)"
            )
            summary = run_llm(DEEP_RESEARCH_MODEL_CONFIG["researcher"], extraction_prompt)
            new_notes.append(f"### Deep Dive on '{query}':\n{summary}\n")

    return {"research_notes": existing_notes + new_notes}

def writer_node(state: AgentState):
    topic = state["topic"]
    main_topic = state["main_topic"]
    flat_notes = "\n".join(state["research_notes"])
    current_draft = state.get("current_draft", "")
    loop_count = state.get("loop_count", 0)

    system = "You are a technical writer."

    if loop_count == 0:
        prompt = (
            f"Context: Writing a report on '{main_topic}'.\n"
            f"Current Section to write: {topic}\n"
            f"Research Notes:\n{flat_notes}\n\n"
            "Write this specific section. Do not write a whole intro/conclusion for the whole report, just this part.\n"
            "Use information from ONLY the research notes.\n"
            "If no research notes are relevant, then write a factual paragraph stating no relevant information was found, and write that more research is needed.\n"
            "Use academic tone. Cite sources inline [1].\n"
            "Output ONLY the section text."
        )
    else:
        prompt = (
            f"Refine the section: {topic}.\n"
            f"Current Draft:\n{current_draft}\n\n"
            f"New Notes:\n{flat_notes}\n\n"
            "Integrate new findings. Output the updated section.\n"
            "Determine if the critiques are valid first before integrating them.\n"
            "Use information from ONLY the research notes.\n"
            "If no research notes are relevant, then write a factual paragraph stating no relevant information was found, and write that more research is needed.\n"
            "Also make sure to retain all cited sources and their inline citations [1]."
        )

    content = run_llm(DEEP_RESEARCH_MODEL_CONFIG["writer"], prompt, system_prompt=system, temperature=0.3)
    return {"current_draft": content}

def quorum_node(state: AgentState):
    draft = state["current_draft"]
    critiques = []
    skeptic_models = DEEP_RESEARCH_MODEL_CONFIG["skeptics"]

    console.print(f"[bold red]Running Quorum on Section Draft...[/bold red]")

    for model in skeptic_models:
        identify_prompt = (
            f"Draft:\n{draft[:4000]}...\n\n"
            "Identify one weak/unverified claim. Generate a search query to check it.\n"
            "Output ONLY the search query.\n"
            "Make the query in few words and use KEYWORDS only to make the search."
        )
        # Using generic run_llm but need to support model override?
        # run_llm takes model_name as first arg.
        skeptic_query = run_llm(model, identify_prompt, temperature=0.1).strip().replace('"', '')

        skeptic_evidence = search_tool.invoke(skeptic_query)

        critique_prompt = (
            f"Draft:\n{draft[:4000]}...\n"
            f"Evidence found for '{skeptic_query}':\n{skeptic_evidence}\n\n"
            "Critique the draft based on this evidence. Be harsh but constructive.\n"
            "If the source is not relevant then critique the draft based on weak links.\n"
            "Include all the critiques that you can think of."
        )

        critique = run_llm(model, critique_prompt, temperature=0.3)
        critiques.append(f"[{model}]: {critique}")

    return {"critiques": critiques}

def refiner_node(state: AgentState):
    draft = state["current_draft"]
    critiques = "\n".join(state["critiques"])
    flat_notes = "\n".join(state["research_notes"])

    prompt = (
        f"Original Draft:\n{draft}\n\n"
        f"Critiques:\n{critiques}\n\n"
        f"Notes:\n{flat_notes}\n\n"
        "Rewrite the draft to address critiques. Preserve citations. Output the final section text.\n"
        "First determine if the critique is worth addressing before addressing them."
    )

    content = run_llm(DEEP_RESEARCH_MODEL_CONFIG["writer"], prompt, temperature=0.25)
    return {"current_draft": content, "loop_count": state["loop_count"] + 1}

def section_compiler_node(state: AgentState):
    """Saves the finished section and moves to the next."""
    finished_section_text = state["current_draft"]
    current_idx = state["current_section_idx"]
    topic = state["topic"]

    completed = state["completed_sections"]
    completed.append(f"## {topic}\n\n{finished_section_text}\n\n")

    console.print(Panel(f"Section '{topic}' Completed", style="bold green"))

    return {
        "completed_sections": completed,
        "current_section_idx": current_idx + 1
    }

def final_editor_node(state: AgentState):
    """Stitches everything together."""
    main_topic = state["main_topic"]
    raw_body = "\n".join(state["completed_sections"])

    console.print(Panel("Finalizing Report...", title="[bold magenta]FINAL EDITOR[/bold magenta]"))

    prompt = (
        f"Topic: {main_topic}\n"
        f"Here are the drafted sections:\n{raw_body}\n\n"
        "Instructions:\n"
        "1. Write a strong Introduction summarizing the topic.\n"
        "2. Include the provided sections in order.\n"
        "3. Write a Conclusion.\n"
        "4. Smooth out transitions between sections if they feel disjointed.\n"
        "5. Compile a 'References' section at the bottom based on the URLs found in the text.\n"
        "6. Preserve the citations [1] where they belong.\n"
        "7. Turn bullet points into FULL PARAGRAPHS.\n"
        "Output the final Markdown report."
    )

    final_report = run_llm(DEEP_RESEARCH_MODEL_CONFIG["editor"], prompt, temperature=0.2)
    return {"final_report": final_report}

# --- EDGES & ROUTING ---

def check_section_loop(state: AgentState):
    if state["loop_count"] < max_loop:
        return "loop"
    return "done"

def check_global_progress(state: AgentState):
    if state["current_section_idx"] < len(state["section_plan"]):
        return "next_section"
    return "finalize"

# 1. SETUP GRAPH
workflow = StateGraph(AgentState)

# Global Nodes
workflow.add_node("global_planner", global_planner_node)
workflow.add_node("section_initiator", section_initiator_node)
workflow.add_node("section_compiler", section_compiler_node)
workflow.add_node("final_editor", final_editor_node)

# Deep Research Sub-Nodes
workflow.add_node("deep_researcher", deep_researcher_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("quorum", quorum_node)
workflow.add_node("refiner", refiner_node)

# Edges
workflow.set_entry_point("global_planner")
workflow.add_edge("global_planner", "section_initiator")

# Start Local Loop
workflow.add_edge("section_initiator", "deep_researcher")
workflow.add_edge("deep_researcher", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "quorum")
workflow.add_edge("quorum", "refiner")

# Inner Loop Conditional
workflow.add_conditional_edges(
    "refiner",
    check_section_loop,
    {
        "loop": "deep_researcher",
        "done": "section_compiler"
    }
)

# Global Loop Conditional
workflow.add_conditional_edges(
    "section_compiler",
    check_global_progress,
    {
        "next_section": "section_initiator",
        "finalize": "final_editor"
    }
)

workflow.add_edge("final_editor", END)

# Compile
research_app = workflow.compile()

def main():
    console.print(Panel.fit("[bold white]Deep Research Agent v2[/bold white]", style="blue"))

    # 1. SETUP PROMPT SESSION (Robust line editing + History)
    history_file = os.path.expanduser("~/.deep_research_history")
    session = PromptSession(history=FileHistory(history_file))

    with SqliteSaver.from_conn_string("checkpoints.sqlite") as memory:
        # Re-compile with checkpointer if running directly
        # Note: We need to use 'global' research_app or return it if we want to expose it,
        # but here we are just running the main loop.
        # We shadow the global research_app with the checkpointer version for this run.
        app_with_memory = workflow.compile(checkpointer=memory)

        # 2. INPUT 1: SESSION ID
        console.print("[bold yellow]Enter Session ID[/bold yellow] (or press Enter):", end=" ")
        session_id = session.prompt("").strip()

        topic = ""

        if not session_id:
            session_id = str(uuid.uuid4())[:8]

            # 3. INPUT 2: TOPIC
            console.print("[bold yellow]Enter MAIN research topic:[/bold yellow]")
            topic = session.prompt("> ").strip()

            console.print(f"[dim]Starting new session: {session_id}[/dim]")

            inputs = {
                "main_topic": topic,
                "section_plan": [],
                "completed_sections": [],
                "current_section_idx": 0,
                "topic": "init",
                "research_plan": [],
                "research_notes": [],
                "current_draft": "",
                "critiques": [],
                "loop_count": 0
            }
            config = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": 150
            }
            final_state = app_with_memory.invoke(inputs, config=config)
        else:
            config = {
                "configurable": {"thread_id": session_id},
                "recursion_limit": 150
            }
            console.print("[dim]Resuming...[/dim]")
            final_state = app_with_memory.invoke(None, config=config)

        filename = f"final_report_{session_id}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_state["final_report"])

        console.print(f"\n[bold green]Done![/bold green] Report saved to '{filename}'")

# --- RUN ---
if __name__ == "__main__":
    main()
