import operator
from typing import Annotated, List, TypedDict

from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# --- CONFIGURATION ---
OLLAMA_MODEL = "qwen3-vl:8b" 
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

# We use 'DuckDuckGoSearchResults' instead of 'Run' to get the URL/Title metadata
search_tool = DuckDuckGoSearchResults() 

# --- STATE DEFINITION ---
class ResearchState(TypedDict):
    topic: str
    plan: List[str]
    # We aggregate "notes" which will contain the raw content + source URLs
    content: Annotated[List[str], operator.add] 
    final_report: str

# --- NODES ---

def planner_node(state: ResearchState):
    print(f"--- PLANNING: {state['topic']} ---")
    
    system_prompt = (
        "You are a research planning assistant. Given a topic, generate a list "
        "of 3 targeted search queries to gather comprehensive information. "
        "Return ONLY the queries, separated by newlines."
    )
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=state['topic'])]
    response = llm.invoke(messages)
    
    plan = [line.strip() for line in response.content.split('\n') if line.strip()]
    return {"plan": plan}

def researcher_node(state: ResearchState):
    current_plan = state["plan"]
    query = current_plan[0]
    remaining_plan = current_plan[1:]
    
    print(f"--- RESEARCHING: {query} ---")
    
    # 1. Get structured search results (List of [Snippet, Title, Link])
    try:
        raw_results = search_tool.invoke(query)
    except Exception as e:
        print("Search results failed!!!!")
        raw_results = f"Error: {str(e)}"

    # 2. Summarize while strictly preserving URLs
    summary_prompt = ChatPromptTemplate.from_template(
        "You are a researcher. Your goal is to extract facts from the search results below. "
        "IMPORTANT: You must include the source URL for every fact you extract. "
        "Format your notes like this:\n"
        "- [Fact or finding] (Source: [URL])\n\n"
        "Search Results:\n{results}\n\n"
        "Research Notes:"
    )
    
    chain = summary_prompt | llm
    summary = chain.invoke({"results": raw_results})
    
    # We append the query context so the writer knows what this section is about
    section_data = f"### Sources for '{query}':\n{summary.content}\n\n"
    
    return {
        "plan": remaining_plan, 
        "content": [section_data]
    }

def writer_node(state: ResearchState):
    print("--- WRITING FINAL REPORT ---")
    
    full_context = "\n".join(state["content"])
    
    writer_prompt = ChatPromptTemplate.from_template(
        "You are a technical writer. Write a detailed markdown report on '{topic}'.\n\n"
        "RULES FOR CITATION:\n"
        "1. You MUST use inline citations in the text like [1], [2].\n"
        "2. You MUST create a 'References' section at the very end.\n"
        "3. Every [n] citation must correspond to a real URL provided in the Research Notes.\n"
        "4. Do not make up links. Only use the ones provided.\n\n"
        "Research Notes:\n{context}\n\n"
        "Final Report:"
    )
    
    chain = writer_prompt | llm
    report = chain.invoke({"topic": state["topic"], "context": full_context})
    
    return {"final_report": report.content}

# --- EDGES & GRAPH ---

def should_continue(state: ResearchState):
    if state["plan"]:
        return "research"
    return "write"

workflow = StateGraph(ResearchState)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_conditional_edges("researcher", should_continue, {"research": "researcher", "write": "writer"})
workflow.add_edge("writer", END)

app = workflow.compile()

# --- EXECUTION ---

if __name__ == "__main__":
    topic = input("Enter research topic: ")
    inputs = {"topic": topic, "plan": [], "content": []}
    
    print(f"\nStarting Deep Research on: {topic}...\n")
    
    # Run to completion
    result = app.invoke(inputs)
    
    print("\n" + "="*50)
    print("FINAL REPORT WITH CITATIONS")
    print("="*50 + "\n")
    print(result["final_report"])
    
    # Save to file
    with open("cited_report.md", "w", encoding="utf-8") as f:
        f.write(result["final_report"])
