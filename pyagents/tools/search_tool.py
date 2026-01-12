import json
import warnings
from typing import List, Dict, Optional
from duckduckgo_search import DDGS
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pyagents.config import SEARCH_MODEL, MAX_SEARCH_RESULTS, MAX_READ_COUNT
from pyagents.utils.crawler import crawl_url

# --- WARNING SUPPRESSION ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

class WebScout:
    def __init__(self):
        self.llm = ChatOllama(model=SEARCH_MODEL, temperature=0.2)
        # Headers no longer needed for crawl4ai

    def generate_queries(self, objective: str) -> List[str]:
        """Brainstorms search queries."""
        print(f"[WebScout] 🧠 Brainstorming queries for: '{objective}'")
        system_prompt = (
            "You are an expert Researcher. Generate 2 distinct search queries to find this info.\n"
            "Output ONLY a JSON list of strings. Example: [\"query1\", \"query2\"]"
        )
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Topic: {objective}")
            ])
            content = response.content.strip()
            if "```json" in content: content = content.split("```json")[1].split("```")[0]
            elif "```" in content: content = content.split("```")[1].split("```")[0]
            return json.loads(content)
        except:
            return [objective]

    def search_metadata(self, queries: List[str]) -> List[Dict]:
        """Gets titles and snippets using a Context Manager."""
        aggregated = []
        seen = set()
        print(f"[WebScout] 🕸️  Searching...")

        with DDGS() as ddgs:
            for q in queries:
                try:
                    results = ddgs.text(q, max_results=5)
                    for r in results:
                        if r['href'] not in seen:
                            seen.add(r['href'])
                            aggregated.append(r)
                except Exception as e:
                    print(f"Search failed for {q}: {e}")

        return aggregated

    def select_best_links(self, objective: str, results: List[Dict]) -> List[Dict]:
        """Asks LLM to pick the most relevant links to read."""
        if not results: return []

        print(f"[WebScout] 🧐 Selecting best {MAX_READ_COUNT} links to read...")

        candidates = "\n".join([f"{i}. {r['title']} ({r['href']})\nSnippet: {r['body']}" for i, r in enumerate(results)])

        system_prompt = (
            "Analyze the search results. Return the indices of the top 3 most relevant results to read.\n"
            "Output ONLY a JSON list of integers. Example: [0, 4, 2]"
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Objective: {objective}\n\nResults:\n{candidates}")
            ])
            content = response.content.strip()
            content = ''.join(c for c in content if c in '0123456789[],')
            indices = json.loads(content)

            selected = [results[i] for i in indices if i < len(results)]
            return selected[:MAX_READ_COUNT]
        except:
            return results[:MAX_READ_COUNT]

    def scrape_page(self, url: str) -> str:
        """Fetches and cleans text from a URL using crawl4ai."""
        text = crawl_url(url)
        return text[:4000]

    def synthesize_report(self, objective: str, data: List[Dict]) -> str:
        """Writes the final prose summary with citations."""
        print(f"[WebScout] ✍️  Synthesizing report...")

        context = ""
        for i, item in enumerate(data):
            context += f"\n--- SOURCE [{i+1}] ---\n"
            context += f"URL: {item['href']}\n"
            context += f"CONTENT: {item['content']}\n"

        system_prompt = (
            "You are a Research Assistant.\n"
            "Write a concise summary answering the user's objective based on the provided sources.\n"
            "Rules:\n"
            "1. Use prose (paragraphs).\n"
            "2. Use inline citations like [1], [2] to refer to sources.\n"
            "3. At the end, list the references."
        )

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Objective: {objective}\n\nResearch Data:\n{context}")
        ])

        return response.content.strip()

    def run(self, objective: str):
        queries = self.generate_queries(objective)
        raw_results = self.search_metadata(queries)
        targets = self.select_best_links(objective, raw_results)

        scraped_data = []
        for t in targets:
            print(f"[WebScout] 📖 Reading: {t['title']}...")
            content = self.scrape_page(t['href'])
            t['content'] = content
            scraped_data.append(t)

        if not scraped_data:
            return "No relevant data found to read."

        return self.synthesize_report(objective, scraped_data)

def main():
    scout = WebScout()
    topic = input("Research Topic: ")
    print("\n" + scout.run(topic))

if __name__ == "__main__":
    main()
