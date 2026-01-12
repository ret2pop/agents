import json
import requests
import warnings
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pyagents.config import SEARCH_MODEL, MAX_SEARCH_RESULTS, MAX_READ_COUNT
from pyagents.tools.search_api import HybridSearchProvider

# --- WARNING SUPPRESSION ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

class WebScout:
    def __init__(self):
        self.llm = ChatOllama(model=SEARCH_MODEL, temperature=0.2)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.search_provider = HybridSearchProvider()

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
        """Gets titles and snippets using the Search Provider."""
        aggregated = []
        seen = set()
        print(f"[WebScout] 🕸️  Searching...")

        for q in queries:
            try:
                results = self.search_provider.search(q, max_results=5)
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
        """Fetches and cleans text from a URL."""
        try:
            with requests.Session() as session:
                resp = session.get(url, headers=self.headers, timeout=5)

            if resp.status_code != 200:
                return f"Error: Status {resp.status_code}"

            soup = BeautifulSoup(resp.content, 'html.parser')

            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text[:4000]
        except Exception as e:
            return f"Scrape Error: {str(e)}"

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
