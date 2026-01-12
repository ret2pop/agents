from abc import ABC, abstractmethod
from typing import List, Dict
import requests
from duckduckgo_search import DDGS
from pyagents.config import BRAVE_API_KEY, MAX_SEARCH_RESULTS

class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Executes a search query and returns a list of results.
        Each result should be a dictionary with 'title', 'href', and 'body' keys.
        """
        pass

class BraveSearchProvider(SearchProvider):
    def __init__(self, api_key: str = BRAVE_API_KEY):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        if not self.api_key:
            raise ValueError("Brave API key not provided")

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        params = {
            "q": query,
            "count": max_results
        }

        try:
            response = requests.get(self.base_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            if 'web' in data and 'results' in data['web']:
                for item in data['web']['results']:
                    results.append({
                        'title': item.get('title', ''),
                        'href': item.get('url', ''),
                        'body': item.get('description', '')
                    })
            return results
        except requests.exceptions.RequestException as e:
            print(f"[BraveSearchProvider] Error: {e}")
            raise e

class DuckDuckGoSearchProvider(SearchProvider):
    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        results = []
        try:
            with DDGS() as ddgs:
                ddgs_results = ddgs.text(query, max_results=max_results)
                for r in ddgs_results:
                    results.append({
                        'title': r.get('title', ''),
                        'href': r.get('href', ''),
                        'body': r.get('body', '')
                    })
        except Exception as e:
            print(f"[DuckDuckGoSearchProvider] Error: {e}")
            raise e
        return results

class HybridSearchProvider(SearchProvider):
    def __init__(self):
        self.primary = BraveSearchProvider()
        self.fallback = DuckDuckGoSearchProvider()

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        # Try primary (Brave)
        try:
            print(f"[HybridSearchProvider] Trying Brave Search for '{query}'...")
            return self.primary.search(query, max_results)
        except Exception as e:
            print(f"[HybridSearchProvider] Brave Search failed: {e}")

        # Fallback to secondary (DuckDuckGo)
        try:
            print(f"[HybridSearchProvider] Falling back to DuckDuckGo for '{query}'...")
            return self.fallback.search(query, max_results)
        except Exception as e:
            print(f"[HybridSearchProvider] DuckDuckGo Search failed: {e}")
            return []
