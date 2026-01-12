from abc import ABC, abstractmethod
from typing import List, Dict
import requests
from duckduckgo_search import DDGS
from pyagents.config import BRAVE_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID, MAX_SEARCH_RESULTS

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

class GoogleSearchProvider(SearchProvider):
    def __init__(self, api_key: str = GOOGLE_API_KEY, cse_id: str = GOOGLE_CSE_ID):
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        if not self.api_key or not self.cse_id:
            raise ValueError("Google API key or CSE ID not provided")

        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.cse_id,
            "num": max_results
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            if 'items' in data:
                for item in data['items']:
                    results.append({
                        'title': item.get('title', ''),
                        'href': item.get('link', ''),
                        'body': item.get('snippet', '')
                    })
            return results
        except requests.exceptions.RequestException as e:
            print(f"[GoogleSearchProvider] Error: {e}")
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
        self.brave = BraveSearchProvider()
        self.google = GoogleSearchProvider()
        self.ddg = DuckDuckGoSearchProvider()

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        # Try Brave
        try:
            print(f"[HybridSearchProvider] Trying Brave Search for '{query}'...")
            return self.brave.search(query, max_results)
        except Exception as e:
            print(f"[HybridSearchProvider] Brave Search failed: {e}")

        # Try Google
        try:
            print(f"[HybridSearchProvider] Falling back to Google Search for '{query}'...")
            return self.google.search(query, max_results)
        except Exception as e:
            print(f"[HybridSearchProvider] Google Search failed: {e}")

        # Fallback to DuckDuckGo
        try:
            print(f"[HybridSearchProvider] Falling back to DuckDuckGo for '{query}'...")
            return self.ddg.search(query, max_results)
        except Exception as e:
            print(f"[HybridSearchProvider] DuckDuckGo Search failed: {e}")
            return []
