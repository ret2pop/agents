import os
from abc import ABC, abstractmethod
from typing import List, Dict
import requests
from dotenv import load_dotenv
from duckduckgo_search import DDGS

# Load environment variables from .env file
load_dotenv()

# Set default max results from env or default to 5
DEFAULT_MAX_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict]:
        """
        Executes a search query and returns a list of results.
        Each result should be a dictionary with 'title', 'href', and 'body' keys.
        """
        pass

class TavilySearchProvider(SearchProvider):
    def __init__(self, api_key: str = None):
        self.api_key = (api_key or os.getenv("TAVILY_API_KEY", "")).strip()
        self.base_url = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict]:
        if not self.api_key:
            raise ValueError("Tavily API key not provided in init or .env")

        # Tavily payload
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False
        }

        try:
            # Tavily uses a POST request
            response = requests.post(self.base_url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            if 'results' in data:
                for item in data['results']:
                    results.append({
                        'title': item.get('title', ''),
                        'href': item.get('url', ''),
                        'body': item.get('content', '')
                    })
            return results
        except requests.exceptions.RequestException as e:
            # Silently re-raise to be caught by Hybrid provider
            raise e

class BraveSearchProvider(SearchProvider):
    def __init__(self, api_key: str = None):
        self.api_key = (api_key or os.getenv("BRAVE_API_KEY", "")).strip()
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict]:
        if not self.api_key:
            raise ValueError("Brave API key not provided in init or .env")

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        
        # Brave limits 'count' to 20 per request
        safe_count = min(max_results, 20)
        
        params = {
            "q": query,
            "count": safe_count
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
            raise e

class GoogleSearchProvider(SearchProvider):
    def __init__(self, api_key: str = None, cse_id: str = None):
        self.api_key = (api_key or os.getenv("GOOGLE_API_KEY", "")).strip()
        self.cse_id = (cse_id or os.getenv("GOOGLE_CSE_ID", "")).strip()
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict]:
        if not self.api_key or not self.cse_id:
            raise ValueError("Google API key or CSE ID not provided in init or .env")

        # Google API 'num' must be <= 10
        safe_num = min(max_results, 10)

        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.cse_id,
            "num": safe_num
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
            raise e

class DuckDuckGoSearchProvider(SearchProvider):
    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> List[Dict]:
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
        except Exception:
            # Silently fail or re-raise; since this is the fallback, returning empty is appropriate
            pass
        return results

class HybridSearchProvider(SearchProvider):
    def __init__(self, num_results=10):
        self.tavily = TavilySearchProvider()
        self.brave = BraveSearchProvider()
        self.google = GoogleSearchProvider()
        self.ddg = DuckDuckGoSearchProvider()
        self.num_results = num_results

    def search(self, query: str, max_results: int = self.num_results) -> List[Dict]:
        # 1. Try Tavily (Primary)
        try:
            return self.tavily.search(query, max_results)
        except Exception:
            pass

        # 2. Try Google
        try:
            return self.google.search(query, max_results)
        except Exception:
            pass

        # 3. Try Brave
        try:
            return self.brave.search(query, max_results)
        except Exception:
            pass

        # 4. Fallback to DuckDuckGo (always works, or returns empty list)
        return self.ddg.search(query, max_results)
