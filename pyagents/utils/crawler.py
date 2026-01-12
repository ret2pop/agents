import asyncio
from crawl4ai import AsyncWebCrawler
from typing import Optional

async def _crawl(url: str) -> str:
    """
    Asynchronously crawls a URL using crawl4ai and returns the markdown content.
    """
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url)
        return result.markdown

def crawl_url(url: str) -> str:
    """
    Synchronous wrapper for crawling a URL.
    Returns markdown content or error message.
    """
    try:
        return asyncio.run(_crawl(url))
    except Exception as e:
        return f"Error crawling {url}: {str(e)}"
