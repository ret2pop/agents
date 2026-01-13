import asyncio
import nest_asyncio
from crawl4ai import AsyncWebCrawler

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def crawl_url(url: str) -> str:
    """
    Crawls a URL using crawl4ai and returns the cleaned markdown content.
    """
    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(url=url)
        if result.success:
            return result.markdown
        else:
            return f"Error crawling {url}: {result.error_message}"

def scrape_text_crawl4ai(url: str) -> str:
    """
    Synchronous wrapper for crawl_url.
    Checks if an event loop is already running and handles it.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Loop is running, use run_until_complete or create task if possible,
        # but since we want to return the result synchronously, we might need to rely on nest_asyncio
        # which patches the loop to allow re-entry.
        return loop.run_until_complete(crawl_url(url))
    else:
        return asyncio.run(crawl_url(url))
