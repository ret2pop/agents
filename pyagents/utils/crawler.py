import asyncio
from crawl4ai import AsyncWebCrawler

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
    """
    return asyncio.run(crawl_url(url))
