"""
Brave Search API client for fetching live news articles.
Fallback live news source when NewsAPI fails or returns insufficient results.
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import httpx

from src.config.settings import settings
from src.data.news_aggregator import NewsArticle
from src.utils.logging_setup import TradingLoggerMixin


class BraveNewsClient(TradingLoggerMixin):
    """Client for fetching news from Brave Search API."""

    BASE_URL = "https://api.search.brave.com/res/v1/web"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or settings.api.brave_api_key
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch_articles(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[NewsArticle]:
        """
        Fetch news articles matching the query from Brave Search.

        Args:
            query: Search query (typically the market title)
            max_results: Maximum number of articles to return

        Returns:
            List of NewsArticle objects
        """
        if not self._api_key:
            self.logger.warning("Brave Search API key not configured, skipping live news fetch")
            return []

        url = f"{self.BASE_URL}/search"
        headers = {
            "X-Subscription-Token": self._api_key,
            "Accept": "application/json",
        }
        params = {
            "q": query,
            "count": max_results,
            "search_filter": "news",  # Focus on news results
        }

        try:
            client = await self._get_client()
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()

            web_results = data.get("web", {}).get("results", [])
            if not web_results:
                self.logger.info("Brave Search returned no results", query=query)
                return []

            articles = []
            for item in web_results:
                article = self._parse_article(item)
                if article:
                    articles.append(article)

            self.logger.info(
                "Brave Search fetch complete",
                query=query,
                num_articles=len(articles),
            )
            return articles

        except httpx.HTTPStatusError as e:
            self.logger.error(
                "Brave Search HTTP error",
                status_code=e.response.status_code,
                error=str(e),
            )
            return []
        except asyncio.TimeoutError:
            self.logger.warning("Brave Search request timed out", query=query)
            return []
        except Exception as e:
            self.logger.error(
                "Brave Search fetch failed",
                query=query,
                error=str(e),
            )
            return []

    def _parse_article(self, item: dict) -> Optional[NewsArticle]:
        try:
            title = item.get("title", "")
            if not title:
                return None

            description = item.get("description", item.get("snippet", ""))
            url = item.get("url", "")
            age = item.get("age", "")
            source = item.get("source", "Unknown")

            if isinstance(source, dict):
                source = source.get("name", "Unknown")

            published_dt: Optional[datetime] = None
            if age:
                try:
                    published_dt = datetime.fromisoformat(age.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass
            else:
                published_dt = datetime.now(timezone.utc)

            return NewsArticle(
                title=title,
                summary=description,
                source=source,
                published=published_dt,
                url=url,
                category="live_news",
            )

        except Exception as e:
            self.logger.warning("Failed to parse Brave Search article", error=str(e))
            return None