"""
NewsAPI client for fetching live news articles.
Primary live news source for the sentiment analysis pipeline.
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

import httpx

from src.config.settings import settings
from src.data.news_aggregator import NewsArticle
from src.utils.logging_setup import TradingLoggerMixin


class NewsAPIClient(TradingLoggerMixin):
    """Client for fetching articles from NewsAPI.org."""

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or settings.api.newsapi_key
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
        max_results: int = 10,
        language: str = "en",
    ) -> List[NewsArticle]:
        """
        Fetch articles matching the query from NewsAPI.

        Args:
            query: Search query (typically the market title)
            max_results: Maximum number of articles to return
            language: Language filter (default: en)

        Returns:
            List of NewsArticle objects
        """
        if not self._api_key:
            self.logger.warning("NewsAPI key not configured, skipping live news fetch")
            return []

        url = f"{self.BASE_URL}/everything"
        params = {
            "q": query,
            "pageSize": max_results,
            "language": language,
            "sortBy": "publishedAt",
            "apiKey": self._api_key,
        }

        try:
            client = await self._get_client()
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "ok":
                self.logger.warning(
                    "NewsAPI returned error",
                    status=data.get("status"),
                    code=data.get("code"),
                    message=data.get("message"),
                )
                return []

            articles = []
            for item in data.get("articles", []):
                article = self._parse_article(item)
                if article:
                    articles.append(article)

            self.logger.info(
                "NewsAPI fetch complete",
                query=query,
                num_articles=len(articles),
            )
            return articles

        except httpx.HTTPStatusError as e:
            self.logger.error(
                "NewsAPI HTTP error",
                status_code=e.response.status_code,
                error=str(e),
            )
            return []
        except asyncio.TimeoutError:
            self.logger.warning("NewsAPI request timed out", query=query)
            return []
        except Exception as e:
            self.logger.error(
                "NewsAPI fetch failed",
                query=query,
                error=str(e),
            )
            return []

    def _parse_article(self, item: dict) -> Optional[NewsArticle]:
        try:
            title = item.get("title", "")
            if not title or title == "[Removed]":
                return None

            description = item.get("description", item.get("content", ""))
            source = item.get("source", {}).get("name", "Unknown")
            url = item.get("url", "")
            published_at = item.get("publishedAt", "")

            published_dt: Optional[datetime] = None
            if published_at:
                try:
                    published_dt = datetime.fromisoformat(
                        published_at.replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            return NewsArticle(
                title=title,
                summary=description,
                source=source,
                published=published_dt,
                url=url,
                category="live_news",
            )

        except Exception as e:
            self.logger.warning("Failed to parse NewsAPI article", error=str(e))
            return None