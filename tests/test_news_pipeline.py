"""
Tests for the news pipeline components: NewsAPI client, Brave Search client,
SentimentAnalyzer integration, and config fixes.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

from src.config.settings import settings
from src.data.news_aggregator import NewsArticle
from src.data.newsapi_client import NewsAPIClient
from src.data.brave_search_client import BraveNewsClient


class TestConfigNewsAttributes:
    """Tests that config attributes are properly defined (regression for AttributeError bug)."""

    def test_skip_news_for_low_volume_in_trading_config(self):
        assert hasattr(settings.trading, "skip_news_for_low_volume")
        assert isinstance(settings.trading.skip_news_for_low_volume, bool)

    def test_news_search_volume_threshold_in_trading_config(self):
        assert hasattr(settings.trading, "news_search_volume_threshold")
        assert isinstance(settings.trading.news_search_volume_threshold, float)

    def test_newsapi_key_in_api_config(self):
        assert hasattr(settings.api, "newsapi_key")

    def test_brave_api_key_in_api_config(self):
        assert hasattr(settings.api, "brave_api_key")


class TestNewsAPIClient:
    """Tests for the NewsAPI client."""

    @pytest.mark.asyncio
    async def test_fetch_articles_no_api_key(self):
        with patch.object(settings.api, "newsapi_key", ""):
            client = NewsAPIClient(api_key="")
            articles = await client.fetch_articles("test query")
            assert articles == []

    @pytest.mark.asyncio
    async def test_fetch_articles_parses_response(self):
        mock_response = {
            "status": "ok",
            "articles": [
                {
                    "title": "Test Article",
                    "description": "Test description",
                    "source": {"name": "Test Source"},
                    "url": "https://example.com",
                    "publishedAt": "2024-01-15T10:00:00Z",
                }
            ],
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response_obj)
            mock_client_class.return_value = mock_client

            client = NewsAPIClient(api_key="test_key")
            articles = await client.fetch_articles("test query")

            assert len(articles) == 1
            assert articles[0].title == "Test Article"
            assert articles[0].summary == "Test description"
            assert articles[0].source == "Test Source"

    @pytest.mark.asyncio
    async def test_fetch_articles_handles_error_status(self):
        mock_response = {
            "status": "error",
            "code": "apiKeyInvalid",
            "message": "API key is invalid",
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response_obj)
            mock_client_class.return_value = mock_client

            client = NewsAPIClient(api_key="test_key")
            articles = await client.fetch_articles("test query")

            assert articles == []

    @pytest.mark.asyncio
    async def test_fetch_articles_skips_removed(self):
        mock_response = {
            "status": "ok",
            "articles": [
                {"title": "[Removed]", "description": "Removed", "source": {"name": "Test"}, "url": "", "publishedAt": ""},
                {
                    "title": "Valid Article",
                    "description": "Valid description",
                    "source": {"name": "Test Source"},
                    "url": "https://example.com",
                    "publishedAt": "2024-01-15T10:00:00Z",
                },
            ],
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response_obj)
            mock_client_class.return_value = mock_client

            client = NewsAPIClient(api_key="test_key")
            articles = await client.fetch_articles("test query")

            assert len(articles) == 1
            assert articles[0].title == "Valid Article"


class TestBraveSearchClient:
    """Tests for the Brave Search client."""

    @pytest.mark.asyncio
    async def test_fetch_articles_no_api_key(self):
        with patch.object(settings.api, "brave_api_key", ""):
            client = BraveNewsClient(api_key="")
            articles = await client.fetch_articles("test query")
            assert articles == []

    @pytest.mark.asyncio
    async def test_fetch_articles_parses_response(self):
        mock_response = {
            "web": {
                "results": [
                    {
                        "title": "Test Article",
                        "description": "Test description",
                        "url": "https://example.com",
                        "source": "Test Source",
                        "age": "2024-01-15T10:00:00Z",
                    }
                ]
            }
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response_obj)
            mock_client_class.return_value = mock_client

            client = BraveNewsClient(api_key="test_key")
            articles = await client.fetch_articles("test query")

            assert len(articles) == 1
            assert articles[0].title == "Test Article"
            assert articles[0].summary == "Test description"

    @pytest.mark.asyncio
    async def test_fetch_articles_handles_empty_results(self):
        mock_response = {"web": {"results": []}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response_obj = MagicMock()
            mock_response_obj.json.return_value = mock_response
            mock_response_obj.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response_obj)
            mock_client_class.return_value = mock_client

            client = BraveNewsClient(api_key="test_key")
            articles = await client.fetch_articles("test query")

            assert articles == []


class TestSentimentAnalyzerLiveNews:
    """Tests for SentimentAnalyzer integration with live news APIs."""

    @pytest.mark.asyncio
    async def test_fetch_live_news_newsapi_fallback(self):
        with patch("src.data.newsapi_client.NewsAPIClient.fetch_articles") as mock_newsapi, \
             patch("src.data.brave_search_client.BraveNewsClient.fetch_articles") as mock_brave:
            
            # NewsAPI returns results
            mock_newsapi.return_value = [
                NewsArticle(
                    title="Test Article",
                    summary="Test summary",
                    source="Test Source",
                    published=datetime.now(timezone.utc),
                    url="https://example.com",
                )
            ]
            
            from src.data.sentiment_analyzer import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            
            # Inject clients to avoid creating new ones
            analyzer._newsapi_client = MagicMock()
            analyzer._newsapi_client.fetch_articles = AsyncMock(return_value=mock_newsapi.return_value)
            
            result = await analyzer._fetch_live_news("test query")
            
            assert len(result) == 1
            assert result[0].title == "Test Article"
            # Brave should not be called if NewsAPI has results

    @pytest.mark.asyncio
    async def test_fetch_live_news_brave_fallback(self):
        with patch("src.data.newsapi_client.NewsAPIClient.fetch_articles") as mock_newsapi, \
             patch("src.data.brave_search_client.BraveNewsClient.fetch_articles") as mock_brave:
            
            # NewsAPI returns empty, Brave has results
            mock_newsapi.return_value = []
            mock_brave.return_value = [
                NewsArticle(
                    title="Brave Article",
                    summary="Brave summary",
                    source="Brave Source",
                    published=datetime.now(timezone.utc),
                    url="https://brave.com",
                )
            ]
            
            from src.data.sentiment_analyzer import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            
            analyzer._newsapi_client = MagicMock()
            analyzer._newsapi_client.fetch_articles = AsyncMock(return_value=[])
            
            analyzer._brave_client = MagicMock()
            analyzer._brave_client.fetch_articles = AsyncMock(return_value=mock_brave.return_value)
            
            result = await analyzer._fetch_live_news("test query")
            
            assert len(result) == 1
            assert result[0].title == "Brave Article"

    @pytest.mark.asyncio
    async def test_get_market_sentiment_summary_with_live_news(self):
        with patch("src.data.news_aggregator.NewsAggregator.fetch_all") as mock_fetch, \
             patch("src.data.news_aggregator.NewsAggregator.get_relevant_articles") as mock_relevant, \
             patch.object(settings.sentiment, "enabled", True):
            
            # RSS returns empty (simulating the bug scenario)
            mock_fetch.return_value = []
            mock_relevant.return_value = []  # Empty from RSS
            
            # Mock live news to return results
            from src.data.sentiment_analyzer import SentimentAnalyzer
            analyzer = SentimentAnalyzer()
            
            mock_live_articles = [
                NewsArticle(
                    title="Live News Article",
                    summary="Live summary about CPI and inflation",
                    source="NewsAPI",
                    published=datetime.now(timezone.utc),
                    url="https://example.com",
                )
            ]
            
            with patch.object(analyzer, "_fetch_live_news", return_value=mock_live_articles):
                result = await analyzer.get_market_sentiment_summary("Will CPI exceed 3%?")
                
                # Should NOT return "No relevant news" message
                assert "No relevant news articles found" not in result
                assert "Live News Article" in result or "Sentiment analysis unavailable" not in result


class TestNewsArticleModel:
    """Tests for the NewsArticle model used across clients."""

    def test_normalized_title(self):
        article = NewsArticle(
            title="  Test Article Title  ",
            summary="Summary",
            source="Source",
            published=None,
            url="",
        )
        assert article.normalized_title == "test article title"

    def test_category_defaults_to_empty(self):
        article = NewsArticle(
            title="Test",
            summary="Summary",
            source="Source",
            published=None,
            url="",
        )
        assert article.category == ""