"""
AI-powered sentiment analysis for the Kalshi trading system.
Uses OpenRouter (via the openai library) with a fast/cheap model to score
news article sentiment relative to prediction market questions.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from src.config.settings import settings
from src.data.news_aggregator import NewsAggregator, NewsArticle
from src.data.newsapi_client import NewsAPIClient
from src.data.brave_search_client import BraveNewsClient
from src.utils.logging_setup import TradingLoggerMixin


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SentimentResult:
    """Result of sentiment analysis on a single piece of text."""
    score: float  # -1.0 (very negative) to 1.0 (very positive)
    confidence: float  # 0.0 to 1.0
    reasoning: str


@dataclass
class ArticleSentiment:
    """Sentiment result tied to a specific article and relevance score."""
    article: NewsArticle
    sentiment: SentimentResult
    relevance_score: float


@dataclass
class MarketSentiment:
    """Aggregated sentiment analysis for a market question."""
    overall_score: float  # Simple average of article sentiments
    article_sentiments: List[ArticleSentiment]
    relevance_weighted_score: float  # Weighted average by relevance
    num_articles: int


# ---------------------------------------------------------------------------
# Sentiment Analyzer
# ---------------------------------------------------------------------------

class SentimentAnalyzer(TradingLoggerMixin):
    """
    AI-powered sentiment scorer using OpenRouter with a fast/cheap model.
    Analyses news articles for sentiment relative to prediction market questions,
    caches results, and tracks API cost.
    """

    def __init__(self, news_aggregator: Optional[NewsAggregator] = None) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.api.openrouter_api_key,
            base_url=settings.api.openrouter_base_url,
            timeout=30.0,
            max_retries=2,
        )
        self._model = settings.sentiment.sentiment_model
        self._news = news_aggregator or NewsAggregator()

        # Live news API clients
        self._newsapi_client: Optional[NewsAPIClient] = None
        self._brave_client: Optional[BraveNewsClient] = None

        # Sentiment cache: hash(text + market_title) -> SentimentResult
        self._cache: Dict[str, SentimentResult] = {}

        # Cost tracking
        self.total_cost: float = 0.0
        self.request_count: int = 0

        self.logger.info(
            "SentimentAnalyzer initialized",
            model=self._model,
            cache_ttl_minutes=settings.sentiment.cache_ttl_minutes,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_sentiment(self, text: str, market_context: str = "") -> SentimentResult:
        """
        Analyse the sentiment of *text*, optionally in the context of a
        market question.

        Args:
            text: The text to analyse (typically an article title + summary).
            market_context: Optional market title for contextual scoring.

        Returns:
            A SentimentResult with score, confidence, and reasoning.
        """
        cache_key = self._cache_key(text, market_context)
        if cache_key in self._cache:
            self.logger.debug("Returning cached sentiment result", cache_key=cache_key[:16])
            return self._cache[cache_key]

        result = await self._call_sentiment_model(text, market_context)
        self._cache[cache_key] = result
        return result

    async def analyze_market_sentiment(
        self,
        market_title: str,
        articles: List[NewsArticle],
        relevance_scores: Optional[List[float]] = None,
    ) -> MarketSentiment:
        """
        Score each article's sentiment relative to *market_title* and
        return an aggregated MarketSentiment.

        Args:
            market_title: The prediction market question.
            articles: List of NewsArticle objects to analyse.
            relevance_scores: Optional pre-computed relevance scores (same order
                              as *articles*). If None, all articles are weighted
                              equally.

        Returns:
            MarketSentiment with per-article and aggregate scores.
        """
        if not articles:
            return MarketSentiment(
                overall_score=0.0,
                article_sentiments=[],
                relevance_weighted_score=0.0,
                num_articles=0,
            )

        if relevance_scores is None:
            relevance_scores = [1.0] * len(articles)

        # Analyse all articles concurrently
        tasks = [
            self.analyze_sentiment(
                text=f"{a.title}. {a.summary}",
                market_context=market_title,
            )
            for a in articles
        ]
        results: List[SentimentResult] = list(await asyncio.gather(*tasks, return_exceptions=True))  # type: ignore[assignment]

        article_sentiments: List[ArticleSentiment] = []
        scores: List[float] = []
        weighted_sum: float = 0.0
        weight_total: float = 0.0

        for article, result, relevance in zip(articles, results, relevance_scores):
            if isinstance(result, Exception):
                self.logger.warning(
                    "Sentiment analysis failed for article, skipping",
                    article_title=article.title[:80],
                    error=str(result),
                )
                continue

            article_sentiments.append(
                ArticleSentiment(
                    article=article,
                    sentiment=result,
                    relevance_score=relevance,
                )
            )
            scores.append(result.score)
            weighted_sum += result.score * relevance
            weight_total += relevance

        overall = sum(scores) / len(scores) if scores else 0.0
        weighted = weighted_sum / weight_total if weight_total > 0 else 0.0

        self.logger.info(
            "Market sentiment analysis complete",
            market_title=market_title[:80],
            num_articles=len(article_sentiments),
            overall_score=round(overall, 3),
            weighted_score=round(weighted, 3),
        )

        return MarketSentiment(
            overall_score=round(overall, 4),
            article_sentiments=article_sentiments,
            relevance_weighted_score=round(weighted, 4),
            num_articles=len(article_sentiments),
        )

    async def get_market_sentiment_summary(self, market_title: str) -> str:
        """
        High-level convenience method: fetch relevant articles, run sentiment
        analysis, and return a concise text summary suitable for inclusion in
        a trading decision prompt.

        Args:
            market_title: The prediction market question.

        Returns:
            A human-readable summary string.
        """
        if not settings.sentiment.enabled:
            return "Sentiment analysis is disabled."

        try:
            # Fetch latest RSS articles
            await self._news.fetch_all()

            # Find relevant articles from RSS
            relevant = self._news.get_relevant_articles(market_title, max_articles=5)

            # Try to supplement with live API results if RSS didn't find enough
            if len(relevant) < 3:
                live_articles = await self._fetch_live_news(market_title)
                if live_articles:
                    live_with_scores = [
                        (art, 1.0) for art in live_articles
                    ]
                    relevant.extend(live_with_scores)

            if not relevant:
                return (
                    f"No relevant news articles found for: {market_title}. "
                    "Sentiment: neutral (no signal)."
                )

            articles = [pair[0] for pair in relevant]
            relevance_scores = [pair[1] for pair in relevant]

            # Analyse sentiment
            market_sentiment = await self.analyze_market_sentiment(
                market_title=market_title,
                articles=articles,
                relevance_scores=relevance_scores,
            )

            # Build summary
            return self._format_summary(market_title, market_sentiment)

        except Exception as exc:
            self.logger.error(
                "Failed to generate market sentiment summary",
                market_title=market_title[:80],
                error=str(exc),
            )
            return (
                f"Sentiment analysis unavailable (error: {exc}). "
                "Proceed without sentiment signal."
            )

    async def close(self) -> None:
        """Close the underlying OpenAI client."""
        await self._client.close()
        self.logger.info(
            "SentimentAnalyzer closed",
            total_cost=round(self.total_cost, 4),
            total_requests=self.request_count,
        )

    async def _fetch_live_news(self, market_title: str) -> List[NewsArticle]:
        """
        Fetch live news from NewsAPI (primary) and Brave Search (fallback).

        Args:
            market_title: The market title to search for

        Returns:
            List of NewsArticle objects from live APIs
        """
        all_articles: List[NewsArticle] = []

        # Try NewsAPI first (primary)
        if not self._newsapi_client:
            self._newsapi_client = NewsAPIClient()

        newsapi_articles = await self._newsapi_client.fetch_articles(
            query=market_title,
            max_results=10,
        )

        if newsapi_articles:
            self.logger.info(
                "NewsAPI returned articles",
                query=market_title,
                count=len(newsapi_articles),
            )
            all_articles.extend(newsapi_articles)
        else:
            # Fallback to Brave Search if NewsAPI returned nothing
            self.logger.info("NewsAPI returned no results, trying Brave Search")
            if not self._brave_client:
                self._brave_client = BraveNewsClient()

            brave_articles = await self._brave_client.fetch_articles(
                query=market_title,
                max_results=5,
            )

            if brave_articles:
                self.logger.info(
                    "Brave Search returned articles",
                    query=market_title,
                    count=len(brave_articles),
                )
                all_articles.extend(brave_articles)

        # Deduplicate based on normalized title
        if all_articles:
            seen = set()
            deduplicated = []
            for article in all_articles:
                key = article.normalized_title
                if key not in seen:
                    seen.add(key)
                    deduplicated.append(article)
            return deduplicated

        return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_sentiment_model(
        self,
        text: str,
        market_context: str,
    ) -> SentimentResult:
        """Send text to the sentiment model and parse the JSON response."""
        system_prompt = (
            "You are a financial sentiment analysis engine. You will be given a "
            "news article excerpt and optionally a prediction market question for "
            "context. Your job is to assess the sentiment of the article as it "
            "relates to the market question (if provided) or in general financial "
            "terms.\n\n"
            "Respond with ONLY a valid JSON object (no markdown, no extra text) "
            "with exactly these keys:\n"
            '  "score": a float from -1.0 (very negative) to 1.0 (very positive),\n'
            '  "confidence": a float from 0.0 to 1.0 indicating how confident '
            "you are in the score,\n"
            '  "reasoning": a brief one-sentence explanation.\n\n'
            "Example response:\n"
            '{"score": 0.6, "confidence": 0.8, "reasoning": "The article reports '
            'strong economic growth which supports a YES outcome."}'
        )

        user_content = f"Article text:\n{text}"
        if market_context:
            user_content += f"\n\nMarket question for context:\n{market_context}"

        try:
            start = time.monotonic()
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            elapsed = time.monotonic() - start

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Sentiment model returned empty response")

            content = content.strip()

            # Track cost
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            # Gemini Flash via OpenRouter is very cheap; approximate pricing
            cost = (input_tokens * 0.0000001) + (output_tokens * 0.0000004)
            self.total_cost += cost
            self.request_count += 1

            self.logger.debug(
                "Sentiment model response received",
                model=self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=round(cost, 6),
                elapsed_seconds=round(elapsed, 2),
            )

            return self._parse_sentiment_response(content)

        except Exception as exc:
            self.logger.error(
                "Sentiment model call failed",
                model=self._model,
                error=str(exc),
            )
            raise

    @staticmethod
    def _parse_sentiment_response(content: str) -> SentimentResult:
        """Parse the JSON response from the sentiment model."""
        # Strip markdown code fences if the model wraps the JSON
        cleaned = content.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (e.g. ```json)
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse sentiment model response as JSON: {exc}. "
                f"Raw content: {content[:200]}"
            ) from exc

        score = float(data.get("score", 0.0))
        confidence = float(data.get("confidence", 0.5))
        reasoning = str(data.get("reasoning", "No reasoning provided."))

        # Clamp values to expected ranges
        score = max(-1.0, min(1.0, score))
        confidence = max(0.0, min(1.0, confidence))

        return SentimentResult(
            score=score,
            confidence=confidence,
            reasoning=reasoning,
        )

    @staticmethod
    def _cache_key(text: str, market_context: str) -> str:
        """Create a deterministic cache key from text and market context."""
        raw = f"{text}|||{market_context}"
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _format_summary(market_title: str, sentiment: MarketSentiment) -> str:
        """Format a MarketSentiment into a human-readable summary string."""
        if sentiment.num_articles == 0:
            return (
                f"No relevant news articles found for: {market_title}. "
                "Sentiment: neutral (no signal)."
            )

        # Determine sentiment label
        ws = sentiment.relevance_weighted_score
        if ws > 0.3:
            label = "positive"
        elif ws < -0.3:
            label = "negative"
        elif ws > 0.1:
            label = "slightly positive"
        elif ws < -0.1:
            label = "slightly negative"
        else:
            label = "neutral"

        lines = [
            f"News Sentiment for '{market_title}':",
            f"  Overall sentiment: {label} (weighted score: {ws:+.2f})",
            f"  Based on {sentiment.num_articles} relevant article(s).",
            "  Key signals:",
        ]

        for item in sentiment.article_sentiments[:3]:
            direction = "positive" if item.sentiment.score > 0 else "negative"
            lines.append(
                f"    - [{direction} {item.sentiment.score:+.2f}] "
                f"{item.article.title[:90]} "
                f"(relevance: {item.relevance_score:.0%})"
            )
            if item.sentiment.reasoning:
                lines.append(f"      Reasoning: {item.sentiment.reasoning}")

        return "\n".join(lines)
