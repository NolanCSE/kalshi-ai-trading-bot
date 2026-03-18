"""
Knowledge Researcher Agent -- retrieves relevant context for the Trader.

This agent retrieves relevant passages from the user's knowledge library
and surfaces worldview context to help the Trader synthesize the debate.

Key difference from previous IdeologyAgent: This agent does NOT make predictions.
It only retrieves knowledge and context. The Trader uses this to interpret the debate.

Key features:
- RAG-based retrieval from personal knowledge library
- Worldview context surfacing
- Relevant theory identification
- News context gathering
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import yaml

from src.agents.base_agent import BaseAgent
from src.utils.knowledge_library import get_knowledge_library, RetrievedPassage, WebArticle
from src.data.news_aggregator import NewsAggregator
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("knowledge_researcher")


class KnowledgeResearcher(BaseAgent):
    """
    Retrieves relevant knowledge and worldview context for the Trader.
    
    This agent:
    1. Retrieves relevant theory from the user's knowledge library
    2. Surfaces the user's worldview framework as context
    3. Gathers relevant news for context
    4. Does NOT generate predictions - that's the Trader's job
    
    The Trader receives this context and uses it to synthesize the debate
    through the worldview lens.
    """
    
    AGENT_NAME = "knowledge_researcher"
    AGENT_ROLE = "knowledge_researcher"
    # Use Mistral Nemo via OpenRouter for low-censorship retrieval.
    # mistral-7b-instruct was delisted from OpenRouter; Nemo is the current
    # lightweight Mistral model with the same open-weights spirit.
    # Free fallback: google/gemma-3-12b-it:free
    DEFAULT_MODEL = "mistralai/mistral-nemo"
    
    SYSTEM_PROMPT = (
        "You are a Knowledge Researcher. Your job is to find and present "
        "relevant information from a knowledge library to help a decision-maker.\n\n"
        "You do NOT make predictions or decisions.\n"
        "You do NOT evaluate whether the knowledge is 'correct'.\n"
        "You simply RETRIEVE and SUMMARIZE relevant context.\n\n"
        "Your output should help someone understand:\n"
        "1. What frameworks/theories from the knowledge library apply here\n"
        "2. What the worldview perspective includes\n"
        "3. What relevant news/context exists\n\n"
        "Return your research as JSON:\n"
        '  "worldview_applies": boolean (does this market touch the worldview?),\n'
        '  "retrieved_passages": list (relevant texts from knowledge library),\n'
        '  "worldview_context": string (summary of applicable beliefs/frameworks),\n'
        '  "news_summary": string (relevant current news),\n'
        '  "key_frameworks": list (which theoretical frameworks apply),\n'
        '  "reasoning": string (why these passages are relevant)'
    )
    
    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name)
        self.worldview = self._load_worldview()
        self.knowledge_library = None
        self.news_aggregator = None
        
        # Web research configuration
        web_research_config = self.worldview.get("web_research", {})
        self.web_research_enabled = web_research_config.get("enabled", True)
        self.target_domains = web_research_config.get("target_domains", [])
        self.max_web_results = web_research_config.get("search", {}).get("max_results_per_query", 5)
        self.max_articles_to_ingest = web_research_config.get("ingestion", {}).get("max_articles_per_market", 10)
        
    def _load_worldview(self) -> Dict:
        """Load worldview configuration from YAML."""
        worldview_path = Path("config/worldview.yaml")
        if not worldview_path.exists():
            logger.warning("No worldview.yaml found, using empty worldview")
            return {"ideology_agent": {"enabled": False}}
        
        try:
            with open(worldview_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load worldview: {e}")
            return {"ideology_agent": {"enabled": False}}
    
    async def _get_knowledge_library(self):
        """Lazy initialization of knowledge library."""
        if self.knowledge_library is None:
            self.knowledge_library = await get_knowledge_library()
        return self.knowledge_library
    
    async def _get_news_aggregator(self):
        """Lazy initialization of news aggregator."""
        if self.news_aggregator is None:
            self.news_aggregator = NewsAggregator()
        return self.news_aggregator
    
    async def _reformulate_queries(
        self,
        market_data: Dict,
        get_completion: callable
    ) -> Dict[str, List[str]]:
        """Generate optimized search queries from market data.
        
        Uses an LLM to reformulate the market title into:
        - web_search_queries: 2-3 short keyword phrases for DuckDuckGo search
        - knowledge_retrieval_queries: 5-6 semantic queries for vector search
        
        Knowledge retrieval queries are more detailed and semantic since they need
        to match against embedded document chunks, not just find web articles.
        
        Returns a dict with 'web_search_queries' and 'knowledge_retrieval_queries' lists.
        Falls back to basic queries on failure.
        """
        try:
            title = market_data.get("title", "")
            category = market_data.get("category", "")
            rules = market_data.get("rules", "")
            
            prompt = f"""You are a query reformulation assistant. Given a prediction market, 
generate optimized search queries for two different purposes:

1. WEB SEARCH QUERIES: Generate 2-3 short keyword phrases (not questions) that would 
   find relevant blog posts or articles on sites like LessWrong, Mises Institute, or Substack.
   - Strip "Will X happen?" or "Will X yes/no?" framing
   - Remove specific dates like "March 2026" or "Q1 2026"
   - Remove "basis points", specific prices, or platform-specific language
   - Focus on the core topic and concepts
   - Keep these short and keyword-like (e.g., "China Taiwan invasion military")
   
2. KNOWLEDGE RETRIEVAL QUERIES: Generate 5-6 detailed semantic phrases that would find 
   relevant passages in an academic library or embedded knowledge base through vector similarity search.
   - These should be longer and more descriptive
   - Include the original market question rephrased as a statement
   - Use formal academic-style terminology
   - Focus on underlying theories, frameworks, and historical precedents
   - Include relevant domain keywords and concepts
   - Example: "What are the historical precedents for China invading Taiwan and what military and economic factors influence this decision?"

Return ONLY a JSON object with this structure:
{{
  "web_search_queries": ["query1", "query2", "query3"],
  "knowledge_retrieval_queries": ["query1", "query2", "query3", "query4", "query5", "query6"]
}}

Market Title: {title}
Market Category: {category}
Market Rules: {rules}

Generate the queries:"""

            response = await get_completion(prompt)
            
            if response is None:
                logger.warning("Query reformulation returned None, using fallbacks")
                return self._get_fallback_queries(market_data)
            
            parsed = self._extract_json(response)
            if parsed is None:
                logger.warning("Query reformulation failed to parse JSON, using fallbacks")
                return self._get_fallback_queries(market_data)
            
            web_queries = parsed.get("web_search_queries", [])
            knowledge_queries = parsed.get("knowledge_retrieval_queries", [])
            
            # Validate we got proper lists
            if not isinstance(web_queries, list) or not isinstance(knowledge_queries, list):
                logger.warning("Query reformulation returned invalid structure, using fallbacks")
                return self._get_fallback_queries(market_data)
            
            # Ensure we have minimum queries (2-3 for web, 5-6 for knowledge)
            if len(web_queries) < 2:
                web_queries = web_queries + [title] if web_queries else [title]
            web_queries = web_queries[:3]  # Max 3 for web
            
            if len(knowledge_queries) < 3:
                knowledge_queries = knowledge_queries + [title] if knowledge_queries else [title]
            knowledge_queries = knowledge_queries[:6]  # Max 6 for knowledge retrieval
            
            # Add original title as fallback for knowledge retrieval (captures direct matches)
            if title not in knowledge_queries:
                knowledge_queries = [title] + knowledge_queries[:5]
            
            logger.info(f"Reformulated queries - Web: {web_queries}, Knowledge: {knowledge_queries}")
            
            return {
                "web_search_queries": web_queries,
                "knowledge_retrieval_queries": knowledge_queries
            }
            
        except Exception as e:
            logger.error(f"Query reformulation failed: {e}")
            return self._get_fallback_queries(market_data)
    
    def _get_fallback_queries(self, market_data: Dict) -> Dict[str, List[str]]:
        """Generate basic fallback queries when reformulation fails."""
        title = market_data.get("title", "")
        category = market_data.get("category", "")
        
        # Simple fallback: use title + category as both query types
        basic_query = f"{title} {category}".strip()
        
        return {
            "web_search_queries": [basic_query],
            "knowledge_retrieval_queries": [basic_query]
        }
    
    async def _search_web_articles(self, query: str) -> List[WebArticle]:
        """Search for relevant articles on configured target domains.
        
        Uses DuckDuckGo to search within specified domains.
        """
        if not self.web_research_enabled:
            logger.debug("Web research disabled")
            return []
        
        if not self.target_domains:
            logger.warning("No target domains configured for web research")
            return []
        
        try:
            from ddgs import DDGS
            
            articles = []
            
            with DDGS() as ddgs:
                for domain in self.target_domains:
                    # Search with site: operator
                    search_query = f"site:{domain} {query}"
                    
                    try:
                        results = ddgs.text(
                            search_query,
                            max_results=self.max_web_results
                        )
                        
                        for r in results:
                            # Check if this URL is already in results
                            url = r.get("href", "")
                            if not url:
                                continue
                            
                            # Check if domain matches (with www variant)
                            domain_match = any(
                                d.replace("www.", "") in url.replace("www.", "")
                                for d in [domain]
                            )
                            
                            if not domain_match:
                                continue
                            
                            articles.append(WebArticle(
                                url=url,
                                title=r.get("title", ""),
                                snippet=r.get("body", ""),
                                relevance_score=0.5  # Default, could be improved
                            ))
                            
                    except Exception as e:
                        logger.warning(f"Search failed for domain {domain}: {e}")
                        continue
            
            logger.info(f"Found {len(articles)} web articles for query: {query[:50]}")
            return articles
            
        except ImportError:
            logger.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
            return []
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []

    async def _ingest_web_articles(self, articles: List[WebArticle]) -> Dict:
        """Ingest found web articles into the knowledge library.
        
        Returns stats about what was ingested.
        """
        if not articles:
            return {"ingested": 0, "skipped": 0, "failed": 0}
        
        try:
            kl = await self._get_knowledge_library()
            
            urls = [a.url for a in articles]
            
            # Use the knowledge library's batch ingest method
            stats = await kl.ingest_multiple_urls(
                urls=urls,
                category="web_articles",
                max_to_ingest=self.max_articles_to_ingest
            )
            
            logger.info(f"Web article ingestion complete: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to ingest web articles: {e}")
            return {"ingested": 0, "skipped": 0, "failed": len(articles)}

    async def _get_web_context(
        self,
        market_data: Dict,
        web_queries: List[str]
    ) -> str:
        """Search for and ingest relevant web articles, then return context.
        
        Uses reformulated web search queries instead of raw market title.
        
        Args:
            market_data: The market data dict
            web_queries: List of reformulated queries from _reformulate_queries
            
        Returns a summary of what was found/ingested.
        """
        if not self.web_research_enabled or not self.target_domains:
            return "Web research disabled or no target domains configured"
        
        try:
            # Search for articles using each reformulated query
            all_articles = []
            seen_urls = set()
            
            for query in web_queries:
                articles = await self._search_web_articles(query)
                
                # Deduplicate across queries
                for article in articles:
                    if article.url not in seen_urls:
                        all_articles.append(article)
                        seen_urls.add(article.url)
            
            if not all_articles:
                return f"No relevant articles found on configured domains: {', '.join(self.target_domains)}"
            
            # Ingest articles (deduplication happens inside)
            ingest_stats = await self._ingest_web_articles(all_articles)
            
            # Build context string
            lines = [f"=== WEB RESEARCH ==="]
            lines.append(f"Searched domains: {', '.join(self.target_domains)}")
            lines.append(f"Search queries used: {web_queries}")
            lines.append(f"Found {len(all_articles)} unique articles")
            lines.append(f"Ingested: {ingest_stats['ingested']}, Skipped (duplicate): {ingest_stats['skipped']}, Failed: {ingest_stats['failed']}")
            lines.append("")
            lines.append("Top articles found:")
            
            for i, article in enumerate(all_articles[:5], 1):
                lines.append(f"  {i}. {article.title}")
                lines.append(f"     URL: {article.url}")
                if article.snippet:
                    lines.append(f"     Snippet: {article.snippet[:200]}...")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Web research failed: {e}")
            return f"Web research error: {str(e)}"
    
    def _should_apply_worldview(self, market_data: Dict) -> bool:
        """Determine if worldview applies to this market."""
        ideology_config = self.worldview.get("ideology_agent", {})
        if not ideology_config.get("enabled", False):
            return False
        
        category = market_data.get("category", "").lower()
        title = market_data.get("title", "").lower()
        
        # Check excluded categories
        excluded = self.worldview.get("excluded_markets", [])
        if category in excluded:
            return False
        
        # Check if category is in applicable list
        applicable = self.worldview.get("applicable_categories", [])
        if category not in applicable:
            return False
        
        # Skip obvious non-ideological markets
        non_ideological_keywords = [
            "will it rain", "temperature", "sports", "game", "match",
            "oscars", "grammy", "box office", "rotten tomatoes",
            "super bowl", "world cup", "nba", "nfl", "mlb"
        ]
        if any(kw in title for kw in non_ideological_keywords):
            return False
        
        return True
    
    async def _retrieve_relevant_knowledge(
        self,
        market_data: Dict,
        knowledge_queries: List[str]
    ) -> List[RetrievedPassage]:
        """Retrieve relevant passages from knowledge library using multiple queries.
        
        Runs each query against the vector store, then merges and deduplicates
        the results, keeping the highest similarity score for any duplicates.
        
        Args:
            market_data: The market data dict
            knowledge_queries: List of reformulated queries from _reformulate_queries
            
        Returns:
            List of RetrievedPassage objects, deduplicated and ranked by similarity
        """
        try:
            kl = await self._get_knowledge_library()
            
            all_passages = []
            seen_sources = set()  # Track unique sources
            
            # Run each query and collect passages (increased top_k for more diversity)
            for query in knowledge_queries:
                passages = await kl.retrieve_relevant_passages(
                    query=query,
                    top_k=15  # Increased from 5 for more candidate diversity
                )
                
                # Collect passages from all queries
                for p in passages:
                    all_passages.append(p)
            
            # Deduplicate by source - but ensure at least 1 from each unique source
            # First pass: collect top passage from each unique source
            source_to_best_passage = {}
            for p in all_passages:
                src = p.source
                if src not in source_to_best_passage:
                    source_to_best_passage[src] = p
                elif p.similarity_score > source_to_best_passage[src].similarity_score:
                    source_to_best_passage[src] = p
            
            # Get unique sources that appeared in results
            unique_sources = list(source_to_best_passage.keys())
            
            # Second pass: get top-scoring passages from each source
            # Sort all passages by score, then ensure we include at least 1 from each source
            all_passages.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Build final list: include at least 1 from each unique source, then fill with top scores
            final_passages = []
            sources_added = set()
            
            # First, add top passage from each unique source
            for src in unique_sources:
                if src not in sources_added:
                    final_passages.append(source_to_best_passage[src])
                    sources_added.add(src)
            
            # Then add remaining top-scoring passages until we have 5
            for p in all_passages:
                if p.source not in sources_added:
                    final_passages.append(p)
                    sources_added.add(p.source)
                if len(final_passages) >= 5:
                    break
            
            # Sort final by similarity score
            final_passages.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(f"Retrieved {len(final_passages)} passages from {len(unique_sources)} unique sources")
            return final_passages
            
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge: {e}")
            return []
    
    async def _get_relevant_news(self, market_data: Dict) -> str:
        """Fetch relevant news for context."""
        try:
            # Check if news integration is enabled
            news_config = self.worldview.get("news_integration", {})
            if not news_config.get("enabled", True):
                return "News integration disabled"
            
            # Get news aggregator
            aggregator = await self._get_news_aggregator()
            
            # Fetch all RSS feeds first to populate cache
            await aggregator.fetch_all()
            
            # Get relevant news for this market
            query = market_data.get("title", "")
            news_items_with_scores = aggregator.get_relevant_articles(query, max_results=5)
            news_items = [item for item, score in news_items_with_scores]
            
            if not news_items:
                return "No relevant news found"
            
            # Compile news text
            news_text = "\n\n".join([
                f"Headline: {item.get('title', 'N/A')}\n"
                f"Source: {item.get('source', 'N/A')}\n"
                f"Summary: {item.get('summary', item.get('description', 'N/A'))}"
                for item in news_items
            ])
            
            return news_text
            
        except Exception as e:
            logger.error(f"Failed to get news: {e}")
            return f"Error fetching news: {str(e)}"
    
    def _build_worldview_context(self, market_data: Dict) -> str:
        """Build context string from worldview configuration."""
        lines = []
        
        worldview = self.worldview.get("worldview", {})
        
        # Add framework descriptions
        for category, data in worldview.items():
            if isinstance(data, dict):
                framework = data.get("framework", "unknown")
                confidence = data.get("confidence_in_framework", 0.5)
                lines.append(f"{category.upper()}: {framework} (confidence: {confidence})")
                
                # Add key beliefs
                for belief in data.get("key_beliefs", []):
                    lines.append(f"  - {belief}")
        
        # Add domain expertise
        expertise = self.worldview.get("domain_expertise", [])
        if expertise:
            lines.append("\nDOMAIN EXPERTISE:")
            for domain in expertise:
                area = domain.get("area", "unknown")
                level = domain.get("expertise_level", "beginner")
                years = domain.get("years_experience", 0)
                lines.append(f"  {area}: {level} ({years} years)")
                
                for insight in domain.get("specific_insights", [])[:3]:
                    lines.append(f"    - {insight}")
        
        return "\n".join(lines) if lines else "No worldview context configured"
    
    def _build_knowledge_context(self, passages: List[RetrievedPassage]) -> str:
        """Build readable context from retrieved passages."""
        if not passages:
            return "No relevant knowledge passages found."
        
        lines = ["=== RELEVANT KNOWLEDGE ===\n"]
        
        for i, p in enumerate(passages, 1):
            lines.append(f"\n--- Passage {i} ---\n")
            lines.append(p.text)
        
        return "".join(lines)
    
    def _build_citations(self, passages: List[RetrievedPassage]) -> List[Dict]:
        """Build citation info for logging."""
        return [
            {
                "source": p.source,
                "page_number": p.page_number,
                "relevance_score": p.similarity_score
            }
            for p in passages
        ]
    
    def _build_prompt(self, market_data: Dict, context: Dict) -> str:
        """Build research prompt."""
        summary = self.format_market_summary(market_data)
        
        # Check if worldview applies
        worldview_applies = self._should_apply_worldview(market_data)
        
        if not worldview_applies:
            return f"""
Market: {market_data.get('title', 'Unknown')}

This market is outside the scope of the knowledge library worldview
(sports, weather, entertainment, or pure chance).

Return minimal research:
```json
{{
  "worldview_applies": false,
  "retrieved_passages": [],
  "worldview_context": "Worldview framework does not apply to this market category",
  "news_summary": "N/A",
  "key_frameworks": [],
  "reasoning": "Market category excluded from worldview application"
}}
```
"""
        
        # Get worldview context
        worldview_context = self._build_worldview_context(market_data)
        
        return f"""
=== MARKET ===
{summary}

=== WORLDVIEW CONTEXT ===
{worldview_context}

[RETRIEVED_PASSAGES_WILL_BE_INSERTED_HERE]

[CURRENT_NEWS_WILL_BE_INSERTED_HERE]

[WEB_ARTICLES_WILL_BE_INSERTED_HERE]

=== YOUR TASK ===
You are a research assistant. Your job is to:

1. Identify which passages from the knowledge library are most relevant
2. Summarize what theoretical frameworks apply to this market
3. Note what the worldview perspective includes
4. Summarize relevant current news

Do NOT make predictions.
Do NOT evaluate if the worldview is 'correct'.
Simply PRESENT the relevant context.

Return your research as a JSON object inside a ```json``` code block.
"""
    
    async def analyze(
        self,
        market_data: dict,
        context: dict,
        get_completion: callable,
    ) -> dict:
        """
        Conduct research and return relevant context.
        
        This does NOT generate predictions - only retrieves context.
        """
        import time
        start_time = time.time()
        
        try:
            # Build base prompt
            base_prompt = self._build_prompt(market_data, context)
            
            # Check if worldview applies
            worldview_applies = self._should_apply_worldview(market_data)
            
            if not worldview_applies:
                return {
                    "worldview_applies": False,
                    "retrieved_passages": [],
                    "worldview_context": "Worldview framework does not apply",
                    "news_summary": "N/A",
                    "key_frameworks": [],
                    "reasoning": "Market category excluded from worldview application",
                    "_agent": self.name,
                    "_model": self.model_name,
                    "_elapsed_seconds": round(time.time() - start_time, 2),
                }
            
            # Reformulate queries for better web search and knowledge retrieval
            reformulated = await self._reformulate_queries(market_data, get_completion)
            web_queries = reformulated.get("web_search_queries", [market_data.get("title", "")])
            knowledge_queries = reformulated.get("knowledge_retrieval_queries", [market_data.get("title", "")])
            
            # Get web articles context (search + ingest) using reformulated queries
            web_context = ""
            if worldview_applies:
                web_context = await self._get_web_context(market_data, web_queries)
            
            # Retrieve knowledge using reformulated queries
            passages = await self._retrieve_relevant_knowledge(market_data, knowledge_queries)
            passages_context = self._build_knowledge_context(passages)
            
            # Get news
            news_summary = await self._get_relevant_news(market_data)
            
            # Inject into prompt
            full_prompt = base_prompt.replace(
                "[RETRIEVED_PASSAGES_WILL_BE_INSERTED_HERE]",
                f"=== RETRIEVED KNOWLEDGE PASSAGES ===\n{passages_context}"
            ).replace(
                "[CURRENT_NEWS_WILL_BE_INSERTED_HERE]",
                f"=== CURRENT NEWS ===\n{news_summary[:1000]}"  # Truncate
            ).replace(
                "[WEB_ARTICLES_WILL_BE_INSERTED_HERE]",
                f"\n\n{web_context}"
            )
            
            # Add system prompt
            if self.SYSTEM_PROMPT:
                full_prompt = f"{self.SYSTEM_PROMPT}\n\n{full_prompt}"
            
            # Get completion
            raw_response = await get_completion(full_prompt)
            
            if raw_response is None:
                return self._error_result("Model returned None")
            
            # Parse response
            parsed = self._extract_json(raw_response)
            if parsed is None:
                return self._error_result(f"Failed to extract JSON: {raw_response[:300]}")
            
            result = self._parse_result(parsed)

            # Add passages to result
            result["retrieved_passages"] = passages_context
            result["knowledge_citations"] = self._build_citations(passages)
            result["news_summary"] = news_summary[:500]  # Truncate
            
            # Add metadata
            result["_agent"] = self.name
            result["_model"] = self.model_name
            result["_elapsed_seconds"] = round(time.time() - start_time, 2)
            
            return result
            
        except Exception as e:
            return self._error_result(str(e))
    
    def _parse_result(self, raw_json: Dict) -> Dict:
        """Parse and validate researcher output."""
        worldview_applies = bool(raw_json.get("worldview_applies", True))
        retrieved_passages = raw_json.get("retrieved_passages", [])
        worldview_context = str(raw_json.get("worldview_context", ""))
        news_summary = str(raw_json.get("news_summary", ""))
        key_frameworks = raw_json.get("key_frameworks", [])
        reasoning = str(raw_json.get("reasoning", ""))
        
        return {
            "worldview_applies": worldview_applies,
            "retrieved_passages": retrieved_passages,
            "worldview_context": worldview_context,
            "news_summary": news_summary,
            "key_frameworks": key_frameworks,
            "reasoning": reasoning,
        }


# Backwards compatibility alias
IdeologyAgent = KnowledgeResearcher
