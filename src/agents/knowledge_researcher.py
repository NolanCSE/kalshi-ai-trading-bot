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

from typing import Dict, List, Optional
from pathlib import Path

import yaml

from src.agents.base_agent import BaseAgent
from src.utils.knowledge_library import get_knowledge_library, RetrievedPassage
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
    # Use Mistral via OpenRouter for low-censorship retrieval
    # This allows processing controversial/truth-seeking material without refusal
    DEFAULT_MODEL = "mistralai/mistral-7b-instruct"
    
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
    
    async def _retrieve_relevant_knowledge(self, market_data: Dict) -> List[RetrievedPassage]:
        """Retrieve relevant passages from knowledge library."""
        try:
            kl = await self._get_knowledge_library()
            
            # Build query from market title and category
            query_parts = [
                market_data.get("title", ""),
                market_data.get("category", ""),
                market_data.get("rules", "")
            ]
            query = " ".join(filter(None, query_parts))
            
            # Retrieve passages
            passages = await kl.retrieve_relevant_passages(
                query=query,
                top_k=5
            )
            
            logger.info(f"Retrieved {len(passages)} passages from knowledge library")
            return passages
            
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
            
            # Search for relevant news
            query = market_data.get("title", "")
            news_items = await aggregator.search_news(query, max_results=5)
            
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
    
    def _build_knowledge_context(self, passages: List[RetrievedPassage]) -> List[Dict]:
        """Build structured context from retrieved passages."""
        if not passages:
            return []
        
        return [
            {
                "text": p.text[:500],  # Truncate long passages
                "source": p.source,
                "category": p.category,
                "relevance": p.similarity_score
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
            
            # Retrieve knowledge
            passages = await self._retrieve_relevant_knowledge(market_data)
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
