"""
Ideology Agent -- applies user's worldview to market analysis with RAG support.

This agent retrieves relevant passages from the user's knowledge library,
integrates current news through their ideological framework, and generates
predictions based on their worldview.

Key features:
- RAG-based retrieval from personal knowledge library
- Confidence inference from belief statement wording
- Worldview-informed news interpretation
- Divergence detection and override triggers
"""

import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import yaml

from src.agents.base_agent import BaseAgent
from src.utils.knowledge_library import get_knowledge_library, RetrievedPassage
from src.data.news_aggregator import NewsAggregator
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("ideology_agent")


class IdeologyAgent(BaseAgent):
    """
    Applies user's worldview to generate probability estimates.
    
    This agent:
    1. Retrieves relevant theory from the user's knowledge library
    2. Fetches and interprets current news through the worldview lens
    3. Infers confidence from belief statement wording
    4. Generates predictions that may diverge from consensus
    5. Triggers manual review when divergence exceeds thresholds
    """
    
    AGENT_NAME = "ideology_analyst"
    AGENT_ROLE = "ideology_analyst"
    DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
    
    SYSTEM_PROMPT = (
        "You are a sophisticated analyst who holds a specific worldview framework. "
        "You do NOT analyze this framework from the outside - you think FROM WITHIN IT.\n\n"
        "Your beliefs include:\n"
        "- The theoretical frameworks in your knowledge library\n"
        "- The domain expertise areas you possess\n"
        "- The key beliefs that shape your understanding\n\n"
        "When analyzing markets:\n"
        "1. Reason as someone who GENUINELY HOLDS these beliefs\n"
        "2. Use retrieved knowledge as YOUR source of understanding\n"
        "3. Interpret news THROUGH your ideological framework (not as an observer of it)\n"
        "4. Generate the probability YOU would assign based on YOUR perspective\n"
        "5. Let your confidence reflect the strength of YOUR convictions\n\n"
        "IMPORTANT:\n"
        "- Do NOT question whether your worldview is 'correct'\n"
        "- Do NOT comment on potential biases in your framework\n"
        "- Do NOT note contradictions with other perspectives\n"
        "- Simply PREDICT from your perspective with conviction\n"
        "- Other agents in the ensemble represent different viewpoints\n\n"
        "Return your analysis as JSON:\n"
        '  "probability": float (0.0-1.0, YOUR prediction),\n'
        '  "confidence": float (0.0-1.0, YOUR confidence in this view),\n'
        '  "worldview_applies": boolean (does this market touch your expertise?),\n'
        '  "contrarian_signal": boolean (your view differs from consensus),\n'
        '  "ensemble_probability": float (what others predicted - for reference),\n'
        '  "worldview_adjustment": float (how much you differ),\n'
        '  "retrieved_theories": list (which sources shaped YOUR thinking),\n'
        '  "news_interpretation": string (how YOU interpret current events),\n'
        '  "reasoning": string (YOUR reasoning from YOUR perspective),\n'
        '  "caveats": string (uncertainties within YOUR framework, not about it)'
    )
    
    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name)
        self.worldview = self._load_worldview()
        self.trigger_config = self.worldview.get("ideology_agent", {}).get("override_triggers", {})
        self.confidence_config = self.worldview.get("ideology_agent", {}).get("confidence_inference", {})
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
            logger.debug(f"Worldview excluded for category: {category}")
            return False
        
        # Check if category is in applicable list
        applicable = self.worldview.get("applicable_categories", [])
        if category not in applicable:
            logger.debug(f"Worldview not applicable for category: {category}")
            return False
        
        # Skip obvious non-ideological markets
        non_ideological_keywords = [
            "will it rain", "temperature", "sports", "game", "match",
            "oscars", "grammy", "box office", "rotten tomatoes",
            "super bowl", "world cup", "nba", "nfl", "mlb"
        ]
        if any(kw in title for kw in non_ideological_keywords):
            logger.debug(f"Worldview excluded for non-ideological market: {title[:50]}")
            return False
        
        return True
    
    def _infer_confidence(self, reasoning: str) -> float:
        """
        Infer confidence from belief statement wording.
        
        Analyzes the reasoning text for confidence-indicating phrases
        and returns a confidence score.
        """
        if not self.confidence_config.get("enabled", True):
            return 0.50  # Default neutral
        
        text_lower = reasoning.lower()
        
        high_phrases = self.confidence_config.get("high_confidence_phrases", [])
        medium_phrases = self.confidence_config.get("medium_confidence_phrases", [])
        low_phrases = self.confidence_config.get("low_confidence_phrases", [])
        
        # Count occurrences
        high_count = sum(1 for phrase in high_phrases if phrase in text_lower)
        medium_count = sum(1 for phrase in medium_phrases if phrase in text_lower)
        low_count = sum(1 for phrase in low_phrases if phrase in text_lower)
        
        # Calculate base confidence
        total_indicators = high_count + medium_count + low_count
        if total_indicators == 0:
            return 0.50  # Default when no indicators
        
        # Weighted calculation
        weighted_sum = (high_count * 0.85) + (medium_count * 0.60) + (low_count * 0.35)
        base_confidence = weighted_sum / total_indicators
        
        # Adjust based on number of indicators (more indicators = more confident)
        confidence_boost = min(0.10, total_indicators * 0.02)
        
        final_confidence = min(0.95, base_confidence + confidence_boost)
        
        logger.debug(f"Inferred confidence: {final_confidence:.2f} "
                    f"(high={high_count}, med={medium_count}, low={low_count})")
        
        return final_confidence
    
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
    
    async def _get_worldview_informed_news(
        self,
        market_data: Dict,
        get_completion
    ) -> str:
        """
        Fetch and interpret news through worldview lens.
        
        Gets current news and filters it through the ideological framework.
        """
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
            
            # Get worldview framework for interpretation
            worldview_text = self._build_worldview_context(market_data)
            
            # Create interpretation prompt - from the worldview's perspective
            interpretation_prompt = f"""
You are reading current news. You hold the following beliefs and expertise:

YOUR WORLDVIEW:
{worldview_text}

CURRENT NEWS:
{news_text}

TASK - Read this news AS SOMEONE WHO HOLDS THESE BELIEFS:
1. What facts does the news present?
2. How do YOU interpret these events given YOUR framework?
3. What do you EXPECT to happen based on YOUR understanding?
4. What would SURPRISE you given YOUR beliefs?

Do NOT analyze your worldview from the outside. Simply interpret the news FROM YOUR PERSPECTIVE.

Provide YOUR interpretation of these events.
"""
            
            # Get interpretation from LLM
            interpretation = await get_completion(interpretation_prompt)
            
            return interpretation or "News interpretation unavailable"
            
        except Exception as e:
            logger.error(f"Failed to get worldview-informed news: {e}")
            return f"Error fetching news: {str(e)}"
    
    def _build_worldview_context(self, market_data: Dict) -> str:
        """Build context string from worldview configuration."""
        lines = ["WORLDVIEW FRAMEWORK:"]
        
        worldview = self.worldview.get("worldview", {})
        
        # Add framework descriptions
        for category, data in worldview.items():
            if isinstance(data, dict):
                framework = data.get("framework", "unknown")
                confidence = data.get("confidence_in_framework", 0.5)
                lines.append(f"\n{category.upper()}: {framework} (confidence: {confidence})")
                
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
                
                for insight in domain.get("specific_insights", [])[:3]:  # Limit to 3
                    lines.append(f"    - {insight}")
        
        return "\n".join(lines)
    
    def _build_knowledge_context(self, passages: List[RetrievedPassage]) -> str:
        """Build context string from retrieved knowledge passages."""
        if not passages:
            return "No relevant passages retrieved from knowledge library."
        
        lines = ["RETRIEVED KNOWLEDGE:"]
        
        for i, passage in enumerate(passages, 1):
            lines.append(f"\n[{i}] Source: {passage.source} (relevance: {passage.similarity_score:.2f})")
            lines.append(f"Category: {passage.category}")
            lines.append(f"Text: {passage.text[:500]}...")  # Truncate long passages
        
        return "\n".join(lines)
    
    def _get_ensemble_consensus(self, context: Dict) -> Tuple[float, str]:
        """
        Extract ensemble consensus from context.
        
        Returns:
            Tuple of (consensus_probability, consensus_reasoning)
        """
        probabilities = []
        
        # Check for forecaster result
        if context.get("forecaster_result"):
            fc = context["forecaster_result"]
            if "probability" in fc:
                probabilities.append(("forecaster", fc.get("probability", 0.5)))
        
        # Check for bull result
        if context.get("bull_result"):
            bull = context["bull_result"]
            if "probability" in bull:
                probabilities.append(("bull", bull.get("probability", 0.5)))
        
        # Check for bear result
        if context.get("bear_result"):
            bear = context["bear_result"]
            if "probability" in bear:
                probabilities.append(("bear", bear.get("probability", 0.5)))
        
        if not probabilities:
            return 0.5, "No ensemble consensus available"
        
        # Calculate simple average
        avg_prob = sum(p for _, p in probabilities) / len(probabilities)
        
        reasoning = f"Ensemble consensus from {len(probabilities)} agents: " + \
                   ", ".join([f"{name}={p:.2f}" for name, p in probabilities])
        
        return avg_prob, reasoning
    
    def _build_prompt(self, market_data: Dict, context: Dict) -> str:
        """Build comprehensive prompt with worldview, knowledge, and news."""
        
        # Check if worldview applies
        if not self._should_apply_worldview(market_data):
            return self._build_neutral_prompt(market_data)
        
        summary = self.format_market_summary(market_data)
        
        # Get worldview context
        worldview_context = self._build_worldview_context(market_data)
        
        # Get ensemble consensus
        ensemble_prob, ensemble_reasoning = self._get_ensemble_consensus(context)
        
        # Build the main prompt (knowledge and news added dynamically in analyze)
        return f"""
=== MARKET ===
{summary}

=== WHAT OTHER ANALYSTS THINK ===
Other analysts predict: {ensemble_prob:.2f} YES probability
Their reasoning: {ensemble_reasoning}

=== YOUR BELIEFS AND EXPERTISE ===
{worldview_context}

[KNOWLEDGE_PASSAGES_WILL_BE_INSERTED_HERE]

[CURRENT_NEWS_INTERPRETATION_WILL_BE_INSERTED_HERE]

=== YOUR TASK ===
You are analyzing this market from YOUR perspective, using YOUR knowledge and beliefs.

1. Based on the sources YOU'VE read (retrieved passages), what do YOU think?
2. Given YOUR interpretation of current events, what's YOUR prediction?
3. What probability do YOU assign? (Not what others think - what YOU think)
4. How confident are YOU in this view?

You are ONE voice in a debate. Other agents represent different perspectives. Your job is to provide YOUR genuine view, not to critique or question your own framework.

Return ONLY a JSON object inside a ```json``` code block with YOUR prediction.
"""
    
    def _build_neutral_prompt(self, market_data: Dict) -> str:
        """Build neutral prompt when worldview doesn't apply."""
        return f"""
Market: {market_data.get('title', 'Unknown')}

This market appears to be outside the scope of the worldview framework
(sports, weather, entertainment, or pure chance).

Return neutral analysis:
```json
{{
  "probability": {market_data.get('yes_price', 0.5)},
  "confidence": 0.0,
  "worldview_applies": false,
  "contrarian_signal": false,
  "ensemble_probability": {market_data.get('yes_price', 0.5)},
  "worldview_adjustment": 0.0,
  "retrieved_theories": [],
  "news_interpretation": "Worldview framework does not apply to this market category.",
  "reasoning": "Worldview framework does not apply to this market category.",
  "caveats": "N/A - worldview not applied"
}}
```
"""
    
    async def analyze(
        self,
        market_data: dict,
        context: dict,
        get_completion: callable,
    ) -> dict:
        """
        Override analyze to inject knowledge and news dynamically.
        """
        # Build base prompt
        base_prompt = self._build_prompt(market_data, context)
        
        # Check if worldview applies
        if not self._should_apply_worldview(market_data):
            return {
                "probability": market_data.get("yes_price", 0.5),
                "confidence": 0.0,
                "worldview_applies": False,
                "contrarian_signal": False,
                "ensemble_probability": market_data.get("yes_price", 0.5),
                "worldview_adjustment": 0.0,
                "retrieved_theories": [],
                "news_interpretation": "Worldview framework does not apply",
                "reasoning": "Worldview framework does not apply to this market category",
                "caveats": "N/A",
                "_agent": self.name,
                "_model": self.model_name,
            }
        
        # Retrieve knowledge
        passages = await self._retrieve_relevant_knowledge(market_data)
        knowledge_context = self._build_knowledge_context(passages)
        
        # Get worldview-informed news
        news_interpretation = await self._get_worldview_informed_news(
            market_data, get_completion
        )
        
        # Inject into prompt
        full_prompt = base_prompt.replace(
            "[KNOWLEDGE_PASSAGES_WILL_BE_INSERTED_HERE]",
            knowledge_context
        ).replace(
            "[CURRENT_NEWS_INTERPRETATION_WILL_BE_INSERTED_HERE]",
            f"=== WORLDVIEW-INFORMED NEWS ANALYSIS ===\n{news_interpretation}"
        )
        
        # Add system prompt
        if self.SYSTEM_PROMPT:
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{full_prompt}"
        
        # Get completion
        import time
        start_time = time.time()
        
        try:
            raw_response = await get_completion(full_prompt)
            
            if raw_response is None:
                return self._error_result("Model returned None")
            
            elapsed = time.time() - start_time
            
            # Parse response
            parsed = self._extract_json(raw_response)
            if parsed is None:
                return self._error_result(f"Failed to extract JSON: {raw_response[:300]}")
            
            result = self._parse_result(parsed)
            
            # Add retrieved theories to result
            result["retrieved_theories"] = [
                {"source": p.source, "category": p.category, "relevance": p.similarity_score}
                for p in passages
            ]
            
            # Add news interpretation
            result["news_interpretation"] = news_interpretation[:500]  # Truncate
            
            # Add metadata
            result["_agent"] = self.name
            result["_model"] = self.model_name
            result["_elapsed_seconds"] = round(elapsed, 2)
            
            # Check if trigger conditions are met
            result["_triggers_override"] = self._check_override_triggers(
                result, market_data, context
            )
            
            return result
            
        except Exception as e:
            return self._error_result(str(e))
    
    def _parse_result(self, raw_json: Dict) -> Dict:
        """Parse and validate ideology agent output."""
        probability = self.clamp(raw_json.get("probability", 0.5))
        confidence = self.clamp(raw_json.get("confidence", 0.5))
        worldview_applies = bool(raw_json.get("worldview_applies", True))
        contrarian_signal = bool(raw_json.get("contrarian_signal", False))
        ensemble_prob = float(raw_json.get("ensemble_probability", 0.5))
        adjustment = float(raw_json.get("worldview_adjustment", 0.0))
        
        # If worldview doesn't apply, force neutral
        if not worldview_applies:
            probability = 0.5
            confidence = 0.0
        
        # Infer confidence from reasoning if not provided or low
        reasoning = str(raw_json.get("reasoning", ""))
        if confidence < 0.3 and reasoning:
            inferred = self._infer_confidence(reasoning)
            confidence = max(confidence, inferred)
        
        return {
            "probability": probability,
            "confidence": confidence,
            "worldview_applies": worldview_applies,
            "contrarian_signal": contrarian_signal,
            "ensemble_probability": ensemble_prob,
            "worldview_adjustment": adjustment,
            "retrieved_theories": raw_json.get("retrieved_theories", []),
            "news_interpretation": str(raw_json.get("news_interpretation", "")),
            "reasoning": reasoning,
            "caveats": str(raw_json.get("caveats", "Uncertainties within my framework")),
        }
    
    def _check_override_triggers(
        self,
        result: Dict,
        market_data: Dict,
        context: Dict
    ) -> bool:
        """
        Check if override conditions are met for manual review.
        
        Returns True if manual review should be triggered.
        """
        if not self.trigger_config:
            return False
        
        # Get values
        worldview_prob = result.get("probability", 0.5)
        worldview_conf = result.get("confidence", 0.0)
        ensemble_prob = result.get("ensemble_probability", 0.5)
        market_price = market_data.get("yes_price", 0.5)
        
        # Check divergence from ensemble
        min_divergence = self.trigger_config.get("min_divergence_from_ensemble", 0.20)
        if abs(worldview_prob - ensemble_prob) >= min_divergence:
            if worldview_conf >= self.trigger_config.get("min_worldview_confidence", 0.70):
                logger.info(
                    "Override trigger: Divergence from ensemble",
                    divergence=abs(worldview_prob - ensemble_prob),
                    worldview_prob=worldview_prob,
                    ensemble_prob=ensemble_prob
                )
                return True
        
        # Check divergence from market
        min_market_div = self.trigger_config.get("min_divergence_from_market", 0.15)
        if abs(worldview_prob - market_price) >= min_market_div:
            if worldview_conf >= self.trigger_config.get("min_worldview_confidence", 0.70):
                logger.info(
                    "Override trigger: Divergence from market",
                    divergence=abs(worldview_prob - market_price),
                    worldview_prob=worldview_prob,
                    market_price=market_price
                )
                return True
        
        return False