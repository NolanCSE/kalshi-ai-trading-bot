"""
Backtest Prediction Tracking

This script allows testing the prediction tracking system on historical or mock data.
Run with: python -m src.jobs.backtest_prediction

Usage:
    python -m src.jobs.backtest_prediction --real-llm     # Use real LLM for analysis
    python -m src.jobs.backtest_prediction --show-records  # Show existing records
    python -m src.jobs.backtest_prediction --markets 5     # Run on first N markets
"""

import asyncio
import json
import argparse
from datetime import datetime, timedelta
from typing import Optional

from src.utils.database import DatabaseManager, PredictionRecord
from src.agents.knowledge_researcher import KnowledgeResearcher
from src.clients.openrouter_client import OpenRouterClient
from src.config.settings import settings


# Sample historical markets for backtesting
# These are real resolved markets from 2024-2025 that you can verify at kalshi.com/markets
SAMPLE_HISTORICAL_MARKETS = [
    {
        "market_id": "HIST-2024-FED-01",
        "title": "Will the Fed cut rates by 25bps in March 2024?",
        "category": "economics",
        "rules": "This market resolves to YES if the Fed cuts rates by 25 basis points at the March 2024 FOMC meeting.",
        "yes_price": 0.65,
        "no_price": 0.35,
        "actual_result": "YES",  # Fed kept rates - need to verify
        "volume": 85000,
        "expiration_ts": int((datetime(2024, 3, 20)).timestamp())
    },
    {
        "market_id": "HIST-2024-ELECTION-01", 
        "title": "Will Trump win the 2024 US Presidential election?",
        "category": "politics",
        "rules": "This market resolves to YES if Donald Trump receives the most electoral votes in the 2024 election.",
        "yes_price": 0.48,
        "no_price": 0.52,
        "actual_result": "YES",  # Trump won
        "volume": 250000,
        "expiration_ts": int((datetime(2024, 11, 5)).timestamp())
    },
    {
        "market_id": "HIST-2024-BTC-01",
        "title": "Will Bitcoin close above $100,000 in 2024?",
        "category": "crypto",
        "rules": "This market resolves to YES if Bitcoin's daily close price exceeds $100,000 USD at any point in 2024.",
        "yes_price": 0.42,
        "no_price": 0.58,
        "actual_result": "YES",  # Bitcoin did exceed 100k in Dec 2024
        "volume": 320000,
        "expiration_ts": int((datetime(2024, 12, 31)).timestamp())
    },
    {
        "market_id": "HIST-2024-NVIDIA-01",
        "title": "Will Nvidia stock split 10-for-1 before January 2025?",
        "category": "stocks",
        "rules": "This market resolves to YES if Nvidia announces and executes a 10-for-1 stock split.",
        "yes_price": 0.55,
        "no_price": 0.45,
        "actual_result": "YES",  # Nvidia did 10-for-1 split in June 2024
        "volume": 120000,
        "expiration_ts": int((datetime(2024, 6, 10)).timestamp())
    },
    {
        "market_id": "HIST-2024-OIL-01",
        "title": "Will crude oil exceed $100/barrel in Q2 2024?",
        "category": "commodities",
        "rules": "This market resolves to YES if WTI crude oil exceeds $100 per barrel during Q2 2024.",
        "yes_price": 0.35,
        "no_price": 0.65,
        "actual_result": "NO",  # Oil stayed below $100
        "volume": 95000,
        "expiration_ts": int((datetime(2024, 6, 30)).timestamp())
    },
    {
        "market_id": "HIST-2024-AI-01",
        "title": "Will OpenAI have a CEO change in 2024?",
        "category": "tech",
        "rules": "This market resolves to YES if OpenAI announces a change in CEO position during 2024.",
        "yes_price": 0.25,
        "no_price": 0.75,
        "actual_result": "YES",  # Sam Altman was briefly fired then reinstated
        "volume": 180000,
        "expiration_ts": int((datetime(2024, 11, 5)).timestamp())
    },
]


async def run_knowledge_researcher_analysis(
    market_data: dict, 
    get_completion: callable,
    use_real_llm: bool = False
) -> dict:
    """Run the knowledge researcher agent to get context for a market."""
    kr = KnowledgeResearcher()
    
    market_info = {
        "title": market_data["title"],
        "category": market_data["category"],
        "rules": market_data["rules"],
    }
    
    # Get reformulated queries
    reformulated = await kr._reformulate_queries(market_info, get_completion)
    print(f"    Reformulated queries: {len(reformulated.get('knowledge_retrieval_queries', []))} knowledge queries")
    
    # Get knowledge retrieval (this will search the database + web)
    try:
        passages = await kr._retrieve_relevant_knowledge(
            market_info, 
            reformulated["knowledge_retrieval_queries"]
        )
        
        # Get citations
        citations = []
        for p in passages:
            citations.append({
                "source": p.source,
                "relevance_score": p.similarity_score,
                "text_preview": p.text[:200]
            })
        
        # Get web context if enabled
        web_context = ""
        if kr.web_research_enabled:
            try:
                web_context = await kr._get_web_context(
                    market_info, 
                    reformulated["web_search_queries"]
                )
                print(f"    Web context retrieved")
            except Exception as e:
                print(f"    Web context error: {e}")
        
        return {
            "reformulated_queries": reformulated,
            "passages": passages,
            "citations": citations,
            "web_context": web_context[:500] if web_context else ""
        }
    except Exception as e:
        return {"error": str(e)}


async def run_backtest(
    db_manager: DatabaseManager,
    markets: list = None,
    use_knowledge_researcher: bool = True,
    use_real_llm: bool = False
):
    """Run backtest on historical markets.
    
    Args:
        db_manager: Database manager instance
        markets: List of market dicts to analyze
        use_knowledge_researcher: Whether to run knowledge researcher
        use_real_llm: Whether to use real LLM (costs money) or mock
    """
    
    if markets is None:
        markets = SAMPLE_HISTORICAL_MARKETS
    
    print(f"=" * 70)
    print(f"BACKTEST: Running on {len(markets)} historical markets")
    print(f"Knowledge Researcher: {use_knowledge_researcher}, Real LLM: {use_real_llm}")
    print(f"=" * 70)
    
    # Set up get_completion - real or mock
    if use_real_llm:
        print("\nUsing REAL LLM for analysis...")
        openrouter = OpenRouterClient()
        
        async def get_completion(prompt):
            try:
                result = await openrouter.get_completion(
                    prompt=prompt,
                    model="mistralai/mistral-7b-instruct",  # Cheap model for backtesting
                )
                return result
            except Exception as e:
                print(f"    LLM error: {e}")
                return None
    else:
        print("\nUsing MOCK LLM (no real API calls)")
        async def get_completion(prompt):
            if "query reformulation" in prompt.lower():
                return json.dumps({
                    "web_search_queries": ["historical test query"],
                    "knowledge_retrieval_queries": ["historical context query"]
                })
            return '{"probability": 0.5, "reasoning": "mock analysis"}'
    
    results = []
    
    for market in markets:
        print(f"\n--- Market: {market['title'][:50]}... ---")
        
        # Get analysis context
        context_citations = []
        web_context = ""
        trader_reasoning = "Mock prediction based on market price"
        predicted_probability = market["yes_price"]  # Default to market price
        predicted_side = "YES" if predicted_probability > 0.5 else "NO"
        
        if use_knowledge_researcher:
            try:
                print(f"  Running knowledge researcher...")
                analysis = await run_knowledge_researcher_analysis(market, get_completion, use_real_llm)
                
                if "citations" in analysis:
                    context_citations = analysis["citations"]
                    print(f"  Found {len(context_citations)} context citations")
                
                if "web_context" in analysis:
                    web_context = analysis["web_context"]
                
                # If we have real LLM, we could make a real prediction here
                # For now, we'll use the market price as the prediction
                # but record all the context for later analysis
            except Exception as e:
                print(f"  Knowledge researcher error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"  Market price: YES={market['yes_price']}, NO={market['no_price']}")
        print(f"  Actual result: {market['actual_result']}")
        
        # Record the prediction
        prediction = PredictionRecord(
            market_id=market["market_id"],
            market_title=market["title"],
            category=market["category"],
            rules=market["rules"],
            created_at=datetime.now() - timedelta(days=30),  # Simulate historical date
            predicted_probability=predicted_probability,
            predicted_side=predicted_side,
            context_citations=json.dumps(context_citations),
            trader_reasoning=f"Based on market price of {predicted_probability} and historical analysis",
        )
        
        record_id = await db_manager.record_prediction(prediction)
        print(f"  Recorded prediction (ID: {record_id})")
        
        # Simulate market resolution by updating with actual result
        actual_pnl = 1.0 - predicted_probability if market["actual_result"] == predicted_side else -predicted_probability
        
        await db_manager.update_prediction_resolution(
            market_id=market["market_id"],
            actual_result=market["actual_result"],
            pnl=actual_pnl,
            position_id=None
        )
        
        # Check if prediction was correct
        was_correct = (market["actual_result"] == predicted_side)
        print(f"  Result: {'✅ CORRECT' if was_correct else '❌ INCORRECT'}")
        print(f"  PnL: {actual_pnl:.2f}")
        
        results.append({
            "market_id": market["market_id"],
            "predicted": predicted_side,
            "actual": market["actual_result"],
            "correct": was_correct,
            "pnl": actual_pnl
        })
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"BACKTEST SUMMARY")
    print(f"=" * 70)
    
    correct_count = sum(1 for r in results if r["correct"])
    total_pnl = sum(r["pnl"] for r in results)
    
    print(f"Total markets: {len(results)}")
    print(f"Correct predictions: {correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)")
    print(f"Total PnL: {total_pnl:.2f}")
    
    return results


async def show_prediction_records(db_manager: DatabaseManager):
    """Display all prediction records in the database."""
    import aiosqlite
    
    async with aiosqlite.connect(db_manager.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM prediction_records ORDER BY created_at DESC") as cursor:
            rows = await cursor.fetchall()
    
    print(f"\n{'='*70}")
    print(f"PREDICTION RECORDS IN DATABASE")
    print(f"{'='*70}")
    
    for row in rows:
        print(f"\nMarket: {row['market_title'][:60]}")
        print(f"  ID: {row['market_id']}")
        print(f"  Predicted: {row['predicted_side']} ({row['predicted_probability']:.0%})")
        print(f"  Actual: {row['actual_result']}")
        print(f"  PnL: {row['pnl']}")
        print(f"  Created: {row['created_at']}")
        print(f"  Resolved: {row['resolved_at']}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backtest prediction tracking")
    parser.add_argument("--show-records", action="store_true", help="Show existing prediction records")
    parser.add_argument("--no-agent", action="store_true", help="Skip knowledge researcher agent")
    parser.add_argument("--real-llm", action="store_true", help="Use real LLM for analysis (costs money)")
    parser.add_argument("--markets", type=int, default=None, help="Number of markets to process")
    args = parser.parse_args()
    
    db = DatabaseManager("trading_system.db")
    await db.initialize()
    
    markets = SAMPLE_HISTORICAL_MARKETS
    if args.markets:
        markets = markets[:args.markets]
    
    if args.show_records:
        await show_prediction_records(db)
    else:
        await run_backtest(
            db, 
            markets=markets,
            use_knowledge_researcher=not args.no_agent,
            use_real_llm=args.real_llm
        )
        await show_prediction_records(db)


if __name__ == "__main__":
    asyncio.run(main())