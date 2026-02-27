"""
Quick test for KnowledgeResearcher agent.
Run: python scripts/test_knowledge_researcher.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.agents.knowledge_researcher import KnowledgeResearcher


async def main():
    researcher = KnowledgeResearcher()
    
    market_data = {
        'title': 'Will the federal reserve cut interest rates by March 2025?',
        'category': 'economics',
        'rules': 'Rate cut must occur by March 31, 2025 to resolve YES.',
        'yes_price': 65,
        'no_price': 35,
    }
    
    async def mock_get_completion(prompt: str):
        print("=" * 60)
        print("PROMPT SENT TO LLM:")
        print("=" * 60)
        print(prompt[:1500])
        print("\n[prompt truncated...]")
        
        return """{
            "worldview_applies": true,
            "retrieved_passages": [],
            "worldview_context": "Test",
            "key_frameworks": [],
            "reasoning": "Test reasoning"
        }"""
    
    print("Testing KnowledgeResearcher with interest rate market...")
    print(f"Market: {market_data['title']}")
    print(f"Category: {market_data['category']}")
    print()
    
    result = await researcher.analyze(market_data, {}, mock_get_completion)
    
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(f"worldview_applies: {result.get('worldview_applies')}")
    print(f"knowledge_citations: {result.get('knowledge_citations')}")
    print(f"elapsed_seconds: {result.get('_elapsed_seconds')}")
    print("\n--- Retrieved Passages Preview ---")
    passages = result.get('retrieved_passages', '')
    print(passages[:1500] if passages else "None")


if __name__ == "__main__":
    asyncio.run(main())
