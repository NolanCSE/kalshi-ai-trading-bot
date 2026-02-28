#!/usr/bin/env python3
"""Show prediction records from the trading database."""

import sqlite3
import json
import sys

def main():
    conn = sqlite3.connect('trading_system.db')
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute('''
        SELECT * FROM prediction_records 
        ORDER BY created_at DESC
    ''')
    
    rows = cursor.fetchall()
    print(f"Total predictions: {len(rows)}\n")
    
    for row in rows:
        print('='*70)
        print(row['market_title'])
        print('='*70)
        print(f"ID:        {row['market_id']}")
        print(f"Category:  {row['category']}")
        print(f"Predicted: {row['predicted_side']} ({row['predicted_probability']:.0%})")
        print(f"Actual:    {row['actual_result']}")
        print(f"PnL:       {row['pnl']}")
        print(f"Created:   {row['created_at']}")
        print()
        print('REASONING:')
        print(row['trader_reasoning'] if row['trader_reasoning'] else '(none)')
        print()
        
        if row['context_citations']:
            try:
                citations = json.loads(row['context_citations'])
                print(f'CITATIONS ({len(citations)} sources):')
                for c in citations:
                    src = c.get('source', '?')
                    score = c.get('relevance_score', 0)
                    text = c.get('text_preview', '')[:100]
                    print(f'  [{score:.3f}] {src}')
                    print(f'          {text}...')
            except Exception as e:
                print(f'(could not parse citations: {e})')
        print()
        
        if row['knowledge_researcher_result']:
            try:
                kr_result = json.loads(row['knowledge_researcher_result'])
                print('KNOWLEDGE RESEARCHER:')
                if 'reformulated_queries' in kr_result:
                    print(f"  Web queries: {kr_result['reformulated_queries'].get('web_search_queries', [])}")
                    print(f"  Knowledge queries: {kr_result['reformulated_queries'].get('knowledge_retrieval_queries', [])}")
            except:
                pass
        print()
    
    conn.close()

if __name__ == '__main__':
    main()
