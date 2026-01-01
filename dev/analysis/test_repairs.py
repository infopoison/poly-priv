#!/usr/bin/env python3
"""
Quick test script to verify repair_winner_from_api works.
Grabs a few condition IDs from your markets CSV and tests the repair function.
"""

import requests
import json
import csv
import os

BASE_DIR = "../../"
MARKETS_CSV = os.path.join(BASE_DIR, 'markets_past_year.csv')

def repair_winner_from_api(condition_id, token_ids=None):
    """
    Fetch market metadata from API and reconstruct winner.
    """
    print(f"\n{'='*60}")
    print(f"[REPAIR] Testing condition: {condition_id}")
    print(f"{'='*60}")
    
    try:
        params = {'condition_ids': condition_id}
        print(f"  API URL: https://gamma-api.polymarket.com/markets")
        print(f"  Params: {params}")
        
        response = requests.get('https://gamma-api.polymarket.com/markets', 
                              params=params, timeout=10)
        
        print(f"  Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"  FAIL: Bad status code")
            return None
        
        data = response.json()
        print(f"  Response Type: {type(data).__name__}")
        print(f"  Response Length: {len(data) if isinstance(data, list) else 'N/A (dict)'}")
        
        # Find matching market
        market = None
        if isinstance(data, list):
            for m in data:
                if m.get('conditionId', '').lower() == condition_id.lower():
                    market = m
                    break
            if not market and len(data) > 0:
                print(f"  No exact match, using first result")
                market = data[0]
        elif isinstance(data, dict):
            market = data
        
        if not market:
            print(f"  FAIL: No market found")
            return None
        
        print(f"\n  Market Found:")
        print(f"    Question: {market.get('question', 'N/A')[:60]}...")
        print(f"    Resolved: {market.get('resolved', 'N/A')}")
        print(f"    End Date: {market.get('endDate', 'N/A')}")
        
        # Get clobTokenIds and outcomePrices
        clob_ids = market.get('clobTokenIds', [])
        outcome_prices = market.get('outcomePrices', [])

        print(f"\n  Raw clobTokenIds: {clob_ids}")
        print(f"  Raw outcomePrices: {outcome_prices}")
        
        # Parse if they're JSON strings
        if isinstance(clob_ids, str):
            clob_ids = json.loads(clob_ids)
        if isinstance(outcome_prices, str):
            outcome_prices = json.loads(outcome_prices)
        
        print(f"  Parsed clobTokenIds: {clob_ids}")
        print(f"  Parsed outcomePrices: {outcome_prices}")
        
        if not (isinstance(clob_ids, list) and len(clob_ids) >= 2 and 
                isinstance(outcome_prices, list) and len(outcome_prices) >= 2):
            print(f"  FAIL: Invalid structure")
            return None
        
        yes_token_id = str(clob_ids[0])
        no_token_id = str(clob_ids[1])
        
        yes_price = float(outcome_prices[0])
        no_price = float(outcome_prices[1])

        print(f"\n  Token Mapping:")
        print(f"    YES (Index 0): {yes_token_id[:20]}...")
        print(f"    NO  (Index 1): {no_token_id[:20]}...")
        print(f"  Settlement Prices:")
        print(f"    YES: ${yes_price}")
        print(f"    NO:  ${no_price}")
        
        if yes_price > no_price:
            winner_token_id = yes_token_id
            decision = "YES Won"
        elif no_price > yes_price:
            winner_token_id = no_token_id
            decision = "NO Won"
        else:
            print(f"  FAIL: Ambiguous (prices equal)")
            return None
        
        print(f"\n  DECISION: {decision}")
        print(f"  Winner Token: {winner_token_id[:20]}...")
        
        # Build repair map
        if token_ids:
            repair_map = {}
            for tid in token_ids:
                repair_map[str(tid)] = (str(tid) == winner_token_id)
            print(f"  Repair Map: {repair_map}")
            return repair_map
        else:
            return {'winner': winner_token_id, 'decision': decision}
        
    except Exception as e:
        print(f"  FAIL: Exception - {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("Loading markets CSV...")
    
    # Load a few condition IDs from the markets CSV
    condition_ids = []
    with open(MARKETS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            cid = row.get('condition_id') or row.get('conditionId')
            if cid:
                condition_ids.append(cid)
            if len(condition_ids) >= 5:
                break
    
    print(f"Found {len(condition_ids)} condition IDs to test")
    
    # Test repair on each
    successes = 0
    failures = 0
    
    for cid in condition_ids:
        result = repair_winner_from_api(cid)
        if result:
            successes += 1
        else:
            failures += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {successes} successes, {failures} failures")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()