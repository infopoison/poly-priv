#!/usr/bin/env python3
"""
Probe: Validate OOS Markets
===========================
Randomly sample markets from oos_backfill_tokens.csv and verify:
1. Do they exist in CLOB API?
2. What is their actual end_date_iso?
3. Did they actually resolve in our target window (after Dec 8, 2025)?

This will tell us if the discovery script collected the right data.
"""

import pandas as pd
import requests
import random
import time
from datetime import datetime, timezone

CLOB_API_URL = "https://clob.polymarket.com/markets"
TARGET_AFTER = datetime(2025, 12, 8, tzinfo=timezone.utc)

def probe_market(condition_id):
    """Fetch market metadata from CLOB API"""
    url = f"{CLOB_API_URL}/{condition_id}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return {'status': 'NOT_FOUND'}
        elif resp.status_code != 200:
            return {'status': f'HTTP_{resp.status_code}'}
        
        data = resp.json()
        return {
            'status': 'OK',
            'question': data.get('question', '')[:80],
            'end_date_iso': data.get('end_date_iso'),
            'closed': data.get('closed'),
            'active': data.get('active'),
            'tokens': len(data.get('tokens', [])),
            'tags': data.get('tags', [])
        }
    except Exception as e:
        return {'status': f'ERROR: {e}'}

def main():
    print("="*70)
    print("OOS MARKETS VALIDATION PROBE")
    print("="*70)
    
    # Load the backfill tokens
    df = pd.read_csv('oos_collection/oos_backfill_tokens_filtered.csv')
    
    print(f"\nLoaded {len(df):,} tokens")
    print(f"Unique conditions: {df['condition_id'].nunique():,}")
    
    # Get unique conditions
    conditions = df['condition_id'].unique().tolist()
    
    # Sample 30 random conditions
    sample_size = min(30, len(conditions))
    sample = random.sample(conditions, sample_size)
    
    print(f"\nSampling {sample_size} random markets...")
    print(f"Target window: after {TARGET_AFTER.strftime('%Y-%m-%d')}")
    print("-"*70)
    
    results = {
        'in_window': 0,
        'before_window': 0,
        'no_date': 0,
        'not_found': 0,
        'errors': 0
    }
    
    before_window_examples = []
    in_window_examples = []
    
    for i, cond_id in enumerate(sample):
        print(f"\n[{i+1}/{sample_size}] {cond_id[:40]}...")
        
        time.sleep(1)  # Rate limit
        info = probe_market(cond_id)
        
        if info['status'] == 'NOT_FOUND':
            print(f"   ❌ NOT FOUND in CLOB")
            results['not_found'] += 1
            continue
        elif info['status'] != 'OK':
            print(f"   ❌ {info['status']}")
            results['errors'] += 1
            continue
        
        end_date_str = info.get('end_date_iso')
        
        if not end_date_str:
            print(f"   ⚠️ No end_date_iso")
            print(f"   Question: {info['question']}")
            results['no_date'] += 1
            continue
        
        # Parse end date
        try:
            end_date_str = end_date_str.replace('Z', '+00:00')
            end_dt = datetime.fromisoformat(end_date_str)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
        except Exception as e:
            print(f"   ⚠️ Can't parse date: {end_date_str}")
            results['no_date'] += 1
            continue
        
        # Check if in window
        if end_dt >= TARGET_AFTER:
            print(f"   ✅ IN WINDOW: {end_dt.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Question: {info['question']}")
            print(f"   Tags: {info.get('tags', [])[:3]}")
            results['in_window'] += 1
            in_window_examples.append({
                'date': end_dt.strftime('%Y-%m-%d'),
                'question': info['question'][:50]
            })
        else:
            print(f"   ❌ BEFORE WINDOW: {end_dt.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Question: {info['question']}")
            results['before_window'] += 1
            before_window_examples.append({
                'date': end_dt.strftime('%Y-%m-%d'),
                'question': info['question'][:50]
            })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nSampled {sample_size} markets:")
    print(f"  ✅ In window (>= Dec 8, 2025):  {results['in_window']} ({results['in_window']/sample_size*100:.1f}%)")
    print(f"  ❌ Before window:               {results['before_window']} ({results['before_window']/sample_size*100:.1f}%)")
    print(f"  ⚠️ No date / parse error:       {results['no_date']}")
    print(f"  ❌ Not found in CLOB:           {results['not_found']}")
    print(f"  ❌ Other errors:                {results['errors']}")
    
    if results['before_window'] > 0:
        print(f"\n⚠️ WARNING: {results['before_window']/sample_size*100:.1f}% of markets are OUTSIDE the target window!")
        print("   This suggests the discovery script has a bug in date filtering.")
        print("\n   Examples of markets BEFORE the window:")
        for ex in before_window_examples[:5]:
            print(f"     - {ex['date']}: {ex['question']}...")
    
    if results['in_window'] > sample_size * 0.9:
        print(f"\n✅ VALIDATION PASSED: {results['in_window']/sample_size*100:.1f}% of markets are in the target window")
    
    # Extrapolate
    if results['in_window'] + results['before_window'] > 0:
        valid_pct = results['in_window'] / (results['in_window'] + results['before_window'])
        estimated_valid = int(df['condition_id'].nunique() * valid_pct)
        print(f"\n📊 Extrapolation:")
        print(f"   If {valid_pct*100:.1f}% of all {df['condition_id'].nunique():,} markets are valid...")
        print(f"   Estimated valid markets: ~{estimated_valid:,}")
        print(f"   Estimated valid tokens: ~{estimated_valid * 2:,}")

if __name__ == "__main__":
    main()