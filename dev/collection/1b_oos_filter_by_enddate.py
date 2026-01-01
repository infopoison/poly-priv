#!/usr/bin/env python3
"""
OOS Fast Filter - Using Gamma API bulk endpoint
================================================

Instead of 100K individual CLOB calls (28 hours), this uses Gamma API's
server-side date filtering and pagination (100 markets/request).

~370 requests instead of 100K = ~6 minutes instead of 28 hours.

Usage:
    python oos_fast_filter.py --after-date 2025-12-08 --input oos_collection/oos_backfill_tokens.csv
"""

import argparse
import csv
import json
import os
import sys
import time
import requests
from datetime import datetime, timezone

GAMMA_API = "https://gamma-api.polymarket.com/markets"
BATCH_SIZE = 100
DELAY = 0.5  # seconds between requests

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")
    sys.stdout.flush()

def fetch_valid_conditions(after_date_str):
    """
    Fetch all closed markets with end_date >= after_date from Gamma API.
    Returns set of valid condition_ids.
    """
    log(f"Fetching closed markets with end_date >= {after_date_str} from Gamma API...")
    
    valid_conditions = set()
    offset = 0
    batch_num = 0
    
    while True:
        batch_num += 1
        
        params = {
            'closed': 'true',
            'end_date_min': after_date_str,
            'limit': BATCH_SIZE,
            'offset': offset
        }
        
        try:
            time.sleep(DELAY)
            resp = requests.get(GAMMA_API, params=params, timeout=30)
            
            if resp.status_code != 200:
                log(f"HTTP {resp.status_code} at offset {offset}", "ERROR")
                break
            
            markets = resp.json()
            
            if not markets:
                break
            
            for market in markets:
                cond_id = market.get('conditionId')
                if cond_id:
                    valid_conditions.add(cond_id)
            
            if batch_num % 10 == 0:
                log(f"  Batch {batch_num}: offset {offset}, total valid: {len(valid_conditions):,}")
            
            if len(markets) < BATCH_SIZE:
                break
            
            offset += BATCH_SIZE
            
        except Exception as e:
            log(f"Error at offset {offset}: {e}", "ERROR")
            time.sleep(5)
            continue
    
    log(f"✓ Found {len(valid_conditions):,} markets with end_date >= {after_date_str}")
    return valid_conditions

def main():
    parser = argparse.ArgumentParser(description="Fast filter OOS markets using Gamma API")
    parser.add_argument('--after-date', type=str, required=True, help='Keep markets ending after this date (YYYY-MM-DD)')
    parser.add_argument('--input', type=str, required=True, help='Input CSV (oos_backfill_tokens.csv)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV (default: input_filtered.csv)')
    args = parser.parse_args()
    
    # Output path
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_filtered{ext}"
    
    log("="*70)
    log("OOS FAST FILTER (Gamma API)")
    log("="*70)
    log(f"Input:   {args.input}")
    log(f"Output:  {output_path}")
    log(f"Filter:  end_date >= {args.after_date}")
    
    # Step 1: Fetch valid conditions from Gamma
    log("\n" + "-"*70)
    log("STEP 1: Fetch valid conditions from Gamma API")
    log("-"*70)
    
    start_time = time.time()
    valid_conditions = fetch_valid_conditions(args.after_date)
    fetch_time = time.time() - start_time
    
    log(f"Fetched in {fetch_time:.1f}s")
    
    # Step 2: Load and filter input
    log("\n" + "-"*70)
    log("STEP 2: Filter input CSV")
    log("-"*70)
    
    log(f"Loading {args.input}...")
    
    input_rows = []
    with open(args.input, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            input_rows.append(row)
    
    log(f"Loaded {len(input_rows):,} tokens")
    
    # Get unique conditions in input
    input_conditions = set(row['condition_id'] for row in input_rows)
    log(f"Unique conditions in input: {len(input_conditions):,}")
    
    # Find overlap
    valid_in_input = input_conditions & valid_conditions
    log(f"Valid conditions (in window): {len(valid_in_input):,}")
    
    # Filter rows
    filtered_rows = [row for row in input_rows if row['condition_id'] in valid_conditions]
    
    log(f"Filtered tokens: {len(filtered_rows):,}")
    
    # Step 3: Write output
    log("\n" + "-"*70)
    log("STEP 3: Write output")
    log("-"*70)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    log(f"✓ Wrote {output_path}")
    
    # Summary
    log("\n" + "="*70)
    log("SUMMARY")
    log("="*70)
    log(f"Original:  {len(input_rows):,} tokens ({len(input_conditions):,} markets)")
    log(f"Filtered:  {len(filtered_rows):,} tokens ({len(valid_in_input):,} markets)")
    log(f"Removed:   {len(input_rows) - len(filtered_rows):,} tokens")
    
    pct_kept = len(filtered_rows) / len(input_rows) * 100 if input_rows else 0
    log(f"\nKept {pct_kept:.1f}% of tokens")
    
    # Estimate fetch time
    queries = len(filtered_rows) * 2 * 1.5
    hours = queries * 5 / 3600
    log(f"\nEstimated order history fetch time: ~{hours:.0f} hours ({hours/24:.1f} days)")
    
    log("\n" + "="*70)
    log("Next step:")
    log(f"  python oos_fetch_order_history.py --input {output_path} --output-dir oos_collection/order_history_batches/")
    log("="*70)

if __name__ == "__main__":
    main()