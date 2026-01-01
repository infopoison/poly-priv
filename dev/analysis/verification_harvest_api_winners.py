#!/usr/bin/env python3
"""
SIDECAR REPAIR SCRIPT: Harvest API-Derived Winners
===================================================
Purpose: Build authoritative winner data from Polymarket API and store
         in a sidecar file for later merge with parquet data.

This is conceptually separate from calibration analysis - it's a data
repair/enrichment operation that should be run once to create the
authoritative winner mapping.

Output Schema (data/repair/api_derived_winners.parquet):
    - condition_id: str (market identifier)
    - token_id: str (CLOB token identifier)
    - api_derived_winner: bool (True if this token won, False otherwise)
    - market_question: str (human-readable market title)
    - outcome_label: str ('YES' or 'NO' based on token position)
    - settlement_price: float (final price: 1.0 for winner, 0.0 for loser)
    - yes_token_id: str (reference: the YES token for this market)
    - no_token_id: str (reference: the NO token for this market)
    - repair_timestamp: str (ISO format timestamp of API call)
    - repair_status: str ('SUCCESS', 'API_ERROR', 'PARSE_ERROR', 'AMBIGUOUS')

Usage:
    # Diagnostic mode - process 10 conditions with verbose output
    python harvest_api_winners.py --sample 10
    
    # Full run
    python harvest_api_winners.py
    
    # Custom paths
    python harvest_api_winners.py --batch-dir /path/to/batches --output-dir /path/to/repair

Directory Structure:
    polymarket_data/
    ├── data/
    │   ├── raw/          # symlink to parquet files (read-only)
    │   ├── processed/
    │   └── repair/       # OUTPUT: sidecar files go here
    │       └── api_derived_winners.parquet
    └── order_history_batches/
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import glob
import json
import requests
import argparse
import os
import sys
import time
from datetime import datetime
from collections import defaultdict
import gc

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = "../../"  # Relative to scripts directory
DEFAULT_BATCH_DIR = os.path.join(BASE_DIR, 'order_history_batches')
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, 'data/repair')

API_BASE_URL = 'https://gamma-api.polymarket.com/markets'
API_TIMEOUT = 15
API_RETRY_DELAY = 1.0  # seconds between retries
API_MAX_RETRIES = 3

# Rate limiting - be nice to the API
API_CALLS_PER_BATCH = 50
API_BATCH_DELAY = 2.0  # seconds between batches of API calls

# ==============================================================================
# LOGGING
# ==============================================================================

def log(msg, level='INFO'):
    """Timestamped logging"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [{level}] {msg}", flush=True)

def log_diagnostic(msg):
    """Diagnostic logging - only in sample/verbose mode"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] [DIAG] {msg}", flush=True)

# ==============================================================================
# PASS 1: BUILD CONDITION/TOKEN INDEX
# ==============================================================================

def build_condition_token_index(batch_files, sample_conditions=None):
    """
    Scan parquet files to build mapping: condition_id -> set of token_ids
    
    Memory efficient: only stores unique IDs, not row data.
    
    In sample mode, stops scanning as soon as we have enough conditions.
    
    Returns:
        dict: {condition_id: set(token_id, ...)}
    """
    log("="*70)
    log("PASS 1: Building condition/token index from parquet files")
    log("="*70)
    
    sample_mode = sample_conditions is not None
    
    if sample_mode:
        log(f"SAMPLE MODE: Will stop after finding {sample_conditions} conditions")
        log(f"Files available: {len(batch_files)}")
    else:
        log(f"Files to scan: {len(batch_files)}")
    
    condition_tokens = defaultdict(set)
    files_processed = 0
    total_rows_scanned = 0
    
    for filepath in batch_files:
        # EARLY EXIT for sample mode
        if sample_mode and len(condition_tokens) >= sample_conditions:
            log(f"  Sample target reached ({sample_conditions} conditions) - stopping scan")
            break
        
        files_processed += 1
        
        # Progress logging - more frequent in sample mode
        log_interval = 50 if sample_mode else 500
        if files_processed % log_interval == 0:
            log(f"  Progress: {files_processed} files, {len(condition_tokens)} conditions found")
        
        try:
            # Read only the columns we need
            df = pq.read_table(filepath, columns=['condition_id', 'token_id']).to_pandas()
            total_rows_scanned += len(df)
            
            # Group by condition and collect unique tokens
            for condition_id in df['condition_id'].unique():
                # Skip if we already have enough in sample mode
                if sample_mode and len(condition_tokens) >= sample_conditions:
                    if condition_id not in condition_tokens:
                        continue
                
                mask = df['condition_id'] == condition_id
                tokens = df.loc[mask, 'token_id'].astype(str).unique()
                condition_tokens[condition_id].update(tokens)
            
            del df
            
        except Exception as e:
            # Skip problematic files silently in production
            continue
    
    # In sample mode, trim to exact count (in case last file added extras)
    if sample_mode and len(condition_tokens) > sample_conditions:
        condition_list = list(condition_tokens.items())[:sample_conditions]
        condition_tokens = dict(condition_list)
    
    log(f"  Index complete:")
    log(f"    Files processed: {files_processed}")
    log(f"    Rows scanned: {total_rows_scanned:,}")
    log(f"    Unique conditions: {len(condition_tokens)}")
    log(f"    Total token mappings: {sum(len(t) for t in condition_tokens.values()):,}")
    
    gc.collect()
    return condition_tokens

# ==============================================================================
# PASS 2: API HARVESTING
# ==============================================================================

def fetch_market_from_api(condition_id, retry_count=0):
    """
    Fetch market metadata from Polymarket API.
    
    Returns:
        tuple: (market_dict, error_string or None)
    """
    try:
        params = {'condition_ids': condition_id}
        response = requests.get(API_BASE_URL, params=params, timeout=API_TIMEOUT)
        
        if response.status_code == 429:  # Rate limited
            if retry_count < API_MAX_RETRIES:
                time.sleep(API_RETRY_DELAY * (retry_count + 1))
                return fetch_market_from_api(condition_id, retry_count + 1)
            return None, "RATE_LIMITED"
        
        if response.status_code != 200:
            return None, f"HTTP_{response.status_code}"
        
        data = response.json()
        
        # Handle response format
        if isinstance(data, list) and len(data) > 0:
            # Find exact match or use first result
            for m in data:
                if m.get('conditionId', '').lower() == condition_id.lower():
                    return m, None
            return data[0], None
        elif isinstance(data, dict) and data:
            return data, None
        else:
            return None, "EMPTY_RESPONSE"
            
    except requests.exceptions.Timeout:
        if retry_count < API_MAX_RETRIES:
            time.sleep(API_RETRY_DELAY)
            return fetch_market_from_api(condition_id, retry_count + 1)
        return None, "TIMEOUT"
    except Exception as e:
        return None, f"EXCEPTION:{str(e)[:50]}"


def derive_winner_from_market(market, token_ids):
    """
    Given API market data and list of token_ids, derive winner status for each.
    
    Returns:
        list of dicts, one per token, with all sidecar fields
    """
    results = []
    timestamp = datetime.now().isoformat()
    
    # Extract market metadata
    question = market.get('question', 'Unknown')[:500]  # Truncate long questions
    clob_ids = market.get('clobTokenIds', [])
    outcome_prices = market.get('outcomePrices', [])
    
    # Parse JSON strings if needed
    if isinstance(clob_ids, str):
        try:
            clob_ids = json.loads(clob_ids)
        except:
            clob_ids = []
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = json.loads(outcome_prices)
        except:
            outcome_prices = []
    
    # Validate structure
    if not (isinstance(clob_ids, list) and len(clob_ids) >= 2 and 
            isinstance(outcome_prices, list) and len(outcome_prices) >= 2):
        # Return error records for all tokens
        for tid in token_ids:
            results.append({
                'condition_id': market.get('conditionId', 'unknown'),
                'token_id': str(tid),
                'api_derived_winner': None,
                'market_question': question,
                'outcome_label': None,
                'settlement_price': None,
                'yes_token_id': None,
                'no_token_id': None,
                'repair_timestamp': timestamp,
                'repair_status': 'PARSE_ERROR',
            })
        return results
    
    # Parse token mapping
    yes_token_id = str(clob_ids[0])
    no_token_id = str(clob_ids[1])
    
    try:
        yes_price = float(outcome_prices[0])
        no_price = float(outcome_prices[1])
    except (ValueError, TypeError):
        for tid in token_ids:
            results.append({
                'condition_id': market.get('conditionId', 'unknown'),
                'token_id': str(tid),
                'api_derived_winner': None,
                'market_question': question,
                'outcome_label': None,
                'settlement_price': None,
                'yes_token_id': yes_token_id,
                'no_token_id': no_token_id,
                'repair_timestamp': timestamp,
                'repair_status': 'PARSE_ERROR',
            })
        return results
    
    # Determine winner
    if yes_price > no_price:
        winner_outcome = 'YES'
        winner_token_id = yes_token_id
    elif no_price > yes_price:
        winner_outcome = 'NO'
        winner_token_id = no_token_id
    else:
        # Ambiguous - both same price (shouldn't happen for resolved markets)
        for tid in token_ids:
            results.append({
                'condition_id': market.get('conditionId', 'unknown'),
                'token_id': str(tid),
                'api_derived_winner': None,
                'market_question': question,
                'outcome_label': 'YES' if str(tid) == yes_token_id else 'NO' if str(tid) == no_token_id else None,
                'settlement_price': yes_price if str(tid) == yes_token_id else no_price if str(tid) == no_token_id else None,
                'yes_token_id': yes_token_id,
                'no_token_id': no_token_id,
                'repair_timestamp': timestamp,
                'repair_status': 'AMBIGUOUS',
            })
        return results
    
    # Build result for each token
    for tid in token_ids:
        tid_str = str(tid)
        is_winner = (tid_str == winner_token_id)
        
        # Determine outcome label
        if tid_str == yes_token_id:
            outcome_label = 'YES'
            settlement_price = yes_price
        elif tid_str == no_token_id:
            outcome_label = 'NO'
            settlement_price = no_price
        else:
            # Token not in API response - might be from a different market format
            outcome_label = 'UNKNOWN'
            settlement_price = None
        
        results.append({
            'condition_id': market.get('conditionId', 'unknown'),
            'token_id': tid_str,
            'api_derived_winner': is_winner,
            'market_question': question,
            'outcome_label': outcome_label,
            'settlement_price': settlement_price,
            'yes_token_id': yes_token_id,
            'no_token_id': no_token_id,
            'repair_timestamp': timestamp,
            'repair_status': 'SUCCESS',
        })
    
    return results


def harvest_api_winners(condition_tokens, sample_mode=False):
    """
    PASS 2: Call API for each condition and harvest winner data.
    
    Args:
        condition_tokens: dict {condition_id: set(token_ids)}
        sample_mode: if True, print verbose diagnostic output
    
    Returns:
        list of dicts (sidecar records)
    """
    log("="*70)
    log("PASS 2: Harvesting winner data from API")
    log("="*70)
    log(f"Conditions to process: {len(condition_tokens)}")
    
    all_records = []
    conditions_processed = 0
    api_successes = 0
    api_errors = 0
    
    condition_list = list(condition_tokens.items())
    
    for idx, (condition_id, token_ids) in enumerate(condition_list):
        conditions_processed += 1
        
        # Progress logging every 500 (or every condition in sample mode)
        if sample_mode or conditions_processed % 500 == 0:
            log(f"  Progress: {conditions_processed}/{len(condition_list)} conditions, "
                f"Success: {api_successes}, Errors: {api_errors}")
        
        # Diagnostic output in sample mode
        if sample_mode:
            log_diagnostic(f"Processing condition: {condition_id[:24]}...")
            log_diagnostic(f"  Tokens: {[t[:16]+'...' for t in list(token_ids)[:4]]}")
        
        # Fetch from API
        market, error = fetch_market_from_api(condition_id)
        
        if error:
            api_errors += 1
            if sample_mode:
                log_diagnostic(f"  API ERROR: {error}")
            
            # Create error records for all tokens
            timestamp = datetime.now().isoformat()
            for tid in token_ids:
                all_records.append({
                    'condition_id': condition_id,
                    'token_id': str(tid),
                    'api_derived_winner': None,
                    'market_question': None,
                    'outcome_label': None,
                    'settlement_price': None,
                    'yes_token_id': None,
                    'no_token_id': None,
                    'repair_timestamp': timestamp,
                    'repair_status': f'API_ERROR:{error}',
                })
            continue
        
        # Derive winner
        records = derive_winner_from_market(market, token_ids)
        all_records.extend(records)
        
        # Count successes
        success_count = sum(1 for r in records if r['repair_status'] == 'SUCCESS')
        if success_count > 0:
            api_successes += 1
        else:
            api_errors += 1
        
        # Diagnostic output
        if sample_mode:
            log_diagnostic(f"  Market: {market.get('question', 'N/A')[:60]}...")
            log_diagnostic(f"  YES token: {records[0]['yes_token_id'][:16] if records[0]['yes_token_id'] else 'N/A'}...")
            log_diagnostic(f"  NO token: {records[0]['no_token_id'][:16] if records[0]['no_token_id'] else 'N/A'}...")
            for r in records:
                winner_str = '✓ WINNER' if r['api_derived_winner'] else '✗ LOSER' if r['api_derived_winner'] is False else '? UNKNOWN'
                log_diagnostic(f"    {r['token_id'][:16]}... -> {r['outcome_label']} -> {winner_str}")
        
        # Rate limiting - pause every N API calls
        if conditions_processed % API_CALLS_PER_BATCH == 0:
            if sample_mode:
                log_diagnostic(f"  Rate limit pause: {API_BATCH_DELAY}s")
            time.sleep(API_BATCH_DELAY)
    
    log(f"  Harvest complete:")
    log(f"    Conditions processed: {conditions_processed}")
    log(f"    API successes: {api_successes}")
    log(f"    API errors: {api_errors}")
    log(f"    Total records: {len(all_records)}")
    
    return all_records

# ==============================================================================
# PASS 3: WRITE SIDECAR FILE
# ==============================================================================

def write_sidecar_parquet(records, output_path):
    """
    Write harvested records to parquet sidecar file.
    """
    log("="*70)
    log("PASS 3: Writing sidecar parquet file")
    log("="*70)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Define schema with appropriate types
    # api_derived_winner can be True, False, or None
    df['api_derived_winner'] = df['api_derived_winner'].astype('object')  # Allows None
    
    log(f"  Records to write: {len(df)}")
    log(f"  Columns: {list(df.columns)}")
    
    # Status distribution
    status_counts = df['repair_status'].value_counts()
    log(f"  Status distribution:")
    for status, count in status_counts.items():
        log(f"    {status}: {count}")
    
    # Winner distribution (for SUCCESS records)
    success_df = df[df['repair_status'] == 'SUCCESS']
    if len(success_df) > 0:
        winner_counts = success_df['api_derived_winner'].value_counts()
        log(f"  Winner distribution (SUCCESS only):")
        for winner, count in winner_counts.items():
            log(f"    {winner}: {count}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write parquet
    df.to_parquet(output_path, index=False, engine='pyarrow')
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log(f"  Written: {output_path}")
    log(f"  File size: {file_size_mb:.2f} MB")
    
    return output_path

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Harvest API-derived winner data into sidecar file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnostic mode - process 10 conditions with verbose output
  python harvest_api_winners.py --sample 10
  
  # Full production run
  python harvest_api_winners.py
  
  # Custom paths
  python harvest_api_winners.py --batch-dir ../order_history_batches --output-dir ../data/repair
        """
    )
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N conditions (diagnostic mode with verbose output)')
    parser.add_argument('--batch-dir', '-b', default=DEFAULT_BATCH_DIR,
                        help=f'Path to parquet batch files (default: {DEFAULT_BATCH_DIR})')
    parser.add_argument('--output-dir', '-o', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory for sidecar file (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--output-file', '-f', default='api_derived_winners.parquet',
                        help='Output filename (default: api_derived_winners.parquet)')
    
    args = parser.parse_args()
    
    sample_mode = args.sample is not None
    
    log("="*70)
    log("SIDECAR REPAIR: Harvest API-Derived Winners")
    log("="*70)
    log(f"Batch directory: {args.batch_dir}")
    log(f"Output directory: {args.output_dir}")
    log(f"Sample mode: {sample_mode}" + (f" (N={args.sample})" if sample_mode else ""))
    
    # Validate batch directory
    batch_files = sorted(glob.glob(os.path.join(args.batch_dir, 'batch_*.parquet')))
    if not batch_files:
        log(f"ERROR: No batch files found in {args.batch_dir}", level='ERROR')
        sys.exit(1)
    
    log(f"Found {len(batch_files)} batch files")
    
    # PASS 1: Build index
    condition_tokens = build_condition_token_index(batch_files, sample_conditions=args.sample)
    
    if len(condition_tokens) == 0:
        log("ERROR: No conditions found in parquet files", level='ERROR')
        sys.exit(1)
    
    # PASS 2: Harvest from API
    records = harvest_api_winners(condition_tokens, sample_mode=sample_mode)
    
    if len(records) == 0:
        log("ERROR: No records harvested", level='ERROR')
        sys.exit(1)
    
    # PASS 3: Write sidecar
    output_path = os.path.join(args.output_dir, args.output_file)
    write_sidecar_parquet(records, output_path)
    
    log("="*70)
    log("COMPLETE")
    log("="*70)
    log(f"Sidecar file: {output_path}")
    log(f"")
    log(f"Next steps:")
    log(f"  1. Inspect sidecar: python -c \"import pandas as pd; print(pd.read_parquet('{output_path}').head(20))\"")
    log(f"  2. Merge with analysis (future script)")


if __name__ == "__main__":
    main()