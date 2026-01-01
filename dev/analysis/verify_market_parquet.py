#!/usr/bin/env python3
"""
DIAGNOSTIC SCRIPT: Parquet vs API Winner Data
==============================================
Purpose: Examine all parquet data for a specific condition_id and compare
         against the authoritative API repair data.

Usage:
    python diagnose_market_parquet.py <condition_id>
    python diagnose_market_parquet.py <condition_id> --sample 50  # Limit files scanned
    python diagnose_market_parquet.py --token <token_id>          # Search by token instead

Output:
    - All parquet fields for matching rows
    - token_winner values found in parquet
    - API repair data (authoritative)
    - Discrepancy analysis
"""

import pyarrow.parquet as pq
import glob
import json
import requests
import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime

# ==============================================================================
# CONFIGURATION (match main script)
# ==============================================================================

BASE_DIR = "../../"
BATCH_DIR = os.path.join(BASE_DIR, 'order_history_batches')

# ==============================================================================
# API REPAIR (copied from main script for reference)
# ==============================================================================

def fetch_api_metadata(condition_id):
    """
    Fetch full market metadata from API.
    Returns the raw API response for inspection.
    """
    print(f"\n{'='*70}")
    print(f"API METADATA FETCH")
    print(f"{'='*70}")
    print(f"Condition ID: {condition_id}")
    
    try:
        params = {'condition_ids': condition_id}
        response = requests.get('https://gamma-api.polymarket.com/markets', 
                              params=params, timeout=10)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"ERROR: Bad status code")
            return None
        
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            market = data[0]
        elif isinstance(data, dict):
            market = data
        else:
            print(f"ERROR: Unexpected response format")
            return None
        
        return market
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def derive_winner_from_api(market):
    """
    Given API market data, derive the winner token_id.
    Returns dict with detailed analysis.
    """
    result = {
        'question': market.get('question', 'N/A'),
        'clob_token_ids_raw': market.get('clobTokenIds'),
        'outcome_prices_raw': market.get('outcomePrices'),
        'tokens': market.get('tokens', []),
        'yes_token_id': None,
        'no_token_id': None,
        'yes_price': None,
        'no_price': None,
        'winner_outcome': None,
        'winner_token_id': None,
        'repair_map': {},
    }
    
    # Parse clobTokenIds and outcomePrices
    clob_ids = market.get('clobTokenIds', [])
    outcome_prices = market.get('outcomePrices', [])
    
    if isinstance(clob_ids, str):
        clob_ids = json.loads(clob_ids)
    if isinstance(outcome_prices, str):
        outcome_prices = json.loads(outcome_prices)
    
    if not (isinstance(clob_ids, list) and len(clob_ids) >= 2 and 
            isinstance(outcome_prices, list) and len(outcome_prices) >= 2):
        result['error'] = "Invalid clob_ids or outcome_prices structure"
        return result
    
    result['yes_token_id'] = str(clob_ids[0])
    result['no_token_id'] = str(clob_ids[1])
    result['yes_price'] = float(outcome_prices[0])
    result['no_price'] = float(outcome_prices[1])
    
    # Determine winner
    if result['yes_price'] > result['no_price']:
        result['winner_outcome'] = 'YES'
        result['winner_token_id'] = result['yes_token_id']
    elif result['no_price'] > result['yes_price']:
        result['winner_outcome'] = 'NO'
        result['winner_token_id'] = result['no_token_id']
    else:
        result['winner_outcome'] = 'AMBIGUOUS'
        result['winner_token_id'] = None
    
    # Build repair map
    if result['winner_token_id']:
        result['repair_map'][result['yes_token_id']] = (result['winner_outcome'] == 'YES')
        result['repair_map'][result['no_token_id']] = (result['winner_outcome'] == 'NO')
    
    return result


# ==============================================================================
# PARQUET SCANNING
# ==============================================================================

def scan_parquet_for_condition(condition_id, batch_files, max_files=None):
    """
    Scan all parquet files for rows matching the condition_id.
    Returns detailed breakdown of what's in the parquet files.
    """
    print(f"\n{'='*70}")
    print(f"PARQUET SCAN")
    print(f"{'='*70}")
    print(f"Searching for condition_id: {condition_id}")
    print(f"Files to scan: {len(batch_files) if max_files is None else min(max_files, len(batch_files))}")
    
    results = {
        'files_scanned': 0,
        'files_with_matches': [],
        'total_rows': 0,
        'tokens_found': {},  # token_id -> {row_count, winner_values, sample_data}
        'schema_sample': None,
        'all_columns': None,
    }
    
    files_to_scan = batch_files[:max_files] if max_files else batch_files
    
    for idx, filepath in enumerate(files_to_scan):
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{len(files_to_scan)} files...")
        
        results['files_scanned'] += 1
        
        try:
            pf = pq.ParquetFile(filepath)
            
            # Capture schema once
            if results['schema_sample'] is None:
                results['all_columns'] = list(pf.schema.names)
                results['schema_sample'] = str(pf.schema)
            
            # Read file
            table = pf.read()
            df = table.to_pandas()
            
            # Filter to our condition
            matches = df[df['condition_id'] == condition_id]
            
            if len(matches) > 0:
                results['files_with_matches'].append({
                    'filepath': filepath,
                    'filename': os.path.basename(filepath),
                    'row_count': len(matches),
                })
                results['total_rows'] += len(matches)
                
                # Aggregate by token
                for token_id in matches['token_id'].unique():
                    token_rows = matches[matches['token_id'] == token_id]
                    tid_str = str(token_id)
                    
                    if tid_str not in results['tokens_found']:
                        results['tokens_found'][tid_str] = {
                            'row_count': 0,
                            'token_winner_values': set(),
                            'sample_row': None,
                        }
                    
                    results['tokens_found'][tid_str]['row_count'] += len(token_rows)
                    
                    # Collect token_winner values
                    if 'token_winner' in token_rows.columns:
                        winner_vals = token_rows['token_winner'].dropna().unique()
                        for v in winner_vals:
                            results['tokens_found'][tid_str]['token_winner_values'].add(str(v))
                    
                    # Keep a sample row
                    if results['tokens_found'][tid_str]['sample_row'] is None:
                        sample = token_rows.iloc[0].to_dict()
                        # Convert numpy types to native for display
                        results['tokens_found'][tid_str]['sample_row'] = {
                            k: (str(v) if hasattr(v, 'item') else v) 
                            for k, v in sample.items()
                        }
        
        except Exception as e:
            continue
    
    print(f"  Scan complete: {results['total_rows']} rows found across {len(results['files_with_matches'])} files")
    
    return results


def scan_parquet_for_token(token_id, batch_files, max_files=None):
    """
    Scan all parquet files for rows matching a specific token_id.
    """
    print(f"\n{'='*70}")
    print(f"PARQUET SCAN (by Token)")
    print(f"{'='*70}")
    print(f"Searching for token_id: {token_id}")
    
    results = {
        'files_scanned': 0,
        'total_rows': 0,
        'condition_ids_found': set(),
        'token_winner_values': set(),
        'sample_row': None,
    }
    
    files_to_scan = batch_files[:max_files] if max_files else batch_files
    
    for idx, filepath in enumerate(files_to_scan):
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{len(files_to_scan)} files...")
        
        results['files_scanned'] += 1
        
        try:
            pf = pq.ParquetFile(filepath)
            table = pf.read()
            df = table.to_pandas()
            
            # Convert token_id column to string for matching
            df['token_id'] = df['token_id'].astype(str)
            matches = df[df['token_id'] == str(token_id)]
            
            if len(matches) > 0:
                results['total_rows'] += len(matches)
                
                for cid in matches['condition_id'].unique():
                    results['condition_ids_found'].add(cid)
                
                if 'token_winner' in matches.columns:
                    for v in matches['token_winner'].dropna().unique():
                        results['token_winner_values'].add(str(v))
                
                if results['sample_row'] is None:
                    sample = matches.iloc[0].to_dict()
                    results['sample_row'] = {
                        k: (str(v) if hasattr(v, 'item') else v) 
                        for k, v in sample.items()
                    }
        
        except Exception as e:
            continue
    
    return results


# ==============================================================================
# REPORTING
# ==============================================================================

def print_diagnostic_report(parquet_results, api_result, derived):
    """
    Print comprehensive diagnostic report comparing parquet vs API.
    """
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC REPORT")
    print(f"{'='*70}")
    
    # -------------------------------------------------------------------------
    # PARQUET SUMMARY
    # -------------------------------------------------------------------------
    print(f"\n--- PARQUET FILE DATA ---")
    print(f"Files scanned:      {parquet_results['files_scanned']}")
    print(f"Files with matches: {len(parquet_results['files_with_matches'])}")
    print(f"Total rows found:   {parquet_results['total_rows']}")
    print(f"Tokens found:       {len(parquet_results['tokens_found'])}")
    
    if parquet_results['all_columns']:
        print(f"\nAvailable columns in parquet:")
        for col in parquet_results['all_columns']:
            print(f"  - {col}")
    
    print(f"\n--- TOKEN BREAKDOWN ---")
    for tid, tdata in parquet_results['tokens_found'].items():
        print(f"\nToken: {tid}")
        print(f"  Rows in parquet:  {tdata['row_count']}")
        print(f"  token_winner values: {tdata['token_winner_values'] or 'N/A (column missing or all null)'}")
        
        if tdata['sample_row']:
            print(f"  Sample row fields:")
            for k, v in tdata['sample_row'].items():
                # Truncate long values
                v_str = str(v)
                if len(v_str) > 60:
                    v_str = v_str[:57] + "..."
                print(f"    {k}: {v_str}")
    
    # -------------------------------------------------------------------------
    # API DATA
    # -------------------------------------------------------------------------
    print(f"\n--- API METADATA ---")
    if api_result is None:
        print(f"ERROR: Could not fetch API data")
    else:
        print(f"Question: {derived['question'][:80]}...")
        print(f"\nclobTokenIds (raw):   {derived['clob_token_ids_raw']}")
        print(f"outcomePrices (raw):  {derived['outcome_prices_raw']}")
        print(f"\nParsed mapping:")
        print(f"  YES token (index 0): {derived['yes_token_id']}")
        print(f"  NO token (index 1):  {derived['no_token_id']}")
        print(f"\nSettlement prices:")
        print(f"  YES price: ${derived['yes_price']}")
        print(f"  NO price:  ${derived['no_price']}")
        print(f"\nDerived winner:")
        print(f"  Outcome: {derived['winner_outcome']}")
        print(f"  Winner token_id: {derived['winner_token_id']}")
    
    # -------------------------------------------------------------------------
    # DISCREPANCY ANALYSIS
    # -------------------------------------------------------------------------
    print(f"\n--- DISCREPANCY ANALYSIS ---")
    
    if derived.get('winner_token_id') is None:
        print(f"Cannot analyze: API winner could not be determined")
        return
    
    for tid, tdata in parquet_results['tokens_found'].items():
        expected_winner = derived['repair_map'].get(tid)
        parquet_values = tdata['token_winner_values']
        
        print(f"\nToken: {tid[:16]}...")
        print(f"  API says winner: {expected_winner}")
        print(f"  Parquet token_winner values: {parquet_values}")
        
        if not parquet_values:
            print(f"  STATUS: ⚠️  MISSING - No token_winner in parquet (needs repair)")
        else:
            # Check if parquet values match expected
            parquet_bool_values = set()
            for v in parquet_values:
                v_lower = str(v).lower()
                if v_lower in ('true', '1', '1.0'):
                    parquet_bool_values.add(True)
                elif v_lower in ('false', '0', '0.0'):
                    parquet_bool_values.add(False)
                else:
                    parquet_bool_values.add(f"UNKNOWN:{v}")
            
            if expected_winner in parquet_bool_values and len(parquet_bool_values) == 1:
                print(f"  STATUS: ✓ CONSISTENT - Parquet matches API")
            elif expected_winner in parquet_bool_values:
                print(f"  STATUS: ⚠️  INCONSISTENT - Multiple values in parquet: {parquet_bool_values}")
            else:
                print(f"  STATUS: ❌ WRONG - Parquet has {parquet_bool_values}, API says {expected_winner}")
    
    # -------------------------------------------------------------------------
    # REPAIR RECOMMENDATION
    # -------------------------------------------------------------------------
    print(f"\n--- REPAIR RECOMMENDATION ---")
    print(f"Repair map to apply to parquet files:")
    for tid, is_winner in derived['repair_map'].items():
        print(f"  {tid}: token_winner = {is_winner}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Diagnose parquet data for a specific market')
    parser.add_argument('condition_id', nargs='?', help='Condition ID to examine')
    parser.add_argument('--token', '-t', help='Search by token_id instead of condition_id')
    parser.add_argument('--sample', '-s', type=int, default=None, help='Max files to scan')
    parser.add_argument('--batch-dir', '-b', default=BATCH_DIR, help='Batch directory path')
    args = parser.parse_args()
    
    if not args.condition_id and not args.token:
        parser.print_help()
        print("\nExample:")
        print("  python diagnose_market_parquet.py 0x88a79ea86071ec340d9a79410c1940411de99659a556df81dd1d42bc96c248f9")
        sys.exit(1)
    
    # Load batch files
    batch_files = sorted(glob.glob(f"{args.batch_dir}/batch_*.parquet"))
    if not batch_files:
        print(f"ERROR: No batch files found in {args.batch_dir}")
        sys.exit(1)
    
    print(f"Found {len(batch_files)} batch files")
    
    if args.token:
        # Token-based search
        results = scan_parquet_for_token(args.token, batch_files, args.sample)
        print(f"\n--- RESULTS ---")
        print(f"Total rows: {results['total_rows']}")
        print(f"Condition IDs found: {results['condition_ids_found']}")
        print(f"token_winner values: {results['token_winner_values']}")
        if results['sample_row']:
            print(f"\nSample row:")
            for k, v in results['sample_row'].items():
                print(f"  {k}: {v}")
        
        # If we found a condition, offer to run full diagnostic
        if results['condition_ids_found']:
            cid = list(results['condition_ids_found'])[0]
            print(f"\nTo run full diagnostic for associated condition:")
            print(f"  python diagnose_market_parquet.py {cid}")
    else:
        # Condition-based search
        condition_id = args.condition_id
        
        # Scan parquet files
        parquet_results = scan_parquet_for_condition(condition_id, batch_files, args.sample)
        
        # Fetch API metadata
        api_result = fetch_api_metadata(condition_id)
        
        # Derive winner from API
        if api_result:
            derived = derive_winner_from_api(api_result)
        else:
            derived = {'question': 'N/A', 'winner_token_id': None, 'repair_map': {}}
        
        # Print report
        print_diagnostic_report(parquet_results, api_result, derived)


if __name__ == "__main__":
    main()