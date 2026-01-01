#!/usr/bin/env python3
"""
Task 0: Data Boundary Probe for Out-of-Sample Planning
======================================================
Scans existing parquet files to determine:
1. Resolution time boundaries (when markets resolved)
2. Trade timestamp boundaries (when trades occurred)
3. Token and condition counts
4. Data quality summary

This tells us exactly where OOS collection should start.
"""

import pandas as pd
import glob
import os
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_DIR = '../../order_history_batches'
SAMPLE_SIZE_FOR_DISTRIBUTIONS = 10  # Files to sample for distribution stats
FULL_SCAN_BATCH_SIZE = 100  # Process this many files at a time for boundary scan

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def ts_to_date(ts):
    """Convert unix timestamp to readable date string"""
    try:
        if pd.isna(ts):
            return "N/A"
        ts = int(ts)
        # Handle milliseconds vs seconds
        if ts > 3e11:
            ts = ts // 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    except:
        return f"Invalid: {ts}"

def ts_to_datetime(ts):
    """Convert unix timestamp to datetime object"""
    try:
        if pd.isna(ts):
            return None
        ts = int(ts)
        if ts > 3e11:
            ts = ts // 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except:
        return None

def probe_boundaries():
    log("Starting Data Boundary Probe for OOS Planning...")
    
    # ==========================================
    # 1. DIRECTORY CHECK
    # ==========================================
    if not os.path.exists(BATCH_DIR):
        log(f"ERROR: Directory not found: {BATCH_DIR}")
        log(f"       Looking in: {os.path.abspath(BATCH_DIR)}")
        return None
    
    parquet_files = sorted(glob.glob(os.path.join(BATCH_DIR, "batch_*.parquet")))
    if not parquet_files:
        log(f"ERROR: No batch files found in {BATCH_DIR}")
        return None
    
    log(f"Found {len(parquet_files):,} batch files")
    
    # ==========================================
    # 2. FULL SCAN FOR TEMPORAL BOUNDARIES
    # ==========================================
    print("\n" + "="*70)
    print("📊 SCANNING ALL FILES FOR TEMPORAL BOUNDARIES")
    print("="*70)
    
    # Track global extremes
    min_resolution_time = float('inf')
    max_resolution_time = float('-inf')
    min_trade_timestamp = float('inf')
    max_trade_timestamp = float('-inf')
    
    # Track unique identifiers
    all_token_ids = set()
    all_condition_ids = set()
    
    # Track enrichment status
    total_rows = 0
    rows_with_winner = 0
    rows_with_outcome = 0
    
    # Track resolution times for distribution
    resolution_times = []
    
    log("Scanning all files (this may take a few minutes)...")
    
    for i, filepath in enumerate(parquet_files):
        try:
            # Read only columns we need for efficiency
            cols_to_read = ['token_id', 'condition_id', 'timestamp', 'resolution_time']
            
            # Check if enrichment columns exist
            df_check = pd.read_parquet(filepath, columns=None)
            available_cols = df_check.columns.tolist()
            
            if 'token_winner' in available_cols:
                cols_to_read.append('token_winner')
            if 'token_outcome' in available_cols:
                cols_to_read.append('token_outcome')
            
            df = df_check  # Use already-loaded data
            
            # Update counts
            total_rows += len(df)
            
            # Track unique IDs
            all_token_ids.update(df['token_id'].unique())
            all_condition_ids.update(df['condition_id'].unique())
            
            # Track temporal boundaries
            if 'resolution_time' in df.columns:
                res_times = pd.to_numeric(df['resolution_time'], errors='coerce')
                valid_res = res_times.dropna()
                if len(valid_res) > 0:
                    min_resolution_time = min(min_resolution_time, valid_res.min())
                    max_resolution_time = max(max_resolution_time, valid_res.max())
                    # Sample some resolution times for distribution
                    if len(resolution_times) < 10000:
                        resolution_times.extend(valid_res.sample(min(100, len(valid_res))).tolist())
            
            if 'timestamp' in df.columns:
                trade_times = pd.to_numeric(df['timestamp'], errors='coerce')
                valid_trade = trade_times.dropna()
                if len(valid_trade) > 0:
                    min_trade_timestamp = min(min_trade_timestamp, valid_trade.min())
                    max_trade_timestamp = max(max_trade_timestamp, valid_trade.max())
            
            # Track enrichment
            if 'token_winner' in df.columns:
                rows_with_winner += df['token_winner'].notna().sum()
            if 'token_outcome' in df.columns:
                rows_with_outcome += (df['token_outcome'].notna() & (df['token_outcome'] != 'UNKNOWN')).sum()
            
            # Progress
            if (i + 1) % 500 == 0:
                log(f"  Processed {i+1:,}/{len(parquet_files):,} files...")
                
        except Exception as e:
            log(f"  WARNING: Error reading {filepath}: {e}")
            continue
    
    # ==========================================
    # 3. REPORT FINDINGS
    # ==========================================
    print("\n" + "="*70)
    print("📅 TEMPORAL BOUNDARIES")
    print("="*70)
    
    print("\n[Resolution Times] (when markets resolved)")
    print(f"  Earliest: {ts_to_date(min_resolution_time)}")
    print(f"  Latest:   {ts_to_date(max_resolution_time)}")
    print(f"  Raw timestamps: {int(min_resolution_time)} → {int(max_resolution_time)}")
    
    # Calculate span
    earliest_dt = ts_to_datetime(min_resolution_time)
    latest_dt = ts_to_datetime(max_resolution_time)
    if earliest_dt and latest_dt:
        span = latest_dt - earliest_dt
        print(f"  Span: {span.days} days ({span.days/30.44:.1f} months)")
    
    print("\n[Trade Timestamps] (when trades occurred)")
    print(f"  Earliest: {ts_to_date(min_trade_timestamp)}")
    print(f"  Latest:   {ts_to_date(max_trade_timestamp)}")
    print(f"  Raw timestamps: {int(min_trade_timestamp)} → {int(max_trade_timestamp)}")
    
    print("\n" + "="*70)
    print("📈 DATA VOLUME")
    print("="*70)
    print(f"\n  Total batch files:    {len(parquet_files):,}")
    print(f"  Total trade rows:     {total_rows:,}")
    print(f"  Unique tokens:        {len(all_token_ids):,}")
    print(f"  Unique conditions:    {len(all_condition_ids):,}")
    print(f"  Avg rows per token:   {total_rows / len(all_token_ids):.1f}")
    
    print("\n" + "="*70)
    print("✅ ENRICHMENT STATUS")
    print("="*70)
    enrichment_pct = (rows_with_winner / total_rows * 100) if total_rows > 0 else 0
    outcome_pct = (rows_with_outcome / total_rows * 100) if total_rows > 0 else 0
    print(f"\n  Rows with token_winner: {rows_with_winner:,} ({enrichment_pct:.1f}%)")
    print(f"  Rows with token_outcome: {rows_with_outcome:,} ({outcome_pct:.1f}%)")
    
    if enrichment_pct < 90:
        print(f"\n  ⚠️  WARNING: {100-enrichment_pct:.1f}% of rows missing winner data")
    
    # ==========================================
    # 4. OOS RECOMMENDATIONS
    # ==========================================
    print("\n" + "="*70)
    print("🎯 OUT-OF-SAMPLE RECOMMENDATIONS")
    print("="*70)
    
    # The OOS window should start after the latest resolution in training data
    oos_start = ts_to_datetime(max_resolution_time)
    if oos_start:
        # Add 1 day buffer
        oos_start_buffered = oos_start.replace(hour=0, minute=0, second=0)
        print(f"\n  Training data ends:     {latest_dt.strftime('%Y-%m-%d')}")
        print(f"  Recommended OOS start:  {oos_start_buffered.strftime('%Y-%m-%d')}")
        print(f"  OOS start timestamp:    {int(oos_start_buffered.timestamp())}")
        
        # Estimate how many markets we might find
        now = datetime.now(tz=timezone.utc)
        days_available = (now - oos_start_buffered).days
        avg_conditions_per_day = len(all_condition_ids) / span.days if span.days > 0 else 0
        estimated_oos_conditions = int(avg_conditions_per_day * days_available)
        estimated_oos_tokens = estimated_oos_conditions * 2
        
        print(f"\n  Days available for OOS: {days_available}")
        print(f"  Avg conditions/day:     {avg_conditions_per_day:.1f}")
        print(f"  Estimated OOS conditions: ~{estimated_oos_conditions:,}")
        print(f"  Estimated OOS tokens:     ~{estimated_oos_tokens:,}")
        
        # Runtime estimate
        queries_per_token = 3  # maker + taker + some pagination
        seconds_per_query = 5  # conservative with rate limiting
        estimated_hours = (estimated_oos_tokens * queries_per_token * seconds_per_query) / 3600
        print(f"\n  Estimated collection time: {estimated_hours:.1f} hours ({estimated_hours/24:.1f} days)")
    
    print("\n" + "="*70)
    print("📋 SUMMARY FOR PITCH")
    print("="*70)
    print(f"""
  Training Dataset:
    - Period: {ts_to_date(min_resolution_time)[:10]} to {ts_to_date(max_resolution_time)[:10]}
    - Markets (conditions): {len(all_condition_ids):,}
    - Tokens: {len(all_token_ids):,}
    - Trade events: {total_rows:,}
    - Batch files: {len(parquet_files):,}
""")
    
    print("="*70)
    
    # Return key values for programmatic use
    return {
        'min_resolution_time': int(min_resolution_time),
        'max_resolution_time': int(max_resolution_time),
        'min_trade_timestamp': int(min_trade_timestamp),
        'max_trade_timestamp': int(max_trade_timestamp),
        'total_rows': total_rows,
        'unique_tokens': len(all_token_ids),
        'unique_conditions': len(all_condition_ids),
        'batch_files': len(parquet_files),
        'enrichment_pct': enrichment_pct
    }

if __name__ == "__main__":
    results = probe_boundaries()
    
    if results:
        # Save results to JSON for later use
        import json
        with open('data_boundaries.json', 'w') as f:
            json.dump(results, f, indent=2)
        log(f"Results saved to data_boundaries.json")