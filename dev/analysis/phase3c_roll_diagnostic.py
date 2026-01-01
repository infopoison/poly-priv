#!/usr/bin/env python3
"""
Quick diagnostic: Why are Roll spread estimates so large?

Check:
1. Trade-to-trade price change distribution
2. Time gaps between trades (sparsity)
3. Autocovariance structure
4. Sample trade sequences
"""

import pyarrow.parquet as pq
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import os

BASE_DIR = "../../"
BATCH_DIR = os.path.join(BASE_DIR, 'order_history_batches')

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    log("Roll Measure Diagnostic")
    log("="*60)
    
    batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, '*.parquet')))
    log(f"Found {len(batch_files):,} batch files")
    
    # Sample first 500 files
    sample_files = batch_files[:500]
    
    # Collect trade sequences
    all_price_changes = []
    all_time_gaps = []
    all_autocovs = []
    sample_sequences = []
    
    tokens_seen = 0
    
    for filepath in sample_files:
        try:
            df = pq.read_table(filepath, columns=['timestamp', 'price', 'token_id']).to_pandas()
            
            if len(df) == 0:
                continue
            
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            if df['timestamp'].iloc[0] > 3e9:
                df['timestamp'] = df['timestamp'] / 1000.0
            
            df['token_id'] = df['token_id'].astype(str)
            
            for token_id, group in df.groupby('token_id', sort=False):
                if len(group) < 20:
                    continue
                
                tokens_seen += 1
                
                group = group.sort_values('timestamp')
                prices = group['price'].values
                timestamps = group['timestamp'].values
                
                # Price changes
                delta_p = np.diff(prices)
                all_price_changes.extend(delta_p)
                
                # Time gaps (in minutes)
                time_gaps = np.diff(timestamps) / 60
                all_time_gaps.extend(time_gaps)
                
                # Autocovariance
                if len(delta_p) >= 2:
                    delta_p_t = delta_p[1:]
                    delta_p_lag = delta_p[:-1]
                    cov = np.mean((delta_p_t - np.mean(delta_p_t)) * (delta_p_lag - np.mean(delta_p_lag)))
                    all_autocovs.append(cov)
                
                # Save sample sequences
                if len(sample_sequences) < 10 and len(prices) >= 20:
                    sample_sequences.append({
                        'token_id': token_id[:20],
                        'prices': prices[:20],
                        'timestamps': timestamps[:20],
                        'n_trades': len(prices),
                    })
                    
        except Exception as e:
            continue
    
    log(f"\nAnalyzed {tokens_seen:,} tokens")
    
    # ==========================================================================
    # 1. Price Change Distribution
    # ==========================================================================
    log("\n" + "="*60)
    log("1. PRICE CHANGE DISTRIBUTION (trade-to-trade)")
    log("="*60)
    
    price_changes = np.array(all_price_changes)
    abs_changes = np.abs(price_changes)
    
    log(f"\n  N price changes: {len(price_changes):,}")
    log(f"\n  Absolute price changes:")
    log(f"    Mean:   {np.mean(abs_changes):.4f} ({np.mean(abs_changes)*10000:.1f} bps)")
    log(f"    Median: {np.median(abs_changes):.4f} ({np.median(abs_changes)*10000:.1f} bps)")
    log(f"    P10:    {np.percentile(abs_changes, 10):.4f}")
    log(f"    P25:    {np.percentile(abs_changes, 25):.4f}")
    log(f"    P75:    {np.percentile(abs_changes, 75):.4f}")
    log(f"    P90:    {np.percentile(abs_changes, 90):.4f}")
    log(f"    P99:    {np.percentile(abs_changes, 99):.4f}")
    log(f"    Max:    {np.max(abs_changes):.4f}")
    
    log(f"\n  Signed price changes:")
    log(f"    Mean:   {np.mean(price_changes):.6f}")
    log(f"    Std:    {np.std(price_changes):.4f}")
    
    # Zero changes (same price trades)
    zero_pct = np.sum(abs_changes == 0) / len(abs_changes) * 100
    log(f"\n  Zero changes (same price): {zero_pct:.1f}%")
    
    # Small changes
    small_pct = np.sum(abs_changes <= 0.01) / len(abs_changes) * 100
    log(f"  Changes <= 1 cent: {small_pct:.1f}%")
    
    large_pct = np.sum(abs_changes > 0.05) / len(abs_changes) * 100
    log(f"  Changes > 5 cents: {large_pct:.1f}%")
    
    # ==========================================================================
    # 2. Time Gaps
    # ==========================================================================
    log("\n" + "="*60)
    log("2. TIME GAPS BETWEEN TRADES")
    log("="*60)
    
    time_gaps = np.array(all_time_gaps)
    time_gaps = time_gaps[time_gaps > 0]  # Remove same-second trades
    
    log(f"\n  Time gaps (minutes):")
    log(f"    Mean:   {np.mean(time_gaps):.1f} min")
    log(f"    Median: {np.median(time_gaps):.1f} min")
    log(f"    P10:    {np.percentile(time_gaps, 10):.1f} min")
    log(f"    P25:    {np.percentile(time_gaps, 25):.1f} min")
    log(f"    P75:    {np.percentile(time_gaps, 75):.1f} min")
    log(f"    P90:    {np.percentile(time_gaps, 90):.1f} min")
    
    # Categorize
    under_1min = np.sum(time_gaps < 1) / len(time_gaps) * 100
    under_5min = np.sum(time_gaps < 5) / len(time_gaps) * 100
    under_1hr = np.sum(time_gaps < 60) / len(time_gaps) * 100
    over_1hr = np.sum(time_gaps >= 60) / len(time_gaps) * 100
    
    log(f"\n  Gap distribution:")
    log(f"    < 1 min:  {under_1min:.1f}%")
    log(f"    < 5 min:  {under_5min:.1f}%")
    log(f"    < 1 hour: {under_1hr:.1f}%")
    log(f"    >= 1 hour: {over_1hr:.1f}%")
    
    # ==========================================================================
    # 3. Autocovariance Structure
    # ==========================================================================
    log("\n" + "="*60)
    log("3. AUTOCOVARIANCE OF PRICE CHANGES")
    log("="*60)
    
    autocovs = np.array(all_autocovs)
    
    log(f"\n  N tokens: {len(autocovs):,}")
    log(f"\n  Autocovariance distribution:")
    log(f"    Mean:   {np.mean(autocovs):.6f}")
    log(f"    Median: {np.median(autocovs):.6f}")
    log(f"    Std:    {np.std(autocovs):.6f}")
    log(f"    P10:    {np.percentile(autocovs, 10):.6f}")
    log(f"    P90:    {np.percentile(autocovs, 90):.6f}")
    
    neg_cov_pct = np.sum(autocovs < 0) / len(autocovs) * 100
    log(f"\n  Negative covariance: {neg_cov_pct:.1f}%")
    log(f"  Positive covariance: {100-neg_cov_pct:.1f}%")
    
    # What Roll would estimate from median covariance
    median_cov = np.median(autocovs)
    if median_cov < 0:
        implied_spread = 2 * np.sqrt(-median_cov)
        log(f"\n  Roll spread from median cov: {implied_spread:.4f} ({implied_spread*10000:.0f} bps)")
    
    # Compare to actual price change magnitude
    log(f"\n  SANITY CHECK:")
    log(f"    Median |ΔP|: {np.median(abs_changes):.4f}")
    log(f"    If spread = 2*median|ΔP|: {2*np.median(abs_changes):.4f} ({2*np.median(abs_changes)*10000:.0f} bps)")
    
    # ==========================================================================
    # 4. Sample Sequences
    # ==========================================================================
    log("\n" + "="*60)
    log("4. SAMPLE TRADE SEQUENCES")
    log("="*60)
    
    for i, seq in enumerate(sample_sequences[:5]):
        log(f"\n  Token {i+1}: {seq['token_id']}... ({seq['n_trades']} total trades)")
        prices = seq['prices']
        timestamps = seq['timestamps']
        
        log(f"    First 15 prices: {[f'{p:.2f}' for p in prices[:15]]}")
        
        # Show deltas
        deltas = np.diff(prices[:15])
        log(f"    Price changes:   {[f'{d:+.2f}' for d in deltas]}")
        
        # Time gaps
        gaps = np.diff(timestamps[:15]) / 60
        log(f"    Time gaps (min): {[f'{g:.1f}' for g in gaps]}")
        
        # Compute Roll for this sequence
        if len(prices) >= 20:
            delta_p = np.diff(prices)
            delta_p_t = delta_p[1:]
            delta_p_lag = delta_p[:-1]
            cov = np.mean((delta_p_t - np.mean(delta_p_t)) * (delta_p_lag - np.mean(delta_p_lag)))
            if cov < 0:
                spread = 2 * np.sqrt(-cov)
                log(f"    Roll spread: {spread:.4f} ({spread*10000:.0f} bps)")
            else:
                log(f"    Roll spread: UNDEFINED (cov={cov:.6f} > 0)")
    
    # ==========================================================================
    # 5. THE PROBLEM
    # ==========================================================================
    log("\n" + "="*60)
    log("5. DIAGNOSIS")
    log("="*60)
    
    log(f"""
  Roll measure assumes:
    - Trades alternate between bid and ask (bounce)
    - Negative autocorrelation comes from this bounce
    - Spread ≈ 2 * sqrt(-Cov)
    
  What's actually happening:
    - Trades are sparse (median gap: {np.median(time_gaps):.0f} min)
    - Large price changes between trades (median: {np.median(abs_changes)*100:.1f} cents)
    - Mix of information arrival + spread + price impact
    
  The negative autocovariance captures:
    - Bid-ask bounce (what we want)
    - Mean reversion after information (not spread)
    - Temporary price impact reversing (not spread)
    
  This is why Roll gives ~600-3000 bps "spreads":
    It's measuring total price reversal, not bid-ask spread.
    
  RECOMMENDATION:
    Don't use Roll for cost estimation.
    Use order book data or Phase 3B conditional fill analysis.
""")

if __name__ == "__main__":
    main()