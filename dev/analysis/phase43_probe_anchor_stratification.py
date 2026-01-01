#!/usr/bin/env python3
"""
Probe: Anchor Stratification Analysis for Sub-51% Longshots
============================================================

OBJECTIVE:
  Investigate whether low-anchor signals (e.g., 10-20% tokens) contribute 
  positive edge or are noise. The concern: for a 15% anchor, a 10pp drop 
  to 5% might still be microstructure noise rather than signal.

QUESTIONS ANSWERED:
  1. What % of sub_51 signals come from each anchor bucket?
  2. What is the win rate and edge by anchor bucket?
  3. Is there a clear anchor floor below which edge degrades?
  4. Recommendation: minimum anchor threshold for Strategy B?

INTERPRETATION VERIFICATION:
  The backtest uses ABSOLUTE thresholds (10pp = 0.10), not relative.
  This probe will verify: can a 10% anchor even trigger a 10pp drop?
  (Answer: only to 0%, so very few signals expected from sub-10% anchors)

USAGE:
  python probe_anchor_stratification.py --returns-dir /path/to/phase4d_returns_data_TIMESTAMP
  python probe_anchor_stratification.py --diagnostic  # Uses sample data path
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Fine-grained anchor buckets for sub_51 analysis
ANCHOR_BUCKETS = [
    ('0-10%', 0.00, 0.10),
    ('10-20%', 0.10, 0.20),
    ('20-30%', 0.20, 0.30),
    ('30-40%', 0.30, 0.40),
    ('40-51%', 0.40, 0.51),
]

# Focus on the key threshold/interval combinations from Strategy B
FOCUS_INTERVALS = ['48h_to_24h', '24h_to_12h']
FOCUS_THRESHOLDS = [0.05, 0.10, 0.15]
FOCUS_TERCILE = 'early'

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def get_anchor_bucket(entry_price):
    """Determine which anchor bucket an entry_price falls into."""
    for label, lo, hi in ANCHOR_BUCKETS:
        if lo <= entry_price < hi:
            return label
    return None

def compute_stats(df):
    """Compute summary statistics for a returns DataFrame."""
    if len(df) == 0:
        return {
            'n': 0, 'n_wins': 0, 'win_rate': np.nan,
            'mean_bps': np.nan, 'median_bps': np.nan, 'std_bps': np.nan,
            'p10': np.nan, 'p25': np.nan, 'p75': np.nan, 'p90': np.nan,
        }
    
    n = len(df)
    n_wins = df['is_winner'].sum()
    win_rate = n_wins / n
    
    returns = df['return_bps']
    return {
        'n': n,
        'n_wins': n_wins,
        'win_rate': win_rate,
        'mean_bps': returns.mean(),
        'median_bps': returns.median(),
        'std_bps': returns.std(),
        'p10': returns.quantile(0.10),
        'p25': returns.quantile(0.25),
        'p75': returns.quantile(0.75),
        'p90': returns.quantile(0.90),
    }

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_returns_data(returns_dir):
    """Load all returns parquet chunks from directory."""
    log(f"Loading returns data from: {returns_dir}")
    
    chunk_files = sorted(glob.glob(os.path.join(returns_dir, 'returns_chunk_*.parquet')))
    
    if not chunk_files:
        log("  ERROR: No returns chunk files found.")
        return None
    
    log(f"  Found {len(chunk_files)} chunk files")
    
    dfs = []
    for f in chunk_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            log(f"  Warning: Failed to load {f}: {e}")
    
    if not dfs:
        log("  ERROR: No data loaded.")
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    log(f"  Loaded {len(combined):,} total returns")
    
    return combined

# ==============================================================================
# ANALYSIS
# ==============================================================================

def analyze_anchor_stratification(df):
    """
    Main analysis: stratify sub_51 by anchor bucket and compute edge.
    """
    log("\n" + "="*70)
    log("ANCHOR STRATIFICATION ANALYSIS")
    log("="*70)
    
    # Filter to sub_51 only
    sub51 = df[df['prob_bucket'] == 'sub_51'].copy()
    log(f"\nTotal sub_51 observations: {len(sub51):,}")
    
    if len(sub51) == 0:
        log("  No sub_51 data found!")
        return None
    
    # Assign anchor buckets
    sub51['anchor_bucket'] = sub51['entry_price'].apply(get_anchor_bucket)
    
    # Check for null anchor buckets (shouldn't happen)
    null_anchors = sub51['anchor_bucket'].isna().sum()
    if null_anchors > 0:
        log(f"  Warning: {null_anchors} rows with null anchor bucket")
        sub51 = sub51.dropna(subset=['anchor_bucket'])
    
    results = []
    
    # ==========================================================================
    # SECTION 1: Overall anchor distribution
    # ==========================================================================
    log("\n" + "-"*70)
    log("SECTION 1: ANCHOR DISTRIBUTION (All intervals/thresholds)")
    log("-"*70)
    
    print(f"\n{'Anchor Bucket':<15} {'Count':>10} {'% of Total':>12} {'Win Rate':>10} {'Mean Edge':>12}")
    print("-" * 60)
    
    total = len(sub51)
    for label, lo, hi in ANCHOR_BUCKETS:
        bucket_data = sub51[sub51['anchor_bucket'] == label]
        n = len(bucket_data)
        pct = (n / total * 100) if total > 0 else 0
        
        stats = compute_stats(bucket_data)
        
        wr_str = f"{stats['win_rate']*100:.1f}%" if not np.isnan(stats['win_rate']) else "N/A"
        edge_str = f"{stats['mean_bps']:+.1f}" if not np.isnan(stats['mean_bps']) else "N/A"
        
        print(f"{label:<15} {n:>10,} {pct:>11.1f}% {wr_str:>10} {edge_str:>12}")
        
        results.append({
            'section': 'overall',
            'anchor_bucket': label,
            **stats
        })
    
    # ==========================================================================
    # SECTION 2: Focus analysis (Strategy B parameters)
    # ==========================================================================
    log("\n" + "-"*70)
    log("SECTION 2: STRATEGY B FOCUS (early tercile, 10% threshold)")
    log("-"*70)
    
    focus = sub51[
        (sub51['tercile'] == FOCUS_TERCILE) & 
        (sub51['threshold'] == 0.10)
    ]
    log(f"\nFocus subset (early, 10%): {len(focus):,} observations")
    
    if len(focus) > 0:
        print(f"\n{'Anchor':<12} {'Interval':<15} {'n':>8} {'Wins':>8} {'WR':>8} {'Mean':>10} {'Median':>10}")
        print("-" * 75)
        
        for interval in FOCUS_INTERVALS:
            interval_data = focus[focus['interval'] == interval]
            
            for label, lo, hi in ANCHOR_BUCKETS:
                bucket_data = interval_data[interval_data['anchor_bucket'] == label]
                stats = compute_stats(bucket_data)
                
                if stats['n'] > 0:
                    wr_str = f"{stats['win_rate']*100:.1f}%"
                    mean_str = f"{stats['mean_bps']:+.0f}"
                    median_str = f"{stats['median_bps']:+.0f}"
                else:
                    wr_str = mean_str = median_str = "-"
                
                print(f"{label:<12} {interval:<15} {stats['n']:>8} {stats['n_wins']:>8} {wr_str:>8} {mean_str:>10} {median_str:>10}")
                
                results.append({
                    'section': 'strategy_b_focus',
                    'interval': interval,
                    'threshold': 0.10,
                    'tercile': 'early',
                    'anchor_bucket': label,
                    **stats
                })
    
    # ==========================================================================
    # SECTION 3: Threshold sensitivity by anchor
    # ==========================================================================
    log("\n" + "-"*70)
    log("SECTION 3: THRESHOLD SENSITIVITY BY ANCHOR")
    log("-"*70)
    log("(Can low anchors even trigger larger thresholds?)")
    
    for threshold in FOCUS_THRESHOLDS:
        thresh_data = sub51[sub51['threshold'] == threshold]
        
        print(f"\n{int(threshold*100)}pp Threshold:")
        print(f"  {'Anchor':<12} {'Count':>10} {'Physically Possible?':>22}")
        print("  " + "-" * 46)
        
        for label, lo, hi in ANCHOR_BUCKETS:
            bucket_data = thresh_data[thresh_data['anchor_bucket'] == label]
            n = len(bucket_data)
            
            # Check if this threshold is even possible
            # For a 10% anchor with 10pp threshold, would need to go to 0%
            max_anchor = hi
            can_trigger = max_anchor > threshold
            possible_str = "Yes" if can_trigger else f"No (needs >{threshold*100:.0f}%)"
            
            print(f"  {label:<12} {n:>10,} {possible_str:>22}")
    
    # ==========================================================================
    # SECTION 4: Noise analysis - price precision at low anchors
    # ==========================================================================
    log("\n" + "-"*70)
    log("SECTION 4: ENTRY PRICE DISTRIBUTION BY ANCHOR")
    log("-"*70)
    
    for label, lo, hi in ANCHOR_BUCKETS[:3]:  # Focus on lowest three
        bucket_data = sub51[sub51['anchor_bucket'] == label]
        
        if len(bucket_data) > 0:
            prices = bucket_data['entry_price']
            print(f"\n{label}:")
            print(f"  n = {len(bucket_data):,}")
            print(f"  Entry price range: {prices.min():.4f} to {prices.max():.4f}")
            print(f"  Entry price mean: {prices.mean():.4f}")
            print(f"  Entry price std: {prices.std():.4f}")
            
            # Show the absolute move in pp
            if 'fill_price' in bucket_data.columns:
                moves = bucket_data['entry_price'] - bucket_data['fill_price']
                print(f"  Actual move (entry-fill) mean: {moves.mean()*100:.2f}pp")
                print(f"  Actual move (entry-fill) std: {moves.std()*100:.2f}pp")
    
    return pd.DataFrame(results)

def generate_recommendations(results_df, sub51_df):
    """Generate actionable recommendations based on analysis."""
    log("\n" + "="*70)
    log("RECOMMENDATIONS")
    log("="*70)
    
    # Focus on Strategy B parameters
    focus = results_df[
        (results_df['section'] == 'strategy_b_focus') & 
        (results_df['threshold'] == 0.10)
    ]
    
    print("\n1. ANCHOR FLOOR RECOMMENDATION:")
    print("-" * 40)
    
    # Check which anchor buckets have positive edge
    positive_edge = focus[focus['mean_bps'] > 0]
    negative_edge = focus[focus['mean_bps'] <= 0]
    
    if len(positive_edge) > 0:
        print("\n   Positive edge buckets:")
        for _, row in positive_edge.iterrows():
            print(f"     {row['anchor_bucket']}: {row['mean_bps']:+.0f} bps (n={row['n']})")
    
    if len(negative_edge) > 0:
        print("\n   Negative/zero edge buckets:")
        for _, row in negative_edge.iterrows():
            if row['n'] > 0:
                print(f"     {row['anchor_bucket']}: {row['mean_bps']:+.0f} bps (n={row['n']})")
    
    print("\n2. SIGNAL CONCENTRATION:")
    print("-" * 40)
    
    # What % of signals come from each bucket?
    overall = results_df[results_df['section'] == 'overall']
    total_n = overall['n'].sum()
    
    for _, row in overall.iterrows():
        pct = (row['n'] / total_n * 100) if total_n > 0 else 0
        edge_contribution = row['n'] * row['mean_bps'] / 10000  # rough $ contribution at $100
        print(f"   {row['anchor_bucket']}: {pct:.1f}% of signals, contributes ~{edge_contribution:.2f} edge-weighted signals")
    
    print("\n3. OPERATIONAL RECOMMENDATIONS:")
    print("-" * 40)
    
    # Check if low anchors have enough samples
    low_anchor_n = overall[overall['anchor_bucket'].isin(['0-10%', '10-20%'])]['n'].sum()
    low_anchor_pct = (low_anchor_n / total_n * 100) if total_n > 0 else 0
    
    if low_anchor_pct < 5:
        print(f"   ✓ Low anchors (<20%) represent only {low_anchor_pct:.1f}% of signals")
        print("     → Anchor floor likely NOT critical (few signals anyway)")
    else:
        print(f"   ⚠ Low anchors (<20%) represent {low_anchor_pct:.1f}% of signals")
        print("     → Consider anchor floor if edge is negative in these buckets")
    
    # Check for impossible triggers
    print("\n4. THRESHOLD FEASIBILITY:")
    print("-" * 40)
    print("   10pp threshold requires anchor > 10% (else can't drop 10pp)")
    print("   15pp threshold requires anchor > 15%")
    print("   20pp threshold requires anchor > 20%")
    
    # Count how many signals would be excluded by anchor floors
    if sub51_df is not None and 'entry_price' in sub51_df.columns:
        for floor in [0.10, 0.15, 0.20]:
            excluded = (sub51_df['entry_price'] < floor).sum()
            pct = excluded / len(sub51_df) * 100
            print(f"   Anchor floor ≥{floor*100:.0f}%: would exclude {excluded:,} ({pct:.1f}%) signals")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Probe: Anchor Stratification Analysis for Sub-51% Longshots'
    )
    parser.add_argument('--returns-dir', type=str, required=True,
                        help='Path to phase4d returns data directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path for results (optional)')
    
    args = parser.parse_args()
    
    log("="*70)
    log("PROBE: ANCHOR STRATIFICATION ANALYSIS")
    log("="*70)
    log(f"Returns directory: {args.returns_dir}")
    
    # Load data
    df = load_returns_data(args.returns_dir)
    
    if df is None:
        log("ERROR: Failed to load data.")
        sys.exit(1)
    
    # Quick data summary
    log(f"\nData summary:")
    log(f"  Total returns: {len(df):,}")
    log(f"  Prob buckets: {df['prob_bucket'].unique().tolist()}")
    log(f"  Intervals: {df['interval'].unique().tolist()}")
    log(f"  Thresholds: {df['threshold'].unique().tolist()}")
    log(f"  Terciles: {df['tercile'].unique().tolist()}")
    
    # Filter to sub_51 for detailed analysis
    sub51 = df[df['prob_bucket'] == 'sub_51'].copy()
    
    # Run analysis
    results_df = analyze_anchor_stratification(df)
    
    if results_df is not None:
        generate_recommendations(results_df, sub51)
        
        if args.output:
            results_df.to_csv(args.output, index=False)
            log(f"\nResults saved to: {args.output}")
    
    log("\n" + "="*70)
    log("PROBE COMPLETE")
    log("="*70)

if __name__ == "__main__":
    main()