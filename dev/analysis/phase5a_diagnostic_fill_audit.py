#!/usr/bin/env python3
"""
Diagnostic: Fill Price Audit
Version: 1.0

PURPOSE:
  Identify and quantify suspicious fill price anomalies that may be inflating
  reported edge statistics. Specifically looking for:
  
  1. Large negative slippage (fill_price << entry_price) combined with wins
  2. Extreme returns (>5000 bps) that may represent data artifacts
  3. Fill prices that are implausible given the probability bucket
  
OUTPUT:
  - Summary statistics on slippage distribution
  - Extreme case examples for manual verification
  - Impact quantification: what happens to mean edge if we exclude anomalies
  - Specific trade-level details for the most suspicious cases

USAGE:
  python diagnostic_fill_audit.py --baseline-dir /path/to/volume_returns_baseline
  python diagnostic_fill_audit.py --baseline-dir /path/to/volume_returns_baseline --strategy-a
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Strategy A defaults (from framing document)
STRATEGY_A_PARAMS = {
    'prob_buckets': ['75_90', '90_99'],
    'intervals': ['6h_to_4h', '8h_to_4h', '9h_to_6h'],
    'min_threshold': 0.10,
    'terciles': ['early'],
    'volume_quintiles': ['Q1_low', 'Q2'],
}

# Anomaly thresholds
LARGE_SLIPPAGE_THRESHOLD = -0.10  # Fill price 10%+ below entry price
EXTREME_RETURN_THRESHOLD = 5000   # 50% return seems suspicious for favorites
IMPLAUSIBLE_FILL_THRESHOLD = 0.20 # Fill at <20% for a 75%+ bucket starter

VOLUME_QUINTILE_LABELS = ['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high']


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_baseline_data(baseline_dir):
    """Load all parquet chunks from baseline directory."""
    log(f"Loading baseline data from: {baseline_dir}")
    
    if not os.path.exists(baseline_dir):
        log(f"  ERROR: Directory not found: {baseline_dir}")
        return None
    
    chunk_files = sorted(glob.glob(os.path.join(baseline_dir, 'volume_returns_chunk_*.parquet')))
    
    if not chunk_files:
        log(f"  ERROR: No chunk files found")
        return None
    
    log(f"  Found {len(chunk_files)} chunk files")
    
    dfs = []
    for chunk_file in chunk_files:
        df = pd.read_parquet(chunk_file)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    log(f"  Loaded {len(combined):,} total observations")
    
    return combined


def compute_volume_quintiles(df):
    """Add volume quintile column if not present."""
    if 'volume_quintile' in df.columns:
        return df
    
    valid_volume = df[df['volume_at_dip'] > 0]['volume_at_dip']
    
    if len(valid_volume) == 0:
        df['volume_quintile'] = None
        return df
    
    q20 = np.percentile(valid_volume, 20)
    q40 = np.percentile(valid_volume, 40)
    q60 = np.percentile(valid_volume, 60)
    q80 = np.percentile(valid_volume, 80)
    
    def assign_quintile(vol):
        if vol <= 0:
            return None
        elif vol <= q20:
            return 'Q1_low'
        elif vol <= q40:
            return 'Q2'
        elif vol <= q60:
            return 'Q3'
        elif vol <= q80:
            return 'Q4'
        else:
            return 'Q5_high'
    
    df['volume_quintile'] = df['volume_at_dip'].apply(assign_quintile)
    return df


def apply_strategy_a_filter(df, params):
    """Apply Strategy A filter."""
    # Exclude 'all' bucket
    df = df[df['prob_bucket'] != 'all'].copy()
    
    mask = (
        (df['prob_bucket'].isin(params['prob_buckets'])) &
        (df['interval'].isin(params['intervals'])) &
        (df['threshold'] >= params['min_threshold']) &
        (df['tercile'].isin(params['terciles'])) &
        (df['volume_quintile'].isin(params['volume_quintiles']))
    )
    
    return df[mask]


# ==============================================================================
# DIAGNOSTIC ANALYSIS
# ==============================================================================

def compute_slippage_metrics(df):
    """Compute slippage (fill_price - entry_price) metrics."""
    df = df.copy()
    df['slippage'] = df['fill_price'] - df['entry_price']
    df['slippage_pct'] = df['slippage'] / df['entry_price']
    
    return df


def analyze_slippage_distribution(df, label=""):
    """Analyze the distribution of slippage."""
    log(f"\n{'='*70}")
    log(f"SLIPPAGE DISTRIBUTION ANALYSIS {label}")
    log(f"{'='*70}")
    
    slippage = df['slippage']
    slippage_pct = df['slippage_pct']
    
    log(f"\nSlippage (fill_price - entry_price):")
    log(f"  n:      {len(slippage):,}")
    log(f"  Mean:   {slippage.mean():+.4f}")
    log(f"  Median: {slippage.median():+.4f}")
    log(f"  Std:    {slippage.std():.4f}")
    log(f"  Min:    {slippage.min():+.4f}")
    log(f"  Max:    {slippage.max():+.4f}")
    log(f"  P5:     {np.percentile(slippage, 5):+.4f}")
    log(f"  P95:    {np.percentile(slippage, 95):+.4f}")
    
    log(f"\nSlippage as % of entry price:")
    log(f"  Mean:   {slippage_pct.mean()*100:+.2f}%")
    log(f"  Median: {slippage_pct.median()*100:+.2f}%")
    log(f"  P5:     {np.percentile(slippage_pct, 5)*100:+.2f}%")
    log(f"  P95:    {np.percentile(slippage_pct, 95)*100:+.2f}%")
    
    # Count anomalies
    large_neg_slippage = (slippage < LARGE_SLIPPAGE_THRESHOLD).sum()
    log(f"\n  Trades with slippage < {LARGE_SLIPPAGE_THRESHOLD}: {large_neg_slippage:,} ({large_neg_slippage/len(df)*100:.2f}%)")
    
    return df


def identify_suspicious_trades(df):
    """Identify trades with suspicious characteristics."""
    log(f"\n{'='*70}")
    log(f"SUSPICIOUS TRADE IDENTIFICATION")
    log(f"{'='*70}")
    
    suspicious = {}
    
    # Category 1: Large negative slippage + winner
    cat1 = df[(df['slippage'] < LARGE_SLIPPAGE_THRESHOLD) & (df['is_winner'] == True)]
    suspicious['large_slippage_winners'] = cat1
    log(f"\n1. Large negative slippage (< {LARGE_SLIPPAGE_THRESHOLD}) + Winner:")
    log(f"   Count: {len(cat1):,} ({len(cat1)/len(df)*100:.2f}% of trades)")
    if len(cat1) > 0:
        log(f"   Mean return: {cat1['return_bps'].mean():+.1f} bps")
        log(f"   Contribution to overall mean: {cat1['return_bps'].sum() / len(df):.1f} bps")
    
    # Category 2: Extreme returns (>5000 bps)
    cat2 = df[df['return_bps'] > EXTREME_RETURN_THRESHOLD]
    suspicious['extreme_returns'] = cat2
    log(f"\n2. Extreme returns (> {EXTREME_RETURN_THRESHOLD} bps):")
    log(f"   Count: {len(cat2):,} ({len(cat2)/len(df)*100:.2f}% of trades)")
    if len(cat2) > 0:
        log(f"   Mean return: {cat2['return_bps'].mean():+.1f} bps")
        log(f"   Contribution to overall mean: {cat2['return_bps'].sum() / len(df):.1f} bps")
    
    # Category 3: Implausibly low fill price for favorites bucket
    favorites = df[df['prob_bucket'].isin(['75_90', '90_99'])]
    cat3 = favorites[favorites['fill_price'] < IMPLAUSIBLE_FILL_THRESHOLD]
    suspicious['implausible_fills'] = cat3
    log(f"\n3. Implausibly low fills (< {IMPLAUSIBLE_FILL_THRESHOLD}) for 75%+ starters:")
    log(f"   Count: {len(cat3):,} ({len(cat3)/len(favorites)*100:.2f}% of favorite trades)")
    if len(cat3) > 0:
        log(f"   Mean return: {cat3['return_bps'].mean():+.1f} bps")
        log(f"   Contribution to favorites mean: {cat3['return_bps'].sum() / len(favorites):.1f} bps")
    
    # Category 4: The most extreme case - very low fill + winner + favorites bucket
    cat4 = df[
        (df['prob_bucket'].isin(['75_90', '90_99'])) &
        (df['fill_price'] < IMPLAUSIBLE_FILL_THRESHOLD) &
        (df['is_winner'] == True)
    ]
    suspicious['implausible_fill_winners'] = cat4
    log(f"\n4. Implausibly low fills + Winner (favorites only):")
    log(f"   Count: {len(cat4):,}")
    if len(cat4) > 0:
        log(f"   Mean return: {cat4['return_bps'].mean():+.1f} bps")
        log(f"   These are the most suspicious cases.")
    
    return suspicious


def show_extreme_examples(df, n=20):
    """Show the most extreme examples for manual verification."""
    log(f"\n{'='*70}")
    log(f"EXTREME EXAMPLES FOR MANUAL VERIFICATION")
    log(f"{'='*70}")
    
    # Top returns
    log(f"\nTop {n} highest returns:")
    log("-" * 100)
    top_returns = df.nlargest(n, 'return_bps')[
        ['interval', 'prob_bucket', 'threshold', 'tercile', 
         'entry_price', 'fill_price', 'slippage', 'is_winner', 'return_bps']
    ]
    log(f"{'interval':<12} {'bucket':<8} {'thresh':<6} {'terc':<6} "
        f"{'entry':>8} {'fill':>8} {'slip':>8} {'winner':<6} {'return':>10}")
    log("-" * 100)
    for _, row in top_returns.iterrows():
        log(f"{row['interval']:<12} {row['prob_bucket']:<8} {row['threshold']:<6.2f} {row['tercile']:<6} "
            f"{row['entry_price']:>8.4f} {row['fill_price']:>8.4f} {row['slippage']:>+8.4f} "
            f"{str(row['is_winner']):<6} {row['return_bps']:>+10.1f}")
    
    # Largest negative slippage
    log(f"\n\nTop {n} largest negative slippage:")
    log("-" * 100)
    worst_slippage = df.nsmallest(n, 'slippage')[
        ['interval', 'prob_bucket', 'threshold', 'tercile',
         'entry_price', 'fill_price', 'slippage', 'is_winner', 'return_bps']
    ]
    log(f"{'interval':<12} {'bucket':<8} {'thresh':<6} {'terc':<6} "
        f"{'entry':>8} {'fill':>8} {'slip':>8} {'winner':<6} {'return':>10}")
    log("-" * 100)
    for _, row in worst_slippage.iterrows():
        log(f"{row['interval']:<12} {row['prob_bucket']:<8} {row['threshold']:<6.2f} {row['tercile']:<6} "
            f"{row['entry_price']:>8.4f} {row['fill_price']:>8.4f} {row['slippage']:>+8.4f} "
            f"{str(row['is_winner']):<6} {row['return_bps']:>+10.1f}")
    
    return top_returns, worst_slippage


def compute_impact_of_exclusions(df, suspicious):
    """Compute what happens to statistics if we exclude suspicious trades."""
    log(f"\n{'='*70}")
    log(f"IMPACT OF EXCLUSIONS ON REPORTED STATISTICS")
    log(f"{'='*70}")
    
    original_n = len(df)
    original_mean = df['return_bps'].mean()
    original_median = df['return_bps'].median()
    original_winrate = df['is_winner'].mean()
    
    log(f"\nOriginal statistics:")
    log(f"  n:        {original_n:,}")
    log(f"  Mean:     {original_mean:+.1f} bps")
    log(f"  Median:   {original_median:+.1f} bps")
    log(f"  Win rate: {original_winrate*100:.1f}%")
    
    # Exclusion scenarios
    scenarios = [
        ('Exclude extreme returns (>{} bps)'.format(EXTREME_RETURN_THRESHOLD), 
         df[df['return_bps'] <= EXTREME_RETURN_THRESHOLD]),
        ('Exclude large slippage (<{})'.format(LARGE_SLIPPAGE_THRESHOLD),
         df[df['slippage'] >= LARGE_SLIPPAGE_THRESHOLD]),
        ('Exclude implausible fills (favorites < {})'.format(IMPLAUSIBLE_FILL_THRESHOLD),
         df[~((df['prob_bucket'].isin(['75_90', '90_99'])) & (df['fill_price'] < IMPLAUSIBLE_FILL_THRESHOLD))]),
        ('Exclude all suspicious (union)',
         df[~(
             (df['return_bps'] > EXTREME_RETURN_THRESHOLD) |
             (df['slippage'] < LARGE_SLIPPAGE_THRESHOLD) |
             ((df['prob_bucket'].isin(['75_90', '90_99'])) & (df['fill_price'] < IMPLAUSIBLE_FILL_THRESHOLD))
         )]),
    ]
    
    log(f"\nImpact of exclusion scenarios:")
    log("-" * 90)
    log(f"{'Scenario':<50} {'n':>8} {'Mean':>10} {'Median':>10} {'WinRate':>10}")
    log("-" * 90)
    
    for scenario_name, scenario_df in scenarios:
        if len(scenario_df) > 0:
            n = len(scenario_df)
            mean = scenario_df['return_bps'].mean()
            median = scenario_df['return_bps'].median()
            winrate = scenario_df['is_winner'].mean()
            log(f"{scenario_name:<50} {n:>8,} {mean:>+10.1f} {median:>+10.1f} {winrate*100:>9.1f}%")
        else:
            log(f"{scenario_name:<50} {'(empty)':>8}")
    
    log("-" * 90)
    log(f"{'Original':<50} {original_n:>8,} {original_mean:>+10.1f} {original_median:>+10.1f} {original_winrate*100:>9.1f}%")


def analyze_fill_mechanics(df):
    """Deep dive into fill price mechanics."""
    log(f"\n{'='*70}")
    log(f"FILL MECHANICS ANALYSIS")
    log(f"{'='*70}")
    
    # By probability bucket
    log(f"\nFill price statistics by probability bucket:")
    log("-" * 80)
    log(f"{'Bucket':<10} {'n':>8} {'Entry Mean':>12} {'Fill Mean':>12} {'Slippage':>12} {'Win Rate':>10}")
    log("-" * 80)
    
    for bucket in ['sub_51', '51_60', '60_75', '75_90', '90_99', '99_plus']:
        subset = df[df['prob_bucket'] == bucket]
        if len(subset) > 0:
            log(f"{bucket:<10} {len(subset):>8,} {subset['entry_price'].mean():>12.4f} "
                f"{subset['fill_price'].mean():>12.4f} {subset['slippage'].mean():>+12.4f} "
                f"{subset['is_winner'].mean()*100:>9.1f}%")
    
    # Check: Is fill_price ever equal to entry_price?
    exact_fills = (df['fill_price'] == df['entry_price']).sum()
    log(f"\n\nExact fills (fill_price == entry_price): {exact_fills:,} ({exact_fills/len(df)*100:.2f}%)")
    
    # Check: Fill price above entry price (should be rare for buy orders)
    above_entry = (df['fill_price'] > df['entry_price']).sum()
    log(f"Fills above entry price: {above_entry:,} ({above_entry/len(df)*100:.2f}%)")
    
    # Distribution of time from crossing to fill (if we had that data)
    # Note: This would require access to the raw data with fill_time
    
    return


def run_diagnostic(baseline_dir, apply_strategy_filter=False):
    """Run the full diagnostic analysis."""
    
    # Load data
    df = load_baseline_data(baseline_dir)
    if df is None:
        return
    
    # Add volume quintiles if needed
    df = compute_volume_quintiles(df)
    
    # Exclude 'all' bucket to avoid double counting
    df = df[df['prob_bucket'] != 'all'].copy()
    log(f"\nAfter excluding 'all' bucket: {len(df):,} observations")
    
    # Optionally apply Strategy A filter
    if apply_strategy_filter:
        log(f"\nApplying Strategy A filter...")
        df = apply_strategy_a_filter(df, STRATEGY_A_PARAMS)
        log(f"After Strategy A filter: {len(df):,} observations")
        label = "(Strategy A)"
    else:
        label = "(Full Dataset)"
    
    # Compute slippage
    df = compute_slippage_metrics(df)
    
    # Run analyses
    analyze_slippage_distribution(df, label)
    suspicious = identify_suspicious_trades(df)
    show_extreme_examples(df)
    compute_impact_of_exclusions(df, suspicious)
    analyze_fill_mechanics(df)
    
    # Summary
    log(f"\n{'='*70}")
    log(f"DIAGNOSTIC SUMMARY")
    log(f"{'='*70}")
    
    cat4 = suspicious.get('implausible_fill_winners', pd.DataFrame())
    if len(cat4) > 0:
        log(f"\n⚠️  CRITICAL: Found {len(cat4)} trades where:")
        log(f"   - Token started at 75%+ probability")
        log(f"   - Fill occurred at <{IMPLAUSIBLE_FILL_THRESHOLD*100:.0f}% price")
        log(f"   - Token still WON")
        log(f"\n   These trades contribute {cat4['return_bps'].sum()/len(df):.1f} bps to mean edge.")
        log(f"   This pattern is extremely suspicious and warrants manual verification.")
    
    extreme = suspicious.get('extreme_returns', pd.DataFrame())
    if len(extreme) > 0:
        log(f"\n⚠️  Found {len(extreme)} trades with returns >{EXTREME_RETURN_THRESHOLD} bps")
        log(f"   These contribute {extreme['return_bps'].sum()/len(df):.1f} bps to mean edge.")
    
    log(f"\n{'='*70}")
    log(f"DIAGNOSTIC COMPLETE")
    log(f"{'='*70}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic: Fill Price Audit',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--baseline-dir', type=str, required=True,
                        help='Path to volume_returns_baseline directory')
    parser.add_argument('--strategy-a', action='store_true',
                        help='Apply Strategy A filter before analysis')
    
    args = parser.parse_args()
    
    run_diagnostic(args.baseline_dir, apply_strategy_filter=args.strategy_a)


if __name__ == "__main__":
    main()