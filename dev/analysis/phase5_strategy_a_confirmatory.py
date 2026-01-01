#!/usr/bin/env python3
"""
Strategy A Confirmatory Script - Favorites Fading (Batch-Viable)
Version: 2.0

CHANGELOG v2.0:
  - CRITICAL FIX: Returns now calculated from entry_price (limit price), not 
    the incorrectly simulated fill_price. Previous version used market trade 
    price which gave unrealistic price improvement.
  - Updated defaults based on Phase 4E delay analysis findings
  - Changed default windows to batch-viable (48h→24h, 24h→12h)
  - Increased minimum threshold to 15% (favorites need larger dips)
  - Added --reaction-delay parameter for rebuild mode
  - Terciles now defaults to all (removed as primary filter)
  - Removed volume quintile restriction (needs revalidation post-bugfix)

OBJECTIVE:
  Validate that the formalized Strategy A rules extract consistent positive edge
  before layering on position sizing, temporal sequencing, and portfolio metrics.
  This is a "unit test" for the strategy logic.

STRATEGY A RULES (Updated from Phase 4E Delay Analysis):
  - Prob buckets: 75_90, 90_99 (confirmed positive edge in delay analysis)
  - Windows: 48h→24h, 24h→12h (batch-viable, edge retained at 15-30min delay)
  - Threshold: ≥15% (favorites need larger dips to signal genuine overreaction)
  - Terciles: All (no longer primary discriminator - use delay instead)
  - Volume: All quintiles (volume discrimination needs revalidation)
  - Reaction delay: Configurable, default 0s for baseline comparison

KEY FINDINGS FROM DELAY ANALYSIS:
  - 75_90 at 48h→24h, 15% threshold: +222 bps at 0s, slower decay than longshots
  - 90_99 at 48h→24h, 15% threshold: +551 bps at 0s, but half-life ~5min
  - Close-to-resolution windows (6h→4h) show +221 bps at 0s but collapse to 
    negative within 5 minutes - these require real-time infrastructure
  - For batch execution, stick to 48h→24h and 24h→12h windows

DATA SOURCE:
  Uses pre-computed volume returns data from phase4d_volume_stratification.py
  Default path: ../../analysis/outputs/volume_returns_baseline/
  
  NOTE: Baseline data contains incorrectly computed fill_price (market price).
  This script recalculates returns from entry_price (the actual limit price).

USAGE:
  # Quick run with defaults
  python phase5_strategy_a_confirmatory_v2.py

  # Diagnostic mode
  python phase5_strategy_a_confirmatory_v2.py --diagnostic

  # Custom parameters
  python phase5_strategy_a_confirmatory_v2.py --min-threshold 0.20

  # Force rebuild with reaction delay
  python phase5_strategy_a_confirmatory_v2.py --rebuild --reaction-delay 300
"""

import pyarrow.parquet as pq
import glob
import json
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from collections import defaultdict
import psutil
import gc
import os
import sys
import warnings
import argparse

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = "../../"

# Data paths
BATCH_DIR = os.path.join(BASE_DIR, 'order_history_batches')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis/outputs')
INDEX_CACHE_FILE = os.path.join(BASE_DIR, 'analysis/market_index_cache_full.pkl')
SIDECAR_FILE = os.path.join(BASE_DIR, 'data/repair/api_derived_winners.parquet')

# Baseline volume returns data (from phase4d_volume_stratification)
VOLUME_BASELINE_DIR = os.path.join(OUTPUT_DIR, 'volume_returns_baseline')

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ------------------------------------------------------------------------------
# STRATEGY A DEFAULT PARAMETERS (UPDATED from Phase 4E Delay Analysis)
# ------------------------------------------------------------------------------

STRATEGY_A_DEFAULTS = {
    # Favorites buckets - confirmed positive edge in delay analysis
    'prob_buckets': ['75_90', '90_99'],
    
    # Batch-viable windows only (edge retained at 15-30min delay)
    # Excluded: 6h→4h, 8h→4h, 9h→6h (edge collapses within 5min - need real-time)
    # Note: 90_99 has faster decay than 75_90 even in these windows
    'intervals': ['48h_to_24h', '24h_to_12h'],
    
    # 15% minimum (favorites need larger dips to signal overreaction vs informed flow)
    'min_threshold': 0.15,
    
    # All terciles - no longer primary discriminator
    # Delay-based analysis supersedes tercile stratification
    'terciles': ['early', 'mid', 'late'],
    
    # All volume quintiles (volume discrimination needs revalidation post-bugfix)
    'volume_quintiles': ['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high'],
}

# ------------------------------------------------------------------------------
# ANALYSIS PARAMETERS (must match volume_stratification for compatibility)
# ------------------------------------------------------------------------------

INTERVAL_PAIRS = [
    (48, 24, '48h_to_24h'),
    (24, 12, '24h_to_12h'),
    (12, 6, '12h_to_6h'),
    (9, 6, '9h_to_6h'),
    (8, 4, '8h_to_4h'),
    (6, 4, '6h_to_4h'),
]

MOVE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]

TERCILE_LABELS = ['early', 'mid', 'late']
TERCILE_BOUNDS = [(0.0, 0.333), (0.333, 0.666), (0.666, 1.0)]

PROB_BUCKETS = [
    ('sub_51', 0.0, 0.51),
    ('51_60', 0.51, 0.60),
    ('60_75', 0.60, 0.75),
    ('75_90', 0.75, 0.90),
    ('90_99', 0.90, 0.99),
    ('99_plus', 0.99, 1.01),
]
PROB_BUCKET_LABELS = [b[0] for b in PROB_BUCKETS]

VOLUME_QUINTILE_LABELS = ['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high']

# Volume window parameters (must match baseline data)
VOLUME_WINDOW_BEFORE = 1800  # 30 minutes before crossing
VOLUME_WINDOW_AFTER = 1800   # 30 minutes after crossing

MIN_TRADES = 10
MIN_SAMPLES_PER_CELL = 20

REQUIRED_COLUMNS = ['timestamp', 'price', 'token_id', 'condition_id', 'resolution_time']
VOLUME_COLUMNS = ['size_tokens', 'maker_amount', 'size']


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def log_memory():
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024**2
    return f"Memory: {mem_mb:.0f}MB"

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_prob_bucket(start_price):
    """Determine which probability bucket a start_price falls into."""
    for label, lo, hi in PROB_BUCKETS:
        if lo <= start_price < hi:
            return label
    return None


def get_tercile(fraction):
    """Determine tercile from fraction of interval."""
    for label, (lo, hi) in zip(TERCILE_LABELS, TERCILE_BOUNDS):
        if lo <= fraction < hi or (label == 'late' and fraction == 1.0):
            return label
    return None


def get_volume_quintile(volume, quintile_thresholds):
    """Determine volume quintile from volume value and pre-computed thresholds."""
    if quintile_thresholds is None or volume <= 0:
        return None
    
    q20, q40, q60, q80 = quintile_thresholds
    
    if volume <= q20:
        return 'Q1_low'
    elif volume <= q40:
        return 'Q2'
    elif volume <= q60:
        return 'Q3'
    elif volume <= q80:
        return 'Q4'
    else:
        return 'Q5_high'


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_volume_returns_baseline(baseline_dir, diagnostic=False):
    """
    Load pre-computed volume returns data from baseline directory.
    
    CRITICAL: This function recalculates return_bps from entry_price (the limit
    price) rather than using the stored return_bps which was computed from the
    incorrect fill_price (market trade price).
    
    Returns DataFrame with columns:
        interval, threshold, tercile, prob_bucket, entry_price, fill_price,
        is_winner, return_bps, return_bps_corrected, volume_at_dip
    """
    log(f"Loading volume returns baseline from: {baseline_dir}")
    
    if not os.path.exists(baseline_dir):
        log(f"  ERROR: Baseline directory not found: {baseline_dir}")
        return None
    
    chunk_files = sorted(glob.glob(os.path.join(baseline_dir, 'volume_returns_chunk_*.parquet')))
    
    if not chunk_files:
        log(f"  ERROR: No chunk files found in {baseline_dir}")
        return None
    
    log(f"  Found {len(chunk_files)} chunk files")
    
    if diagnostic:
        # Load only first 3 chunks for validation
        chunk_files = chunk_files[:3]
        log(f"  DIAGNOSTIC: Loading only {len(chunk_files)} chunks")
    
    dfs = []
    for chunk_file in chunk_files:
        df = pd.read_parquet(chunk_file)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    log(f"  Loaded {len(combined):,} returns observations")
    
    # =========================================================================
    # CRITICAL FIX: Recalculate returns from entry_price (limit price)
    # =========================================================================
    # The baseline data has:
    #   - entry_price: the crossing price = our limit price (CORRECT)
    #   - fill_price: the market trade price when filled (INCORRECT for returns)
    #   - return_bps: computed from fill_price (INCORRECT)
    #
    # We need returns computed from entry_price (our actual fill price as limit order)
    
    log("\n  *** APPLYING CRITICAL FIX: Recalculating returns from limit price ***")
    log(f"  Old return_bps stats: mean={combined['return_bps'].mean():+.1f}, median={combined['return_bps'].median():+.1f}")
    
    # Store old returns for comparison
    combined['return_bps_old'] = combined['return_bps']
    
    # Recalculate: return = (outcome - entry_price) * 10000
    # outcome is 1.0 if winner, 0.0 if loser
    combined['return_bps'] = (combined['is_winner'].astype(float) - combined['entry_price']) * 10000
    
    log(f"  New return_bps stats: mean={combined['return_bps'].mean():+.1f}, median={combined['return_bps'].median():+.1f}")
    
    avg_correction = (combined['return_bps'] - combined['return_bps_old']).mean()
    log(f"  Average correction: {avg_correction:+.1f} bps (positive = old was inflated)")
    
    log(f"  {log_memory()}")
    
    # Validate schema
    expected_columns = ['interval', 'threshold', 'tercile', 'prob_bucket', 
                        'entry_price', 'fill_price', 'is_winner', 'return_bps', 'volume_at_dip']
    missing = set(expected_columns) - set(combined.columns)
    if missing:
        log(f"  WARNING: Missing columns: {missing}")
    
    return combined


def compute_volume_quintiles(returns_df):
    """
    Compute volume quintiles globally across all observations.
    Returns the modified dataframe with 'volume_quintile' column and thresholds.
    """
    log("\nComputing volume quintiles...")
    
    # Filter out zero volume
    valid_volume = returns_df[returns_df['volume_at_dip'] > 0]['volume_at_dip']
    
    if len(valid_volume) == 0:
        log("  WARNING: No valid volume data found")
        returns_df['volume_quintile'] = None
        return returns_df, None
    
    # Compute quintile thresholds
    q20 = np.percentile(valid_volume, 20)
    q40 = np.percentile(valid_volume, 40)
    q60 = np.percentile(valid_volume, 60)
    q80 = np.percentile(valid_volume, 80)
    
    quintile_thresholds = (q20, q40, q60, q80)
    
    log(f"  Volume quintile thresholds:")
    log(f"    Q1 (low):    <= {q20:.2f}")
    log(f"    Q2:          {q20:.2f} - {q40:.2f}")
    log(f"    Q3:          {q40:.2f} - {q60:.2f}")
    log(f"    Q4:          {q60:.2f} - {q80:.2f}")
    log(f"    Q5 (high):   > {q80:.2f}")
    
    # Assign quintiles
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
    
    returns_df['volume_quintile'] = returns_df['volume_at_dip'].apply(assign_quintile)
    
    # Log distribution
    quintile_counts = returns_df['volume_quintile'].value_counts()
    log(f"\n  Volume quintile distribution:")
    for q in VOLUME_QUINTILE_LABELS:
        count = quintile_counts.get(q, 0)
        log(f"    {q}: {count:,}")
    
    return returns_df, quintile_thresholds


# ==============================================================================
# STRATEGY A FILTERING
# ==============================================================================

def apply_strategy_a_filter(returns_df, params):
    """
    Apply Strategy A filter rules to returns dataframe.
    
    Args:
        returns_df: DataFrame with all returns (must have volume_quintile column)
        params: Dict with filter parameters:
            - prob_buckets: list of probability bucket labels
            - intervals: list of interval labels
            - min_threshold: minimum threshold value
            - terciles: list of tercile labels
            - volume_quintiles: list of volume quintile labels
    
    Returns:
        Filtered DataFrame
    """
    log("\nApplying Strategy A filter...")
    log(f"  Parameters:")
    log(f"    Prob buckets:     {params['prob_buckets']}")
    log(f"    Intervals:        {params['intervals']}")
    log(f"    Min threshold:    {params['min_threshold']:.0%}")
    log(f"    Volume quintiles: {params['volume_quintiles']}")
    
    initial_n = len(returns_df)
    
    # Exclude 'all' bucket (it's a summary bucket, would double-count)
    df = returns_df[returns_df['prob_bucket'] != 'all'].copy()
    
    # Apply filters
    mask = (
        (df['prob_bucket'].isin(params['prob_buckets'])) &
        (df['interval'].isin(params['intervals'])) &
        (df['threshold'] >= params['min_threshold']) &
        (df['tercile'].isin(params['terciles'])) &
        (df['volume_quintile'].isin(params['volume_quintiles']))
    )
    
    filtered_df = df[mask]
    
    log(f"\n  Filter results:")
    log(f"    Initial (excl 'all' bucket): {len(df):,}")
    log(f"    After filter:                {len(filtered_df):,}")
    log(f"    Pass rate:                   {len(filtered_df)/len(df)*100:.1f}%" if len(df) > 0 else "    Pass rate: N/A")
    
    return filtered_df


def compute_aggregate_stats(returns_df, label=""):
    """
    Compute aggregate statistics for a returns distribution.
    
    Returns dict with: n, win_rate, mean, median, std, p10, p25, p75, p90, min, max
    """
    if len(returns_df) == 0:
        return None
    
    returns = returns_df['return_bps'].values
    
    stats = {
        'label': label,
        'n': len(returns),
        'win_rate': returns_df['is_winner'].mean(),
        'mean': np.mean(returns),
        'median': np.median(returns),
        'std': np.std(returns),
        'p10': np.percentile(returns, 10),
        'p25': np.percentile(returns, 25),
        'p75': np.percentile(returns, 75),
        'p90': np.percentile(returns, 90),
        'min': np.min(returns),
        'max': np.max(returns),
    }
    
    return stats


# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_strategy_a_report(filtered_stats, unfiltered_stats, params, output_dir, 
                                quintile_thresholds, diagnostic=False, reaction_delay=0):
    """
    Generate comprehensive text report for Strategy A confirmatory analysis.
    """
    suffix = '_DIAGNOSTIC' if diagnostic else ''
    report_path = os.path.join(output_dir, f'strategyA_confirmatory_v2_report{suffix}_{TIMESTAMP}.txt')
    
    log(f"\nGenerating report: {report_path}")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STRATEGY A CONFIRMATORY ANALYSIS v2.0: FAVORITES FADING (BATCH-VIABLE)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if diagnostic:
            f.write("*** DIAGNOSTIC MODE - PARTIAL DATA ***\n")
        f.write("\n")
        
        # ---------------------------------------------------------------------
        # VERSION 2.0 CRITICAL FIXES
        # ---------------------------------------------------------------------
        f.write("-"*80 + "\n")
        f.write("VERSION 2.0 CRITICAL FIXES\n")
        f.write("-"*80 + "\n\n")
        
        f.write("1. FILL PRICE FIX:\n")
        f.write("   Previous version incorrectly used market trade price as fill price,\n")
        f.write("   giving unrealistic price improvement. This version uses entry_price\n")
        f.write("   (the limit/crossing price) as the actual fill price.\n\n")
        
        f.write("2. WINDOW UPDATE:\n")
        f.write("   Changed to batch-viable windows (48h→24h, 24h→12h) where edge\n")
        f.write("   is retained at 15-30 minute reaction delays.\n")
        f.write("   Excluded 6h→4h, 8h→4h, 9h→6h (edge collapses within 5min).\n")
        f.write("   NOTE: Close-to-resolution windows CAN work but require real-time\n")
        f.write("   infrastructure with <5min latency.\n\n")
        
        f.write("3. THRESHOLD UPDATE:\n")
        f.write("   Increased minimum threshold to 15% (10% showed more noise in\n")
        f.write("   favorites; larger dips needed to distinguish overreaction from\n")
        f.write("   informed flow).\n\n")
        
        f.write("4. VOLUME FILTER REMOVED:\n")
        f.write("   Previous Q1-Q2 volume restriction was based on pre-bugfix analysis.\n")
        f.write("   Now using all volume quintiles pending revalidation.\n\n")
        
        # ---------------------------------------------------------------------
        # FILTER PARAMETERS
        # ---------------------------------------------------------------------
        f.write("-"*80 + "\n")
        f.write("FILTER PARAMETERS\n")
        f.write("-"*80 + "\n\n")
        
        f.write(f"  Prob buckets:      {params['prob_buckets']}\n")
        f.write(f"  Intervals:         {params['intervals']}\n")
        f.write(f"  Min threshold:     {params['min_threshold']:.0%}\n")
        f.write(f"  Volume quintiles:  {params['volume_quintiles']}\n")
        f.write(f"  Reaction delay:    {reaction_delay}s (baseline data assumes ~0s)\n")
        f.write("\n")
        
        # Volume quintile thresholds
        if quintile_thresholds:
            q20, q40, q60, q80 = quintile_thresholds
            f.write("  Volume quintile boundaries (tokens traded in ±30min window):\n")
            f.write(f"    Q1 (low):  <= {q20:.2f}\n")
            f.write(f"    Q2:        {q20:.2f} - {q40:.2f}\n")
            f.write(f"    Q3:        {q40:.2f} - {q60:.2f}\n")
            f.write(f"    Q4:        {q60:.2f} - {q80:.2f}\n")
            f.write(f"    Q5 (high): > {q80:.2f}\n")
        f.write("\n")
        
        # ---------------------------------------------------------------------
        # STRATEGY A RESULTS (Primary Output)
        # ---------------------------------------------------------------------
        f.write("="*80 + "\n")
        f.write("STRATEGY A RESULTS (CORRECTED FILL PRICES)\n")
        f.write("="*80 + "\n\n")
        
        if filtered_stats:
            # Primary metrics block
            f.write("```\n")
            f.write(f"Trades passing filter:     n = {filtered_stats['n']:,}\n")
            f.write(f"Win rate:                  {filtered_stats['win_rate']*100:.1f}%\n")
            f.write(f"Mean return:               {filtered_stats['mean']:+.1f} bps\n")
            f.write(f"Median return:             {filtered_stats['median']:+.1f} bps\n")
            f.write(f"Std dev:                   {filtered_stats['std']:.1f} bps\n")
            f.write(f"P10 (worst decile):        {filtered_stats['p10']:+.1f} bps\n")
            f.write(f"P90 (best decile):         {filtered_stats['p90']:+.1f} bps\n")
            f.write("```\n")
            
            # Extended percentiles
            f.write("\nExtended percentiles:\n")
            f.write(f"  P25:                     {filtered_stats['p25']:+.1f} bps\n")
            f.write(f"  P75:                     {filtered_stats['p75']:+.1f} bps\n")
            f.write(f"  Min:                     {filtered_stats['min']:+.1f} bps\n")
            f.write(f"  Max:                     {filtered_stats['max']:+.1f} bps\n")
        else:
            f.write("ERROR: No trades passed the filter.\n")
        f.write("\n")
        
        # ---------------------------------------------------------------------
        # COMPARISON TO UNFILTERED
        # ---------------------------------------------------------------------
        f.write("-"*80 + "\n")
        f.write("COMPARISON TO UNFILTERED BASELINE\n")
        f.write("-"*80 + "\n\n")
        
        if unfiltered_stats and filtered_stats:
            f.write(f"Unfiltered universe (all threshold crossings, excl 'all' bucket):\n")
            f.write(f"  n = {unfiltered_stats['n']:,}\n")
            f.write(f"  Win rate = {unfiltered_stats['win_rate']*100:.1f}%\n")
            f.write(f"  Mean = {unfiltered_stats['mean']:+.1f} bps\n")
            f.write(f"  Median = {unfiltered_stats['median']:+.1f} bps\n")
            f.write("\n")
            
            f.write(f"Strategy A filter applied:\n")
            f.write(f"  n = {filtered_stats['n']:,} ({filtered_stats['n']/unfiltered_stats['n']*100:.1f}% of universe)\n")
            f.write(f"  Win rate = {filtered_stats['win_rate']*100:.1f}%\n")
            f.write(f"  Mean = {filtered_stats['mean']:+.1f} bps\n")
            f.write(f"  Median = {filtered_stats['median']:+.1f} bps\n")
            f.write("\n")
            
            edge_improvement = filtered_stats['mean'] - unfiltered_stats['mean']
            winrate_improvement = (filtered_stats['win_rate'] - unfiltered_stats['win_rate']) * 100
            
            f.write(f"Improvement from filtering:\n")
            f.write(f"  Mean edge improvement:     {edge_improvement:+.1f} bps\n")
            f.write(f"  Win rate improvement:      {winrate_improvement:+.1f} pp\n")
        f.write("\n")
        
        # ---------------------------------------------------------------------
        # EXPECTED PERFORMANCE WITH DELAY
        # ---------------------------------------------------------------------
        f.write("-"*80 + "\n")
        f.write("EXPECTED EDGE DEGRADATION WITH REACTION DELAY\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Based on Phase 4E delay analysis for favorites at 48h→24h:\n\n")
        f.write("75_90 bucket, 15% threshold:\n")
        f.write("  Delay       Edge (bps)    Notes\n")
        f.write("  ---------   ----------    -----\n")
        f.write("  0s (inst)   +222          Baseline\n")
        f.write("  5min        ~+150         Moderate decay\n")
        f.write("  30min       ~+50          Still positive\n")
        f.write("\n")
        f.write("90_99 bucket, 15% threshold:\n")
        f.write("  Delay       Edge (bps)    Notes\n")
        f.write("  ---------   ----------    -----\n")
        f.write("  0s (inst)   +551          High baseline but fast decay\n")
        f.write("  5min        ~+200         Half-life ~5min\n")
        f.write("  30min       ~marginal     Edge largely gone\n")
        f.write("\n")
        f.write("RECOMMENDATION: \n")
        f.write("  - 75_90: Batch job at 15-30min intervals is viable\n")
        f.write("  - 90_99: Requires faster reaction (<10min) to capture meaningful edge\n")
        f.write("  - Consider separating 75_90 and 90_99 into distinct operational strategies\n")
        f.write("\n")
        
        # ---------------------------------------------------------------------
        # RETURN DISTRIBUTION CHARACTERIZATION
        # ---------------------------------------------------------------------
        f.write("-"*80 + "\n")
        f.write("RETURN DISTRIBUTION CHARACTERIZATION\n")
        f.write("-"*80 + "\n\n")
        
        if filtered_stats:
            spread = filtered_stats['p90'] - filtered_stats['p10']
            skew = (filtered_stats['mean'] - filtered_stats['median'])
            
            f.write(f"P90 - P10 spread:          {spread:.1f} bps\n")
            f.write(f"Mean - Median (skew):      {skew:+.1f} bps\n")
            f.write("\n")
            
            # Strategy A note: different risk profile than B
            f.write("Strategy A Risk Profile Notes:\n")
            f.write("  - Higher win rate expected (betting on favorites to recover)\n")
            f.write("  - Wins are typically smaller per trade (buying near fair value)\n")
            f.write("  - Losses can be large when favorite actually loses (rare but painful)\n")
            f.write("\n")
            
            if filtered_stats['p10'] < -5000:
                f.write("⚠️  WARNING: P10 < -5000 bps indicates fat-tailed losses.\n")
                f.write("    Consider stop-loss rules or position sizing adjustments.\n")
            elif filtered_stats['p10'] < -3000:
                f.write("⚠️  CAUTION: P10 < -3000 bps indicates significant loss potential.\n")
                f.write("    Monitor position sizes carefully.\n")
            else:
                f.write("✓  Loss distribution appears manageable (P10 > -3000 bps).\n")
            
            if skew < -200:
                f.write("⚠️  Negative skew detected (mean < median).\n")
                f.write("    Occasional large losses drag down average returns.\n")
                f.write("    This is expected for favorites strategy - wins are frequent but small,\n")
                f.write("    losses are rare but large.\n")
            
            f.write("\n")
            
            # Favorites characterization
            f.write("Favorites Payoff Structure:\n")
            f.write(f"  Win rate:                {filtered_stats['win_rate']*100:.1f}%\n")
            if filtered_stats['win_rate'] > 0:
                f.write(f"  Typical loss (P10):      {filtered_stats['p10']:+.1f} bps\n")
                f.write(f"  Typical win (P90):       {filtered_stats['p90']:+.1f} bps\n")
                
                # Loss/win ratio
                if filtered_stats['p90'] != 0:
                    loss_win_ratio = abs(filtered_stats['p10']) / abs(filtered_stats['p90'])
                    f.write(f"  Loss/Win ratio:          {loss_win_ratio:.1f}x\n")
        f.write("\n")
        
        # ---------------------------------------------------------------------
        # METHODOLOGY NOTES
        # ---------------------------------------------------------------------
        f.write("-"*80 + "\n")
        f.write("METHODOLOGY NOTES\n")
        f.write("-"*80 + "\n\n")
        
        f.write("1. Returns are calculated as: (outcome - entry_price) * 10000\n")
        f.write("   where entry_price is the crossing/limit price.\n\n")
        
        f.write("2. This assumes fill at limit price (conservative). In reality,\n")
        f.write("   you may get price improvement if the market trades through.\n\n")
        
        f.write("3. Baseline data was computed with ~0s reaction delay. The delay\n")
        f.write("   degradation table above is from separate delay analysis.\n\n")
        
        f.write("4. Win rate reflects binary outcomes: 1 if token wins, 0 otherwise.\n")
        f.write("   For favorites, high win rate is expected but each loss is larger.\n\n")
        
        # ---------------------------------------------------------------------
        # CAPACITY ESTIMATE
        # ---------------------------------------------------------------------
        f.write("-"*80 + "\n")
        f.write("CAPACITY ESTIMATE\n")
        f.write("-"*80 + "\n\n")
        
        if filtered_stats:
            # Rough annualization (assuming data spans ~1 year)
            trades_per_month = filtered_stats['n'] / 12  # crude estimate
            f.write(f"Total trades in dataset:   {filtered_stats['n']:,}\n")
            f.write(f"Est. trades per month:     ~{trades_per_month:.0f} (assumes 1-year dataset)\n")
            f.write(f"Est. monthly capital:      ~${trades_per_month * 100:,.0f} (at $100/trade)\n")
            f.write("\n")
            f.write("Note: Strategy A (favorites) typically has lower capacity than Strategy B\n")
            f.write("      because large dips in high-probability markets are less common.\n")
            f.write("      The 15% threshold further reduces opportunity count but improves\n")
            f.write("      signal quality.\n")
        f.write("\n")
        
        # ---------------------------------------------------------------------
        # SUMMARY
        # ---------------------------------------------------------------------
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        if filtered_stats and filtered_stats['mean'] > 0:
            f.write("✓  Strategy A shows POSITIVE expected edge.\n")
            f.write(f"   Mean return: {filtered_stats['mean']:+.1f} bps per trade\n")
            f.write(f"   Win rate: {filtered_stats['win_rate']*100:.1f}%\n")
            f.write("\n")
            f.write("RECOMMENDATION: Proceed to position sizing and temporal backtest.\n")
            f.write("                Consider splitting 75_90 vs 90_99 for operational purposes\n")
            f.write("                (different latency requirements).\n")
        elif filtered_stats:
            f.write("✗  Strategy A shows NEGATIVE or ZERO expected edge.\n")
            f.write(f"   Mean return: {filtered_stats['mean']:+.1f} bps per trade\n")
            f.write("\n")
            f.write("RECOMMENDATION: Review filter parameters or investigate data.\n")
        else:
            f.write("✗  No trades passed the filter. Check parameters.\n")
        
        f.write("\n" + "="*80 + "\n")
    
    return report_path


# ==============================================================================
# REBUILD FUNCTIONS (for --rebuild mode)
# ==============================================================================

def load_winner_sidecar(sidecar_path):
    """Load winner status lookup from sidecar file."""
    log(f"Loading winner sidecar from: {sidecar_path}")
    
    if not os.path.exists(sidecar_path):
        log(f"  ERROR: Sidecar file not found: {sidecar_path}")
        return None
    
    try:
        df = pd.read_parquet(sidecar_path)
        
        # Build lookup dict
        winner_lookup = {}
        for _, row in df.iterrows():
            token_id = row.get('token_id')
            winner = row.get('winner', row.get('is_winner', None))
            if token_id is not None and winner is not None:
                winner_lookup[token_id] = bool(winner)
        
        log(f"  Loaded {len(winner_lookup):,} winner statuses")
        return winner_lookup
        
    except Exception as e:
        log(f"  ERROR loading sidecar: {e}")
        return None


def load_market_index(index_cache_path, batch_files):
    """Load or build market index cache."""
    if os.path.exists(index_cache_path):
        log(f"Loading market index from cache: {index_cache_path}")
        try:
            with open(index_cache_path, 'rb') as f:
                market_index = pickle.load(f)
            log(f"  Loaded index with {len(market_index):,} conditions")
            return market_index
        except Exception as e:
            log(f"  ERROR loading cache: {e}")
    
    return None


def extract_price_at_horizon(trades, resolution_time, hours_before):
    """Extract price at a specific time horizon before resolution."""
    target_time = resolution_time - (hours_before * 3600)
    
    # Find trades before and after target time
    before_trades = [(ts, p, s) for ts, p, s in trades if ts <= target_time]
    after_trades = [(ts, p, s) for ts, p, s in trades if ts > target_time]
    
    if not before_trades:
        if after_trades:
            return after_trades[0][0], after_trades[0][1]
        return None, None
    
    # Use the last trade before or at target time
    last_before = before_trades[-1]
    return last_before[0], last_before[1]


def find_first_threshold_crossing(trades, start_time, end_time, start_price, threshold, direction='drop'):
    """Find first threshold crossing within interval."""
    interval_trades = [(ts, p, s) for ts, p, s in trades 
                       if start_time < ts <= end_time]
    
    if not interval_trades:
        return {'crossed': False}
    
    interval_length = end_time - start_time
    
    for ts, price, size in interval_trades:
        if direction == 'drop':
            move = (start_price - price) / start_price if start_price > 0 else 0
        else:
            move = (price - start_price) / start_price if start_price > 0 else 0
        
        if move >= threshold:
            fraction = (ts - start_time) / interval_length if interval_length > 0 else 0
            return {
                'crossed': True,
                'crossing_time': ts,
                'crossing_price': price,
                'fraction_of_interval': fraction,
                'time_to_crossing_hours': (ts - start_time) / 3600,
                'velocity': (move / ((ts - start_time) / 3600)) if (ts - start_time) > 0 else 0,
                'move_at_crossing': move,
            }
    
    # Check if final price crossed threshold
    final_ts, final_price, _ = interval_trades[-1]
    if direction == 'drop':
        final_move = (start_price - final_price) / start_price if start_price > 0 else 0
    else:
        final_move = (final_price - start_price) / start_price if start_price > 0 else 0
    
    if final_move >= threshold:
        return {
            'crossed': True,
            'crossing_time': end_time,
            'crossing_price': final_price,
            'fraction_of_interval': 1.0,
            'time_to_crossing_hours': interval_length / 3600,
            'velocity': (threshold / (interval_length / 3600)) if interval_length > 0 else 0,
            'move_at_crossing': final_move,
        }
    
    return {'crossed': False}


def calculate_volume_at_crossing(trades, crossing_time, window_before=1800, window_after=1800):
    """Calculate volume in a window around the crossing time."""
    window_start = crossing_time - window_before
    window_end = crossing_time + window_after
    
    window_trades = [(ts, p, s) for ts, p, s in trades 
                     if window_start <= ts <= window_end]
    
    if not window_trades:
        return 0.0
    
    total_volume = sum(s for _, _, s in window_trades)
    return total_volume


def simulate_limit_order_fill(trades, placement_time, limit_price, is_buy=True):
    """
    Simulate whether a limit order would be filled.
    
    CRITICAL FIX in v2.0: Returns limit_price as fill_price, not the market
    trade price. When you place a limit buy at 0.70, you get filled at 0.70,
    not at whatever the market traded at.
    """
    future_trades = [(ts, p, s) for ts, p, s in trades if ts > placement_time]
    
    if not future_trades:
        return {
            'filled': False,
            'fill_time': None,
            'fill_price': None,
            'time_to_fill': None,
        }
    
    for ts, price, size in future_trades:
        if is_buy and price <= limit_price:
            return {
                'filled': True,
                'fill_time': ts,
                'fill_price': limit_price,  # FIXED: Use limit price, not market price
                'time_to_fill': ts - placement_time,
            }
        elif not is_buy and price >= limit_price:
            return {
                'filled': True,
                'fill_time': ts,
                'fill_price': limit_price,  # FIXED: Use limit price, not market price
                'time_to_fill': ts - placement_time,
            }
    
    return {
        'filled': False,
        'fill_time': None,
        'fill_price': None,
        'time_to_fill': None,
    }


class VolumeReturnsWriter:
    """Memory-safe streaming writer for returns data with volume."""
    
    def __init__(self, output_dir, chunk_size=10000):
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.buffer = []
        self.chunk_counter = 0
        self.total_written = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        existing = glob.glob(os.path.join(output_dir, 'volume_returns_chunk_*.parquet'))
        if existing:
            max_idx = max(int(f.split('_')[-1].replace('.parquet', '')) for f in existing)
            self.chunk_counter = max_idx + 1
            log(f"  Found {len(existing)} existing chunks, starting at {self.chunk_counter}")
    
    def add_return(self, interval_label, threshold, tercile, prob_bucket,
                   entry_price, fill_price, is_winner, return_bps, volume_at_dip):
        self.buffer.append({
            'interval': interval_label,
            'threshold': threshold,
            'tercile': tercile,
            'prob_bucket': prob_bucket,
            'entry_price': entry_price,
            'fill_price': fill_price,
            'is_winner': is_winner,
            'return_bps': return_bps,
            'volume_at_dip': volume_at_dip,
        })
        
        if len(self.buffer) >= self.chunk_size:
            self.flush()
    
    def flush(self):
        if not self.buffer:
            return
        
        df = pd.DataFrame(self.buffer)
        
        chunk_path = os.path.join(self.output_dir, f'volume_returns_chunk_{self.chunk_counter:06d}.parquet')
        while os.path.exists(chunk_path):
            self.chunk_counter += 1
            chunk_path = os.path.join(self.output_dir, f'volume_returns_chunk_{self.chunk_counter:06d}.parquet')
        
        df.to_parquet(chunk_path, index=False)
        
        self.total_written += len(self.buffer)
        self.chunk_counter += 1
        self.buffer = []
        
        gc.collect()
    
    def finalize(self):
        self.flush()
        return {
            'total_written': self.total_written,
            'num_chunks': self.chunk_counter,
            'output_dir': self.output_dir,
        }


class VolumeTimingAccumulator:
    """Accumulates trade data for timing and volume analysis."""
    
    def __init__(self, token_id, condition_id, resolution_time, winner_status):
        self.token_id = token_id
        self.condition_id = condition_id
        self.resolution_time = resolution_time
        self.winner_status = winner_status
        self.trades = []
    
    def add_trades(self, trades_batch):
        self.trades.extend(trades_batch)
    
    def compute_returns_with_volume(self, returns_writer, reaction_delay=0):
        """
        Compute returns for all cells WITH VOLUME AT DIP.
        
        Args:
            returns_writer: VolumeReturnsWriter instance
            reaction_delay: Seconds to delay order placement after crossing detection
        """
        if len(self.trades) < MIN_TRADES:
            return 0
        
        self.trades.sort(key=lambda x: x[0])
        
        n_returns = 0
        
        for start_h, end_h, interval_label in INTERVAL_PAIRS:
            start_time, start_price = extract_price_at_horizon(
                self.trades, self.resolution_time, start_h
            )
            end_time, end_price = extract_price_at_horizon(
                self.trades, self.resolution_time, end_h
            )
            
            if start_price is None or end_price is None:
                continue
            if start_time is None or end_time is None:
                continue
            
            prob_bucket = get_prob_bucket(start_price)
            if prob_bucket is None:
                continue
            
            for threshold in MOVE_THRESHOLDS:
                crossing = find_first_threshold_crossing(
                    self.trades, start_time, end_time, start_price, threshold, direction='drop'
                )
                
                if not crossing or not crossing.get('crossed'):
                    continue
                
                fraction = crossing['fraction_of_interval']
                tercile = get_tercile(fraction)
                if tercile is None:
                    continue
                
                crossing_time = crossing['crossing_time']
                crossing_price = crossing['crossing_price']
                
                # Apply reaction delay
                placement_time = crossing_time + reaction_delay
                
                volume_at_dip = calculate_volume_at_crossing(
                    self.trades, 
                    crossing_time,
                    window_before=VOLUME_WINDOW_BEFORE,
                    window_after=VOLUME_WINDOW_AFTER
                )
                
                # Simulate fill with delayed placement
                fill_result = simulate_limit_order_fill(
                    self.trades, placement_time, crossing_price, is_buy=True
                )
                
                if not fill_result.get('filled'):
                    continue
                
                # CRITICAL: fill_price is now the limit price (crossing_price)
                fill_price = fill_result['fill_price']
                
                outcome = 1.0 if self.winner_status else 0.0
                return_bps = (outcome - fill_price) * 10000
                
                for bucket in [prob_bucket, 'all']:
                    returns_writer.add_return(
                        interval_label=interval_label,
                        threshold=threshold,
                        tercile=tercile,
                        prob_bucket=bucket,
                        entry_price=crossing_price,
                        fill_price=fill_price,
                        is_winner=self.winner_status,
                        return_bps=return_bps,
                        volume_at_dip=volume_at_dip,
                    )
                    n_returns += 1
        
        return n_returns


def rebuild_volume_data(sample_files=None, diagnostic=False, reaction_delay=0):
    """
    Rebuild volume returns data from raw batch files.
    
    Args:
        sample_files: If set, only process first N files
        diagnostic: If True, use diagnostic output directory
        reaction_delay: Seconds to delay order placement after crossing
    """
    start_time = datetime.now()
    
    log("="*70)
    log("REBUILDING VOLUME RETURNS DATA FROM RAW FILES (v2.0)")
    log("="*70)
    
    if diagnostic:
        log("\n*** DIAGNOSTIC MODE ***")
    
    log(f"\nReaction delay: {reaction_delay}s")
    
    # Load resources
    log("\nLoading resources...")
    
    winner_lookup = load_winner_sidecar(SIDECAR_FILE)
    if winner_lookup is None:
        log("ERROR: Cannot proceed without winner sidecar.")
        return None
    
    batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, '*.parquet')))
    log(f"  Found {len(batch_files):,} batch files")
    
    if len(batch_files) == 0:
        log("ERROR: No batch files found.")
        return None
    
    market_index = load_market_index(INDEX_CACHE_FILE, batch_files)
    
    if market_index is None:
        log("ERROR: Market index cache not found or invalid.")
        return None
    
    log(f"  Loaded market index: {len(market_index):,} conditions")
    
    # Setup output directory
    delay_suffix = f"_delay{reaction_delay}s" if reaction_delay > 0 else ""
    if diagnostic:
        returns_dir = os.path.join(OUTPUT_DIR, f'strategyA_rebuild_v2_DIAGNOSTIC{delay_suffix}_{TIMESTAMP}')
    else:
        returns_dir = os.path.join(OUTPUT_DIR, f'strategyA_rebuild_v2{delay_suffix}_{TIMESTAMP}')
    
    os.makedirs(returns_dir, exist_ok=True)
    log(f"\nReturns data will be written to: {returns_dir}")
    
    returns_writer = VolumeReturnsWriter(returns_dir, chunk_size=10000)
    
    # Setup tracking
    if sample_files:
        files_to_process_indices = list(range(min(sample_files, len(batch_files))))
        log(f"  SAMPLE MODE: Processing {len(files_to_process_indices)} files")
    else:
        files_to_process_indices = list(range(len(batch_files)))
    
    unique_file_set = set(files_to_process_indices)
    
    condition_remaining_files = {}
    for condition_id, file_list in market_index.items():
        relevant_files = sum(1 for file_idx, _ in file_list if file_idx in unique_file_set)
        if relevant_files > 0:
            condition_remaining_files[condition_id] = relevant_files
    
    file_to_conditions = defaultdict(set)
    for condition_id, file_list in market_index.items():
        for file_idx, _ in file_list:
            if file_idx in unique_file_set:
                file_to_conditions[file_idx].add(condition_id)
    
    log(f"  Conditions to track: {len(condition_remaining_files):,}")
    
    # Processing
    token_accumulators = {}
    condition_tokens = defaultdict(set)
    
    stats = {
        'files_processed': 0,
        'total_rows': 0,
        'tokens_no_winner': 0,
        'tokens_filtered': 0,
        'conditions_flushed': 0,
        'returns_written': 0,
    }
    
    PROGRESS_INTERVAL = 500
    
    for file_idx in files_to_process_indices:
        batch_file = batch_files[file_idx]
        
        try:
            df = pq.read_table(batch_file).to_pandas()
            stats['total_rows'] += len(df)
            
            # Get required columns
            if 'timestamp' not in df.columns:
                continue
            
            # Find volume column
            vol_col = None
            for col in VOLUME_COLUMNS:
                if col in df.columns:
                    vol_col = col
                    break
            
            if vol_col is None:
                vol_col = 'size'
                df[vol_col] = 1.0
            
            # Process each condition in this file
            conditions_in_file = file_to_conditions.get(file_idx, set())
            
            for condition_id in conditions_in_file:
                condition_df = df[df['condition_id'] == condition_id]
                
                if len(condition_df) == 0:
                    continue
                
                # Get resolution time
                resolution_time = condition_df['resolution_time'].iloc[0]
                if pd.isna(resolution_time):
                    continue
                
                resolution_time = float(resolution_time)
                
                # Process tokens in this condition
                for token_id in condition_df['token_id'].unique():
                    token_df = condition_df[condition_df['token_id'] == token_id]
                    
                    # Get winner status
                    winner_status = winner_lookup.get(token_id)
                    if winner_status is None:
                        stats['tokens_no_winner'] += 1
                        continue
                    
                    # Create or update accumulator
                    if token_id not in token_accumulators:
                        token_accumulators[token_id] = VolumeTimingAccumulator(
                            token_id, condition_id, resolution_time, winner_status
                        )
                        condition_tokens[condition_id].add(token_id)
                    
                    # Add trades
                    trades = list(zip(
                        token_df['timestamp'].values,
                        token_df['price'].values,
                        token_df[vol_col].values
                    ))
                    token_accumulators[token_id].add_trades(trades)
                
                # Update remaining files count
                if condition_id in condition_remaining_files:
                    condition_remaining_files[condition_id] -= 1
                    
                    # Flush if all files processed for this condition
                    if condition_remaining_files[condition_id] <= 0:
                        for token_id in condition_tokens[condition_id]:
                            if token_id in token_accumulators:
                                n_ret = token_accumulators[token_id].compute_returns_with_volume(
                                    returns_writer, reaction_delay=reaction_delay
                                )
                                stats['returns_written'] += n_ret
                                del token_accumulators[token_id]
                        
                        del condition_tokens[condition_id]
                        del condition_remaining_files[condition_id]
                        stats['conditions_flushed'] += 1
            
            stats['files_processed'] += 1
            
            if stats['files_processed'] % PROGRESS_INTERVAL == 0:
                log(f"  Processed {stats['files_processed']:,} files, "
                    f"{stats['conditions_flushed']:,} conditions flushed, "
                    f"{stats['returns_written']:,} returns written")
            
            del df
            gc.collect()
            
        except Exception as e:
            log(f"  ERROR processing {batch_file}: {e}")
            continue
    
    # Final flush
    returns_writer.finalize()
    writer_summary = returns_writer.finalize()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    log(f"\nRebuild complete!")
    log(f"Files processed: {stats['files_processed']:,}")
    log(f"Total rows: {stats['total_rows']:,}")
    log(f"Conditions flushed: {stats['conditions_flushed']:,}")
    log(f"Returns written: {writer_summary['total_written']:,}")
    log(f"Elapsed: {format_duration(elapsed)}")
    
    # Save metadata
    metadata = {
        'timestamp': TIMESTAMP,
        'version': '2.0',
        'diagnostic': diagnostic,
        'reaction_delay_seconds': reaction_delay,
        'stats': stats,
        'writer_summary': writer_summary,
        'interval_pairs': [(s, e, l) for s, e, l in INTERVAL_PAIRS],
        'move_thresholds': MOVE_THRESHOLDS,
        'tercile_bounds': list(zip(TERCILE_LABELS, TERCILE_BOUNDS)),
        'prob_buckets': PROB_BUCKETS,
        'volume_window_before': VOLUME_WINDOW_BEFORE,
        'volume_window_after': VOLUME_WINDOW_AFTER,
        'fill_price_method': 'limit_price (v2.0 fix)',
    }
    
    metadata_path = os.path.join(returns_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    log(f"\nMetadata saved: {metadata_path}")
    
    return returns_dir


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Strategy A Confirmatory Analysis v2.0: Favorites Fading (Batch-Viable)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run with defaults (uses existing baseline data)
  python phase5_strategy_a_confirmatory_v2.py

  # Diagnostic mode (sample validation)
  python phase5_strategy_a_confirmatory_v2.py --diagnostic

  # Include additional probability buckets
  python phase5_strategy_a_confirmatory_v2.py --prob-buckets 75_90 90_99 99_plus

  # Use specific baseline directory
  python phase5_strategy_a_confirmatory_v2.py --baseline-dir /path/to/volume_data

  # Force rebuild with reaction delay
  python phase5_strategy_a_confirmatory_v2.py --rebuild --reaction-delay 300 --sample 1000

  # Full rebuild with 5-minute delay
  python phase5_strategy_a_confirmatory_v2.py --rebuild --reaction-delay 300

VERSION 2.0 CHANGES:
  - Fixed fill price to use limit price (not market trade price)
  - Updated defaults: 75_90/90_99, 48h/24h windows, 15%+ threshold
  - Added --reaction-delay parameter for rebuild mode
  - Removed volume quintile restriction (was Q1-Q2, now all)
        """
    )
    
    # Mode arguments
    parser.add_argument('--diagnostic', '-d', action='store_true',
                        help='Run in diagnostic mode with partial data')
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild volume data from raw batch files')
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files (for rebuild)')
    parser.add_argument('--reaction-delay', type=int, default=0,
                        help='Reaction delay in seconds for rebuild (default: 0)')
    
    # Data source
    parser.add_argument('--baseline-dir', type=str, default=VOLUME_BASELINE_DIR,
                        help=f'Path to baseline volume returns data (default: {VOLUME_BASELINE_DIR})')
    
    # Strategy A parameters
    parser.add_argument('--prob-buckets', nargs='+', 
                        default=STRATEGY_A_DEFAULTS['prob_buckets'],
                        help=f"Probability buckets (default: {STRATEGY_A_DEFAULTS['prob_buckets']})")
    parser.add_argument('--intervals', nargs='+',
                        default=STRATEGY_A_DEFAULTS['intervals'],
                        help=f"Interval labels (default: {STRATEGY_A_DEFAULTS['intervals']})")
    parser.add_argument('--min-threshold', type=float,
                        default=STRATEGY_A_DEFAULTS['min_threshold'],
                        help=f"Minimum threshold (default: {STRATEGY_A_DEFAULTS['min_threshold']})")
    parser.add_argument('--terciles', nargs='+',
                        default=STRATEGY_A_DEFAULTS['terciles'],
                        help=f"Tercile labels (default: {STRATEGY_A_DEFAULTS['terciles']})")
    parser.add_argument('--volume-quintiles', nargs='+',
                        default=STRATEGY_A_DEFAULTS['volume_quintiles'],
                        help=f"Volume quintiles (default: {STRATEGY_A_DEFAULTS['volume_quintiles']})")
    
    args = parser.parse_args()
    
    ensure_output_dir()
    
    log("="*70)
    log("STRATEGY A CONFIRMATORY ANALYSIS v2.0")
    log("="*70)
    
    if args.diagnostic:
        log("\n*** DIAGNOSTIC MODE ***\n")
    
    # -------------------------------------------------------------------------
    # STEP 1: Load or rebuild volume returns data
    # -------------------------------------------------------------------------
    
    if args.rebuild:
        log("\nRebuilding volume returns data from raw files...")
        baseline_dir = rebuild_volume_data(
            sample_files=args.sample,
            diagnostic=args.diagnostic,
            reaction_delay=args.reaction_delay
        )
        if baseline_dir is None:
            log("ERROR: Rebuild failed.")
            sys.exit(1)
    else:
        baseline_dir = args.baseline_dir
        
        if not os.path.exists(baseline_dir):
            log(f"\nERROR: Baseline directory not found: {baseline_dir}")
            log("       Use --rebuild to generate from raw batch files")
            log("       Or specify --baseline-dir with a valid path")
            sys.exit(1)
    
    # -------------------------------------------------------------------------
    # STEP 2: Load volume returns data (with corrected returns calculation)
    # -------------------------------------------------------------------------
    
    returns_df = load_volume_returns_baseline(baseline_dir, diagnostic=args.diagnostic)
    
    if returns_df is None:
        log("ERROR: Failed to load volume returns data.")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # STEP 3: Compute volume quintiles
    # -------------------------------------------------------------------------
    
    returns_df, quintile_thresholds = compute_volume_quintiles(returns_df)
    
    # -------------------------------------------------------------------------
    # STEP 4: Compute unfiltered baseline stats
    # -------------------------------------------------------------------------
    
    log("\nComputing unfiltered baseline statistics...")
    
    # Exclude 'all' bucket to avoid double-counting
    unfiltered_df = returns_df[returns_df['prob_bucket'] != 'all']
    unfiltered_stats = compute_aggregate_stats(unfiltered_df, label="Unfiltered")
    
    log(f"  Unfiltered: n={unfiltered_stats['n']:,}, mean={unfiltered_stats['mean']:+.1f} bps")
    
    # -------------------------------------------------------------------------
    # STEP 5: Apply Strategy A filter
    # -------------------------------------------------------------------------
    
    params = {
        'prob_buckets': args.prob_buckets,
        'intervals': args.intervals,
        'min_threshold': args.min_threshold,
        'terciles': args.terciles,
        'volume_quintiles': args.volume_quintiles,
    }
    
    filtered_df = apply_strategy_a_filter(returns_df, params)
    
    # -------------------------------------------------------------------------
    # STEP 6: Compute filtered stats
    # -------------------------------------------------------------------------
    
    log("\nComputing Strategy A filtered statistics...")
    
    filtered_stats = compute_aggregate_stats(filtered_df, label="Strategy A")
    
    if filtered_stats:
        log(f"  Filtered: n={filtered_stats['n']:,}, mean={filtered_stats['mean']:+.1f} bps, "
            f"win_rate={filtered_stats['win_rate']*100:.1f}%")
    else:
        log("  WARNING: No trades passed the filter!")
    
    # -------------------------------------------------------------------------
    # STEP 7: Generate report
    # -------------------------------------------------------------------------
    
    report_path = generate_strategy_a_report(
        filtered_stats=filtered_stats,
        unfiltered_stats=unfiltered_stats,
        params=params,
        output_dir=OUTPUT_DIR,
        quintile_thresholds=quintile_thresholds,
        diagnostic=args.diagnostic,
        reaction_delay=args.reaction_delay
    )
    
    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    
    log("\n" + "="*70)
    log("ANALYSIS COMPLETE")
    log("="*70)
    
    log(f"\nReport saved: {report_path}")
    
    if filtered_stats:
        edge_improvement = filtered_stats['mean'] - unfiltered_stats['mean']
        
        # Print the dedicated output block to console as well
        log("")
        log("```")
        log(f"Trades passing filter:     n = {filtered_stats['n']:,}")
        log(f"Win rate:                  {filtered_stats['win_rate']*100:.1f}%")
        log(f"Mean return:               {filtered_stats['mean']:+.1f} bps")
        log(f"Median return:             {filtered_stats['median']:+.1f} bps")
        log(f"Std dev:                   {filtered_stats['std']:.1f} bps")
        log(f"P10 (worst decile):        {filtered_stats['p10']:+.1f} bps")
        log(f"P90 (best decile):         {filtered_stats['p90']:+.1f} bps")
        log("```")
        log("")
        log(f"Edge vs unfiltered:        {edge_improvement:+.1f} bps")
    
    log("\n" + "="*70)


if __name__ == "__main__":
    main()