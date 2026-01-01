#!/usr/bin/env python3
"""
Strategy B Confirmatory Script - Longshot Fading (Batch-Viable)
Version: 3.0

CHANGELOG v3.0:
  - Added market volume diagnostic mode (--market-volume)
  - VolumeReturnsWriter now captures condition_id for market-level joins
  - Computes total market volume from raw parquets
  - Stratifies edge by market volume quintiles
  - Reports distribution of Strategy B targets by market volume
  
  IMPORTANT DISTINCTION (from v2.0 discussion):
  - volume_at_dip: Trading activity in ±30min window around the dip crossing
    (proxy for "can I get filled at this moment")
  - market_volume: Total lifetime volume for the market
    (proxy for position sizing, NOT a disqualifier)
  
  Low market volume != untradeable. Liquidity (order book depth) determines
  tradeability, not historical volume. Volume is a position sizing input.

CHANGELOG v2.0:
  - CRITICAL FIX: Returns now calculated from entry_price (limit price), not 
    the incorrectly simulated fill_price.
  - Updated defaults based on Phase 4E delay analysis findings
  - Removed 51-75% "dead zone" buckets (negative edge at all delays)
  - Changed default windows to batch-viable (48h→24h, 24h→12h)
  - Increased minimum threshold to 10% (5% was noise)

USAGE:
  # Quick run with defaults (uses existing baseline data)
  python phase5_strategy_b_confirmatory_v3.py

  # Market volume diagnostic (computes market volumes, stratifies edge)
  python phase5_strategy_b_confirmatory_v3.py --market-volume

  # Rebuild with condition_id captured (required for market volume analysis)
  python phase5_strategy_b_confirmatory_v3.py --rebuild --sample 500

  # Full pipeline: rebuild then analyze market volume
  python phase5_strategy_b_confirmatory_v3.py --rebuild --market-volume
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
# STRATEGY B DEFAULT PARAMETERS (UPDATED from Phase 4E Delay Analysis)
# ------------------------------------------------------------------------------

STRATEGY_B_DEFAULTS = {
    'prob_buckets': ['sub_51'],
    'intervals': ['48h_to_24h', '24h_to_12h'],
    'min_threshold': 0.10,
    'terciles': ['early', 'mid', 'late'],
    'volume_quintiles': ['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high'],
}

# ------------------------------------------------------------------------------
# ANALYSIS PARAMETERS
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
MARKET_VOLUME_QUINTILE_LABELS = ['MV_Q1_low', 'MV_Q2', 'MV_Q3', 'MV_Q4', 'MV_Q5_high']

VOLUME_WINDOW_BEFORE = 1800
VOLUME_WINDOW_AFTER = 1800

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

def format_volume(vol):
    """Format volume with K/M suffix."""
    if vol >= 1_000_000:
        return f"${vol/1_000_000:.1f}M"
    elif vol >= 1_000:
        return f"${vol/1_000:.1f}K"
    else:
        return f"${vol:.0f}"


def get_prob_bucket(start_price):
    for label, lo, hi in PROB_BUCKETS:
        if lo <= start_price < hi:
            return label
    return None


def get_tercile(fraction):
    for label, (lo, hi) in zip(TERCILE_LABELS, TERCILE_BOUNDS):
        if lo <= fraction < hi or (label == 'late' and fraction == 1.0):
            return label
    return None


def get_volume_quintile(volume, quintile_thresholds):
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


def get_market_volume_quintile(volume, quintile_thresholds):
    """Assign market volume quintile."""
    if quintile_thresholds is None or volume <= 0:
        return None
    
    q20, q40, q60, q80 = quintile_thresholds
    
    if volume <= q20:
        return 'MV_Q1_low'
    elif volume <= q40:
        return 'MV_Q2'
    elif volume <= q60:
        return 'MV_Q3'
    elif volume <= q80:
        return 'MV_Q4'
    else:
        return 'MV_Q5_high'


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_volume_returns_baseline(baseline_dir, diagnostic=False):
    """
    Load pre-computed volume returns data from baseline directory.
    
    v3.0: Also looks for condition_id column for market volume joins.
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
        chunk_files = chunk_files[:3]
        log(f"  DIAGNOSTIC: Loading only {len(chunk_files)} chunks")
    
    dfs = []
    for chunk_file in chunk_files:
        df = pd.read_parquet(chunk_file)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    log(f"  Loaded {len(combined):,} returns observations")
    
    # Check for condition_id (needed for market volume analysis)
    has_condition_id = 'condition_id' in combined.columns
    log(f"  Has condition_id column: {has_condition_id}")
    
    # Recalculate returns from entry_price
    log("\n  *** APPLYING CRITICAL FIX: Recalculating returns from limit price ***")
    log(f"  Old return_bps stats: mean={combined['return_bps'].mean():+.1f}, median={combined['return_bps'].median():+.1f}")
    
    combined['return_bps_old'] = combined['return_bps']
    combined['return_bps'] = (combined['is_winner'].astype(float) - combined['entry_price']) * 10000
    
    log(f"  New return_bps stats: mean={combined['return_bps'].mean():+.1f}, median={combined['return_bps'].median():+.1f}")
    
    avg_correction = (combined['return_bps'] - combined['return_bps_old']).mean()
    log(f"  Average correction: {avg_correction:+.1f} bps (positive = old was inflated)")
    
    log(f"  {log_memory()}")
    
    return combined


def compute_volume_quintiles(returns_df):
    """Compute volume-at-dip quintiles globally."""
    log("\nComputing volume-at-dip quintiles...")
    
    valid_volume = returns_df[returns_df['volume_at_dip'] > 0]['volume_at_dip']
    
    if len(valid_volume) == 0:
        log("  WARNING: No valid volume data found")
        returns_df['volume_quintile'] = None
        return returns_df, None
    
    q20 = np.percentile(valid_volume, 20)
    q40 = np.percentile(valid_volume, 40)
    q60 = np.percentile(valid_volume, 60)
    q80 = np.percentile(valid_volume, 80)
    
    quintile_thresholds = (q20, q40, q60, q80)
    
    log(f"  Volume-at-dip quintile thresholds:")
    log(f"    Q1 (low):    <= {q20:.2f}")
    log(f"    Q2:          {q20:.2f} - {q40:.2f}")
    log(f"    Q3:          {q40:.2f} - {q60:.2f}")
    log(f"    Q4:          {q60:.2f} - {q80:.2f}")
    log(f"    Q5 (high):   > {q80:.2f}")
    
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
    
    return returns_df, quintile_thresholds


# ==============================================================================
# MARKET VOLUME COMPUTATION (NEW in v3.0)
# ==============================================================================

def compute_market_volumes_from_parquets(sample_files=None, diagnostic=False):
    """
    Compute total lifetime volume per condition_id from raw parquet files.
    
    Returns:
        Dict[condition_id -> total_volume]
    """
    log("\n" + "="*70)
    log("COMPUTING MARKET VOLUMES FROM RAW PARQUETS")
    log("="*70)
    
    batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, '*.parquet')))
    log(f"  Found {len(batch_files):,} batch files")
    
    if len(batch_files) == 0:
        log("  ERROR: No batch files found.")
        return None
    
    if sample_files:
        batch_files = batch_files[:sample_files]
        log(f"  SAMPLE MODE: Processing {len(batch_files)} files")
    
    if diagnostic:
        batch_files = batch_files[:50]
        log(f"  DIAGNOSTIC: Processing {len(batch_files)} files")
    
    # Accumulate volume by condition_id
    condition_volumes = defaultdict(float)
    files_processed = 0
    
    for i, batch_file in enumerate(batch_files):
        try:
            df = pd.read_parquet(batch_file)
            
            # Find volume column
            volume_col = None
            for col in VOLUME_COLUMNS:
                if col in df.columns:
                    volume_col = col
                    break
            
            if volume_col is None:
                continue
            
            if 'condition_id' not in df.columns:
                continue
            
            # Aggregate volume by condition_id
            for cond_id, vol in df.groupby('condition_id')[volume_col].sum().items():
                condition_volumes[cond_id] += vol
            
            files_processed += 1
            
            if (i + 1) % 500 == 0:
                log(f"  Processed {i+1}/{len(batch_files)} files, {len(condition_volumes):,} conditions so far")
                
        except Exception as e:
            continue
    
    log(f"\n  Complete: {files_processed} files processed")
    log(f"  Found volumes for {len(condition_volumes):,} conditions")
    
    if len(condition_volumes) > 0:
        volumes = list(condition_volumes.values())
        log(f"\n  Market volume distribution:")
        log(f"    Min:     {format_volume(min(volumes))}")
        log(f"    p10:     {format_volume(np.percentile(volumes, 10))}")
        log(f"    p25:     {format_volume(np.percentile(volumes, 25))}")
        log(f"    p50:     {format_volume(np.percentile(volumes, 50))}")
        log(f"    p75:     {format_volume(np.percentile(volumes, 75))}")
        log(f"    p90:     {format_volume(np.percentile(volumes, 90))}")
        log(f"    p95:     {format_volume(np.percentile(volumes, 95))}")
        log(f"    Max:     {format_volume(max(volumes))}")
    
    return dict(condition_volumes)


def enrich_with_market_volume(returns_df, market_volumes):
    """
    Join market volume to returns data and compute market volume quintiles.
    
    Requires condition_id column in returns_df.
    """
    log("\nEnriching returns data with market volume...")
    
    if 'condition_id' not in returns_df.columns:
        log("  ERROR: condition_id column not found in returns data.")
        log("         Re-run with --rebuild to capture condition_id.")
        return returns_df, None
    
    # Map market volume
    returns_df['market_volume'] = returns_df['condition_id'].map(market_volumes)
    
    matched = returns_df['market_volume'].notna().sum()
    total = len(returns_df)
    log(f"  Matched market volume for {matched:,}/{total:,} observations ({matched/total*100:.1f}%)")
    
    # Fill missing with 0 (will be excluded from quintile analysis)
    returns_df['market_volume'] = returns_df['market_volume'].fillna(0)
    
    # Compute market volume quintiles
    valid_mv = returns_df[returns_df['market_volume'] > 0]['market_volume']
    
    if len(valid_mv) == 0:
        log("  WARNING: No valid market volume data")
        returns_df['market_volume_quintile'] = None
        return returns_df, None
    
    q20 = np.percentile(valid_mv, 20)
    q40 = np.percentile(valid_mv, 40)
    q60 = np.percentile(valid_mv, 60)
    q80 = np.percentile(valid_mv, 80)
    
    mv_quintile_thresholds = (q20, q40, q60, q80)
    
    log(f"\n  Market volume quintile thresholds (of Strategy B targets):")
    log(f"    MV_Q1 (low):  <= {format_volume(q20)}")
    log(f"    MV_Q2:        {format_volume(q20)} - {format_volume(q40)}")
    log(f"    MV_Q3:        {format_volume(q40)} - {format_volume(q60)}")
    log(f"    MV_Q4:        {format_volume(q60)} - {format_volume(q80)}")
    log(f"    MV_Q5 (high): > {format_volume(q80)}")
    
    def assign_mv_quintile(vol):
        if vol <= 0:
            return None
        elif vol <= q20:
            return 'MV_Q1_low'
        elif vol <= q40:
            return 'MV_Q2'
        elif vol <= q60:
            return 'MV_Q3'
        elif vol <= q80:
            return 'MV_Q4'
        else:
            return 'MV_Q5_high'
    
    returns_df['market_volume_quintile'] = returns_df['market_volume'].apply(assign_mv_quintile)
    
    # Log distribution
    mv_counts = returns_df['market_volume_quintile'].value_counts()
    log(f"\n  Market volume quintile distribution:")
    for q in MARKET_VOLUME_QUINTILE_LABELS:
        count = mv_counts.get(q, 0)
        log(f"    {q}: {count:,}")
    
    return returns_df, mv_quintile_thresholds


def analyze_edge_by_market_volume(returns_df, params, mv_quintile_thresholds):
    """
    Analyze edge stratified by market volume quintile.
    
    This answers: "Is the edge real in high-volume (tradeable) markets,
    or is it concentrated in untradeable low-volume markets?"
    """
    log("\n" + "="*70)
    log("EDGE ANALYSIS BY MARKET VOLUME")
    log("="*70)
    
    if 'market_volume_quintile' not in returns_df.columns:
        log("  ERROR: market_volume_quintile not found. Run --market-volume first.")
        return None
    
    # Apply Strategy B filter (except we'll iterate over market volume quintiles)
    df = returns_df[returns_df['prob_bucket'] != 'all'].copy()
    
    base_mask = (
        (df['prob_bucket'].isin(params['prob_buckets'])) &
        (df['interval'].isin(params['intervals'])) &
        (df['threshold'] >= params['min_threshold']) &
        (df['tercile'].isin(params['terciles'])) &
        (df['volume_quintile'].isin(params['volume_quintiles']))
    )
    
    filtered_df = df[base_mask]
    
    log(f"\nStrategy B filtered sample: {len(filtered_df):,} observations")
    
    # Analyze by market volume quintile
    results = []
    
    log("\n  Edge by Market Volume Quintile:")
    log("  " + "-"*80)
    log(f"  {'Quintile':<15} {'n':>8} {'WinRate':>10} {'Mean':>10} {'Median':>10} {'StdDev':>10} {'Volume Range':<20}")
    log("  " + "-"*80)
    
    for mvq in MARKET_VOLUME_QUINTILE_LABELS + [None]:
        if mvq is None:
            subset = filtered_df[filtered_df['market_volume_quintile'].isna()]
            label = "Unknown"
        else:
            subset = filtered_df[filtered_df['market_volume_quintile'] == mvq]
            label = mvq
        
        if len(subset) < MIN_SAMPLES_PER_CELL:
            continue
        
        returns = subset['return_bps'].values
        win_rate = subset['is_winner'].mean()
        mean_ret = np.mean(returns)
        median_ret = np.median(returns)
        std_ret = np.std(returns)
        
        # Get volume range for this quintile
        if mvq and mv_quintile_thresholds:
            q20, q40, q60, q80 = mv_quintile_thresholds
            if mvq == 'MV_Q1_low':
                vol_range = f"<= {format_volume(q20)}"
            elif mvq == 'MV_Q2':
                vol_range = f"{format_volume(q20)}-{format_volume(q40)}"
            elif mvq == 'MV_Q3':
                vol_range = f"{format_volume(q40)}-{format_volume(q60)}"
            elif mvq == 'MV_Q4':
                vol_range = f"{format_volume(q60)}-{format_volume(q80)}"
            else:
                vol_range = f"> {format_volume(q80)}"
        else:
            vol_range = "N/A"
        
        log(f"  {label:<15} {len(subset):>8,} {win_rate:>10.1%} {mean_ret:>+10.1f} {median_ret:>+10.1f} {std_ret:>10.1f} {vol_range:<20}")
        
        results.append({
            'quintile': label,
            'n': len(subset),
            'win_rate': win_rate,
            'mean': mean_ret,
            'median': median_ret,
            'std': std_ret,
            'vol_range': vol_range,
        })
    
    log("  " + "-"*80)
    
    # Overall stats
    if len(filtered_df) > 0:
        overall_mean = filtered_df['return_bps'].mean()
        overall_wr = filtered_df['is_winner'].mean()
        log(f"\n  Overall Strategy B: n={len(filtered_df):,}, win_rate={overall_wr:.1%}, mean={overall_mean:+.1f} bps")
    
    # Key diagnostic: Is edge concentrated in low-volume markets?
    if len(results) >= 2:
        low_vol = [r for r in results if r['quintile'] in ['MV_Q1_low', 'MV_Q2']]
        high_vol = [r for r in results if r['quintile'] in ['MV_Q4', 'MV_Q5_high']]
        
        if low_vol and high_vol:
            low_mean = np.average([r['mean'] for r in low_vol], weights=[r['n'] for r in low_vol])
            high_mean = np.average([r['mean'] for r in high_vol], weights=[r['n'] for r in high_vol])
            
            log(f"\n  KEY DIAGNOSTIC:")
            log(f"    Low-volume markets (MV_Q1+Q2):  {low_mean:+.1f} bps mean edge")
            log(f"    High-volume markets (MV_Q4+Q5): {high_mean:+.1f} bps mean edge")
            log(f"    Difference: {low_mean - high_mean:+.1f} bps")
            
            if high_mean > 50:
                log(f"\n  ✓ POSITIVE: Edge persists in high-volume markets.")
                log(f"    Strategy is likely tradeable with appropriate position sizing.")
            elif high_mean > 0:
                log(f"\n  ⚠ MARGINAL: Edge is reduced but still positive in high-volume markets.")
                log(f"    Consider focusing on higher-volume subset.")
            else:
                log(f"\n  ✗ CONCERNING: Edge disappears in high-volume markets.")
                log(f"    Strategy may be untradeable or require liquidity filter.")
    
    return results


# ==============================================================================
# STRATEGY B FILTERING
# ==============================================================================

def apply_strategy_b_filter(returns_df, params):
    """Apply Strategy B filter rules to returns dataframe."""
    log("\nApplying Strategy B filter...")
    log(f"  Parameters:")
    log(f"    Prob buckets:     {params['prob_buckets']}")
    log(f"    Intervals:        {params['intervals']}")
    log(f"    Min threshold:    {params['min_threshold']:.0%}")
    log(f"    Terciles:         {params['terciles']}")
    log(f"    Volume quintiles: {params['volume_quintiles']}")
    
    df = returns_df[returns_df['prob_bucket'] != 'all'].copy()
    
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
    """Compute aggregate statistics for a returns distribution."""
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

def generate_market_volume_report(mv_results, filtered_stats, params, output_dir, 
                                   mv_quintile_thresholds, diagnostic=False):
    """Generate market volume analysis report."""
    suffix = '_DIAGNOSTIC' if diagnostic else ''
    report_path = os.path.join(output_dir, f'strategyB_market_volume_report{suffix}_{TIMESTAMP}.txt')
    
    log(f"\nGenerating market volume report: {report_path}")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STRATEGY B MARKET VOLUME ANALYSIS v3.0\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if diagnostic:
            f.write("*** DIAGNOSTIC MODE - PARTIAL DATA ***\n")
        f.write("\n")
        
        # Key distinction
        f.write("-"*80 + "\n")
        f.write("IMPORTANT DISTINCTION\n")
        f.write("-"*80 + "\n\n")
        f.write("This analysis distinguishes two volume metrics:\n\n")
        f.write("1. volume_at_dip: Trading activity in ±30min window around dip crossing\n")
        f.write("   - Proxy for 'can I get filled at this moment'\n")
        f.write("   - Already used in Strategy B filtering (all quintiles included)\n\n")
        f.write("2. market_volume: Total lifetime volume for the market\n")
        f.write("   - Proxy for position sizing, NOT a disqualifier\n")
        f.write("   - Low market volume ≠ untradeable\n")
        f.write("   - Liquidity (order book depth) determines tradeability\n\n")
        
        # Filter parameters
        f.write("-"*80 + "\n")
        f.write("STRATEGY B FILTER PARAMETERS\n")
        f.write("-"*80 + "\n\n")
        f.write(f"  Prob buckets:      {params['prob_buckets']}\n")
        f.write(f"  Intervals:         {params['intervals']}\n")
        f.write(f"  Min threshold:     {params['min_threshold']:.0%}\n")
        f.write(f"  Terciles:          {params['terciles']}\n")
        f.write(f"  Volume quintiles:  {params['volume_quintiles']}\n")
        f.write("\n")
        
        # Market volume quintile thresholds
        if mv_quintile_thresholds:
            q20, q40, q60, q80 = mv_quintile_thresholds
            f.write("-"*80 + "\n")
            f.write("MARKET VOLUME QUINTILE THRESHOLDS (of Strategy B targets)\n")
            f.write("-"*80 + "\n\n")
            f.write(f"  MV_Q1 (low):  <= {format_volume(q20)}\n")
            f.write(f"  MV_Q2:        {format_volume(q20)} - {format_volume(q40)}\n")
            f.write(f"  MV_Q3:        {format_volume(q40)} - {format_volume(q60)}\n")
            f.write(f"  MV_Q4:        {format_volume(q60)} - {format_volume(q80)}\n")
            f.write(f"  MV_Q5 (high): > {format_volume(q80)}\n")
            f.write("\n")
        
        # Results table
        f.write("="*80 + "\n")
        f.write("EDGE BY MARKET VOLUME QUINTILE\n")
        f.write("="*80 + "\n\n")
        
        if mv_results:
            f.write(f"{'Quintile':<15} {'n':>8} {'WinRate':>10} {'Mean':>10} {'Median':>10} {'StdDev':>10} {'Volume Range':<20}\n")
            f.write("-"*90 + "\n")
            
            for r in mv_results:
                f.write(f"{r['quintile']:<15} {r['n']:>8,} {r['win_rate']:>10.1%} {r['mean']:>+10.1f} {r['median']:>+10.1f} {r['std']:>10.1f} {r['vol_range']:<20}\n")
            
            f.write("-"*90 + "\n")
        
        # Key diagnostic
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("KEY DIAGNOSTIC: IS EDGE TRADEABLE?\n")
        f.write("="*80 + "\n\n")
        
        if mv_results and len(mv_results) >= 2:
            low_vol = [r for r in mv_results if r['quintile'] in ['MV_Q1_low', 'MV_Q2']]
            high_vol = [r for r in mv_results if r['quintile'] in ['MV_Q4', 'MV_Q5_high']]
            
            if low_vol and high_vol:
                low_n = sum(r['n'] for r in low_vol)
                high_n = sum(r['n'] for r in high_vol)
                low_mean = np.average([r['mean'] for r in low_vol], weights=[r['n'] for r in low_vol])
                high_mean = np.average([r['mean'] for r in high_vol], weights=[r['n'] for r in high_vol])
                
                f.write(f"Low-volume markets (MV_Q1+Q2):\n")
                f.write(f"  n = {low_n:,}\n")
                f.write(f"  Mean edge = {low_mean:+.1f} bps\n\n")
                
                f.write(f"High-volume markets (MV_Q4+Q5):\n")
                f.write(f"  n = {high_n:,}\n")
                f.write(f"  Mean edge = {high_mean:+.1f} bps\n\n")
                
                f.write(f"Difference: {low_mean - high_mean:+.1f} bps\n\n")
                
                if high_mean > 50:
                    f.write("✓ VERDICT: Edge PERSISTS in high-volume markets.\n")
                    f.write("  Strategy is likely tradeable. Use market volume for position sizing.\n")
                elif high_mean > 0:
                    f.write("⚠ VERDICT: Edge is MARGINAL in high-volume markets.\n")
                    f.write("  Consider focusing on higher-volume subset or reducing position sizes.\n")
                else:
                    f.write("✗ VERDICT: Edge DISAPPEARS in high-volume markets.\n")
                    f.write("  Strategy may be untradeable. Consider liquidity filter.\n")
        
        # Position sizing implications
        f.write("\n")
        f.write("-"*80 + "\n")
        f.write("POSITION SIZING IMPLICATIONS\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Market volume informs position sizing, not trade selection:\n\n")
        f.write("  MV_Q1 (low):  Position size = min($10, 1% of market volume)\n")
        f.write("  MV_Q2:        Position size = min($25, 2% of market volume)\n")
        f.write("  MV_Q3:        Position size = min($50, 3% of market volume)\n")
        f.write("  MV_Q4:        Position size = min($100, 5% of market volume)\n")
        f.write("  MV_Q5 (high): Position size = min($200, 5% of market volume)\n\n")
        
        f.write("These are illustrative. Actual sizing should consider:\n")
        f.write("  - Current order book depth (not observable from historical data)\n")
        f.write("  - Concurrent position limits\n")
        f.write("  - Kelly fraction based on edge estimate\n")
        
        f.write("\n" + "="*80 + "\n")
    
    return report_path


# ==============================================================================
# REBUILD FUNCTIONS (MODIFIED in v3.0 to capture condition_id)
# ==============================================================================

def load_winner_sidecar(sidecar_path):
    """Load winner status lookup from sidecar file."""
    log(f"Loading winner sidecar from: {sidecar_path}")
    
    if not os.path.exists(sidecar_path):
        log(f"  ERROR: Sidecar file not found: {sidecar_path}")
        return None
    
    try:
        df = pd.read_parquet(sidecar_path)
        
        winner_lookup = {}
        for _, row in df.iterrows():
            token_id = row.get('token_id')
            winner = row.get('winner', row.get('api_derived_winner', row.get('is_winner', None)))
            if token_id is not None and winner is not None:
                winner_lookup[str(token_id)] = bool(winner)
        
        log(f"  Loaded {len(winner_lookup):,} winner statuses")
        return winner_lookup
        
    except Exception as e:
        log(f"  ERROR loading sidecar: {e}")
        return None


def load_market_index(index_cache_path, batch_files):
    """Load market index cache."""
    if os.path.exists(index_cache_path):
        log(f"Loading market index from cache: {index_cache_path}")
        try:
            with open(index_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            if isinstance(cache_data, dict) and 'market_index' in cache_data:
                market_index = cache_data['market_index']
            else:
                market_index = cache_data
                
            log(f"  Loaded index with {len(market_index):,} conditions")
            return market_index
        except Exception as e:
            log(f"  ERROR loading cache: {e}")
    
    return None


def extract_price_at_horizon(trades, resolution_time, hours_before):
    """Extract price at a specific time horizon before resolution."""
    target_time = resolution_time - (hours_before * 3600)
    
    before_trades = [(ts, p, s) for ts, p, s in trades if ts <= target_time]
    
    if not before_trades:
        return None, None
    
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
            move = start_price - price
        else:
            move = price - start_price
        
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
    """Simulate whether a limit order would be filled."""
    future_trades = [(ts, p, s) for ts, p, s in trades if ts > placement_time]
    
    if not future_trades:
        return {'filled': False, 'fill_time': None, 'fill_price': None, 'time_to_fill': None}
    
    for ts, price, size in future_trades:
        if is_buy and price <= limit_price:
            return {'filled': True, 'fill_time': ts, 'fill_price': limit_price, 'time_to_fill': ts - placement_time}
        elif not is_buy and price >= limit_price:
            return {'filled': True, 'fill_time': ts, 'fill_price': limit_price, 'time_to_fill': ts - placement_time}
    
    return {'filled': False, 'fill_time': None, 'fill_price': None, 'time_to_fill': None}


class VolumeReturnsWriterV3:
    """
    Memory-safe streaming writer for returns data with volume.
    v3.0: Also captures condition_id for market volume joins.
    """
    
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
                   entry_price, fill_price, is_winner, return_bps, volume_at_dip,
                   condition_id=None):
        """v3.0: Now includes condition_id."""
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
            'condition_id': condition_id,  # NEW in v3.0
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


class VolumeTimingAccumulatorV3:
    """Accumulates trade data for timing and volume analysis. v3.0: passes condition_id."""
    
    def __init__(self, token_id, condition_id, resolution_time, winner_status):
        self.token_id = token_id
        self.condition_id = condition_id
        self.resolution_time = resolution_time
        self.winner_status = winner_status
        self.trades = []
    
    def add_trades(self, trades_batch):
        self.trades.extend(trades_batch)
    
    def compute_returns_with_volume(self, returns_writer, reaction_delay=0):
        """Compute returns for all cells WITH VOLUME AT DIP and CONDITION_ID."""
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
                
                placement_time = crossing_time + reaction_delay
                
                volume_at_dip = calculate_volume_at_crossing(
                    self.trades, 
                    crossing_time,
                    window_before=VOLUME_WINDOW_BEFORE,
                    window_after=VOLUME_WINDOW_AFTER
                )
                
                fill_result = simulate_limit_order_fill(
                    self.trades, placement_time, crossing_price, is_buy=True
                )
                
                if not fill_result.get('filled'):
                    continue
                
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
                        condition_id=self.condition_id,  # NEW in v3.0
                    )
                    n_returns += 1
        
        return n_returns


def rebuild_volume_data(sample_files=None, diagnostic=False, reaction_delay=0):
    """Rebuild volume returns data from raw batch files. v3.0: captures condition_id."""
    start_time = datetime.now()
    
    log("="*70)
    log("REBUILDING VOLUME RETURNS DATA FROM RAW FILES (v3.0)")
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
        returns_dir = os.path.join(OUTPUT_DIR, f'strategyB_rebuild_v3_DIAGNOSTIC{delay_suffix}_{TIMESTAMP}')
    else:
        returns_dir = os.path.join(OUTPUT_DIR, f'strategyB_rebuild_v3{delay_suffix}_{TIMESTAMP}')
    
    os.makedirs(returns_dir, exist_ok=True)
    log(f"\nReturns data will be written to: {returns_dir}")
    
    returns_writer = VolumeReturnsWriterV3(returns_dir, chunk_size=10000)
    
    # Setup tracking
    if sample_files:
        files_to_process_indices = list(range(min(sample_files, len(batch_files))))
        log(f"  SAMPLE MODE: Processing {len(files_to_process_indices)} files")
    else:
        files_to_process_indices = list(range(len(batch_files)))
    
    if diagnostic:
        files_to_process_indices = files_to_process_indices[:100]
        log(f"  DIAGNOSTIC: Limiting to {len(files_to_process_indices)} files")
    
    # Track conditions to process
    condition_file_map = defaultdict(set)
    for cond_id, file_indices in market_index.items():
        for idx in file_indices:
            if idx in files_to_process_indices:
                condition_file_map[cond_id].add(idx)
    
    log(f"  Conditions to process: {len(condition_file_map):,}")
    
    # Process files
    tokens_processed = 0
    returns_computed = 0
    
    accumulators = {}
    
    for i, file_idx in enumerate(files_to_process_indices):
        batch_file = batch_files[file_idx]
        
        try:
            df = pd.read_parquet(batch_file)
            
            # Find volume column
            volume_col = None
            for col in VOLUME_COLUMNS:
                if col in df.columns:
                    volume_col = col
                    break
            
            if volume_col is None:
                continue
            
            for token_id in df['token_id'].unique():
                token_str = str(token_id)
                
                if token_str not in winner_lookup:
                    continue
                
                token_df = df[df['token_id'] == token_id]
                
                if 'resolution_time' not in token_df.columns:
                    continue
                
                resolution_time = token_df['resolution_time'].iloc[0]
                if pd.isna(resolution_time):
                    continue
                
                condition_id = token_df['condition_id'].iloc[0] if 'condition_id' in token_df.columns else None
                
                if token_str not in accumulators:
                    accumulators[token_str] = VolumeTimingAccumulatorV3(
                        token_id=token_str,
                        condition_id=condition_id,
                        resolution_time=resolution_time,
                        winner_status=winner_lookup[token_str]
                    )
                
                trades = list(zip(
                    token_df['timestamp'].values,
                    token_df['price'].values,
                    token_df[volume_col].values
                ))
                accumulators[token_str].add_trades(trades)
            
        except Exception as e:
            continue
        
        if (i + 1) % 500 == 0:
            log(f"  Processed {i+1}/{len(files_to_process_indices)} files, {len(accumulators)} tokens accumulated")
    
    # Compute returns for all accumulators
    log(f"\nComputing returns for {len(accumulators):,} tokens...")
    
    for token_str, acc in accumulators.items():
        n_ret = acc.compute_returns_with_volume(returns_writer, reaction_delay=reaction_delay)
        returns_computed += n_ret
        tokens_processed += 1
        
        if tokens_processed % 5000 == 0:
            log(f"  Processed {tokens_processed:,} tokens, {returns_computed:,} returns computed")
    
    # Finalize
    summary = returns_writer.finalize()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    log(f"\n" + "="*70)
    log(f"REBUILD COMPLETE")
    log(f"="*70)
    log(f"  Tokens processed: {tokens_processed:,}")
    log(f"  Returns computed: {summary['total_written']:,}")
    log(f"  Output chunks: {summary['num_chunks']}")
    log(f"  Output directory: {returns_dir}")
    log(f"  Elapsed time: {format_duration(elapsed)}")
    
    # Save metadata
    metadata = {
        'version': '3.0',
        'timestamp': TIMESTAMP,
        'tokens_processed': tokens_processed,
        'returns_computed': summary['total_written'],
        'reaction_delay': reaction_delay,
        'diagnostic': diagnostic,
        'sample_files': sample_files,
        'has_condition_id': True,  # NEW in v3.0
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
        description='Strategy B Confirmatory Analysis v3.0: Market Volume Diagnostics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run with defaults (uses existing baseline data)
  python phase5_strategy_b_confirmatory_v3.py

  # Market volume diagnostic (requires --rebuild first if no condition_id)
  python phase5_strategy_b_confirmatory_v3.py --market-volume

  # Rebuild with condition_id captured
  python phase5_strategy_b_confirmatory_v3.py --rebuild --sample 500

  # Full pipeline: rebuild then analyze market volume
  python phase5_strategy_b_confirmatory_v3.py --rebuild --market-volume

VERSION 3.0 CHANGES:
  - Added market volume diagnostic mode
  - VolumeReturnsWriter now captures condition_id
  - Stratifies edge by market volume quintile
  - Reports distribution of Strategy B targets by market volume
        """
    )
    
    # Mode arguments
    parser.add_argument('--diagnostic', '-d', action='store_true',
                        help='Run in diagnostic mode with partial data')
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild volume data from raw batch files (captures condition_id)')
    parser.add_argument('--market-volume', action='store_true',
                        help='Compute and analyze market volumes')
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files (for rebuild)')
    parser.add_argument('--reaction-delay', type=int, default=0,
                        help='Reaction delay in seconds for rebuild (default: 0)')
    
    # Data source
    parser.add_argument('--baseline-dir', type=str, default=VOLUME_BASELINE_DIR,
                        help=f'Path to baseline volume returns data')
    
    # Strategy B parameters
    parser.add_argument('--prob-buckets', nargs='+', 
                        default=STRATEGY_B_DEFAULTS['prob_buckets'])
    parser.add_argument('--intervals', nargs='+',
                        default=STRATEGY_B_DEFAULTS['intervals'])
    parser.add_argument('--min-threshold', type=float,
                        default=STRATEGY_B_DEFAULTS['min_threshold'])
    parser.add_argument('--terciles', nargs='+',
                        default=STRATEGY_B_DEFAULTS['terciles'])
    parser.add_argument('--volume-quintiles', nargs='+',
                        default=STRATEGY_B_DEFAULTS['volume_quintiles'])
    
    args = parser.parse_args()
    
    ensure_output_dir()
    
    log("="*70)
    log("STRATEGY B CONFIRMATORY ANALYSIS v3.0")
    log("="*70)
    
    if args.diagnostic:
        log("\n*** DIAGNOSTIC MODE ***\n")
    
    # -------------------------------------------------------------------------
    # STEP 1: Load or rebuild volume returns data
    # -------------------------------------------------------------------------
    
    if args.rebuild:
        log("\nRebuilding volume returns data from raw files (with condition_id)...")
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
            sys.exit(1)
    
    # -------------------------------------------------------------------------
    # STEP 2: Load volume returns data
    # -------------------------------------------------------------------------
    
    returns_df = load_volume_returns_baseline(baseline_dir, diagnostic=args.diagnostic)
    
    if returns_df is None:
        log("ERROR: Failed to load volume returns data.")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # STEP 3: Compute volume-at-dip quintiles
    # -------------------------------------------------------------------------
    
    returns_df, quintile_thresholds = compute_volume_quintiles(returns_df)
    
    # -------------------------------------------------------------------------
    # STEP 4: Market volume analysis (if requested)
    # -------------------------------------------------------------------------
    
    mv_quintile_thresholds = None
    mv_results = None
    
    if args.market_volume:
        # Check if we have condition_id
        if 'condition_id' not in returns_df.columns:
            log("\nERROR: condition_id not found in returns data.")
            log("       Run with --rebuild first to capture condition_id.")
            sys.exit(1)
        
        # Compute market volumes from parquets
        market_volumes = compute_market_volumes_from_parquets(
            sample_files=args.sample,
            diagnostic=args.diagnostic
        )
        
        if market_volumes is None or len(market_volumes) == 0:
            log("ERROR: Failed to compute market volumes.")
            sys.exit(1)
        
        # Enrich returns data with market volume
        returns_df, mv_quintile_thresholds = enrich_with_market_volume(returns_df, market_volumes)
        
        # Analyze edge by market volume
        params = {
            'prob_buckets': args.prob_buckets,
            'intervals': args.intervals,
            'min_threshold': args.min_threshold,
            'terciles': args.terciles,
            'volume_quintiles': args.volume_quintiles,
        }
        
        mv_results = analyze_edge_by_market_volume(returns_df, params, mv_quintile_thresholds)
    
    # -------------------------------------------------------------------------
    # STEP 5: Standard Strategy B analysis
    # -------------------------------------------------------------------------
    
    log("\nComputing unfiltered baseline statistics...")
    
    unfiltered_df = returns_df[returns_df['prob_bucket'] != 'all']
    unfiltered_stats = compute_aggregate_stats(unfiltered_df, label="Unfiltered")
    
    log(f"  Unfiltered: n={unfiltered_stats['n']:,}, mean={unfiltered_stats['mean']:+.1f} bps")
    
    params = {
        'prob_buckets': args.prob_buckets,
        'intervals': args.intervals,
        'min_threshold': args.min_threshold,
        'terciles': args.terciles,
        'volume_quintiles': args.volume_quintiles,
    }
    
    filtered_df = apply_strategy_b_filter(returns_df, params)
    
    log("\nComputing Strategy B filtered statistics...")
    
    filtered_stats = compute_aggregate_stats(filtered_df, label="Strategy B")
    
    if filtered_stats:
        log(f"  Filtered: n={filtered_stats['n']:,}, mean={filtered_stats['mean']:+.1f} bps, "
            f"win_rate={filtered_stats['win_rate']*100:.1f}%")
    else:
        log("  WARNING: No trades passed the filter!")
    
    # -------------------------------------------------------------------------
    # STEP 6: Generate reports
    # -------------------------------------------------------------------------
    
    if args.market_volume and mv_results:
        report_path = generate_market_volume_report(
            mv_results=mv_results,
            filtered_stats=filtered_stats,
            params=params,
            output_dir=OUTPUT_DIR,
            mv_quintile_thresholds=mv_quintile_thresholds,
            diagnostic=args.diagnostic
        )
        log(f"\nMarket volume report saved: {report_path}")
    
    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    
    log("\n" + "="*70)
    log("ANALYSIS COMPLETE")
    log("="*70)
    
    if filtered_stats:
        log("")
        log("```")
        log(f"Trades passing filter:     n = {filtered_stats['n']:,}")
        log(f"Win rate:                  {filtered_stats['win_rate']*100:.1f}%")
        log(f"Mean return:               {filtered_stats['mean']:+.1f} bps")
        log(f"Median return:             {filtered_stats['median']:+.1f} bps")
        log(f"Std dev:                   {filtered_stats['std']:.1f} bps")
        log("```")
    
    log("\n" + "="*70)


if __name__ == "__main__":
    main()