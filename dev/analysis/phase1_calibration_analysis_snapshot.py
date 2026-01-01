#!/usr/bin/env python3
"""
Phase 1: Odds-Stratified Calibration Analysis - SNAPSHOT VERSION
Version: 14.0 - Point-in-Time Snapshot (No TWAP)

METHODOLOGY CHANGE (v14.0):
  - Snapshot at fixed time before resolution (e.g., T-24h)
  - No time-weighted averaging - single point-in-time price
  - Better approximates "tradeable belief" at realistic entry time
  - Avoids TWAP contamination from resolution-period price convergence

SNAPSHOT APPROACH:
  - Find trades within tolerance window around snapshot time
  - Use interpolated price or nearest-trade price
  - More noise than TWAP, but cleaner signal for entry-time belief

ARCHITECTURE (unchanged from v13.0):
  - File-First Accumulator: Each Parquet file read exactly ONCE
  - Sidecar Winner Lookup: O(1) dictionary access, no API calls
  - Streaming Flush: Memory-efficient condition-by-condition processing

DIAGNOSTIC MODE:
  - Use --diagnostic to process ~100 conditions with verbose output
  - Shows snapshot time, nearest trades, interpolated price

TARGET AUDIENCE: Quantitative PMs (Citadel, Jane Street, etc.)
"""

import pyarrow.parquet as pq
import glob
import json
import numpy as np
import pandas as pd 
import math
import pickle
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy import stats
import psutil
import gc
import os
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = "../../"

BATCH_DIR = os.path.join(BASE_DIR, 'order_history_batches')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis/outputs')
MARKETS_CSV = os.path.join(BASE_DIR, 'markets_past_year.csv')

# INDEX CACHE: Will be set dynamically based on mode (see get_cache_filename())
# NEVER share cache between snapshot/TWAP or between diagnostic/full runs
INDEX_CACHE_DIR = os.path.join(BASE_DIR, 'analysis')

# SIDECAR FILE: Pre-computed winner data from harvest_api_winners.py
SIDECAR_FILE = os.path.join(BASE_DIR, 'data/repair/api_derived_winners.parquet')

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Will be set by main() based on run mode
INDEX_CACHE_FILE = None  # Set dynamically

# ------------------------------------------------------------------------------
# ANALYSIS PARAMETERS - SNAPSHOT MODE
# ------------------------------------------------------------------------------

# Liquidity filters - Based on empirical market structure analysis
MIN_VOLUME_USD = 100.0      # Minimum USD volume in tolerance window
MIN_TRADES_PER_MARKET = 50  # Minimum total trades (ensures price discovery)
MIN_TRADES_PER_TOKEN = 2    # Reduced: snapshot needs fewer trades than TWAP

# Snapshot configuration
SNAPSHOT_OFFSET_HOURS = 24   # Time before resolution to take snapshot (e.g., T-24h)
SNAPSHOT_TOLERANCE_HOURS = 2 # Window around snapshot to find trades (+/- this value)

# Derived: tolerance window in seconds
SNAPSHOT_TOLERANCE_SECONDS = SNAPSHOT_TOLERANCE_HOURS * 3600

# Probability buckets - Aligned with charter specifications
ODDS_BUCKETS = [
    # Underdog territory (coarse buckets - higher variance, less liquid)
    (0.00, 0.10, '0-10%', 'longshot'),
    (0.10, 0.25, '10-25%', 'underdog'),
    (0.25, 0.40, '25-40%', 'toss-up-'),
    (0.40, 0.51, '40-51%', 'toss-up+'),
    
    # Favorite territory (refined buckets - higher precision needed)
    (0.51, 0.60, '51-60%', 'mild-fav'),
    (0.60, 0.75, '60-75%', 'moderate-fav'),
    (0.75, 0.90, '75-90%', 'strong-fav'),
    (0.90, 0.99, '90-99%', 'heavy-fav'),
    (0.99, 0.995, '99.0-99.5%', 'near-certain'),
    (0.995, 1.00, '99.5-100%', 'extreme'),
]

# Statistical significance threshold
ALPHA = 0.05

# Column configuration
REQUIRED_COLUMNS = ['timestamp', 'price', 'token_id', 'condition_id', 'resolution_time']
VOLUME_COLUMNS = ['size_tokens', 'maker_amount', 'size']
WINNER_COLUMNS = ['token_winner']

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def log(msg):
    """Timestamped logging to stdout"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def log_memory():
    """Returns current memory usage string"""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024**2
    return f"Memory: {mem_mb:.0f}MB"

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_cache_filename(sample_size=None, diagnostic_mode=False):
    """
    Generate unique cache filename based on run mode.
    
    CRITICAL: Never share caches between:
    - Snapshot vs TWAP runs (different price methodology)
    - Diagnostic vs full runs (different sample sizes)
    - Different sample sizes
    
    This prevents catastrophic cache overwrites.
    """
    base_name = "market_index_cache"
    
    # Snapshot mode gets its own cache (separate from TWAP)
    #base_name += "_snapshot"
    
    # Diagnostic/sample modes get separate caches
    if diagnostic_mode:
        base_name += "_diagnostic"
    elif sample_size is not None:
        base_name += f"_sample{sample_size}"
    else:
        base_name += "_full"
    
    return os.path.join(INDEX_CACHE_DIR, f"{base_name}.pkl")

# ------------------------------------------------------------------------------
# STATISTICAL FUNCTIONS (unchanged from TWAP version)
# ------------------------------------------------------------------------------

def wilson_score_interval(successes, trials, confidence=0.95):
    """Wilson score interval for binomial proportion."""
    if trials == 0:
        return 0.0, 0.0, 0.0
    
    p = successes / trials
    z = 1.96 if confidence == 0.95 else 2.576
    
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) / trials + z**2 / (4 * trials**2))) / denominator
    
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    
    return p, lower, upper

def compute_brier_score(predictions, outcomes):
    """Brier score: Mean squared error between predictions and outcomes."""
    if len(predictions) == 0:
        return None
    return np.mean((np.array(predictions) - np.array(outcomes))**2)

def compute_log_loss(predictions, outcomes, epsilon=1e-15):
    """Log loss (cross-entropy loss)."""
    if len(predictions) == 0:
        return None
    preds = np.clip(np.array(predictions), epsilon, 1 - epsilon)
    outcomes = np.array(outcomes)
    return -np.mean(outcomes * np.log(preds) + (1 - outcomes) * np.log(1 - preds))

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    """Sharpe ratio: Risk-adjusted return metric."""
    if len(returns) < 2:
        return None
    mean_return = np.mean(returns) - risk_free_rate
    std_return = np.std(returns, ddof=1)
    if std_return == 0:
        return None
    return mean_return / std_return

def compute_kelly_fraction(win_prob, implied_prob):
    """Kelly criterion: Optimal bet sizing for log utility maximization."""
    if implied_prob >= 1.0 or implied_prob <= 0.0:
        return 0.0
    
    odds = (1 - implied_prob) / implied_prob
    kelly = (win_prob * (1 + odds) - 1) / odds
    
    return max(0.0, kelly)

def compute_t_statistic(edge, std_dev, n):
    """T-statistic for edge significance testing."""
    if n < 2 or std_dev == 0:
        return 0.0, 1.0
    
    t_stat = edge / (std_dev / np.sqrt(n))
    
    if n > 30:
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    else:
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    return t_stat, p_value

# ------------------------------------------------------------------------------
# SNAPSHOT PRICE COMPUTATION (NEW in v14.0)
# ------------------------------------------------------------------------------

def get_snapshot_price(trades_near_snapshot, snapshot_time, method='interpolate'):
    """
    Get price at snapshot time from nearby trades.
    
    Methods:
    - 'interpolate': Linear interpolation between surrounding trades (preferred)
    - 'nearest': Use price of trade nearest to snapshot time
    - 'before': Use most recent trade before snapshot time
    
    Args:
        trades_near_snapshot: list of (timestamp, price, size) tuples, sorted by time
        snapshot_time: target timestamp for snapshot
        method: interpolation method
    
    Returns:
        tuple: (snapshot_price, method_used, diagnostics_dict)
        Returns (None, None, None) if insufficient data
    """
    if not trades_near_snapshot:
        return None, None, None
    
    if len(trades_near_snapshot) == 1:
        # Only one trade - use it directly
        t, p, _ = trades_near_snapshot[0]
        return p, 'single_trade', {
            'single_trade_time': t,
            'single_trade_price': p,
            'time_diff_seconds': abs(t - snapshot_time)
        }
    
    # Sort by timestamp
    sorted_trades = sorted(trades_near_snapshot, key=lambda x: x[0])
    
    # Find trades before and after snapshot
    before_trades = [(t, p, s) for t, p, s in sorted_trades if t <= snapshot_time]
    after_trades = [(t, p, s) for t, p, s in sorted_trades if t > snapshot_time]
    
    diagnostics = {
        'n_trades_before': len(before_trades),
        'n_trades_after': len(after_trades),
        'snapshot_time': snapshot_time,
    }
    
    if method == 'interpolate' and before_trades and after_trades:
        # Linear interpolation
        t1, p1, _ = before_trades[-1]  # Last trade before snapshot
        t2, p2, _ = after_trades[0]     # First trade after snapshot
        
        # Interpolate
        if t2 != t1:
            weight = (snapshot_time - t1) / (t2 - t1)
            interpolated_price = p1 + weight * (p2 - p1)
        else:
            interpolated_price = (p1 + p2) / 2
        
        diagnostics.update({
            'before_time': t1,
            'before_price': p1,
            'after_time': t2,
            'after_price': p2,
            'interpolation_weight': weight if t2 != t1 else 0.5,
        })
        
        return interpolated_price, 'interpolate', diagnostics
    
    elif method == 'nearest' or (method == 'interpolate' and (not before_trades or not after_trades)):
        # Find nearest trade
        nearest_trade = min(sorted_trades, key=lambda x: abs(x[0] - snapshot_time))
        t, p, _ = nearest_trade
        
        diagnostics.update({
            'nearest_time': t,
            'nearest_price': p,
            'time_diff_seconds': abs(t - snapshot_time),
        })
        
        return p, 'nearest', diagnostics
    
    elif method == 'before':
        if before_trades:
            t, p, _ = before_trades[-1]
            diagnostics.update({
                'before_time': t,
                'before_price': p,
                'time_diff_seconds': snapshot_time - t,
            })
            return p, 'before', diagnostics
        else:
            # Fall back to nearest
            nearest_trade = min(sorted_trades, key=lambda x: abs(x[0] - snapshot_time))
            t, p, _ = nearest_trade
            diagnostics.update({
                'nearest_time': t,
                'nearest_price': p,
                'time_diff_seconds': abs(t - snapshot_time),
            })
            return p, 'nearest_fallback', diagnostics
    
    return None, None, None


def assign_bucket(price, buckets):
    """Assign price to odds bucket"""
    for lower, upper, label, tag in buckets:
        if lower <= price < upper:
            return label, tag
    if price >= 1.0: 
        return buckets[-1][2], buckets[-1][3]
    return None, None

# ==============================================================================
# DATA LOADING & INDEXING (mostly unchanged)
# ==============================================================================

def save_market_index(market_index, batch_files, cache_file):
    """
    Save market index to disk for reuse.
    
    PROTECTION: Will not overwrite a cache with more markets than the new one.
    This prevents diagnostic runs from destroying full-run caches.
    """
    # Check if existing cache has more markets
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                existing = pickle.load(f)
            existing_count = existing.get('num_markets', 0)
            new_count = len(market_index)
            
            if existing_count > new_count:
                log(f"  ⚠️ REFUSING to overwrite cache: existing has {existing_count:,} markets, new has {new_count:,}")
                log(f"     To force overwrite, delete: {cache_file}")
                return
        except Exception:
            pass  # If we can't read existing, proceed with save
    
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'num_files': len(batch_files),
        'num_markets': len(market_index),
        'market_index': market_index
    }
    
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    log(f"  Market index cached: {cache_file}")

def load_market_index(cache_file, batch_files):
    """Load market index from cache if valid"""
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        cached_num_files = cache_data.get('num_files', 0)
        current_num_files = len(batch_files)
        
        log(f"  Cache validation:")
        log(f"    Cached: {cached_num_files} files, {cache_data['num_markets']:,} markets")
        log(f"    Current: {current_num_files} files")
        
        file_diff_pct = abs(cached_num_files - current_num_files) / max(cached_num_files, 1) * 100
        
        if file_diff_pct > 10:
            log(f"    ⚠️ File count diff {file_diff_pct:.1f}% > 10% - rebuilding index")
            return None
        
        log(f"    ✓ Cache valid (diff: {file_diff_pct:.1f}%)")
        return cache_data['market_index']
        
    except Exception as e:
        log(f"  ⚠️ Cache load failed: {e}")
        return None

def build_market_index(batch_files, cache_file, sample_size=None):
    """Pass 1: Build lightweight market index"""
    log("\n" + "="*70)
    log("PASS 1: BUILDING MARKET INDEX")
    log("="*70)
    log(f"  Cache file: {cache_file}")
    
    if sample_size:
        batch_files = batch_files[:sample_size]
        log(f"  SAMPLE MODE: Processing {len(batch_files)} files")
    
    market_index = load_market_index(cache_file, batch_files)
    if market_index is not None:
        log(f"  ✓ Loaded from cache: {len(market_index):,} markets")
        log(f"    {log_memory()}")
        return market_index, batch_files
    
    market_index = defaultdict(list)
    
    log(f"  Processing {len(batch_files)} batch files...")
    log(f"  {log_memory()}")
    
    for idx, filepath in enumerate(batch_files, 1):
        if idx % 1000 == 0:
            log(f"    [{idx}/{len(batch_files)}] {log_memory()}")
        
        try:
            parquet_file = pq.ParquetFile(filepath)
            metadata = parquet_file.metadata
            
            for row_group_idx in range(metadata.num_row_groups):
                batch = parquet_file.read_row_group(row_group_idx, columns=['condition_id'])
                df_light = batch.to_pandas()
                
                for condition_id in df_light['condition_id'].unique():
                    count = (df_light['condition_id'] == condition_id).sum()
                    market_index[condition_id].append((idx - 1, count))
                
                del df_light, batch
                gc.collect()
                
        except Exception as e:
            log(f"    ERROR in {os.path.basename(filepath)}: {e}")
    
    log(f"  ✓ Index built: {len(market_index):,} unique markets")
    log(f"    {log_memory()}")
    
    save_market_index(market_index, batch_files, cache_file)
    
    return market_index, batch_files

def get_unique_file_indices(market_index):
    """Extract unique file indices from market_index"""
    unique_indices = set()
    for condition_id, file_list in market_index.items():
        for file_idx, _ in file_list:
            unique_indices.add(file_idx)
    return sorted(unique_indices)

# ==============================================================================
# SIDECAR WINNER LOOKUP
# ==============================================================================

def load_sidecar_winners(sidecar_path):
    """Load pre-computed winner data from sidecar parquet file."""
    log(f"Loading sidecar winner data from {sidecar_path}...")
    
    if not os.path.exists(sidecar_path):
        log(f"  ERROR: Sidecar file not found: {sidecar_path}")
        log(f"  Run harvest_api_winners.py first to generate it.")
        return None
    
    try:
        df = pd.read_parquet(sidecar_path)
        
        log(f"  Sidecar records: {len(df):,}")
        
        success_df = df[df['repair_status'] == 'SUCCESS']
        log(f"  SUCCESS records: {len(success_df):,}")
        
        winner_lookup = {}
        for _, row in success_df.iterrows():
            token_id = str(row['token_id'])
            is_winner = row['api_derived_winner']
            if is_winner is not None:
                winner_lookup[token_id] = bool(is_winner)
        
        log(f"  Winner lookup size: {len(winner_lookup):,} tokens")
        
        winners = sum(1 for v in winner_lookup.values() if v)
        losers = sum(1 for v in winner_lookup.values() if not v)
        log(f"  Distribution: {winners:,} winners, {losers:,} losers")
        
        mem_mb = sys.getsizeof(winner_lookup) / (1024 * 1024)
        log(f"  Lookup memory: ~{mem_mb:.1f} MB")
        
        return winner_lookup
        
    except Exception as e:
        log(f"  ERROR loading sidecar: {e}")
        return None


def lookup_winner(token_id, winner_lookup):
    """Look up winner status for a token from the sidecar data."""
    return winner_lookup.get(str(token_id), None)

# ==============================================================================
# SNAPSHOT-BASED TOKEN METRICS (NEW in v14.0)
# ==============================================================================

def get_available_columns(filepath):
    """Inspect a Parquet file to determine available columns."""
    parquet_file = pq.ParquetFile(filepath)
    schema = parquet_file.schema
    available = set(schema.names)
    
    columns_to_read = [c for c in REQUIRED_COLUMNS if c in available]
    
    volume_col = None
    for vc in VOLUME_COLUMNS:
        if vc in available:
            volume_col = vc
            columns_to_read.append(vc)
            break
    
    for wc in WINNER_COLUMNS:
        if wc in available:
            columns_to_read.append(wc)
            break
    
    return columns_to_read, volume_col


def compute_token_metrics_snapshot(acc, diagnostic_mode=False):
    """
    Compute snapshot price and metrics for a single token.
    
    SNAPSHOT VERSION: Uses point-in-time price instead of TWAP.
    
    Returns:
        dict with computed metrics, or:
        - None if token should be filtered (liquidity/no trades near snapshot)
        - 'NO_WINNER' string if token missing from sidecar
    """
    trades = acc['trades']
    winner_status = acc['winner_status']
    resolution_time = acc['resolution_time']
    
    # Must have winner status from sidecar
    if winner_status is None:
        return 'NO_WINNER'
    
    # Minimum trades filter
    if len(trades) < MIN_TRADES_PER_TOKEN:
        return None
    
    # Compute snapshot time
    snapshot_time = resolution_time - (SNAPSHOT_OFFSET_HOURS * 3600)
    
    # Filter to trades within tolerance window of snapshot
    window_start = snapshot_time - SNAPSHOT_TOLERANCE_SECONDS
    window_end = snapshot_time + SNAPSHOT_TOLERANCE_SECONDS
    
    trades_near_snapshot = [
        (t, p, s) for t, p, s in trades 
        if window_start <= t <= window_end
    ]
    
    if len(trades_near_snapshot) < 1:
        return None  # No trades near snapshot time
    
    # Compute volume in tolerance window (USDC scaling: divide by 1_000_000)
    total_volume_usd = sum(p * s for _, p, s in trades_near_snapshot) / 1_000_000.0
    
    if total_volume_usd < MIN_VOLUME_USD:
        return None
    
    # Get snapshot price
    snapshot_price, method_used, diagnostics = get_snapshot_price(
        trades_near_snapshot, snapshot_time, method='interpolate'
    )
    
    if snapshot_price is None or not (0 < snapshot_price < 1):
        return None
    
    # Assign to bucket
    bucket_label, bucket_tag = assign_bucket(snapshot_price, ODDS_BUCKETS)
    if bucket_label is None:
        return None
    
    # Calculate metrics
    avg_trade_size = sum(p * s for _, p, s in trades_near_snapshot) / len(trades_near_snapshot) / 1_000_000.0
    
    # Return calculation
    token_won = winner_status
    ret = (1.0 / snapshot_price - 1.0) if token_won else -1.0
    
    # Flag surprising outcomes
    surprising = False
    if snapshot_price >= 0.90 and not token_won:
        surprising = 'HIGH_PROB_LOSER'
    elif snapshot_price <= 0.10 and token_won:
        surprising = 'LOW_PROB_WINNER'
    
    result = {
        'bucket_label': bucket_label,
        'snapshot_price': snapshot_price,
        'outcome': 1 if token_won else 0,
        'volume': total_volume_usd,
        'trade_size': avg_trade_size,
        'return': ret,
        'token_won': token_won,
        'condition_id': acc['condition_id'],
        'trade_count': len(trades_near_snapshot),
        'surprising': surprising,
        'snapshot_method': method_used,
    }
    
    # Include diagnostics in diagnostic mode
    if diagnostic_mode and diagnostics:
        result['snapshot_diagnostics'] = diagnostics
    
    return result


def flush_completed_condition_snapshot(condition_id, token_accumulator, bucket_data, 
                                       condition_tokens, stats_counters, diagnostic_mode=False):
    """
    Aggregate all tokens for a completed condition into bucket_data, then flush from memory.
    SNAPSHOT VERSION.
    """
    tokens_to_flush = condition_tokens.get(condition_id, set())
    
    for token_id in tokens_to_flush:
        if token_id not in token_accumulator:
            continue
        
        acc = token_accumulator[token_id]
        
        # Compute metrics for this token
        metrics = compute_token_metrics_snapshot(acc, diagnostic_mode)
        
        if metrics == 'NO_WINNER':
            stats_counters['tokens_no_winner'] += 1
        elif metrics is not None:
            bucket_label = metrics['bucket_label']
            bucket_data[bucket_label]['snapshot_prices'].append(metrics['snapshot_price'])
            bucket_data[bucket_label]['outcomes'].append(metrics['outcome'])
            bucket_data[bucket_label]['volumes'].append(metrics['volume'])
            bucket_data[bucket_label]['trade_sizes'].append(metrics['trade_size'])
            bucket_data[bucket_label]['returns'].append(metrics['return'])
            
            stats_counters['tokens_analyzed'] += 1
            stats_counters['markets_processed'].add(condition_id)
            
            # Track snapshot methods used
            method = metrics.get('snapshot_method', 'unknown')
            stats_counters['snapshot_methods'][method] = stats_counters['snapshot_methods'].get(method, 0) + 1
            
            if metrics['surprising']:
                stats_counters['surprising_outcomes'].append({
                    'condition_id': condition_id,
                    'token_id': token_id,
                    'snapshot_price': metrics['snapshot_price'],
                    'won': metrics['token_won'],
                    'bucket': bucket_label,
                    'type': metrics['surprising'],
                    'volume': metrics['volume'],
                    'trade_count': metrics['trade_count'],
                    'snapshot_method': method,
                })
        else:
            stats_counters['tokens_filtered'] += 1
        
        del token_accumulator[token_id]
    
    if condition_id in condition_tokens:
        del condition_tokens[condition_id]


def process_files_accumulator_snapshot(batch_files, market_index, winner_lookup, 
                                       sample_size=None, diagnostic_mode=False):
    """
    FILE-FIRST ACCUMULATOR PATTERN - SNAPSHOT VERSION
    
    Collects trades within tolerance window of snapshot time,
    then computes point-in-time snapshot price.
    """
    log("\n" + "="*70)
    log("PASS 2: FILE-FIRST ACCUMULATOR (SNAPSHOT MODE)")
    log("="*70)
    log(f"  Snapshot offset: T-{SNAPSHOT_OFFSET_HOURS}h before resolution")
    log(f"  Tolerance window: ±{SNAPSHOT_TOLERANCE_HOURS}h around snapshot")
    
    unique_file_indices = get_unique_file_indices(market_index)
    log(f"  Unique files to process: {len(unique_file_indices)}")
    
    if sample_size:
        unique_file_indices = unique_file_indices[:sample_size]
        log(f"  SAMPLE MODE: Processing {len(unique_file_indices)} files")
    
    if diagnostic_mode:
        log(f"  DIAGNOSTIC MODE: Extra logging enabled")
    
    valid_conditions = set(market_index.keys())
    log(f"  Valid conditions (markets): {len(valid_conditions):,}")
    log(f"  Winner lookup size: {len(winner_lookup):,} tokens")
    
    # Condition completion tracking
    unique_file_set = set(unique_file_indices)
    
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
    
    single_file_conditions = [c for c, count in condition_remaining_files.items() if count == 1]
    log(f"  Conditions with only 1 file: {len(single_file_conditions)}")
    
    # Accumulators
    token_accumulator = {}
    condition_tokens = defaultdict(set)
    
    bucket_data = defaultdict(lambda: {
        'snapshot_prices': [],  # Changed from 'twaps'
        'outcomes': [],
        'volumes': [],
        'trade_sizes': [],
        'returns': []
    })
    
    stats_counters = {
        'tokens_analyzed': 0,
        'tokens_filtered': 0,
        'tokens_no_winner': 0,
        'markets_processed': set(),
        'surprising_outcomes': [],
        'snapshot_methods': {},  # NEW: track which methods used
    }
    
    files_processed = 0
    total_trades_accumulated = 0
    total_rows_read = 0
    total_rows_filtered = 0
    conditions_flushed = 0
    
    # For diagnostic: track first few tokens in detail
    diagnostic_samples = []
    
    log(f"  {log_memory()}")
    
    for file_idx in unique_file_indices:
        files_processed += 1
        filepath = batch_files[file_idx]
        
        if files_processed % 500 == 0 or files_processed <= 10:
            log(f"    [{files_processed}/{len(unique_file_indices)}] "
                f"Active Tokens: {len(token_accumulator):,} | "
                f"Flushed: {conditions_flushed:,} | "
                f"No Winner: {stats_counters['tokens_no_winner']:,} | "
                f"{log_memory()}")
        
        try:
            columns_to_read, volume_col = get_available_columns(filepath)
            
            if not volume_col:
                continue
            
            df = pq.read_table(filepath, columns=columns_to_read).to_pandas()
            
            if len(df) == 0:
                continue
            
            total_rows_read += len(df)
            
            if volume_col != 'size_tokens':
                df.rename(columns={volume_col: 'size_tokens'}, inplace=True)
            
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            if len(df) == 0:
                continue
            
            # Detect and normalize milliseconds -> seconds
            if df['timestamp'].iloc[0] > 3e9:
                df['timestamp'] = df['timestamp'] / 1000.0
                df['resolution_time'] = df['resolution_time'] / 1000.0
            
            # Filter to valid conditions
            condition_mask = df['condition_id'].isin(valid_conditions)
            df = df.loc[condition_mask]
            
            if len(df) == 0:
                continue
            
            # SNAPSHOT WINDOW FILTER
            # Compute snapshot time for each row
            snapshot_times = df['resolution_time'] - (SNAPSHOT_OFFSET_HOURS * 3600)
            window_start = snapshot_times - SNAPSHOT_TOLERANCE_SECONDS
            window_end = snapshot_times + SNAPSHOT_TOLERANCE_SECONDS
            
            window_mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= window_end)
            filtered_df = df.loc[window_mask]
            
            if len(filtered_df) == 0:
                pass  # Still need to decrement condition counters
            else:
                total_rows_filtered += len(filtered_df)
                
                if files_processed <= 5:
                    log(f"    File {files_processed}: {len(filtered_df)} trades in snapshot window")
                
                filtered_df = filtered_df.copy()
                filtered_df['token_id'] = filtered_df['token_id'].astype(str)
                
                for token_id, group in filtered_df.groupby('token_id', sort=False):
                    condition_id = group['condition_id'].iloc[0]
                    
                    if token_id not in token_accumulator:
                        winner_status = lookup_winner(token_id, winner_lookup)
                        
                        token_accumulator[token_id] = {
                            'condition_id': condition_id,
                            'resolution_time': float(group['resolution_time'].iloc[0]),
                            'trades': [],
                            'winner_status': winner_status,
                        }
                        condition_tokens[condition_id].add(token_id)
                        
                        # Diagnostic logging
                        if diagnostic_mode and len(diagnostic_samples) < 10:
                            resolution_time = float(group['resolution_time'].iloc[0])
                            snapshot_time = resolution_time - (SNAPSHOT_OFFSET_HOURS * 3600)
                            diagnostic_samples.append({
                                'token_id': token_id[:40],
                                'condition_id': condition_id[:40],
                                'resolution_time': datetime.fromtimestamp(resolution_time).isoformat(),
                                'snapshot_time': datetime.fromtimestamp(snapshot_time).isoformat(),
                                'winner_status': winner_status,
                            })
                    
                    acc = token_accumulator[token_id]
                    
                    timestamps = group['timestamp'].values
                    prices = group['price'].values
                    sizes = group['size_tokens'].values
                    
                    trades_batch = list(zip(timestamps, prices, sizes))
                    acc['trades'].extend(trades_batch)
                    total_trades_accumulated += len(trades_batch)
            
            del df
            if 'filtered_df' in dir():
                del filtered_df
            
            # Streaming flush
            conditions_in_this_file = file_to_conditions.get(file_idx, set())
            
            for condition_id in conditions_in_this_file:
                if condition_id in condition_remaining_files:
                    condition_remaining_files[condition_id] -= 1
                    
                    if condition_remaining_files[condition_id] == 0:
                        if diagnostic_mode:
                            tokens_in_condition = condition_tokens.get(condition_id, set())
                            winners = sum(1 for t in tokens_in_condition 
                                        if t in token_accumulator and token_accumulator[t]['winner_status'] is True)
                            losers = sum(1 for t in tokens_in_condition 
                                       if t in token_accumulator and token_accumulator[t]['winner_status'] is False)
                            missing = sum(1 for t in tokens_in_condition 
                                        if t in token_accumulator and token_accumulator[t]['winner_status'] is None)
                            log(f"    [DIAG] Flushing {condition_id[:16]}... "
                                f"tokens={len(tokens_in_condition)} winners={winners} losers={losers} missing={missing}")
                        
                        flush_completed_condition_snapshot(
                            condition_id, token_accumulator, bucket_data,
                            condition_tokens, stats_counters, diagnostic_mode
                        )
                        
                        del condition_remaining_files[condition_id]
                        conditions_flushed += 1
            
        except Exception as e:
            continue
    
    # Final cleanup
    remaining_conditions = list(condition_tokens.keys())
    if remaining_conditions:
        log(f"\n  Flushing {len(remaining_conditions)} remaining conditions...")
    
    for condition_id in remaining_conditions:
        flush_completed_condition_snapshot(
            condition_id, token_accumulator, bucket_data,
            condition_tokens, stats_counters, diagnostic_mode
        )
        conditions_flushed += 1
    
    gc.collect()
    
    log(f"  ✓ Accumulation complete: {log_memory()}")
    log(f"    Files processed: {files_processed:,}")
    log(f"    Total rows read: {total_rows_read:,}")
    log(f"    Rows in snapshot window: {total_rows_filtered:,}")
    log(f"    Conditions flushed: {conditions_flushed:,}")
    log(f"    Tokens analyzed: {stats_counters['tokens_analyzed']:,}")
    log(f"    Tokens filtered (liquidity): {stats_counters['tokens_filtered']:,}")
    log(f"    Tokens missing winner: {stats_counters['tokens_no_winner']:,}")
    
    # Report snapshot method distribution
    log(f"\n  Snapshot method distribution:")
    for method, count in sorted(stats_counters['snapshot_methods'].items()):
        pct = count / stats_counters['tokens_analyzed'] * 100 if stats_counters['tokens_analyzed'] > 0 else 0
        log(f"    {method}: {count:,} ({pct:.1f}%)")
    
    # Diagnostic samples
    if diagnostic_mode and diagnostic_samples:
        log(f"\n  DIAGNOSTIC: First {len(diagnostic_samples)} token samples:")
        for sample in diagnostic_samples:
            log(f"    Token: {sample['token_id']}...")
            log(f"      Resolution: {sample['resolution_time']}")
            log(f"      Snapshot (T-{SNAPSHOT_OFFSET_HOURS}h): {sample['snapshot_time']}")
            log(f"      Winner: {sample['winner_status']}")
    
    # Report surprising outcomes
    surprising = stats_counters['surprising_outcomes']
    if surprising:
        high_prob_losers = [s for s in surprising if s['type'] == 'HIGH_PROB_LOSER']
        low_prob_winners = [s for s in surprising if s['type'] == 'LOW_PROB_WINNER']
        
        log(f"\n  ⚠️  SURPRISING OUTCOMES DETECTED:")
        log(f"    High-prob losers (snapshot >= 90%, lost): {len(high_prob_losers)}")
        log(f"    Low-prob winners (snapshot <= 10%, won): {len(low_prob_winners)}")
        
        if high_prob_losers:
            log(f"\n  HIGH-PROBABILITY LOSERS (investigate these):")
            for s in high_prob_losers[:20]:
                log(f"    condition={s['condition_id'][:24]}...")
                log(f"      snapshot={s['snapshot_price']:.4f} ({s['bucket']}) won={s['won']} "
                    f"method={s['snapshot_method']} vol=${s['volume']:.0f}")
    
    total_metrics = {
        'markets_processed': len(stats_counters['markets_processed']),
        'tokens_analyzed': stats_counters['tokens_analyzed'],
        'tokens_low_liquidity': stats_counters['tokens_filtered'],
        'tokens_no_winner': stats_counters['tokens_no_winner'],
        'surprising_outcomes': len(surprising),
        'snapshot_methods': stats_counters['snapshot_methods'],
        'filter_rate_pct': (stats_counters['tokens_filtered'] / 
                          (stats_counters['tokens_analyzed'] + stats_counters['tokens_filtered']) * 100)
                          if (stats_counters['tokens_analyzed'] + stats_counters['tokens_filtered']) > 0 else 0
    }
    
    return bucket_data, total_metrics, surprising


# ==============================================================================
# CALIBRATION STATISTICS COMPUTATION
# ==============================================================================

def compute_calibration_stats(bucket_data):
    """Compute calibration statistics for each bucket."""
    calibration_stats = []
    
    for lower, upper, label, tag in ODDS_BUCKETS:
        if label not in bucket_data or len(bucket_data[label]['snapshot_prices']) == 0:
            continue
        
        data = bucket_data[label]
        snapshot_prices = np.array(data['snapshot_prices'])
        outcomes = np.array(data['outcomes'])
        volumes = np.array(data['volumes'])
        trade_sizes = np.array(data['trade_sizes'])
        returns = np.array(data['returns'])
        
        n_tokens = len(snapshot_prices)
        n_wins = int(np.sum(outcomes))
        
        # Average implied probability (from snapshot prices)
        avg_implied = float(np.mean(snapshot_prices))
        
        # Realized win rate with Wilson confidence interval
        win_rate, ci_lower, ci_upper = wilson_score_interval(n_wins, n_tokens)
        
        # Edge calculations
        raw_edge = win_rate - avg_implied
        return_edge = raw_edge / avg_implied if avg_implied > 0 else 0
        
        # Z-statistic for binomial proportion test
        p0 = avg_implied
        se = np.sqrt(p0 * (1 - p0) / n_tokens) if n_tokens > 0 else 0
        z_stat = (win_rate - p0) / se if se > 0 else 0
        p_value_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # T-test on returns
        t_stat, p_value_t = compute_t_statistic(np.mean(returns), np.std(returns, ddof=1), n_tokens)
        
        # Sharpe and Kelly
        sharpe = compute_sharpe_ratio(returns)
        kelly = compute_kelly_fraction(win_rate, avg_implied)
        
        # Brier and log loss
        predictions = [avg_implied] * n_tokens
        brier = compute_brier_score(predictions, outcomes.tolist())
        log_loss = compute_log_loss(predictions, outcomes.tolist())
        
        # Volume stats
        avg_volume = float(np.mean(volumes))
        median_volume = float(np.median(volumes))
        total_volume = float(np.sum(volumes))
        avg_trade_size = float(np.mean(trade_sizes))
        
        calibration_stats.append({
            'bucket': label,
            'tag': tag,
            'n_tokens': n_tokens,
            'n_wins': n_wins,
            'avg_implied_prob': avg_implied,
            'realized_win_rate': win_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'raw_edge': raw_edge,
            'raw_edge_bps': raw_edge * 10000,
            'return_edge': return_edge,
            'return_edge_pct': return_edge * 100,
            'z_statistic': z_stat,
            'p_value_z': p_value_z,
            't_statistic': t_stat,
            'p_value_t': p_value_t,
            'sharpe_ratio': sharpe,
            'kelly_fraction': kelly,
            'brier_score': brier,
            'log_loss': log_loss,
            'avg_volume': avg_volume,
            'median_volume': median_volume,
            'total_volume': total_volume,
            'avg_trade_size': avg_trade_size,
            'significant': p_value_z < ALPHA,
        })
    
    return calibration_stats


# ==============================================================================
# OUTPUT GENERATION
# ==============================================================================

def generate_visualization(calibration_stats, total_metrics):
    """Generate calibration visualization."""
    log("\nGenerating visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    buckets = [s['bucket'] for s in calibration_stats]
    implied = [s['avg_implied_prob'] for s in calibration_stats]
    realized = [s['realized_win_rate'] for s in calibration_stats]
    ci_lower = [s['ci_lower'] for s in calibration_stats]
    ci_upper = [s['ci_upper'] for s in calibration_stats]
    edges_bps = [s['raw_edge_bps'] for s in calibration_stats]
    n_tokens = [s['n_tokens'] for s in calibration_stats]
    
    # Plot 1: Calibration curve
    ax1 = axes[0, 0]
    x = np.arange(len(buckets))
    ax1.plot([0, len(buckets)-1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    ax1.errorbar(implied, realized, 
                 yerr=[np.array(realized) - np.array(ci_lower), 
                       np.array(ci_upper) - np.array(realized)],
                 fmt='o-', capsize=5, label='Realized (95% CI)')
    ax1.set_xlabel('Implied Probability (Snapshot)')
    ax1.set_ylabel('Realized Win Rate')
    ax1.set_title('Calibration Curve (Snapshot at T-24h)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Edge by bucket
    ax2 = axes[0, 1]
    colors = ['red' if e < 0 else 'green' for e in edges_bps]
    bars = ax2.bar(x, edges_bps, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(buckets, rotation=45, ha='right')
    ax2.set_ylabel('Raw Edge (basis points)')
    ax2.set_title('Edge by Probability Bucket')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Sample sizes
    ax3 = axes[1, 0]
    ax3.bar(x, n_tokens, color='steelblue', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(buckets, rotation=45, ha='right')
    ax3.set_ylabel('Number of Tokens')
    ax3.set_title('Sample Size by Bucket')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Implied vs Realized scatter
    ax4 = axes[1, 1]
    ax4.scatter(implied, realized, s=100, c=edges_bps, cmap='RdYlGn', alpha=0.8)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax4.set_xlabel('Implied Probability (Snapshot)')
    ax4.set_ylabel('Realized Win Rate')
    ax4.set_title('Implied vs Realized (color = edge)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    viz_path = os.path.join(OUTPUT_DIR, f'phase1_calibration_snapshot_{TIMESTAMP}.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"  ✓ Visualization saved: {viz_path}")


def generate_report(calibration_stats, total_metrics):
    """Generate detailed text report."""
    log("\nGenerating report...")
    
    report_path = os.path.join(OUTPUT_DIR, f'phase1_calibration_snapshot_{TIMESTAMP}_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 1: ODDS-STRATIFIED CALIBRATION ANALYSIS v14.0 (SNAPSHOT)\n")
        f.write("Polymarket Baseline Edge Measurement - Point-in-Time Price\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Target Audience: Quantitative PMs (Citadel, Jane Street, etc.)\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Markets Analyzed: {total_metrics['markets_processed']:,}\n")
        f.write(f"Tokens Analyzed: {total_metrics['tokens_analyzed']:,}\n")
        f.write(f"Liquidity Filter Rate: {total_metrics['filter_rate_pct']:.1f}%\n")
        
        sig_pos = len([s for s in calibration_stats if s['significant'] and s['raw_edge'] > 0])
        sig_neg = len([s for s in calibration_stats if s['significant'] and s['raw_edge'] < 0])
        f.write(f"Significant Positive Edge Buckets: {sig_pos}\n")
        f.write(f"Significant Negative Edge Buckets: {sig_neg}\n\n")
        
        f.write("="*80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. PRICE REPRESENTATION - SNAPSHOT\n")
        f.write("-"*40 + "\n")
        f.write(f"   Metric: Point-in-time snapshot price\n")
        f.write(f"   Snapshot Time: T-{SNAPSHOT_OFFSET_HOURS}h before resolution\n")
        f.write(f"   Tolerance Window: ±{SNAPSHOT_TOLERANCE_HOURS}h around snapshot\n")
        f.write(f"   Method: Linear interpolation between surrounding trades\n")
        f.write("   Rationale:\n")
        f.write("     - Represents 'tradeable belief' at realistic entry time\n")
        f.write("     - Avoids TWAP contamination from resolution-period convergence\n")
        f.write("     - More noise than TWAP, but cleaner causal interpretation\n\n")
        
        f.write("2. SNAPSHOT METHOD DISTRIBUTION\n")
        f.write("-"*40 + "\n")
        for method, count in sorted(total_metrics.get('snapshot_methods', {}).items()):
            pct = count / total_metrics['tokens_analyzed'] * 100 if total_metrics['tokens_analyzed'] > 0 else 0
            f.write(f"   {method}: {count:,} ({pct:.1f}%)\n")
        f.write("\n")
        
        f.write("3. LIQUIDITY FILTERS\n")
        f.write("-"*40 + "\n")
        f.write(f"   Minimum Volume: ${MIN_VOLUME_USD:.1f} per token in window\n")
        f.write(f"   Minimum Market Trades: {MIN_TRADES_PER_MARKET}\n")
        f.write(f"   Minimum Token Trades: {MIN_TRADES_PER_TOKEN}\n\n")
        
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for stat in calibration_stats:
            f.write(f"{stat['bucket']} ({stat['tag']})\n")
            f.write("-"*80 + "\n")
            f.write(f"Sample Size:        {stat['n_tokens']:,} tokens ({stat['n_wins']:,} wins)\n")
            f.write(f"Implied Prob:       {stat['avg_implied_prob']:.4f}\n")
            f.write(f"Realized Win Rate:  {stat['realized_win_rate']:.4f} [{stat['ci_lower']:.4f}, {stat['ci_upper']:.4f}]\n\n")
            
            f.write(f"Raw Edge:           {stat['raw_edge']:+.6f} ({stat['raw_edge_bps']:+.1f} bps)\n")
            f.write(f"Return Edge:        {stat['return_edge_pct']:+.2f}%\n")
            f.write(f"Sharpe Ratio:       {stat['sharpe_ratio']:.4f}\n" if stat['sharpe_ratio'] else "Sharpe Ratio:       N/A\n")
            f.write(f"Kelly Fraction:     {stat['kelly_fraction']:.4f}\n\n")
            
            f.write(f"Z-Statistic:        {stat['z_statistic']:.3f}\n")
            f.write(f"P-Value (z-test):   {stat['p_value_z']:.6f}\n")
            f.write(f"T-Statistic:        {stat['t_statistic']:.3f}\n")
            f.write(f"P-Value (t-test):   {stat['p_value_t']:.6f}\n")
            f.write(f"Significance:       {'*** SIGNIFICANT ***' if stat['significant'] else 'not significant'}\n\n")
            
            f.write(f"Brier Score:        {stat['brier_score']:.6f}\n" if stat['brier_score'] else "")
            f.write(f"Log Loss:           {stat['log_loss']:.6f}\n\n" if stat['log_loss'] else "\n")
            
            f.write(f"Avg Volume:         ${stat['avg_volume']:.2f}\n")
            f.write(f"Median Volume:      ${stat['median_volume']:.2f}\n")
            f.write(f"Total Volume:       ${stat['total_volume']:.2f}\n")
            f.write(f"Avg Trade Size:     ${stat['avg_trade_size']:.2f}\n\n")
            
            if stat['raw_edge'] > 0 and stat['significant']:
                f.write("Interpretation:\n")
                f.write("  → POSITIVE EDGE: Market systematically underprices this probability range\n")
                if stat['kelly_fraction'] > 0.25:
                    f.write("  → STRONG SIGNAL: Kelly fraction suggests substantial position sizing\n")
                if stat['sharpe_ratio'] and stat['sharpe_ratio'] > 1.0:
                    f.write("  → ATTRACTIVE RISK-ADJUSTED: Sharpe > 1.0 indicates strong risk-adjusted returns\n")
            elif stat['raw_edge'] < 0 and stat['significant']:
                f.write("Interpretation:\n")
                f.write("  → NEGATIVE EDGE: Market systematically overprices this probability range\n")
                f.write("  → AVOID: Do not provide liquidity in this bucket\n")
            f.write("\n")
        
        # Summary sections
        f.write("="*80 + "\n")
        f.write("KEY FINDINGS & STRATEGIC IMPLICATIONS\n")
        f.write("="*80 + "\n\n")
        
        sig_positive = [s for s in calibration_stats if s['significant'] and s['raw_edge'] > 0]
        sig_negative = [s for s in calibration_stats if s['significant'] and s['raw_edge'] < 0]
        
        if sig_positive:
            f.write("POSITIVE EDGE OPPORTUNITIES (SNAPSHOT-BASED)\n")
            f.write("-"*80 + "\n\n")
            for s in sig_positive:
                f.write(f"{s['bucket']} ({s['tag']}):\n")
                f.write(f"  Edge:          {s['raw_edge_bps']:+.1f} bps ({s['return_edge_pct']:+.2f}% return)\n")
                f.write(f"  Sharpe:        {s['sharpe_ratio']:.3f}\n" if s['sharpe_ratio'] else "")
                kelly_quarter = s['kelly_fraction'] / 4 if s['kelly_fraction'] else 0
                f.write(f"  Kelly:         {s['kelly_fraction']:.3f} (suggest ¼ Kelly = {kelly_quarter:.3f})\n\n")
        
        if sig_negative:
            f.write("\nNEGATIVE EDGE ZONES (AVOID)\n")
            f.write("-"*80 + "\n")
            for s in sig_negative:
                f.write(f"\n{s['bucket']}: {s['raw_edge_bps']:+.1f} bps ({s['return_edge_pct']:+.2f}% return)\n")
                f.write(f"  → DO NOT provide liquidity in this range\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("COMPARISON NOTE: SNAPSHOT vs TWAP\n")
        f.write("="*80 + "\n\n")
        f.write("This analysis uses SNAPSHOT prices (T-24h) instead of TWAP (24h-0h window).\n\n")
        f.write("Expected differences:\n")
        f.write("  - Snapshot captures belief at realistic entry time\n")
        f.write("  - TWAP is contaminated by resolution-period price convergence\n")
        f.write("  - If edge compresses under snapshot, TWAP edge was overstated\n")
        f.write("  - If edge persists, underlying miscalibration is real\n\n")
        
        f.write("="*80 + "\n")
        f.write("METHODOLOGY NOTES FOR REVIEWERS\n")
        f.write("="*80 + "\n\n")
        
        f.write("SNAPSHOT APPROACH:\n")
        f.write(f"  - Point-in-time: T-{SNAPSHOT_OFFSET_HOURS}h before resolution\n")
        f.write(f"  - Tolerance: ±{SNAPSHOT_TOLERANCE_HOURS}h for trade lookup\n")
        f.write("  - Interpolation: Linear between surrounding trades\n")
        f.write("  - Fallback: Nearest trade if no straddling trades available\n\n")
        
        f.write("STATISTICAL FRAMEWORK:\n")
        f.write("  - Wilson score intervals: Correct coverage near boundaries\n")
        f.write("  - Two-tailed tests: Detecting both positive and negative bias\n")
        f.write("  - Multiple comparisons: No Bonferroni correction (exploratory phase)\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis version: 14.0 - Snapshot-Based (T-{SNAPSHOT_OFFSET_HOURS}h)\n")
        f.write("="*80 + "\n")
    
    log(f"  ✓ Report saved: {report_path}")
    
    # Also save JSON summary
    json_path = os.path.join(OUTPUT_DIR, f'phase1_calibration_snapshot_{TIMESTAMP}_summary.json')
    summary = {
        'version': '14.0-snapshot',
        'snapshot_offset_hours': SNAPSHOT_OFFSET_HOURS,
        'snapshot_tolerance_hours': SNAPSHOT_TOLERANCE_HOURS,
        'total_metrics': {k: v if not isinstance(v, set) else len(v) for k, v in total_metrics.items()},
        'calibration_stats': calibration_stats,
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"  ✓ JSON saved: {json_path}")


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================

def analyze_calibration_snapshot(sample_size=None, diagnostic_mode=False):
    """
    Main calibration analysis with SNAPSHOT-BASED pricing.
    
    v14.0: Point-in-time snapshot instead of TWAP
    """
    log("\n" + "="*70)
    log("PHASE 1: ODDS-STRATIFIED CALIBRATION ANALYSIS")
    log(f"Version 14.0 - SNAPSHOT-BASED (T-{SNAPSHOT_OFFSET_HOURS}h)")
    log("="*70)
    log(f"  Snapshot time: T-{SNAPSHOT_OFFSET_HOURS}h before resolution")
    log(f"  Tolerance window: ±{SNAPSHOT_TOLERANCE_HOURS}h")
    log(f"  Min volume: ${MIN_VOLUME_USD}")
    log(f"  Confidence: 95% (Wilson score)")
    log(f"  Significance: α={ALPHA} (two-tailed)")
    if diagnostic_mode:
        log(f"  DIAGNOSTIC MODE: Enabled")
    
    # Generate unique cache filename based on run mode
    cache_file = get_cache_filename(sample_size, diagnostic_mode)
    log(f"  Cache file: {cache_file}")
    
    # Load sidecar winner data
    winner_lookup = load_sidecar_winners(SIDECAR_FILE)
    if winner_lookup is None:
        log("FATAL: Cannot proceed without sidecar data")
        return
    
    # Load batch files
    batch_files = sorted(glob.glob(f"{BATCH_DIR}/batch_*.parquet"))
    if not batch_files:
        log(f"ERROR: No batch files found in {BATCH_DIR}")
        return
    
    log(f"\n  Total batch files available: {len(batch_files)}")
    
    # Pass 1: Build market index (with unique cache file)
    market_index, batch_files = build_market_index(batch_files, cache_file, sample_size)
    
    # Pass 2: File-First accumulator with snapshot pricing
    bucket_data, total_metrics, surprising_outcomes = process_files_accumulator_snapshot(
        batch_files, market_index, winner_lookup, sample_size, diagnostic_mode
    )
    
    # Compute calibration statistics
    log("\n" + "="*70)
    log("COMPUTING CALIBRATION STATISTICS")
    log("="*70)
    
    calibration_stats = compute_calibration_stats(bucket_data)
    
    if not calibration_stats:
        log("ERROR: No calibration statistics computed")
        return
    
    # Quick summary
    log("\n  QUICK SUMMARY:")
    for stat in calibration_stats:
        sig_marker = "***" if stat['significant'] else ""
        edge_sign = "+" if stat['raw_edge'] > 0 else ""
        log(f"    {stat['bucket']:15s}: n={stat['n_tokens']:5,} | "
            f"implied={stat['avg_implied_prob']:.3f} | "
            f"realized={stat['realized_win_rate']:.3f} | "
            f"edge={edge_sign}{stat['raw_edge_bps']:.0f}bps {sig_marker}")
    
    # Generate outputs
    generate_visualization(calibration_stats, total_metrics)
    generate_report(calibration_stats, total_metrics)
    
    log("\nOutputs generated:")
    log(f"  - Summary JSON:    phase1_calibration_snapshot_{TIMESTAMP}_summary.json")
    log(f"  - Detailed Report: phase1_calibration_snapshot_{TIMESTAMP}_report.txt")
    log(f"  - Visualization:   phase1_calibration_snapshot_{TIMESTAMP}.png")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phase 1 Calibration Analysis - SNAPSHOT VERSION',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnostic mode - process ~100 conditions, verbose output
  python phase1_calibration_snapshot.py --diagnostic
  
  # Sample mode - process N batch files
  python phase1_calibration_snapshot.py --sample 500
  
  # Full run with custom snapshot offset
  python phase1_calibration_snapshot.py --offset 48
  
  # Full run
  python phase1_calibration_snapshot.py
        """
    )
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files')
    parser.add_argument('--diagnostic', '-d', action='store_true',
                        help='Diagnostic mode: ~100 conditions with verbose output')
    parser.add_argument('--offset', '-o', type=int, default=None,
                        help=f'Snapshot offset in hours (default: {SNAPSHOT_OFFSET_HOURS})')
    parser.add_argument('--tolerance', '-t', type=int, default=None,
                        help=f'Tolerance window in hours (default: {SNAPSHOT_TOLERANCE_HOURS})')
    
    # Support legacy positional argument
    parser.add_argument('legacy_sample', nargs='?', type=int, default=None,
                        help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Handle legacy positional argument
    sample = args.sample or args.legacy_sample
    diagnostic_mode = args.diagnostic
    
    # Override snapshot parameters if specified
    if args.offset is not None:
        SNAPSHOT_OFFSET_HOURS = args.offset
        log(f"Using custom snapshot offset: T-{SNAPSHOT_OFFSET_HOURS}h")
    
    if args.tolerance is not None:
        SNAPSHOT_TOLERANCE_HOURS = args.tolerance
        SNAPSHOT_TOLERANCE_SECONDS = SNAPSHOT_TOLERANCE_HOURS * 3600
        log(f"Using custom tolerance: ±{SNAPSHOT_TOLERANCE_HOURS}h")
    
    # Diagnostic mode defaults
    if diagnostic_mode and sample is None:
        sample = 50
        print(f"\n*** DIAGNOSTIC MODE: Processing {sample} batch files with verbose output ***\n")
    elif sample:
        print(f"\n*** RUNNING IN SAMPLE MODE: {sample} batch files ***\n")
    
    # Check for cached index (using dynamic filename)
    cache_file = get_cache_filename(sample, diagnostic_mode)
    if os.path.exists(cache_file):
        try:
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            print(f"NOTE: Found cached index: {os.path.basename(cache_file)}")
            print(f"      Age: {cache_age.seconds // 3600}h {(cache_age.seconds % 3600) // 60}m")
            print(f"      Pass 1 will be skipped if cache is valid")
            print(f"      To force rebuild: rm {cache_file}\n")
        except: 
            pass
    else:
        print(f"NOTE: Cache will be created at: {os.path.basename(cache_file)}")
        print(f"      (Separate from TWAP and other run modes)\n")
    
    ensure_output_dir()
    analyze_calibration_snapshot(sample_size=sample, diagnostic_mode=diagnostic_mode)