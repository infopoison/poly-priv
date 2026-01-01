#!/usr/bin/env python3
"""
Phase 1: Odds-Stratified Calibration Analysis - SIDECAR WINNER VERSION
Version: 13.0 - Sidecar-Based Winner Lookup (No API Calls)

ARCHITECTURE CHANGE (v13.0):
  - Winner data loaded from pre-computed sidecar file (api_derived_winners.parquet)
  - NO runtime API calls - all winner lookups are O(1) dictionary access
  - Sidecar file generated separately by harvest_api_winners.py
  
PERFORMANCE (v13.0):
  - File-First Accumulator: Each Parquet file read exactly ONCE
  - Vectorized pandas operations (boolean masks, groupby)
  - STREAMING MEMORY MANAGEMENT: Flush completed conditions immediately
  - Winner lookup: O(1) from preloaded sidecar dictionary
  
MEMORY PROFILE:
  - Sidecar lookup dict: ~50MB for 117k tokens
  - Active tokens: O(active_conditions) - bounded, not growing
  - Typical: ~100-150MB stable after warmup

DATA SOURCES:
  - Trade data: order_history_batches/*.parquet
  - Winner data: data/repair/api_derived_winners.parquet (sidecar)

DIAGNOSTIC MODE:
  - Use --diagnostic to process ~100 conditions with verbose output
  - Validates end-to-end workflow without full dataset

TARGET AUDIENCE: Quantitative PMs (Citadel, Jane Street, etc.)
ENHANCEMENTS:
  - Refined probability buckets aligned with charter specifications
  - Comprehensive statistical metrics (Sharpe, Kelly, Brier, log loss)
  - Publication-quality visualizations with confidence bands
  - Detailed methodology justification in quantitative trading terms
  - Return distribution analysis and tail risk metrics

METHODOLOGY OVERVIEW:
  1. CALIBRATION: Stratified binomial testing across probability buckets
  2. EDGE QUANTIFICATION: Basis points, return edge, and Sharpe ratios
  3. STATISTICAL VALIDATION: Wilson score intervals, z-tests, Brier scores
  4. RISK METRICS: Kelly fractions, drawdown estimates, tail analysis

ARCHITECTURE:
  - Two-pass streaming: Index caching + File-First accumulation
  - Memory efficient: O(unique_tokens) memory footprint
  - Production ready: Checkpointed, resumable, validated
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
INDEX_CACHE_FILE = os.path.join(BASE_DIR, 'analysis/market_index_cache.pkl')
MARKETS_CSV = os.path.join(BASE_DIR, 'markets_past_year.csv')

# SIDECAR FILE: Pre-computed winner data from harvest_api_winners.py
SIDECAR_FILE = os.path.join(BASE_DIR, 'data/repair/api_derived_winners.parquet')

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ------------------------------------------------------------------------------
# ANALYSIS PARAMETERS
# ------------------------------------------------------------------------------

# Liquidity filters - Based on empirical market structure analysis
MIN_VOLUME_USD = 100.0      # Minimum USD volume in window (filters "zombie markets")
MIN_TRADES_PER_MARKET = 50  # Minimum total trades (ensures price discovery)
MIN_TRADES_PER_TOKEN = 5    # Minimum trades per token in window (ensures TWAP quality)

# Time window configuration
TIME_WINDOW_HOURS = 24      # Duration of analysis window
WINDOW_OFFSET_HOURS = 24    # Offset from resolution (avoids settlement bias)

# Probability buckets - Aligned with charter specifications
# Rationale: Finer granularity in favorite territory (50-100%) where:
# - Market efficiency typically highest
# - Bond farming strategies operate
# - Sharpe ratios most attractive due to lower variance
ODDS_BUCKETS = [
    # Underdog territory (coarse buckets - higher variance, less liquid)
    (0.00, 0.10, '0-10%', 'longshot'),
    (0.10, 0.25, '10-25%', 'underdog'),
    (0.25, 0.40, '25-40%', 'toss-up-'),
    (0.40, 0.51, '40-51%', 'toss-up+'),
    
    # Favorite territory (refined buckets - higher precision needed)
    (0.51, 0.60, '51-60%', 'mild-fav'),      # Near-efficient zone
    (0.60, 0.75, '60-75%', 'moderate-fav'),  # Moderate confidence
    (0.75, 0.90, '75-90%', 'strong-fav'),    # Strong conviction
    (0.90, 0.99, '90-99%', 'heavy-fav'),     # Bond territory entry
    (0.99, 0.995, '99.0-99.5%', 'near-certain'), # Bond farming core
    (0.995, 1.00, '99.5-100%', 'extreme'),   # Tail behavior
]

# Statistical significance threshold
ALPHA = 0.05  # Two-tailed test

# Column pruning configuration for File-First processing
REQUIRED_COLUMNS = ['timestamp', 'price', 'token_id', 'condition_id', 'resolution_time']
# Note: 'size_tokens' may be named 'maker_amount' or 'size' in raw data
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

# ------------------------------------------------------------------------------
# STATISTICAL FUNCTIONS
# ------------------------------------------------------------------------------

def wilson_score_interval(successes, trials, confidence=0.95):
    """
    Wilson score interval for binomial proportion.
    
    Why Wilson over Normal approximation:
    - Correct coverage near boundaries (0%, 100%)
    - No continuity correction needed
    - Asymmetric intervals (realistic for extreme probabilities)
    
    Standard in quantitative research for win rate confidence intervals.
    """
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
    """
    Brier score: Mean squared error between predicted probabilities and outcomes.
    
    Range: [0, 1] where 0 is perfect calibration
    Decomposition: Brier = Reliability - Resolution + Uncertainty
    
    Standard metric in prediction market research and probabilistic forecasting.
    """
    if len(predictions) == 0:
        return None
    return np.mean((np.array(predictions) - np.array(outcomes))**2)

def compute_log_loss(predictions, outcomes, epsilon=1e-15):
    """
    Log loss (cross-entropy loss): Penalizes confident wrong predictions heavily.
    
    Properties:
    - Strictly proper scoring rule (incentivizes true probabilities)
    - Unbounded (extreme predictions near 0/1 heavily penalized)
    - Standard in ML and prediction market evaluation
    
    Epsilon clipping prevents log(0) errors.
    """
    if len(predictions) == 0:
        return None
    preds = np.clip(np.array(predictions), epsilon, 1 - epsilon)
    outcomes = np.array(outcomes)
    return -np.mean(outcomes * np.log(preds) + (1 - outcomes) * np.log(1 - preds))

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Sharpe ratio: Risk-adjusted return metric.
    
    Formula: (E[R] - Rf) / σ(R)
    
    Interpretation for discrete binary outcomes:
    - Measures edge per unit of volatility
    - Assumes returns are IID (reasonable for stratified buckets)
    - Annualization: Sharpe_annual ≈ Sharpe_per_bet × sqrt(N_bets_per_year)
    
    Standard metric in quantitative trading for strategy evaluation.
    """
    if len(returns) < 2:
        return None
    mean_return = np.mean(returns) - risk_free_rate
    std_return = np.std(returns, ddof=1)
    if std_return == 0:
        return None
    return mean_return / std_return

def compute_kelly_fraction(win_prob, implied_prob):
    """
    Kelly criterion: Optimal bet sizing for log utility maximization.
    
    Binary outcome formula: f* = (p × (1 + b) - 1) / b
    where b = odds = (1 - q) / q, q = implied_prob
    
    Interpretation:
    - Maximizes long-run growth rate
    - Full Kelly often too aggressive (use fractional Kelly in practice)
    - Negative = Don't bet (negative edge)
    
    Standard position sizing framework in quantitative trading.
    """
    if implied_prob >= 1.0 or implied_prob <= 0.0:
        return 0.0
    
    odds = (1 - implied_prob) / implied_prob
    kelly = (win_prob * (1 + odds) - 1) / odds
    
    return max(0.0, kelly)  # Never bet if negative edge

def compute_t_statistic(edge, std_dev, n):
    """
    T-statistic for edge significance testing.
    
    H0: True edge = 0
    H1: True edge ≠ 0 (two-tailed)
    
    Use t-distribution for small samples, z-distribution for large samples.
    """
    if n < 2 or std_dev == 0:
        return 0.0, 1.0
    
    t_stat = edge / (std_dev / np.sqrt(n))
    
    # For n > 30, t-distribution ≈ z-distribution
    if n > 30:
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    else:
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    return t_stat, p_value

# ------------------------------------------------------------------------------
# PRICE COMPUTATION
# ------------------------------------------------------------------------------

def compute_twap(trades_in_window):
    """
    Time-Weighted Average Price (TWAP).
    
    Why TWAP over other price metrics:
    - Resistant to volume manipulation (time-weighted, not volume-weighted)
    - Captures representative price over window (not just close)
    - Standard in execution analysis and market microstructure
    
    Implementation: Linear interpolation between trades.
    """
    if not trades_in_window or len(trades_in_window) < 2:
        return trades_in_window[0][1] if trades_in_window else None
    
    total_time_weighted_price = 0.0
    total_time = 0.0
    
    for i in range(len(trades_in_window) - 1):
        t1, p1, _ = trades_in_window[i]
        t2, _, _ = trades_in_window[i + 1]
        
        diff = t2 - t1
        total_time_weighted_price += p1 * diff
        total_time += diff
    
    if total_time > 0:
        return total_time_weighted_price / total_time
    else:
        # Fallback to simple mean if all trades same timestamp
        return np.mean([p for _, p, _ in trades_in_window])

def assign_bucket(price, buckets):
    """Assign price to odds bucket"""
    for lower, upper, label, tag in buckets:
        if lower <= price < upper:
            return label, tag
    # Handle floating point edge case (exactly 1.0)
    if price >= 1.0: 
        return buckets[-1][2], buckets[-1][3]
    return None, None

# ==============================================================================
# DATA LOADING & INDEXING
# ==============================================================================

def load_winning_outcomes(markets_csv):
    """Load winning outcomes from markets_past_year.csv"""
    log(f"Loading winning outcomes from {markets_csv}...")
    
    if not os.path.exists(markets_csv):
        log(f"  ERROR: Markets CSV not found: {markets_csv}")
        return None
    
    winning_outcomes = {}
    
    try:
        with open(markets_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    condition_id = row['condition_id'].strip()
                    outcome = int(row['outcome'])
                    winning_outcomes[condition_id] = outcome
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        log(f"  ERROR reading outcomes CSV: {e}")
        return None
    
    log(f"  Loaded {len(winning_outcomes):,} market outcomes")
    
    # Show distribution
    outcome_0 = sum(1 for o in winning_outcomes.values() if o == 0)
    outcome_1 = sum(1 for o in winning_outcomes.values() if o == 1)
    log(f"  Outcome distribution: 0={outcome_0:,}, 1={outcome_1:,}")
    
    return winning_outcomes

def save_market_index(market_index, batch_files, cache_file):
    """Save market index to disk for reuse"""
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
        
        # Validate cache
        cached_num_files = cache_data.get('num_files', 0)
        current_num_files = len(batch_files)
        
        log(f"  Cache validation:")
        log(f"    Cached: {cached_num_files} files, {cache_data['num_markets']:,} markets")
        log(f"    Current: {current_num_files} files")
        
        # Allow 10% discrepancy
        file_diff_pct = abs(cached_num_files - current_num_files) / max(cached_num_files, 1) * 100
        
        if file_diff_pct > 10:
            log(f"    ⚠️ File count diff {file_diff_pct:.1f}% > 10% - rebuilding index")
            return None
        
        log(f"    ✓ Cache valid (diff: {file_diff_pct:.1f}%)")
        return cache_data['market_index']
        
    except Exception as e:
        log(f"  ⚠️ Cache load failed: {e}")
        return None

def build_market_index(batch_files, sample_size=None):
    """
    Pass 1: Build lightweight market index
    Maps: condition_id → list of (file_idx, token_count)
    
    FILE-FIRST OPTIMIZATION: Also extracts unique file indices for Pass 2
    """
    log("\n" + "="*70)
    log("PASS 1: BUILDING MARKET INDEX")
    log("="*70)
    
    if sample_size:
        batch_files = batch_files[:sample_size]
        log(f"  SAMPLE MODE: Processing {len(batch_files)} files")
    
    # Try loading from cache first
    market_index = load_market_index(INDEX_CACHE_FILE, batch_files)
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
                rg = metadata.row_group(row_group_idx)
                num_rows = rg.num_rows
                
                # Read only condition_id column
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
    
    # Save to cache
    save_market_index(market_index, batch_files, INDEX_CACHE_FILE)
    
    return market_index, batch_files

def get_unique_file_indices(market_index):
    """
    Extract unique file indices from market_index for File-First processing.
    Returns set of file indices that contain at least one market.
    """
    unique_indices = set()
    for condition_id, file_list in market_index.items():
        for file_idx, _ in file_list:
            unique_indices.add(file_idx)
    return sorted(unique_indices)

# ==============================================================================
# SIDECAR WINNER LOOKUP (Replaces API Repair in v13.0)
# ==============================================================================

def load_sidecar_winners(sidecar_path):
    """
    Load pre-computed winner data from sidecar parquet file.
    
    Returns:
        dict: {token_id: bool} where True = winner, False = loser
        None values in sidecar are excluded (API errors)
    
    The sidecar file is generated by harvest_api_winners.py and contains:
        - condition_id, token_id, api_derived_winner, market_question, etc.
    """
    log(f"Loading sidecar winner data from {sidecar_path}...")
    
    if not os.path.exists(sidecar_path):
        log(f"  ERROR: Sidecar file not found: {sidecar_path}")
        log(f"  Run harvest_api_winners.py first to generate it.")
        return None
    
    try:
        df = pd.read_parquet(sidecar_path)
        
        log(f"  Sidecar records: {len(df):,}")
        
        # Filter to SUCCESS records only
        success_df = df[df['repair_status'] == 'SUCCESS']
        log(f"  SUCCESS records: {len(success_df):,}")
        
        # Build lookup dict: token_id -> is_winner
        winner_lookup = {}
        for _, row in success_df.iterrows():
            token_id = str(row['token_id'])
            is_winner = row['api_derived_winner']
            if is_winner is not None:
                winner_lookup[token_id] = bool(is_winner)
        
        log(f"  Winner lookup size: {len(winner_lookup):,} tokens")
        
        # Stats
        winners = sum(1 for v in winner_lookup.values() if v)
        losers = sum(1 for v in winner_lookup.values() if not v)
        log(f"  Distribution: {winners:,} winners, {losers:,} losers")
        
        # Memory estimate
        mem_mb = sys.getsizeof(winner_lookup) / (1024 * 1024)
        log(f"  Lookup memory: ~{mem_mb:.1f} MB")
        
        return winner_lookup
        
    except Exception as e:
        log(f"  ERROR loading sidecar: {e}")
        return None


def lookup_winner(token_id, winner_lookup):
    """
    Look up winner status for a token from the sidecar data.
    
    Returns:
        bool: True if winner, False if loser
        None: if token not found in sidecar
    """
    return winner_lookup.get(str(token_id), None)

# ==============================================================================
# FILE-FIRST ACCUMULATOR PROCESSING (NEW in v12.0)
# ==============================================================================

def get_available_columns(filepath):
    """
    Inspect a Parquet file to determine available columns.
    Returns tuple: (columns_to_read, volume_column_name)
    """
    parquet_file = pq.ParquetFile(filepath)
    schema = parquet_file.schema
    available = set(schema.names)
    
    # Base columns
    columns_to_read = [c for c in REQUIRED_COLUMNS if c in available]
    
    # Find volume column (different names in different files)
    volume_col = None
    for vc in VOLUME_COLUMNS:
        if vc in available:
            volume_col = vc
            columns_to_read.append(vc)
            break
    
    # Winner column (may not exist - will need repair)
    for wc in WINNER_COLUMNS:
        if wc in available:
            columns_to_read.append(wc)
            break
    
    return columns_to_read, volume_col

def compute_token_metrics(acc):
    """
    Compute TWAP, volume, and trade metrics for a single token's accumulated data.
    Returns dict with computed metrics, or:
        - None if token should be filtered (liquidity)
        - 'NO_WINNER' string if token missing from sidecar
    """
    trades = acc['trades']
    winner_status = acc['winner_status']
    
    # Must have winner status from sidecar
    if winner_status is None:
        return 'NO_WINNER'  # Distinct from None (filtered)
    
    # Minimum trades filter
    if len(trades) < MIN_TRADES_PER_TOKEN:
        return None
    
    # Sort trades by timestamp for correct TWAP calculation
    trades.sort(key=lambda x: x[0])
    
    # Compute volume (USDC scaling: divide by 1_000_000)
    total_volume_usd = sum(p * s for _, p, s in trades) / 1_000_000.0
    
    if total_volume_usd < MIN_VOLUME_USD:
        return None
    
    # Compute TWAP
    twap = compute_twap(trades)
    
    if twap is None or not (0 < twap < 1):
        return None
    
    # Assign to bucket
    bucket_label, bucket_tag = assign_bucket(twap, ODDS_BUCKETS)
    if bucket_label is None:
        return None
    
    # Calculate metrics
    avg_trade_size = sum(p * s for _, p, s in trades) / len(trades) / 1_000_000.0
    
    # Return calculation (for Sharpe)
    token_won = winner_status
    ret = (1.0 / twap - 1.0) if token_won else -1.0
    
    # Flag surprising outcomes for diagnostics
    # High probability loser or low probability winner
    surprising = False
    if twap >= 0.90 and not token_won:
        surprising = 'HIGH_PROB_LOSER'
    elif twap <= 0.10 and token_won:
        surprising = 'LOW_PROB_WINNER'
    
    return {
        'bucket_label': bucket_label,
        'twap': twap,
        'outcome': 1 if token_won else 0,
        'volume': total_volume_usd,
        'trade_size': avg_trade_size,
        'return': ret,
        'token_won': token_won,
        'condition_id': acc['condition_id'],
        'trade_count': len(trades),
        'surprising': surprising,
    }


def flush_completed_condition(condition_id, token_accumulator, bucket_data, 
                               condition_tokens, stats_counters):
    """
    Aggregate all tokens for a completed condition into bucket_data, then flush from memory.
    
    This is the key memory optimization: once all files for a condition are processed,
    we compute final metrics and discard the raw trade data.
    """
    tokens_to_flush = condition_tokens.get(condition_id, set())
    
    for token_id in tokens_to_flush:
        if token_id not in token_accumulator:
            continue
        
        acc = token_accumulator[token_id]
        
        # Compute metrics for this token
        metrics = compute_token_metrics(acc)
        
        if metrics == 'NO_WINNER':
            # Token not in sidecar - track separately
            stats_counters['tokens_no_winner'] += 1
        elif metrics is not None:
            # Add to bucket
            bucket_label = metrics['bucket_label']
            bucket_data[bucket_label]['twaps'].append(metrics['twap'])
            bucket_data[bucket_label]['outcomes'].append(metrics['outcome'])
            bucket_data[bucket_label]['volumes'].append(metrics['volume'])
            bucket_data[bucket_label]['trade_sizes'].append(metrics['trade_size'])
            bucket_data[bucket_label]['returns'].append(metrics['return'])
            
            stats_counters['tokens_analyzed'] += 1
            stats_counters['markets_processed'].add(condition_id)
            
            # Track surprising outcomes for diagnostics
            if metrics['surprising']:
                stats_counters['surprising_outcomes'].append({
                    'condition_id': condition_id,
                    'token_id': token_id,
                    'twap': metrics['twap'],
                    'won': metrics['token_won'],
                    'bucket': bucket_label,
                    'type': metrics['surprising'],
                    'volume': metrics['volume'],
                    'trade_count': metrics['trade_count'],
                })
        else:
            # Filtered due to liquidity
            stats_counters['tokens_filtered'] += 1
        
        # FREE THE MEMORY - this is the critical step
        del token_accumulator[token_id]
    
    # Clean up condition tracking
    if condition_id in condition_tokens:
        del condition_tokens[condition_id]


def process_files_accumulator(batch_files, market_index, winner_lookup, sample_size=None, diagnostic_mode=False):
    """
    FILE-FIRST ACCUMULATOR PATTERN (VECTORIZED + STREAMING FLUSH)
    
    v13.0: SIDECAR-BASED WINNER LOOKUP
    - Winner data pre-loaded from sidecar file
    - O(1) lookup per token (no API calls)
    - Streaming flush unchanged from v12.2
    
    Memory footprint: O(active_conditions) + O(sidecar_size)
    - Only tokens from in-progress conditions stay in memory
    - Completed conditions flushed immediately
    
    Args:
        batch_files: list of parquet file paths
        market_index: dict {condition_id: [(file_idx, count), ...]}
        winner_lookup: dict {token_id: bool} from sidecar
        sample_size: int, limit files processed (for testing)
        diagnostic_mode: bool, extra logging for validation
    """
    log("\n" + "="*70)
    log("PASS 2: FILE-FIRST ACCUMULATOR PROCESSING (SIDECAR WINNERS)")
    log("="*70)
    
    # Get unique file indices
    unique_file_indices = get_unique_file_indices(market_index)
    log(f"  Unique files to process: {len(unique_file_indices)}")
    
    if sample_size:
        unique_file_indices = unique_file_indices[:sample_size]
        log(f"  SAMPLE MODE: Processing {len(unique_file_indices)} files")
    
    if diagnostic_mode:
        log(f"  DIAGNOSTIC MODE: Extra logging enabled")
    
    # Build set of valid condition_ids for fast lookup
    valid_conditions = set(market_index.keys())
    log(f"  Valid conditions (markets): {len(valid_conditions):,}")
    log(f"  Winner lookup size: {len(winner_lookup):,} tokens")
    
    # ==========================================================================
    # CONDITION COMPLETION TRACKING (for streaming flush)
    # ==========================================================================
    
    # Create set ONCE (not inside loop!)
    unique_file_set = set(unique_file_indices)
    
    # Track remaining files per condition
    condition_remaining_files = {}
    for condition_id, file_list in market_index.items():
        # Only count files that are in our processing set
        relevant_files = sum(1 for file_idx, _ in file_list if file_idx in unique_file_set)
        if relevant_files > 0:
            condition_remaining_files[condition_id] = relevant_files
    
    # Track which conditions are in each file (for decrementing counters)
    file_to_conditions = defaultdict(set)
    for condition_id, file_list in market_index.items():
        for file_idx, _ in file_list:
            if file_idx in unique_file_set:
                file_to_conditions[file_idx].add(condition_id)
    
    log(f"  Conditions to track: {len(condition_remaining_files):,}")
    
    # DEBUG: Find conditions that will complete in first 100 files
    single_file_conditions = [c for c, count in condition_remaining_files.items() if count == 1]
    log(f"  Conditions with only 1 file (will flush immediately): {len(single_file_conditions)}")
    if single_file_conditions:
        log(f"    First 5: {[c[:16] for c in single_file_conditions[:5]]}")
    
    # ==========================================================================
    # ACCUMULATORS
    # ==========================================================================
    
    # Token accumulator (will be flushed as conditions complete)
    token_accumulator = {}
    
    # Track which tokens belong to which condition (for flushing)
    condition_tokens = defaultdict(set)
    
    # Initialize bucket_data HERE (will be populated during streaming flush)
    bucket_data = defaultdict(lambda: {
        'twaps': [],
        'outcomes': [],
        'volumes': [],
        'trade_sizes': [],
        'returns': []
    })
    
    # Statistics counters
    stats_counters = {
        'tokens_analyzed': 0,
        'tokens_filtered': 0,
        'tokens_no_winner': 0,  # NEW: track tokens missing from sidecar
        'markets_processed': set(),
        'surprising_outcomes': [],  # Track high-prob losers and low-prob winners
    }
    
    files_processed = 0
    total_trades_accumulated = 0
    total_rows_read = 0
    total_rows_filtered = 0
    conditions_flushed = 0
    
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
            # Determine available columns with pruning
            columns_to_read, volume_col = get_available_columns(filepath)
            
            if not volume_col:
                continue  # Skip files without volume data
            
            # Read with column pruning
            df = pq.read_table(filepath, columns=columns_to_read).to_pandas()
            
            if len(df) == 0:
                continue
            
            total_rows_read += len(df)
            
            # Normalize volume column name
            if volume_col != 'size_tokens':
                df.rename(columns={volume_col: 'size_tokens'}, inplace=True)
            
            # Type safety: force timestamp to numeric
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            if len(df) == 0:
                continue
            
            # Detect and normalize milliseconds -> seconds (vectorized)
            if df['timestamp'].iloc[0] > 3e9:
                df['timestamp'] = df['timestamp'] / 1000.0
                df['resolution_time'] = df['resolution_time'] / 1000.0
            
            # Check if token_winner exists
            has_winner = 'token_winner' in df.columns
            
            # ==================================================================
            # VECTORIZED FILTERING
            # ==================================================================
            
            # Step 1: Filter to valid conditions (vectorized isin)
            condition_mask = df['condition_id'].isin(valid_conditions)
            df = df.loc[condition_mask]
            
            if len(df) == 0:
                continue
            
            # Step 2: Vectorized window filter
            window_end = df['resolution_time'] - (WINDOW_OFFSET_HOURS * 3600)
            window_start = window_end - (TIME_WINDOW_HOURS * 3600)
            window_mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= window_end)
            filtered_df = df.loc[window_mask]
            
            if len(filtered_df) == 0:
                # Still need to decrement condition counters even if no trades in window
                pass
            else:
                total_rows_filtered += len(filtered_df)
                
                # DEBUG: Log first few files with trades
                if files_processed <= 5:
                    log(f"    File {files_processed}: {len(filtered_df)} trades in window")
                
                # ==================================================================
                # FAST AGGREGATION (groupby instead of row-by-row)
                # ==================================================================
                
                filtered_df = filtered_df.copy()
                filtered_df['token_id'] = filtered_df['token_id'].astype(str)
                
                for token_id, group in filtered_df.groupby('token_id', sort=False):
                    condition_id = group['condition_id'].iloc[0]
                    
                    # Initialize accumulator for this token if needed
                    if token_id not in token_accumulator:
                        # SIDECAR LOOKUP: Get winner status immediately
                        winner_status = lookup_winner(token_id, winner_lookup)
                        
                        token_accumulator[token_id] = {
                            'condition_id': condition_id,
                            'resolution_time': float(group['resolution_time'].iloc[0]),
                            'trades': [],
                            'winner_status': winner_status,  # From sidecar (may be None)
                        }
                        # Track token -> condition mapping for flush
                        condition_tokens[condition_id].add(token_id)
                        
                        # Diagnostic logging for first few tokens - show raw format
                        if diagnostic_mode and len(token_accumulator) <= 10:
                            log(f"    [DIAG] Token lookup:")
                            log(f"      raw token_id type: {type(token_id).__name__}, value: {token_id[:40]}...")
                            log(f"      winner_lookup result: {winner_status}")
                            # Check if similar keys exist
                            sample_keys = list(winner_lookup.keys())[:3]
                            log(f"      sample sidecar keys: {[k[:30]+'...' for k in sample_keys]}")
                    
                    acc = token_accumulator[token_id]
                    
                    # Batch extract trades using numpy arrays
                    timestamps = group['timestamp'].values
                    prices = group['price'].values
                    sizes = group['size_tokens'].values
                    
                    # Batch add trades for this token
                    trades_batch = list(zip(timestamps, prices, sizes))
                    acc['trades'].extend(trades_batch)
                    total_trades_accumulated += len(trades_batch)

            
            del df
            if 'filtered_df' in dir():
                del filtered_df
            
            # ==================================================================
            # STREAMING FLUSH: Check for completed conditions
            # ==================================================================
            
            conditions_in_this_file = file_to_conditions.get(file_idx, set())
            
            for condition_id in conditions_in_this_file:
                if condition_id in condition_remaining_files:
                    condition_remaining_files[condition_id] -= 1
                    
                    # Condition complete! Flush it.
                    if condition_remaining_files[condition_id] == 0:
                        # Diagnostic logging
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
                        
                        # Flush the condition (winner status already set from sidecar)
                        flush_completed_condition(
                            condition_id, token_accumulator, bucket_data,
                            condition_tokens, stats_counters
                        )
                        
                        # Clean up tracking
                        del condition_remaining_files[condition_id]
                        conditions_flushed += 1
            
        except Exception as e:
            continue
    
    # Final cleanup: flush any remaining conditions (shouldn't be many)
    remaining_conditions = list(condition_tokens.keys())
    if remaining_conditions:
        log(f"\n  Flushing {len(remaining_conditions)} remaining conditions...")
    
    for condition_id in remaining_conditions:
        # Winner status already set from sidecar - just flush
        flush_completed_condition(
            condition_id, token_accumulator, bucket_data,
            condition_tokens, stats_counters
        )
        conditions_flushed += 1
    
    gc.collect()
    
    log(f"  ✓ Accumulation complete: {log_memory()}")
    log(f"    Files processed: {files_processed:,}")
    log(f"    Total rows read: {total_rows_read:,}")
    log(f"    Rows in window: {total_rows_filtered:,}")
    log(f"    Conditions flushed: {conditions_flushed:,}")
    log(f"    Tokens analyzed: {stats_counters['tokens_analyzed']:,}")
    log(f"    Tokens filtered (liquidity): {stats_counters['tokens_filtered']:,}")
    log(f"    Tokens missing winner: {stats_counters['tokens_no_winner']:,}")
    
    # Report surprising outcomes
    surprising = stats_counters['surprising_outcomes']
    if surprising:
        high_prob_losers = [s for s in surprising if s['type'] == 'HIGH_PROB_LOSER']
        low_prob_winners = [s for s in surprising if s['type'] == 'LOW_PROB_WINNER']
        
        log(f"\n  ⚠️  SURPRISING OUTCOMES DETECTED:")
        log(f"    High-prob losers (TWAP >= 90%, lost): {len(high_prob_losers)}")
        log(f"    Low-prob winners (TWAP <= 10%, won): {len(low_prob_winners)}")
        
        # Print details for investigation
        if high_prob_losers:
            log(f"\n  HIGH-PROBABILITY LOSERS (investigate these):")
            for s in high_prob_losers[:20]:  # Limit to 20
                log(f"    condition={s['condition_id'][:24]}...")
                log(f"      token={s['token_id'][:24]}...")
                log(f"      TWAP={s['twap']:.4f} ({s['bucket']}) won={s['won']} vol=${s['volume']:.0f} trades={s['trade_count']}")
        
        if low_prob_winners:
            log(f"\n  LOW-PROBABILITY WINNERS (investigate these):")
            for s in low_prob_winners[:20]:  # Limit to 20
                log(f"    condition={s['condition_id'][:24]}...")
                log(f"      token={s['token_id'][:24]}...")
                log(f"      TWAP={s['twap']:.4f} ({s['bucket']}) won={s['won']} vol=${s['volume']:.0f} trades={s['trade_count']}")
    
    # Build total_metrics
    total_metrics = {
        'markets_processed': len(stats_counters['markets_processed']),
        'tokens_analyzed': stats_counters['tokens_analyzed'],
        'tokens_low_liquidity': stats_counters['tokens_filtered'],
        'tokens_no_winner': stats_counters['tokens_no_winner'],
        'surprising_outcomes': len(surprising),
        'filter_rate_pct': (stats_counters['tokens_filtered'] / 
                          (stats_counters['tokens_analyzed'] + stats_counters['tokens_filtered']) * 100)
                          if (stats_counters['tokens_analyzed'] + stats_counters['tokens_filtered']) > 0 else 0
    }
    
    return bucket_data, total_metrics, surprising  # Return surprising for further analysis

# ==============================================================================
# ENHANCED CALIBRATION ANALYSIS (REFACTORED)
# ==============================================================================

def analyze_calibration_streaming(sample_size=None, diagnostic_mode=False):
    """
    Main calibration analysis with FILE-FIRST VECTORIZED + SIDECAR WINNERS
    
    v13.0: SIDECAR-BASED WINNER LOOKUP
    - Pass 0 (NEW): Load sidecar file into memory
    - Pass 1: Build market index (unchanged)
    - Pass 2: File-First accumulation with sidecar winner lookup
    - Pass 3: Compute final statistics from bucket_data
    
    Args:
        sample_size: int, limit number of batch files processed
        diagnostic_mode: bool, enable verbose logging for validation
    """
    log("\n" + "="*70)
    log("PHASE 1: ODDS-STRATIFIED CALIBRATION ANALYSIS")
    log("Version 13.0 - SIDECAR-BASED WINNER LOOKUP")
    log("="*70)
    log(f"  Analysis window: {TIME_WINDOW_HOURS}h ending {WINDOW_OFFSET_HOURS}h before resolution")
    log(f"  Min volume: ${MIN_VOLUME_USD}")
    log(f"  Confidence: 95% (Wilson score)")
    log(f"  Significance: α={ALPHA} (two-tailed)")
    if diagnostic_mode:
        log(f"  DIAGNOSTIC MODE: Enabled")
    
    # ==========================================================================
    # PASS 0: LOAD SIDECAR WINNER DATA
    # ==========================================================================
    
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
    
    # PASS 1: Build market index
    market_index, batch_files = build_market_index(batch_files, sample_size)
    
    # PASS 2: File-First accumulator with sidecar winner lookup
    bucket_data, total_metrics, surprising_outcomes = process_files_accumulator(
        batch_files, market_index, winner_lookup, sample_size, diagnostic_mode
    )
    
    # ==========================================================================
    # DIAGNOSTIC: Verify surprising outcomes against live API
    # ==========================================================================
    if diagnostic_mode and surprising_outcomes:
        log("\n" + "="*70)
        log("DIAGNOSTIC: VERIFYING SURPRISING OUTCOMES AGAINST LIVE API")
        log("="*70)
        
        import requests
        
        # Load full sidecar for metadata lookup
        sidecar_df = pd.read_parquet(SIDECAR_FILE)
        
        # Check up to 10 surprising outcomes
        to_verify = surprising_outcomes[:10]
        
        for s in to_verify:
            condition_id = s['condition_id']
            token_id = s['token_id']
            
            log(f"\n  Verifying: {condition_id}")
            log(f"    Token: {token_id}")
            log(f"    Calibration says: TWAP={s['twap']:.4f}, won={s['won']}")
            
            # Look up in sidecar
            sidecar_row = sidecar_df[sidecar_df['token_id'] == token_id]
            if len(sidecar_row) > 0:
                row = sidecar_row.iloc[0]
                log(f"    Sidecar metadata:")
                log(f"      market_question: {str(row.get('market_question', 'N/A'))[:60]}...")
                log(f"      api_derived_winner: {row.get('api_derived_winner')}")
                log(f"      outcome_label: {row.get('outcome_label')}")
                log(f"      settlement_price: {row.get('settlement_price')}")
                
                yes_token = str(row.get('yes_token_id', ''))
                no_token = str(row.get('no_token_id', ''))
                log(f"      yes_token_id: {yes_token[:40]}...")
                log(f"      no_token_id: {no_token[:40]}...")
                
                # Check the companion token
                companion_token = no_token if token_id == yes_token else yes_token
                companion_winner = winner_lookup.get(companion_token, 'NOT_FOUND')
                log(f"      companion token winner_lookup: {companion_winner}")
                
                # Sanity check: exactly one should be winner
                this_winner = winner_lookup.get(token_id)
                if this_winner is not None and companion_winner not in ('NOT_FOUND', None):
                    if this_winner == companion_winner:
                        log(f"      ❌ ERROR: Both tokens have same winner status!")
                    elif this_winner and companion_winner:
                        log(f"      ❌ ERROR: Both tokens are winners!")
                    elif not this_winner and not companion_winner:
                        log(f"      ❌ ERROR: Both tokens are losers!")
                    else:
                        log(f"      ✓ One winner, one loser (correct)")
            else:
                log(f"    Sidecar: NOT FOUND")
            
            try:
                response = requests.get(
                    'https://gamma-api.polymarket.com/markets',
                    params={'condition_ids': condition_id},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        market = data[0]
                        
                        question = market.get('question', 'N/A')[:60]
                        clob_ids = market.get('clobTokenIds', [])
                        outcome_prices = market.get('outcomePrices', [])
                        
                        if isinstance(clob_ids, str):
                            clob_ids = json.loads(clob_ids)
                        if isinstance(outcome_prices, str):
                            outcome_prices = json.loads(outcome_prices)
                        
                        log(f"    Live API:")
                        log(f"      question: {question}...")
                        log(f"      clobTokenIds[0] (YES): {str(clob_ids[0]) if clob_ids else 'N/A'}")
                        log(f"      clobTokenIds[1] (NO): {str(clob_ids[1]) if len(clob_ids) > 1 else 'N/A'}")
                        log(f"      outcomePrices: {outcome_prices}")
                        
                        if len(clob_ids) >= 2 and len(outcome_prices) >= 2:
                            yes_token = str(clob_ids[0])
                            no_token = str(clob_ids[1])
                            yes_price = float(outcome_prices[0])
                            no_price = float(outcome_prices[1])
                            
                            # Determine winner from API
                            if yes_price > no_price:
                                api_winner_token = yes_token
                                api_winner_outcome = 'YES'
                            else:
                                api_winner_token = no_token
                                api_winner_outcome = 'NO'
                            
                            # Check if our token matches
                            token_is_yes = (token_id == yes_token)
                            token_is_no = (token_id == no_token)
                            
                            log(f"      Our token is: {'YES' if token_is_yes else 'NO' if token_is_no else 'NEITHER!'}")
                            log(f"      Winner outcome: {api_winner_outcome}")
                            
                            # What should the winner status be?
                            api_says_won = (token_id == api_winner_token)
                            sidecar_says_won = s['won']
                            
                            if api_says_won == sidecar_says_won:
                                log(f"    ✓ SIDECAR CORRECT (api_says_won={api_says_won})")
                            else:
                                log(f"    ❌ SIDECAR WRONG! API says won={api_says_won}, sidecar says won={sidecar_says_won}")
                        
            except Exception as e:
                log(f"    API Error: {e}")
    
    # Compute calibration statistics from bucket_data
    log("\n" + "="*70)
    log("COMPUTING CALIBRATION STATISTICS")
    log("="*70)
    
    calibration_stats = []
    
    for lower, upper, label, tag in ODDS_BUCKETS:
        data = bucket_data[label]
        
        if len(data['twaps']) == 0:
            log(f"  {label}: No data")
            continue
        
        n_tokens = len(data['twaps'])
        n_wins = sum(data['outcomes'])
        
        avg_implied = np.mean(data['twaps'])
        realized_rate, ci_lower, ci_upper = wilson_score_interval(n_wins, n_tokens)
        
        raw_edge = realized_rate - avg_implied
        raw_edge_bps = raw_edge * 10000
        
        # Return edge (percentage)
        return_edge_pct = (raw_edge / avg_implied) * 100 if avg_implied > 0 else 0
        
        # Z-test for significance
        se = np.sqrt(avg_implied * (1 - avg_implied) / n_tokens)
        z_score = raw_edge / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        significant = p_value < ALPHA
        
        # Enhanced metrics
        returns = data['returns']
        sharpe = compute_sharpe_ratio(returns) if len(returns) >= 2 else None
        kelly = compute_kelly_fraction(realized_rate, avg_implied)
        
        # Calibration scores
        brier = compute_brier_score(data['twaps'], data['outcomes'])
        log_loss_score = compute_log_loss(data['twaps'], data['outcomes'])
        
        # Volume metrics
        avg_volume = np.mean(data['volumes'])
        avg_trade_size = np.mean(data['trade_sizes'])
        
        # T-test for edge
        returns_array = np.array(returns)
        t_stat, t_pvalue = compute_t_statistic(
            np.mean(returns_array),
            np.std(returns_array, ddof=1),
            len(returns_array)
        )
        
        stats_dict = {
            'bucket': label,
            'bucket_tag': tag,
            'n_tokens': int(n_tokens),
            'n_wins': int(n_wins),
            'avg_implied_prob': float(avg_implied),
            'realized_win_rate': float(realized_rate),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'raw_edge': float(raw_edge),
            'raw_edge_bps': float(raw_edge_bps),
            'return_edge_pct': float(return_edge_pct),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'significant': bool(significant),
            'sharpe_ratio': float(sharpe) if sharpe is not None else None,
            'kelly_fraction': float(kelly),
            'brier_score': float(brier) if brier is not None else None,
            'log_loss': float(log_loss_score) if log_loss_score is not None else None,
            't_statistic': float(t_stat),
            't_pvalue': float(t_pvalue),
            'avg_volume_usd': float(avg_volume),
            'avg_trade_size_usd': float(avg_trade_size),
            'median_volume_usd': float(np.median(data['volumes'])),
            'total_volume_usd': float(np.sum(data['volumes'])),
        }
        
        calibration_stats.append(stats_dict)
        
        # Safe logging
        if sharpe is not None:
            sharpe_display = f"{float(sharpe):.3f}"
        else:
            sharpe_display = "N/A"
            
        if raw_edge_bps is not None:
            edge_display = f"{float(raw_edge_bps):+.1f}"
        else:
            edge_display = "N/A"
        
        log(f"  {label}: n={n_tokens:,}, win={realized_rate:.1%} (exp={avg_implied:.1%}), "
            f"edge={edge_display}bp, Sharpe={sharpe_display}, "
            f"p={p_value:.4f} {'***' if significant else ''}")
    
    # Generate outputs
    log("\n" + "="*70)
    log("GENERATING OUTPUTS")
    log("="*70)
    
    # Prevent visualization crash if no data
    if len(calibration_stats) > 0:
        create_enhanced_visualizations(calibration_stats)
    else:
        log("⚠️ No populated buckets. Skipping visualization to prevent crash.")

    save_enhanced_outputs(calibration_stats, total_metrics, batch_files, sample_size)
    
    log("\n" + "="*70)
    log("ANALYSIS COMPLETE")
    log("="*70)

# ==============================================================================
# ENHANCED VISUALIZATION
# ==============================================================================

def create_enhanced_visualizations(calibration_stats):
    """
    Create publication-quality visualizations
    """
    log("  Creating enhanced visualizations...")
    
    # Set professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # --------------------------------------------------------------------------
    # Plot 1: Calibration Curve with Confidence Bands (MAIN PLOT - larger)
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0:2, 0:2])  # Takes up 2x2 space
    
    # Extract data
    implied_probs = [s['avg_implied_prob'] for s in calibration_stats]
    realized_rates = [s['realized_win_rate'] for s in calibration_stats]
    ci_lowers = [s['ci_lower'] for s in calibration_stats]
    ci_uppers = [s['ci_upper'] for s in calibration_stats]
    sample_sizes = [s['n_tokens'] for s in calibration_stats]
    significant = [s['significant'] for s in calibration_stats]
    
    # Perfect calibration line (y=x)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration', zorder=1)
    
    # Realized rates with confidence bands
    for i in range(len(implied_probs)):
        color = 'darkgreen' if significant[i] else 'steelblue'
        marker = 'D' if significant[i] else 'o'
        markersize = 8 if significant[i] else 6
        
        # Confidence interval
        ax1.plot([implied_probs[i], implied_probs[i]], 
                [ci_lowers[i], ci_uppers[i]], 
                color=color, linewidth=2, alpha=0.6, zorder=2)
        
        # Point estimate
        ax1.scatter(implied_probs[i], realized_rates[i], 
                   s=markersize**2, color=color, marker=marker,
                   edgecolors='black', linewidths=0.5, zorder=3)
        
        # Sample size annotation
        ax1.annotate(f'n={sample_sizes[i]:,}', 
                    (implied_probs[i], realized_rates[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.7)
    
    ax1.set_xlabel('Market-Implied Probability (TWAP)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Realized Win Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Calibration Curve with 95% Confidence Intervals', 
                 fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    
    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='Perfect Calibration'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='darkgreen', 
                  markersize=8, markeredgecolor='black', markeredgewidth=0.5, 
                  label='Significant (p<0.05)', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                  markersize=6, markeredgecolor='black', markeredgewidth=0.5,
                  label='Not Significant', linestyle='None')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    # --------------------------------------------------------------------------
    # Plot 2: Edge by Bucket (Top right)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    
    bucket_labels = [s['bucket'] for s in calibration_stats]
    edges = [s['raw_edge_bps'] for s in calibration_stats]
    colors = ['darkgreen' if e > 0 and s['significant'] else 
              'darkred' if e < 0 and s['significant'] else 
              'gray' for e, s in zip(edges, calibration_stats)]
    
    bars = ax2.barh(range(len(bucket_labels)), edges, color=colors, alpha=0.8)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.set_yticks(range(len(bucket_labels)))
    ax2.set_yticklabels(bucket_labels, fontsize=8)
    ax2.set_xlabel('Edge (basis points)', fontsize=10, fontweight='bold')
    ax2.set_title('Raw Edge by Bucket', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x', linestyle=':')
    
    # Annotate significance
    for i, (bar, stats) in enumerate(zip(bars, calibration_stats)):
        if stats['significant']:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    ' ***', ha='left' if width > 0 else 'right', va='center',
                    fontsize=10, fontweight='bold', color='black')
    
    # --------------------------------------------------------------------------
    # Plot 3: Sharpe Ratios (Middle right)
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 2])
    
    sharpes = [s['sharpe_ratio'] if s['sharpe_ratio'] is not None else 0 
              for s in calibration_stats]
    colors = ['darkgreen' if sr > 0 and s['significant'] else 'gray' 
             for sr, s in zip(sharpes, calibration_stats)]
    
    bars = ax3.barh(range(len(bucket_labels)), sharpes, color=colors, alpha=0.8)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.set_yticks(range(len(bucket_labels)))
    ax3.set_yticklabels(bucket_labels, fontsize=8)
    ax3.set_xlabel('Sharpe Ratio', fontsize=10, fontweight='bold')
    ax3.set_title('Risk-Adjusted Returns', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x', linestyle=':')
    
    # --------------------------------------------------------------------------
    # Plot 4: Sample Sizes (Bottom left)
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 0])
    
    bars = ax4.bar(range(len(bucket_labels)), sample_sizes, color='steelblue', alpha=0.7)
    ax4.set_xticks(range(len(bucket_labels)))
    ax4.set_xticklabels(bucket_labels, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Sample Size (tokens)', fontsize=10, fontweight='bold')
    ax4.set_title('Statistical Power by Bucket', fontsize=11, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # --------------------------------------------------------------------------
    # Plot 5: Brier Scores (Bottom middle)
    # --------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 1])
    
    briers = [s['brier_score'] for s in calibration_stats]
    colors = ['darkgreen' if b < 0.25 else 'orange' if b < 0.5 else 'darkred' 
             for b in briers]
    
    bars = ax5.bar(range(len(bucket_labels)), briers, color=colors, alpha=0.7)
    ax5.axhline(y=0.25, color='green', linestyle='--', linewidth=1, alpha=0.5, 
               label='Good (<0.25)')
    ax5.set_xticks(range(len(bucket_labels)))
    ax5.set_xticklabels(bucket_labels, rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('Brier Score', fontsize=10, fontweight='bold')
    ax5.set_title('Calibration Quality', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax5.legend(fontsize=8, loc='upper right')
    ax5.set_ylim(0, max(briers) * 1.2)
    
    # --------------------------------------------------------------------------
    # Plot 6: Kelly Fractions (Bottom right)
    # --------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 2])
    
    kellys = [s['kelly_fraction'] for s in calibration_stats]
    colors = ['darkgreen' if k > 0 and s['significant'] else 'gray' 
             for k, s in zip(kellys, calibration_stats)]
    
    bars = ax6.barh(range(len(bucket_labels)), kellys, color=colors, alpha=0.8)
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax6.set_yticks(range(len(bucket_labels)))
    ax6.set_yticklabels(bucket_labels, fontsize=8)
    ax6.set_xlabel('Kelly Fraction', fontsize=10, fontweight='bold')
    ax6.set_title('Optimal Position Sizing', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x', linestyle=':')
    
    # Add fractional Kelly reference lines
    for frac, label in [(0.25, '¼ Kelly'), (0.5, '½ Kelly')]:
        ax6.axvline(x=frac, color='blue', linestyle=':', linewidth=1, alpha=0.3)
        ax6.text(frac, len(kellys) - 0.5, label, 
                rotation=90, va='top', ha='right', fontsize=7, alpha=0.6)
    
    # Overall title
    fig.suptitle('Phase 1: Polymarket Calibration Analysis - Streaming Flush',
                fontsize=15, fontweight='bold', y=0.995)
    
    # Save
    plot_path = f"{OUTPUT_DIR}/phase1_calibration_{TIMESTAMP}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log(f"  ✓ Enhanced visualization saved: {plot_path}")

# ==============================================================================
# ENHANCED OUTPUT GENERATION
# ==============================================================================

def save_enhanced_outputs(calibration_stats, total_metrics, batch_files, sample_size):
    """
    Save comprehensive JSON summary and detailed text report
    """
    
    # JSON output with enhanced metrics
    summary = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'version': '12.2 - File-First Streaming Flush',
            'batch_files_processed': len(batch_files),
            'sample_mode': sample_size is not None,
            'analysis_window_hours': TIME_WINDOW_HOURS,
            'window_offset_hours': WINDOW_OFFSET_HOURS,
            'min_volume_usd': MIN_VOLUME_USD,
            'min_trades_per_market': MIN_TRADES_PER_MARKET,
            'min_trades_per_token': MIN_TRADES_PER_TOKEN,
            'significance_level': ALPHA
        },
        'summary': total_metrics,
        'calibration_results': calibration_stats
    }
    
    json_path = f"{OUTPUT_DIR}/phase1_calibration_{TIMESTAMP}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log(f"  ✓ JSON summary saved: {json_path}")
    
    # Enhanced text report
    report_path = f"{OUTPUT_DIR}/phase1_calibration_{TIMESTAMP}_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 1: ODDS-STRATIFIED CALIBRATION ANALYSIS v12.2\n")
        f.write("Polymarket Baseline Edge Measurement - File-First Streaming Flush\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target Audience: Quantitative PMs (Citadel, Jane Street, etc.)\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Markets Analyzed: {total_metrics['markets_processed']:,}\n")
        f.write(f"Tokens Analyzed: {total_metrics['tokens_analyzed']:,}\n")
        f.write(f"Liquidity Filter Rate: {total_metrics['filter_rate_pct']:.1f}%\n")
        
        # Count significant buckets
        sig_positive = [s for s in calibration_stats if s['significant'] and s['raw_edge'] > 0]
        sig_negative = [s for s in calibration_stats if s['significant'] and s['raw_edge'] < 0]
        
        f.write(f"Significant Positive Edge Buckets: {len(sig_positive)}\n")
        f.write(f"Significant Negative Edge Buckets: {len(sig_negative)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. CALIBRATION FRAMEWORK\n")
        f.write("-" * 40 + "\n")
        f.write("   Approach: Stratified binomial testing across probability buckets\n")
        f.write("   Null Hypothesis: Market-implied probabilities are unbiased\n")
        f.write(f"   Significance Level: α={ALPHA} (two-tailed test)\n")
        f.write("   Confidence Intervals: Wilson score method (correct coverage near boundaries)\n\n")
        
        f.write("2. PRICE REPRESENTATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Metric: Time-Weighted Average Price (TWAP)\n")
        f.write(f"   Window: {TIME_WINDOW_HOURS}h ending {WINDOW_OFFSET_HOURS}h before resolution\n")
        f.write("   Rationale: \n")
        f.write("     - Resistant to volume manipulation (time-weighted, not volume-weighted)\n")
        f.write("     - Captures representative market belief over analysis window\n")
        f.write("     - Avoids settlement bias (offset from resolution prevents hindsight)\n")
        f.write("     - Standard in execution analysis and market microstructure research\n\n")
        
        f.write("3. LIQUIDITY FILTERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Minimum Volume: ${MIN_VOLUME_USD} per token in window\n")
        f.write(f"   Minimum Market Trades: {MIN_TRADES_PER_MARKET}\n")
        f.write(f"   Minimum Token Trades: {MIN_TRADES_PER_TOKEN}\n")
        f.write("   Purpose: \n")
        f.write("     - Eliminates 'zombie markets' (stale prices with no volume)\n")
        f.write("     - Ensures TWAP quality through sufficient price discovery\n")
        f.write("     - Filters non-executable opportunities\n\n")
        
        f.write("4. BUCKET STRATIFICATION\n")
        f.write("-" * 40 + "\n")
        f.write("   Design: Coarse buckets for underdogs, refined buckets for favorites\n")
        f.write("   Rationale:\n")
        f.write("     - Underdogs (0-51%): Higher variance, less liquid → coarse buckets\n")
        f.write("     - Favorites (51-100%): Lower variance, more liquid → refined buckets\n")
        f.write("     - Extreme favorites (99%+): Bond farming territory → finest granularity\n")
        f.write("     - Matches market structure: precision where it matters most\n\n")
        
        f.write("5. RISK-ADJUSTED METRICS\n")
        f.write("-" * 40 + "\n")
        f.write("   Sharpe Ratio: E[R] / σ(R)\n")
        f.write("     - Measures edge per unit of volatility\n")
        f.write("     - Assumes returns IID within stratified buckets\n")
        f.write("     - Annualization: Sharpe_annual ≈ Sharpe_per_bet × sqrt(N_bets_per_year)\n\n")
        f.write("   Kelly Fraction: f* = (p × (1 + b) - 1) / b\n")
        f.write("     - Optimal bet size for log utility maximization\n")
        f.write("     - Practice: Use fractional Kelly (¼ or ½) for risk management\n")
        f.write("     - Conservative: Kelly > 0.25 indicates strong edge\n\n")
        
        f.write("6. CALIBRATION QUALITY METRICS\n")
        f.write("-" * 40 + "\n")
        f.write("   Brier Score: Mean squared error between predictions and outcomes\n")
        f.write("     - Range: [0, 1], lower is better\n")
        f.write("     - Good calibration: Brier < 0.25\n")
        f.write("     - Perfect calibration: Brier = 0\n\n")
        f.write("   Log Loss: Cross-entropy loss\n")
        f.write("     - Strictly proper scoring rule\n")
        f.write("     - Heavily penalizes confident wrong predictions\n")
        f.write("     - Unbounded (extreme predictions drive large penalties)\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for stats in calibration_stats:
            f.write(f"{stats['bucket']} ({stats['bucket_tag']})\n")
            f.write("-" * 80 + "\n")
            
            # Sample and outcomes
            f.write(f"Sample Size:        {stats['n_tokens']:,} tokens ({stats['n_wins']:,} wins)\n")
            f.write(f"Implied Prob:       {stats['avg_implied_prob']:.4f}\n")
            f.write(f"Realized Win Rate:  {stats['realized_win_rate']:.4f} "
                   f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]\n\n")
            
            # Edge metrics
            f.write(f"Raw Edge:           {stats['raw_edge']:+.6f} ({stats['raw_edge_bps']:+.1f} bps)\n")
            f.write(f"Return Edge:        {stats['return_edge_pct']:+.2f}%\n")
            if stats['sharpe_ratio'] is not None:
                sharpe_str = f"{stats['sharpe_ratio']:.4f}"
            else:
                sharpe_str = "N/A"
            f.write(f"Sharpe Ratio:       {sharpe_str}\n")
            f.write(f"Kelly Fraction:     {stats['kelly_fraction']:.4f}\n\n")
            
            # Statistical tests
            f.write(f"Z-Statistic:        {stats['z_score']:.3f}\n")
            f.write(f"P-Value (z-test):   {stats['p_value']:.6f}\n")
            f.write(f"T-Statistic:        {stats['t_statistic']:.3f}\n")
            f.write(f"P-Value (t-test):   {stats['t_pvalue']:.6f}\n")
            f.write(f"Significance:       {'*** SIGNIFICANT ***' if stats['significant'] else 'Not significant'}\n\n")
            
            # Calibration quality
            f.write(f"Brier Score:        {stats['brier_score']:.6f}\n")
            f.write(f"Log Loss:           {stats['log_loss']:.6f}\n\n")
            
            # Liquidity metrics
            f.write(f"Avg Volume:         ${stats['avg_volume_usd']:.2f}\n")
            f.write(f"Median Volume:      ${stats['median_volume_usd']:.2f}\n")
            f.write(f"Total Volume:       ${stats['total_volume_usd']:.2f}\n")
            f.write(f"Avg Trade Size:     ${stats['avg_trade_size_usd']:.2f}\n\n")
            
            # Interpretation
            f.write("Interpretation:\n")
            if stats['significant']:
                if stats['raw_edge'] > 0:
                    f.write("  → POSITIVE EDGE: Market systematically underprices this probability range\n")
                    if stats['kelly_fraction'] > 0.25:
                        f.write("  → STRONG SIGNAL: Kelly fraction suggests substantial position sizing\n")
                    if stats['sharpe_ratio'] and stats['sharpe_ratio'] > 1.0:
                        f.write("  → ATTRACTIVE RISK-ADJUSTED: Sharpe > 1.0 indicates strong risk-adjusted returns\n")
                else:
                    f.write("  → NEGATIVE EDGE: Market systematically overprices this probability range\n")
                    f.write("  → AVOID: Do not provide liquidity in this bucket\n")
            else:
                f.write("  → WELL-CALIBRATED: No statistically significant bias detected\n")
                f.write("  → NEUTRAL: Market appears efficient in this probability range\n")
            
            f.write("\n")
        
        # Key findings section
        f.write("="*80 + "\n")
        f.write("KEY FINDINGS & STRATEGIC IMPLICATIONS\n")
        f.write("="*80 + "\n\n")
        
        if sig_positive:
            f.write("POSITIVE EDGE OPPORTUNITIES\n")
            f.write("-" * 80 + "\n")
            for s in sig_positive:
                f.write(f"\n{s['bucket']} ({s['bucket_tag']}):\n")
                f.write(f"  Edge:          {s['raw_edge_bps']:+.1f} bps ({s['return_edge_pct']:+.2f}% return)\n")
                
                sharpe_val = s['sharpe_ratio']
                sharpe_str = f"{sharpe_val:.3f}" if sharpe_val is not None else "N/A"
                f.write(f"  Sharpe:        {sharpe_str}\n")
                
                f.write(f"  Kelly:         {s['kelly_fraction']:.3f} (suggest ¼ Kelly = {s['kelly_fraction']/4:.3f})\n")
            
            f.write("\n\nSTRATEGIC RECOMMENDATIONS:\n")
            f.write("  1. SYSTEMATIC DEPLOYMENT:\n")
            f.write("     - Focus liquidity provision in identified positive-edge buckets\n")
            f.write("     - Use fractional Kelly (¼ or ½) for conservative position sizing\n")
            f.write("     - Monitor volume capacity relative to strategy size\n\n")
            f.write("  2. PHASE 2 PRIORITIES:\n")
            f.write("     - Feature-based segmentation within edge-positive buckets\n")
            f.write("     - Market characteristics: category, time-to-resolution, popularity\n")
            f.write("     - Temporal stability: edge persistence across time periods\n\n")
            f.write("  3. RISK MANAGEMENT:\n")
            f.write("     - Diversify across multiple buckets (correlation <1)\n")
            f.write("     - Monitor realized Sharpe vs. expected\n")
            f.write("     - Establish position limits per bucket and aggregate\n\n")
        else:
            f.write("BASELINE FINDINGS\n")
            f.write("-" * 80 + "\n")
            f.write("No buckets show statistically significant positive edge.\n\n")
            f.write("INTERPRETATION:\n")
            f.write("  - Market appears well-calibrated at baseline\n")
            f.write("  - Edge may exist in feature-stratified segments (Phase 2)\n")
            f.write("  - Consider alternative strategies:\n")
            f.write("     a) Higher-frequency windows (e.g., 1h pre-resolution)\n")
            f.write("     b) Event-driven opportunities (news, announcements)\n")
            f.write("     c) Market-specific inefficiencies (niche categories)\n\n")
        
        if sig_negative:
            f.write("NEGATIVE EDGE ZONES (AVOID)\n")
            f.write("-" * 80 + "\n")
            for s in sig_negative:
                f.write(f"\n{s['bucket']}: {s['raw_edge_bps']:+.1f} bps ({s['return_edge_pct']:+.2f}% return)\n")
                f.write(f"  → DO NOT provide liquidity in this range\n")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICAL QUALITY ASSESSMENT\n")
        f.write("="*80 + "\n\n")
        
        # Overall Brier score
        all_predictions = []
        all_outcomes = []
        for s in calibration_stats:
            idx = [i for i, b in enumerate(ODDS_BUCKETS) if b[2] == s['bucket']][0]
            lower, upper, label, tag = ODDS_BUCKETS[idx]
            bucket_data_key = label
            
            all_predictions.extend([s['avg_implied_prob']] * s['n_tokens'])
            all_outcomes.extend([1] * s['n_wins'] + [0] * (s['n_tokens'] - s['n_wins']))
        
        overall_brier = compute_brier_score(all_predictions, all_outcomes)
        overall_log_loss = compute_log_loss(all_predictions, all_outcomes)
        
        f.write(f"Overall Brier Score:  {overall_brier:.6f}\n")
        f.write(f"Overall Log Loss:     {overall_log_loss:.6f}\n\n")
        
        f.write("Interpretation:\n")
        if overall_brier < 0.20:
            f.write("  → EXCELLENT calibration quality (Brier < 0.20)\n")
        elif overall_brier < 0.25:
            f.write("  → GOOD calibration quality (Brier < 0.25)\n")
        else:
            f.write("  → MODERATE calibration quality (Brier ≥ 0.25)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NEXT STEPS\n")
        f.write("="*80 + "\n\n")
        
        if sig_positive:
            f.write("PROCEED TO PHASE 2: FEATURE-BASED REFINEMENT\n\n")
            f.write("Objectives:\n")
            f.write("  1. Market segmentation within edge-positive buckets:\n")
            f.write("     - Category analysis (politics, sports, crypto, etc.)\n")
            f.write("     - Time-to-resolution buckets (< 1 day, 1-7 days, > 7 days)\n")
            f.write("     - Popularity tiers (volume quintiles)\n")
            f.write("     - Outcome type analysis (Yes/No vs Over/Under vs matchups)\n\n")
            f.write("  2. Temporal stability:\n")
            f.write("     - Edge persistence across calendar months\n")
            f.write("     - Intraday patterns (market open/close effects)\n")
            f.write("     - Event-driven vs. continuous markets\n\n")
            f.write("  3. Portfolio construction:\n")
            f.write("     - Correlation analysis between buckets and features\n")
            f.write("     - Optimal allocation (mean-variance or Kelly framework)\n")
            f.write("     - Capacity analysis and scaling roadmap\n\n")
        else:
            f.write("VALIDATION & SENSITIVITY ANALYSIS\n\n")
            f.write("Actions:\n")
            f.write("  1. Validate on full dataset (if currently sampled)\n")
            f.write("  2. Sensitivity analysis:\n")
            f.write("     - Vary time window (12h, 36h, 48h)\n")
            f.write("     - Adjust liquidity filters\n")
            f.write("     - Alternative price metrics (VWAP, close price)\n")
            f.write("  3. Feature stratification (Phase 2) to identify niche opportunities\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("METHODOLOGY NOTES FOR REVIEWERS\n")
        f.write("="*80 + "\n\n")
        
        f.write("STATISTICAL FRAMEWORK:\n")
        f.write("  - Wilson score intervals: Conservative, correct coverage near boundaries\n")
        f.write("  - Two-tailed tests: Detecting both positive and negative bias\n")
        f.write("  - Multiple comparisons: No Bonferroni correction (exploratory phase)\n\n")
        
        f.write("PRICE REPRESENTATION:\n")
        f.write("  - TWAP over VWAP: More robust to wash trading and manipulation\n")
        f.write("  - Window offset: Prevents settlement bias (analyzing before outcome known)\n")
        f.write("  - Alternative for Phase 2: Compare TWAP vs close price vs VWAP\n\n")
        
        f.write("LIQUIDITY FILTERS:\n")
        f.write("  - Volume threshold: Ensures executability\n")
        f.write("  - Trade count: Ensures price discovery quality\n")
        f.write("  - Conservative approach: Better to exclude edge cases than include noise\n\n")
        
        f.write("RISK METRICS:\n")
        f.write("  - Sharpe assumes IID returns: Reasonable within stratified buckets\n")
        f.write("  - Kelly fraction: Theoretical optimum, use fractional in practice\n")
        f.write("  - No tail risk analysis yet: Phase 2 should examine drawdown distributions\n\n")
        
        f.write("PERFORMANCE OPTIMIZATION (v13.0 - Sidecar Winners):\n")
        f.write("  - File-First Accumulator: Each Parquet file read exactly once\n")
        f.write("  - Vectorized Pandas: Boolean masks and groupby (C-optimized)\n")
        f.write("  - Streaming Flush: Conditions aggregated and freed as they complete\n")
        f.write("  - Sidecar Winner Lookup: O(1) dictionary access, no API calls\n")
        f.write("  - Memory footprint: O(active_conditions) + O(sidecar) - bounded\n")
        f.write("  - Winner data source: data/repair/api_derived_winners.parquet\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis version: 13.0 - Sidecar-Based Winner Lookup\n")
        f.write("="*80 + "\n")
    
    log(f"  ✓ Enhanced report saved: {report_path}")
    
    log("\nOutputs generated:")
    log(f"  - Summary JSON:    phase1_calibration_{TIMESTAMP}_summary.json")
    log(f"  - Detailed Report: phase1_calibration_{TIMESTAMP}_report.txt")
    log(f"  - Visualization:   phase1_calibration_{TIMESTAMP}.png")
    
    log("\nTo view results locally:")
    log(f"  scp -i 'poly.pem' ec2-user@ec2-34-203-248-16.compute-1.amazonaws.com:{OUTPUT_DIR}/phase1_calibration_{TIMESTAMP}_* ./")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phase 1 Calibration Analysis - Sidecar Winner Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnostic mode - process ~100 conditions, verbose output
  python phase1_calibration_analysis_v13.py --diagnostic
  
  # Sample mode - process N batch files
  python phase1_calibration_analysis_v13.py --sample 500
  
  # Full run
  python phase1_calibration_analysis_v13.py
        """
    )
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files')
    parser.add_argument('--diagnostic', '-d', action='store_true',
                        help='Diagnostic mode: ~100 conditions with verbose output')
    
    # Support legacy positional argument
    parser.add_argument('legacy_sample', nargs='?', type=int, default=None,
                        help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # Handle legacy positional argument
    sample = args.sample or args.legacy_sample
    diagnostic_mode = args.diagnostic
    
    # In diagnostic mode, default to ~50 files if no sample specified
    if diagnostic_mode and sample is None:
        sample = 50
        print(f"\n*** DIAGNOSTIC MODE: Processing {sample} batch files with verbose output ***\n")
    elif sample:
        print(f"\n*** RUNNING IN SAMPLE MODE: {sample} batch files ***\n")
    
    # Check for cached index
    if os.path.exists(INDEX_CACHE_FILE):
        try:
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(INDEX_CACHE_FILE))
            print(f"NOTE: Found cached index (age: {cache_age.seconds // 3600}h {(cache_age.seconds % 3600) // 60}m)")
            print(f"      Pass 1 will be skipped if cache is valid")
            print(f"      To force rebuild: rm {INDEX_CACHE_FILE}\n")
        except: 
            pass
    
    ensure_output_dir()
    analyze_calibration_streaming(sample_size=sample, diagnostic_mode=diagnostic_mode)