#!/usr/bin/env python3
"""
Phase 4E: Reaction Delay Analysis - MEMORY SAFE VERSION
Version: 1.0

OBJECTIVE:
  Determine how edge degrades as reaction time increases after detecting
  a threshold crossing. This answers: "Does this strategy require real-time
  infrastructure, or can it run as a periodic batch job?"

KEY QUESTION:
  "If I detect a ≥15% drop in a 90-99% probability market during the 6h→4h 
  window, but I can only react 30 minutes later, how much edge do I lose?"

METHODOLOGY:
  Same threshold crossing detection as Phase 4D, but instead of stratifying
  by tercile (when in interval the crossing occurred), we stratify by 
  reaction delay:
  - At crossing_time + 0s (baseline)
  - At crossing_time + 5min
  - At crossing_time + 15min
  - At crossing_time + 30min
  - At crossing_time + 1hr
  - At crossing_time + 2hr
  
  For each delay, we simulate placing a limit order at the crossing_price
  (the price when threshold was breached) but with delayed placement time.

OUTPUTS:
  - Edge at each delay level
  - Fill rate at each delay
  - "Half-life" of edge: delay at which edge drops to 50% of baseline
  
SUCCESS CRITERIA:
  If edge at delay=30min is still >50% of edge at delay=0 for positive-edge 
  windows, the strategy is viable for batch execution. If edge collapses 
  within 5 minutes, real-time infrastructure is required.

MEMORY SAFETY:
  - Streaming aggregation: no storage of individual results
  - Running statistics via Welford's algorithm
  - Progressive flush after each condition completes
"""

import pyarrow.parquet as pq
import glob
import json
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
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

BATCH_DIR = os.path.join(BASE_DIR, 'order_history_batches')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis/outputs')
INDEX_CACHE_FILE = os.path.join(BASE_DIR, 'analysis/market_index_cache_full.pkl')
SIDECAR_FILE = os.path.join(BASE_DIR, 'data/repair/api_derived_winners.parquet')

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Checkpoint file - UNIQUE to this phase to avoid collisions
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'phase4e_checkpoint.pkl')

# ------------------------------------------------------------------------------
# ANALYSIS PARAMETERS
# ------------------------------------------------------------------------------

# Interval pairs for analysis (all preserved from 4D)
INTERVAL_PAIRS = [
    (48, 24, '48h_to_24h'),
    (24, 12, '24h_to_12h'),
    (12, 6, '12h_to_6h'),
    (9, 6, '9h_to_6h'),
    (8, 4, '8h_to_4h'),
    (6, 4, '6h_to_4h'),
]

# Move thresholds - REMOVED 5%, keeping only 10%, 15%, 20%
MOVE_THRESHOLDS = [0.10, 0.15, 0.20]

# NEW: Reaction delays in seconds (replacing tercile analysis)
REACTION_DELAYS_SECONDS = [0, 300, 900, 1800, 3600, 7200]  # 0, 5min, 15min, 30min, 1hr, 2hr
DELAY_LABELS = {
    0: '0s',
    300: '5min',
    900: '15min',
    1800: '30min',
    3600: '1hr',
    7200: '2hr',
}

# Probability buckets for stratification (preserved from 4D)
PROB_BUCKETS = [
    ('sub_51', 0.0, 0.51),      # Longshots (for completeness)
    ('51_60', 0.51, 0.60),      # Toss-up favoring YES
    ('60_75', 0.60, 0.75),      # Moderate favorite
    ('75_90', 0.75, 0.90),      # Strong favorite  
    ('90_99', 0.90, 0.99),      # Heavy favorite (bond-adjacent)
    ('99_plus', 0.99, 1.01),    # Near-certain (bonds)
]
PROB_BUCKET_LABELS = [b[0] for b in PROB_BUCKETS] + ['all']

MIN_TRADES = 10
MIN_SAMPLES_PER_CELL = 20

REQUIRED_COLUMNS = ['timestamp', 'price', 'token_id', 'condition_id', 'resolution_time']
VOLUME_COLUMNS = ['size_tokens', 'maker_amount', 'size']

PROGRESS_INTERVAL = 1000


# ==============================================================================
# STREAMING STATISTICS CLASS
# ==============================================================================

class StreamingStats:
    """Compute mean, variance incrementally using Welford's algorithm."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.values_for_percentiles = []
        self.max_percentile_samples = 5000
    
    def update(self, value):
        if value is None or not np.isfinite(value):
            return
        
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
        
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        if len(self.values_for_percentiles) < self.max_percentile_samples:
            self.values_for_percentiles.append(value)
        else:
            j = np.random.randint(0, self.n)
            if j < self.max_percentile_samples:
                self.values_for_percentiles[j] = value
    
    def get_stats(self):
        if self.n == 0:
            return None
        
        variance = self.M2 / self.n if self.n > 0 else 0
        std = np.sqrt(variance)
        
        percentiles = {}
        if self.values_for_percentiles:
            arr = np.array(self.values_for_percentiles)
            percentiles = {
                'p10': float(np.percentile(arr, 10)),
                'p25': float(np.percentile(arr, 25)),
                'p50': float(np.percentile(arr, 50)),
                'p75': float(np.percentile(arr, 75)),
                'p90': float(np.percentile(arr, 90)),
            }
        
        return {
            'n': self.n,
            'mean': float(self.mean),
            'std': float(std),
            'min': float(self.min_val) if self.min_val != float('inf') else None,
            'max': float(self.max_val) if self.max_val != float('-inf') else None,
            **percentiles,
        }


# ==============================================================================
# DELAY-BASED AGGREGATOR (REPLACING TERCILE AGGREGATOR)
# ==============================================================================

def _make_delay_cell():
    """Create a cell for one delay bucket."""
    return {
        'n_samples': 0,
        'n_wins': 0,
        'n_fills': 0,
        'n_fill_wins': 0,
        'crossing_prices': StreamingStats(),
        'fill_prices': StreamingStats(),
    }


def _make_timing_cell():
    """Create a cell that tracks all delays for one (interval, threshold, prob_bucket)."""
    return {
        'total_samples': 0,
        'total_with_crossing': 0,
        'delays': {delay: _make_delay_cell() for delay in REACTION_DELAYS_SECONDS},
        'start_price_distribution': StreamingStats(),
    }


def get_prob_bucket(start_price):
    """Determine which probability bucket a start_price falls into."""
    for label, lo, hi in PROB_BUCKETS:
        if lo <= start_price < hi:
            return label
    return None


class DelayAggregator:
    """Aggregate reaction delay results with probability stratification."""
    
    def __init__(self):
        self.n_tokens = 0
        self.n_tokens_with_data = 0
        
        # Main structure: [interval_label][threshold][prob_bucket] -> timing_cell
        # Include 'all' bucket for aggregate view
        self.surface = {}
        for _, _, interval_label in INTERVAL_PAIRS:
            self.surface[interval_label] = {}
            for threshold in MOVE_THRESHOLDS:
                self.surface[interval_label][threshold] = {}
                for prob_label in PROB_BUCKET_LABELS:
                    self.surface[interval_label][threshold][prob_label] = _make_timing_cell()
        
        self.interval_coverage = defaultdict(int)
        
        # Track probability bucket distribution
        self.prob_bucket_counts = defaultdict(int)
    
    def add_result(self, result):
        """Add a single token result to aggregates."""
        self.n_tokens += 1
        
        if not result.get('interval_data'):
            return
        
        self.n_tokens_with_data += 1
        winner = result['winner']
        
        for interval_label, interval_data in result['interval_data'].items():
            self.interval_coverage[interval_label] += 1
            
            # Get start price for probability bucketing
            start_price = interval_data.get('start_price')
            if start_price is None:
                continue
            
            prob_bucket = get_prob_bucket(start_price)
            
            # Process each threshold
            for threshold in MOVE_THRESHOLDS:
                crossing_data = interval_data.get('crossings', {}).get(threshold)
                
                # Update both the specific prob_bucket and 'all'
                buckets_to_update = ['all']
                if prob_bucket:
                    buckets_to_update.append(prob_bucket)
                    self.prob_bucket_counts[prob_bucket] += 1
                
                for bucket in buckets_to_update:
                    cell = self.surface[interval_label][threshold][bucket]
                    cell['total_samples'] += 1
                    cell['start_price_distribution'].update(start_price)
                    
                    if crossing_data is None or not crossing_data.get('crossed'):
                        continue
                    
                    cell['total_with_crossing'] += 1
                    crossing_price = crossing_data['crossing_price']
                    
                    # Process each delay
                    for delay in REACTION_DELAYS_SECONDS:
                        delay_fills = crossing_data.get('delay_fills', {})
                        fill_data = delay_fills.get(delay, {})
                        
                        delay_cell = cell['delays'][delay]
                        delay_cell['n_samples'] += 1
                        
                        if winner:
                            delay_cell['n_wins'] += 1
                        
                        delay_cell['crossing_prices'].update(crossing_price)
                        
                        # Fill simulation for this delay
                        if fill_data and fill_data.get('filled'):
                            delay_cell['n_fills'] += 1
                            if winner:
                                delay_cell['n_fill_wins'] += 1
                            
                            fill_price = fill_data.get('fill_price')
                            if fill_price is not None:
                                delay_cell['fill_prices'].update(fill_price)
    
    def get_summary(self):
        """Get aggregated summary statistics."""
        summary = {
            'n_tokens': self.n_tokens,
            'n_tokens_with_data': self.n_tokens_with_data,
            'interval_coverage': dict(self.interval_coverage),
            'prob_bucket_distribution': dict(self.prob_bucket_counts),
            'surface': {},
        }
        
        for interval_label in [x[2] for x in INTERVAL_PAIRS]:
            summary['surface'][interval_label] = {}
            
            for threshold in MOVE_THRESHOLDS:
                summary['surface'][interval_label][threshold] = {}
                
                for prob_bucket in PROB_BUCKET_LABELS:
                    cell = self.surface[interval_label][threshold][prob_bucket]
                    
                    # Overall stats for this (interval, threshold, prob_bucket)
                    cell_summary = {
                        'total_samples': cell['total_samples'],
                        'total_with_crossing': cell['total_with_crossing'],
                        'crossing_rate': cell['total_with_crossing'] / cell['total_samples'] if cell['total_samples'] > 0 else 0,
                        'start_price_distribution': cell['start_price_distribution'].get_stats(),
                        'delays': {},
                    }
                    
                    # Per-delay stats
                    baseline_edge = None
                    
                    for delay in REACTION_DELAYS_SECONDS:
                        delay_cell = cell['delays'][delay]
                        
                        n = delay_cell['n_samples']
                        n_wins = delay_cell['n_wins']
                        n_fills = delay_cell['n_fills']
                        n_fill_wins = delay_cell['n_fill_wins']
                        
                        if n < MIN_SAMPLES_PER_CELL:
                            cell_summary['delays'][delay] = {
                                'n_samples': n,
                                'n_fills': n_fills,
                                'n_wins': n_wins,
                                'n_fill_wins': n_fill_wins,
                                'status': 'insufficient_data',
                            }
                            continue
                        
                        # Unconditional metrics (based on all crossings)
                        uncond_win_rate = n_wins / n if n > 0 else 0
                        crossing_stats = delay_cell['crossing_prices'].get_stats()
                        avg_crossing_price = crossing_stats['mean'] if crossing_stats else 0
                        
                        uncond_edge_bps = (uncond_win_rate - avg_crossing_price) * 10000
                        
                        # SE for win rate (binomial)
                        se_uncond = np.sqrt(uncond_win_rate * (1 - uncond_win_rate) / n) * 10000 if n > 0 else 0
                        
                        # Fill rate
                        fill_rate = n_fills / n if n > 0 else 0
                        
                        # Conditional metrics (among filled)
                        cond_result = {}
                        if n_fills >= 10:
                            cond_win_rate = n_fill_wins / n_fills
                            fill_stats = delay_cell['fill_prices'].get_stats()
                            avg_fill = fill_stats['mean'] if fill_stats else 0
                            
                            edge_after_fill_bps = (cond_win_rate - avg_fill) * 10000
                            se_cond = np.sqrt(cond_win_rate * (1 - cond_win_rate) / n_fills) * 10000 if n_fills > 0 else 0
                            
                            cond_result = {
                                'conditional_win_rate': cond_win_rate,
                                'avg_fill_price': avg_fill,
                                'edge_after_fill_bps': edge_after_fill_bps,
                                'se_edge_after_fill': se_cond,
                            }
                            
                            # Track baseline for half-life calculation
                            if delay == 0:
                                baseline_edge = edge_after_fill_bps
                        
                        cell_summary['delays'][delay] = {
                            'n_samples': n,
                            'n_fills': n_fills,
                            'n_wins': n_wins,
                            'n_fill_wins': n_fill_wins,
                            'unconditional_win_rate': uncond_win_rate,
                            'avg_crossing_price': avg_crossing_price,
                            'unconditional_edge_bps': uncond_edge_bps,
                            'se_unconditional_edge': se_uncond,
                            'fill_rate': fill_rate,
                            **cond_result,
                            'status': 'ok',
                        }
                    
                    # Compute half-life of edge
                    if baseline_edge is not None and baseline_edge > 0:
                        half_life_seconds = None
                        for delay in REACTION_DELAYS_SECONDS:
                            delay_data = cell_summary['delays'].get(delay, {})
                            if delay_data.get('status') != 'ok':
                                continue
                            edge = delay_data.get('edge_after_fill_bps')
                            if edge is not None and edge <= baseline_edge * 0.5:
                                half_life_seconds = delay
                                break
                        
                        cell_summary['edge_half_life_seconds'] = half_life_seconds
                        cell_summary['baseline_edge_bps'] = baseline_edge
                    
                    summary['surface'][interval_label][threshold][prob_bucket] = cell_summary
        
        return summary


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


# ==============================================================================
# SIDECAR LOADING
# ==============================================================================

def load_winner_sidecar(sidecar_path):
    log(f"Loading winner sidecar from {sidecar_path}...")
    
    if not os.path.exists(sidecar_path):
        log(f"  ERROR: Sidecar file not found: {sidecar_path}")
        return None
    
    try:
        df = pd.read_parquet(sidecar_path)
        log(f"  Raw sidecar: {len(df):,} records")
        
        success_df = df[df['repair_status'] == 'SUCCESS']
        log(f"  SUCCESS records: {len(success_df):,}")
        
        winner_lookup = {}
        for _, row in success_df.iterrows():
            token_id = str(row['token_id'])
            is_winner = row['api_derived_winner']
            if is_winner is not None:
                winner_lookup[token_id] = bool(is_winner)
        
        log(f"  Winner lookup size: {len(winner_lookup):,} tokens")
        return winner_lookup
        
    except Exception as e:
        log(f"  ERROR loading sidecar: {e}")
        return None
        

# ==============================================================================
# CHECKPOINT HANDLING (read-only for cache, new file for phase4e)
# ==============================================================================

def save_checkpoint(aggregator, files_processed, total_files):
    """Save checkpoint for resumption."""
    checkpoint = {
        'aggregator': aggregator,
        'files_processed': files_processed,
        'total_files': total_files,
        'timestamp': datetime.now().isoformat(),
    }
    
    temp_file = CHECKPOINT_FILE + '.tmp'
    try:
        with open(temp_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        os.rename(temp_file, CHECKPOINT_FILE)
        log(f"  Checkpoint saved: {files_processed}/{total_files} files")
    except Exception as e:
        log(f"  WARNING: Checkpoint save failed: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)


def load_checkpoint():
    """Load checkpoint if available."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    
    try:
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint
    except Exception as e:
        log(f"  WARNING: Checkpoint load failed: {e}")
        return None


# ==============================================================================
# PRICE EXTRACTION AND THRESHOLD CROSSING
# ==============================================================================

def extract_price_at_horizon(trades, resolution_time, hours_before):
    """Extract price at approximately T-hours before resolution."""
    target_time = resolution_time - (hours_before * 3600)
    
    relevant_trades = [(ts, p, s) for ts, p, s in trades if ts <= target_time]
    
    if not relevant_trades:
        return None, None
    
    relevant_trades.sort(key=lambda x: x[0])
    closest_trade = relevant_trades[-1]
    
    time_diff = abs(closest_trade[0] - target_time)
    if time_diff > 3600:  # More than 1 hour gap
        return None, None
    
    return closest_trade[0], closest_trade[1]


def find_first_threshold_crossing(trades, start_time, end_time, start_price, threshold, direction='drop'):
    """
    Find the first time price crosses threshold within interval.
    
    Returns crossing details including time, fraction of interval.
    """
    interval_length = end_time - start_time
    if interval_length <= 0:
        return {'crossed': False}
    
    interval_trades = [(ts, p, s) for ts, p, s in trades if start_time < ts <= end_time]
    
    if not interval_trades:
        return {'crossed': False}
    
    interval_trades.sort(key=lambda x: x[0])
    
    for ts, price, size in interval_trades:
        if direction == 'drop':
            move = start_price - price
        else:
            move = price - start_price
        
        if move >= threshold:
            time_elapsed = ts - start_time
            fraction = time_elapsed / interval_length
            time_elapsed_hours = time_elapsed / 3600
            
            return {
                'crossed': True,
                'crossing_time': ts,
                'crossing_price': price,
                'fraction_of_interval': fraction,
                'time_to_crossing_hours': time_elapsed_hours,
                'move_at_crossing': move,
            }
    
    final_price = interval_trades[-1][1]
    if direction == 'drop':
        final_move = start_price - final_price
    else:
        final_move = final_price - start_price
    
    if final_move >= threshold:
        return {
            'crossed': True,
            'crossing_time': end_time,
            'crossing_price': final_price,
            'fraction_of_interval': 1.0,
            'time_to_crossing_hours': interval_length / 3600,
            'move_at_crossing': final_move,
        }
    
    return {'crossed': False}


def simulate_limit_order_fill(trades, placement_time, limit_price, is_buy=True):
    """Simulate whether a limit order would be filled."""
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
                'fill_price': limit_price,
                'time_to_fill': ts - placement_time,
            }
        elif not is_buy and price >= limit_price:
            return {
                'filled': True,
                'fill_time': ts,
                'fill_price': limit_price,
                'time_to_fill': ts - placement_time,
            }
    
    return {
        'filled': False,
        'fill_time': None,
        'fill_price': None,
        'time_to_fill': None,
    }


def compute_interval_data_with_delays(trades, resolution_time):
    """
    Compute interval data with threshold crossing and delay-based fill simulation.
    
    KEY CHANGE FROM 4D: Instead of single fill simulation at crossing_time,
    we simulate fills at crossing_time + delta for each delta in REACTION_DELAYS_SECONDS.
    """
    results = {}
    
    for start_h, end_h, label in INTERVAL_PAIRS:
        start_time, start_price = extract_price_at_horizon(trades, resolution_time, start_h)
        end_time, end_price = extract_price_at_horizon(trades, resolution_time, end_h)
        
        if start_price is None or end_price is None:
            continue
        
        if start_time is None or end_time is None:
            continue
        
        overall_move = end_price - start_price
        
        crossings = {}
        
        for threshold in MOVE_THRESHOLDS:
            crossing = find_first_threshold_crossing(
                trades, start_time, end_time, start_price, threshold, direction='drop'
            )
            
            if crossing and crossing.get('crossed'):
                crossing_time = crossing['crossing_time']
                crossing_price = crossing['crossing_price']
                
                # NEW: Simulate fills at each delay level
                delay_fills = {}
                for delay in REACTION_DELAYS_SECONDS:
                    placement_time = crossing_time + delay
                    fill_result = simulate_limit_order_fill(
                        trades, placement_time, crossing_price, is_buy=True
                    )
                    delay_fills[delay] = fill_result
                
                crossing['delay_fills'] = delay_fills
            
            crossings[threshold] = crossing
        
        results[label] = {
            'start_hours': start_h,
            'end_hours': end_h,
            'start_time': start_time,
            'end_time': end_time,
            'start_price': start_price,
            'end_price': end_price,
            'overall_move': overall_move,
            'crossings': crossings,
        }
    
    return results


# ==============================================================================
# DATA ACCUMULATOR CLASS
# ==============================================================================

class DelayAccumulator:
    """Accumulates trade data for delay analysis."""
    
    def __init__(self, token_id, condition_id, resolution_time, winner_status):
        self.token_id = token_id
        self.condition_id = condition_id
        self.resolution_time = resolution_time
        self.winner_status = winner_status
        self.trades = []
    
    def add_trades(self, trades_batch):
        self.trades.extend(trades_batch)
    
    def compute_delay_data(self):
        """Compute delay data for this token."""
        if len(self.trades) < MIN_TRADES:
            return None
        
        self.trades.sort(key=lambda x: x[0])
        
        interval_data = compute_interval_data_with_delays(self.trades, self.resolution_time)
        
        if not interval_data:
            return None
        
        return {
            'token_id': self.token_id,
            'condition_id': self.condition_id,
            'winner': self.winner_status,
            'n_trades': len(self.trades),
            'interval_data': interval_data,
        }


# ==============================================================================
# MARKET INDEX LOADING (read-only)
# ==============================================================================

def load_market_index(cache_file, batch_files):
    """Load market index from cache. Read-only, never overwrites."""
    if not os.path.exists(cache_file):
        log(f"  Cache file not found: {cache_file}")
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        cached_num_files = cache_data.get('num_files', 0)
        current_num_files = len(batch_files)
        
        file_diff_pct = abs(cached_num_files - current_num_files) / max(cached_num_files, 1) * 100
        
        if file_diff_pct > 10:
            log(f"  WARNING: Cache may be stale ({cached_num_files} vs {current_num_files} files)")
        
        return cache_data['market_index']
        
    except Exception as e:
        log(f"  Cache load failed: {e}")
        return None


def get_available_columns(filepath):
    """Get available columns from parquet file schema."""
    parquet_file = pq.ParquetFile(filepath)
    schema = parquet_file.schema
    available = set(schema.names)
    
    columns_to_read = [c for c in REQUIRED_COLUMNS if c in available]
    
    volume_col = None
    for col in VOLUME_COLUMNS:
        if col in available:
            volume_col = col
            if col not in columns_to_read:
                columns_to_read.append(col)
            break
    
    return columns_to_read, volume_col


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================

def run_analysis(sample_files=None, resume=False, diagnostic=False):
    """Run the reaction delay analysis."""
    global start_time
    start_time = datetime.now()
    
    log("="*70)
    log("PHASE 4E: REACTION DELAY ANALYSIS")
    log("="*70)
    
    # -------------------------------------------------------------------------
    # LOAD RESOURCES
    # -------------------------------------------------------------------------
    
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
        log(f"       Expected: {INDEX_CACHE_FILE}")
        return None
    
    log(f"  Loaded market index: {len(market_index):,} conditions")
    
    # -------------------------------------------------------------------------
    # CONDITION COMPLETION TRACKING
    # -------------------------------------------------------------------------
    
    log("\nSetting up streaming flush tracking...")
    
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
    
    # -------------------------------------------------------------------------
    # STREAMING AGGREGATOR
    # -------------------------------------------------------------------------
    
    start_file_idx = 0
    aggregator = DelayAggregator()
    
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            aggregator = checkpoint['aggregator']
            start_file_idx = checkpoint['files_processed']
            log(f"Resuming from checkpoint: {start_file_idx}/{checkpoint['total_files']} files")
            log(f"  Existing results: {aggregator.n_tokens} tokens")
    
    token_accumulators = {}
    condition_tokens = defaultdict(set)
    
    stats = {
        'files_processed': start_file_idx,
        'total_rows': 0,
        'tokens_no_winner': 0,
        'tokens_filtered': 0,
        'conditions_flushed': 0,
    }
    
    diagnostic_samples = [] if diagnostic else None
    
    log(f"\nProcessing {len(files_to_process_indices) - start_file_idx} files...")
    log(f"  {log_memory()}")
    
    # -------------------------------------------------------------------------
    # MAIN PROCESSING LOOP
    # -------------------------------------------------------------------------
    
    for file_idx in files_to_process_indices[start_file_idx:]:
        stats['files_processed'] += 1
        filepath = batch_files[file_idx]
        
        if stats['files_processed'] % PROGRESS_INTERVAL == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = stats['files_processed'] / elapsed if elapsed > 0 else 0
            eta = (len(files_to_process_indices) - stats['files_processed']) / rate if rate > 0 else 0
            
            log(f"  [{stats['files_processed']:,}/{len(files_to_process_indices):,}] "
                f"Active: {len(token_accumulators):,} | "
                f"Flushed: {stats['conditions_flushed']:,} | "
                f"Results: {aggregator.n_tokens:,} | "
                f"Rate: {rate:.1f}/s | "
                f"ETA: {format_duration(eta)} | "
                f"{log_memory()}")
            
            if stats['files_processed'] % (PROGRESS_INTERVAL * 10) == 0:
                save_checkpoint(aggregator, stats['files_processed'], len(files_to_process_indices))
                gc.collect()
        
        try:
            columns_to_read, volume_col = get_available_columns(filepath)
            
            if not volume_col:
                continue
            
            df = pq.read_table(filepath, columns=columns_to_read).to_pandas()
            
            if len(df) == 0:
                continue
            
            stats['total_rows'] += len(df)
            
            if volume_col != 'size_tokens':
                df.rename(columns={volume_col: 'size_tokens'}, inplace=True)
            
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            if len(df) == 0:
                continue
            
            # Handle millisecond timestamps
            if df['timestamp'].iloc[0] > 3e9:
                df['timestamp'] = df['timestamp'] / 1000.0
                df['resolution_time'] = df['resolution_time'] / 1000.0
            
            df['token_id'] = df['token_id'].astype(str)
            
            for token_id, group in df.groupby('token_id', sort=False):
                condition_id = group['condition_id'].iloc[0]
                
                if token_id not in token_accumulators:
                    winner_status = winner_lookup.get(token_id, None)
                    
                    if winner_status is None:
                        stats['tokens_no_winner'] += 1
                        continue
                    
                    resolution_time = float(group['resolution_time'].iloc[0])
                    
                    token_accumulators[token_id] = DelayAccumulator(
                        token_id, condition_id, resolution_time, winner_status
                    )
                    condition_tokens[condition_id].add(token_id)
                
                timestamps = group['timestamp'].values
                prices = group['price'].values
                sizes = group['size_tokens'].values
                
                trades_batch = list(zip(timestamps, prices, sizes))
                token_accumulators[token_id].add_trades(trades_batch)
                
                # Capture diagnostic samples
                if diagnostic and len(diagnostic_samples) < 5 and len(trades_batch) > 100:
                    diagnostic_samples.append({
                        'token_id': token_id,
                        'winner': winner_status,
                        'trades': trades_batch[:200],
                        'n_trades': len(trades_batch),
                        'resolution_time': float(group['resolution_time'].iloc[0]),
                    })
            
            del df
            
            # -----------------------------------------------------------------
            # STREAMING FLUSH
            # -----------------------------------------------------------------
            
            conditions_in_this_file = file_to_conditions.get(file_idx, set())
            
            for condition_id in conditions_in_this_file:
                if condition_id in condition_remaining_files:
                    condition_remaining_files[condition_id] -= 1
                    
                    if condition_remaining_files[condition_id] == 0:
                        tokens_to_flush = condition_tokens.get(condition_id, set())
                        
                        for token_id in tokens_to_flush:
                            if token_id not in token_accumulators:
                                continue
                            
                            acc = token_accumulators[token_id]
                            delay_result = acc.compute_delay_data()
                            
                            if delay_result is not None:
                                aggregator.add_result(delay_result)
                            else:
                                stats['tokens_filtered'] += 1
                            
                            del token_accumulators[token_id]
                        
                        if condition_id in condition_tokens:
                            del condition_tokens[condition_id]
                        del condition_remaining_files[condition_id]
                        
                        stats['conditions_flushed'] += 1
            
        except Exception as e:
            log(f"  Error processing {filepath}: {e}")
            continue
    
    # -------------------------------------------------------------------------
    # FINAL FLUSH
    # -------------------------------------------------------------------------
    
    remaining_conditions = list(condition_tokens.keys())
    if remaining_conditions:
        log(f"\nFinal flush: {len(remaining_conditions)} remaining conditions...")
        
        for condition_id in remaining_conditions:
            tokens_to_flush = condition_tokens.get(condition_id, set())
            
            for token_id in tokens_to_flush:
                if token_id not in token_accumulators:
                    continue
                
                acc = token_accumulators[token_id]
                delay_result = acc.compute_delay_data()
                
                if delay_result is not None:
                    aggregator.add_result(delay_result)
                else:
                    stats['tokens_filtered'] += 1
                
                del token_accumulators[token_id]
            
            stats['conditions_flushed'] += 1
        
        token_accumulators.clear()
        condition_tokens.clear()
    
    # -------------------------------------------------------------------------
    # DIAGNOSTICS
    # -------------------------------------------------------------------------
    
    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"\nProcessing complete in {format_duration(elapsed)}")
    log(f"  Files processed: {stats['files_processed']:,}")
    log(f"  Total rows: {stats['total_rows']:,}")
    log(f"  Conditions flushed: {stats['conditions_flushed']:,}")
    log(f"  Tokens without winner: {stats['tokens_no_winner']:,}")
    log(f"  Tokens filtered (low trades): {stats['tokens_filtered']:,}")
    log(f"  Final result count: {aggregator.n_tokens:,}")
    log(f"  {log_memory()}")
    
    if diagnostic and diagnostic_samples:
        log("\n" + "="*70)
        log("DIAGNOSTIC SAMPLE ANALYSIS")
        log("="*70)
        
        for sample in diagnostic_samples:
            log(f"\nToken: {sample['token_id'][:20]}...")
            log(f"  Winner: {sample['winner']}")
            log(f"  Total trades: {sample['n_trades']}")
            
            trades = sample['trades']
            resolution_time = sample['resolution_time']
            
            interval_data = compute_interval_data_with_delays(trades, resolution_time)
            
            for label, data in interval_data.items():
                prob_bucket = get_prob_bucket(data['start_price'])
                log(f"\n  {label} [prob_bucket: {prob_bucket}]:")
                log(f"    Start: {data['start_price']:.4f} @ T-{data['start_hours']}h")
                log(f"    End:   {data['end_price']:.4f} @ T-{data['end_hours']}h")
                log(f"    Overall move: {data['overall_move']:+.4f}")
                
                crossings = data.get('crossings', {})
                for threshold, crossing in crossings.items():
                    if crossing and crossing.get('crossed'):
                        log(f"    Threshold {threshold*100:.0f}%: CROSSED")
                        log(f"      Time to crossing: {crossing['time_to_crossing_hours']:.2f}h")
                        log(f"      Price at crossing: {crossing['crossing_price']:.4f}")
                        
                        # Show fill results at each delay
                        delay_fills = crossing.get('delay_fills', {})
                        for delay in REACTION_DELAYS_SECONDS:
                            fill = delay_fills.get(delay, {})
                            delay_label = DELAY_LABELS.get(delay, f'{delay}s')
                            if fill.get('filled'):
                                log(f"      Fill @ +{delay_label}: YES @ {fill['fill_price']:.4f}")
                            else:
                                log(f"      Fill @ +{delay_label}: NO")
                    else:
                        log(f"    Threshold {threshold*100:.0f}%: not crossed")
    
    return aggregator


# ==============================================================================
# REPORTING
# ==============================================================================

def print_delay_analysis(aggregator):
    """Print delay analysis from streaming aggregator."""
    summary = aggregator.get_summary()
    
    log("\n" + "="*70)
    log("REACTION DELAY ANALYSIS RESULTS")
    log("="*70)
    
    log(f"\nSample Size: {summary['n_tokens']:,} tokens")
    log(f"Tokens with interval data: {summary['n_tokens_with_data']:,}")
    
    log("\n" + "-"*50)
    log("PROBABILITY BUCKET DISTRIBUTION")
    log("-"*50)
    
    total_obs = sum(summary['prob_bucket_distribution'].values())
    for bucket in [b[0] for b in PROB_BUCKETS]:
        count = summary['prob_bucket_distribution'].get(bucket, 0)
        pct = count / total_obs * 100 if total_obs > 0 else 0
        log(f"  {bucket:>10}: {count:>8,} ({pct:>5.1f}%)")
    
    log("\n" + "-"*50)
    log("INTERVAL COVERAGE")
    log("-"*50)
    
    for _, _, interval_label in INTERVAL_PAIRS:
        coverage = summary['interval_coverage'].get(interval_label, 0)
        log(f"  {interval_label}: {coverage:,} tokens")
    
    # =========================================================================
    # KEY OUTPUT: Edge degradation by delay
    # =========================================================================
    
    log("\n" + "="*70)
    log("EDGE DEGRADATION BY REACTION DELAY")
    log("="*70)
    log("This shows how edge decays as reaction time increases after detecting the dip.")
    log("Key question: Is this a real-time strategy or can it run as a batch job?")
    
    # Header row
    delay_header = "Delay:      "
    for delay in REACTION_DELAYS_SECONDS:
        delay_label = DELAY_LABELS.get(delay, f'{delay}s')
        delay_header += f" {delay_label:>8}"
    delay_header += " | Half-Life"
    
    # For each (interval, threshold, prob_bucket), show edge at each delay
    for _, _, interval_label in INTERVAL_PAIRS:
        for threshold in MOVE_THRESHOLDS:
            log(f"\n{interval_label} | >= {threshold*100:.0f}% drop")
            log("-"*80)
            log(delay_header)
            log("-"*80)
            
            for prob_bucket in PROB_BUCKET_LABELS:
                cell = summary['surface'].get(interval_label, {}).get(threshold, {}).get(prob_bucket, {})
                
                if cell.get('total_with_crossing', 0) < 50:
                    continue
                
                row = f"{prob_bucket:>10}: "
                
                baseline_edge = None
                for delay in REACTION_DELAYS_SECONDS:
                    delay_data = cell['delays'].get(delay, {})
                    
                    if delay_data.get('status') == 'ok':
                        edge = delay_data.get('edge_after_fill_bps')
                        fill_rate = delay_data.get('fill_rate', 0)
                        
                        if edge is not None:
                            if delay == 0:
                                baseline_edge = edge
                            row += f" {edge:>+7.0f}"
                        else:
                            row += f" {'n/a':>8}"
                    else:
                        row += f" {'--':>8}"
                
                # Half-life
                half_life = cell.get('edge_half_life_seconds')
                if half_life is not None:
                    half_life_label = DELAY_LABELS.get(half_life, f'{half_life}s')
                    row += f" | {half_life_label}"
                elif baseline_edge is not None and baseline_edge > 0:
                    row += f" | >2hr"
                else:
                    row += f" | n/a"
                
                log(row)
    
    # =========================================================================
    # FILL RATE DEGRADATION
    # =========================================================================
    
    log("\n" + "="*70)
    log("FILL RATE BY REACTION DELAY")
    log("="*70)
    log("Shows how fill probability changes as placement is delayed.")
    
    for _, _, interval_label in INTERVAL_PAIRS:
        for threshold in MOVE_THRESHOLDS:
            cell = summary['surface'].get(interval_label, {}).get(threshold, {}).get('all', {})
            
            if cell.get('total_with_crossing', 0) < 100:
                continue
            
            log(f"\n{interval_label} | >= {threshold*100:.0f}% drop (all prob buckets)")
            
            row = "Fill Rate:  "
            for delay in REACTION_DELAYS_SECONDS:
                delay_data = cell['delays'].get(delay, {})
                delay_label = DELAY_LABELS.get(delay, f'{delay}s')
                
                if delay_data.get('status') == 'ok':
                    fill_rate = delay_data.get('fill_rate', 0)
                    row += f" {fill_rate:>7.1%}"
                else:
                    row += f" {'--':>8}"
            
            log(row)
    
    return summary


def print_key_findings(summary):
    """Print key findings and operational implications."""
    log("\n" + "="*70)
    log("KEY FINDINGS & OPERATIONAL IMPLICATIONS")
    log("="*70)
    
    # 1. Identify best windows and their delay tolerance
    log("\n1. EDGE PERSISTENCE ANALYSIS")
    log("-"*50)
    
    best_windows = []
    
    for interval_label in [x[2] for x in INTERVAL_PAIRS]:
        for threshold in MOVE_THRESHOLDS:
            cell = summary['surface'].get(interval_label, {}).get(threshold, {}).get('all', {})
            
            if cell.get('total_with_crossing', 0) < 100:
                continue
            
            delay_0_data = cell['delays'].get(0, {})
            if delay_0_data.get('status') != 'ok':
                continue
            
            baseline_edge = delay_0_data.get('edge_after_fill_bps', 0)
            if baseline_edge <= 0:
                continue
            
            # Check edge at 30min delay
            delay_1800_data = cell['delays'].get(1800, {})
            edge_30min = delay_1800_data.get('edge_after_fill_bps', 0) if delay_1800_data.get('status') == 'ok' else 0
            
            # Check edge at 1hr delay
            delay_3600_data = cell['delays'].get(3600, {})
            edge_1hr = delay_3600_data.get('edge_after_fill_bps', 0) if delay_3600_data.get('status') == 'ok' else 0
            
            half_life = cell.get('edge_half_life_seconds')
            
            retention_30min = edge_30min / baseline_edge if baseline_edge > 0 else 0
            retention_1hr = edge_1hr / baseline_edge if baseline_edge > 0 else 0
            
            best_windows.append({
                'interval': interval_label,
                'threshold': threshold,
                'baseline_edge': baseline_edge,
                'edge_30min': edge_30min,
                'edge_1hr': edge_1hr,
                'retention_30min': retention_30min,
                'retention_1hr': retention_1hr,
                'half_life': half_life,
            })
    
    # Sort by baseline edge
    best_windows.sort(key=lambda x: x['baseline_edge'], reverse=True)
    
    if best_windows:
        log("\nTop Windows by Baseline Edge:")
        log(f"{'Window':<20} | {'Baseline':>10} | {'@ 30min':>10} | {'@ 1hr':>10} | {'Retain 30m':>10} | Half-Life")
        log("-"*85)
        
        for w in best_windows[:10]:
            half_life_str = DELAY_LABELS.get(w['half_life'], '>2hr') if w['half_life'] else '>2hr'
            log(f"{w['interval']} {w['threshold']*100:.0f}%    | {w['baseline_edge']:>+9.0f} | {w['edge_30min']:>+9.0f} | {w['edge_1hr']:>+9.0f} | {w['retention_30min']:>9.1%} | {half_life_str}")
    
    # 2. Operational recommendation
    log("\n" + "-"*50)
    log("2. OPERATIONAL RECOMMENDATION")
    log("-"*50)
    
    # Find windows where edge retention at 30min is >50%
    viable_batch_windows = [w for w in best_windows if w['retention_30min'] > 0.5 and w['baseline_edge'] > 50]
    fast_decay_windows = [w for w in best_windows if w['retention_30min'] <= 0.5 and w['baseline_edge'] > 50]
    
    if viable_batch_windows:
        log(f"\n  BATCH-VIABLE WINDOWS ({len(viable_batch_windows)} found with >50% edge retention at 30min):")
        for w in viable_batch_windows[:5]:
            log(f"    {w['interval']} {w['threshold']*100:.0f}%: {w['baseline_edge']:+.0f} bps -> {w['edge_30min']:+.0f} bps @ 30min ({w['retention_30min']:.0%} retained)")
        
        log("\n  RECOMMENDATION: These windows can run as a periodic batch job (e.g., every 15-30 minutes).")
    
    if fast_decay_windows:
        log(f"\n  REAL-TIME REQUIRED ({len(fast_decay_windows)} windows with rapid edge decay):")
        for w in fast_decay_windows[:5]:
            log(f"    {w['interval']} {w['threshold']*100:.0f}%: {w['baseline_edge']:+.0f} bps -> {w['edge_30min']:+.0f} bps @ 30min ({w['retention_30min']:.0%} retained)")
        
        log("\n  RECOMMENDATION: These windows require real-time detection (<5min latency).")
    
    if not viable_batch_windows and not fast_decay_windows:
        log("\n  Insufficient data to make operational recommendations.")
        log("  Need more samples with positive baseline edge.")
    
    # 3. Probability bucket breakdown
    log("\n" + "-"*50)
    log("3. EDGE BY PROBABILITY BUCKET (at delay=0)")
    log("-"*50)
    
    bucket_summary = defaultdict(list)
    for interval_label in [x[2] for x in INTERVAL_PAIRS]:
        for threshold in MOVE_THRESHOLDS:
            for prob_bucket in [b[0] for b in PROB_BUCKETS]:
                cell = summary['surface'].get(interval_label, {}).get(threshold, {}).get(prob_bucket, {})
                delay_0 = cell.get('delays', {}).get(0, {})
                if delay_0.get('status') == 'ok' and delay_0.get('n_samples', 0) >= 50:
                    bucket_summary[prob_bucket].append(delay_0.get('edge_after_fill_bps', 0))
    
    for bucket in [b[0] for b in PROB_BUCKETS]:
        edges = bucket_summary.get(bucket, [])
        if edges:
            avg_edge = np.mean(edges)
            log(f"  {bucket:>10}: avg edge = {avg_edge:+.0f} bps (across {len(edges)} cells)")


def save_results_json(summary):
    """Save results to JSON file."""
    ensure_output_dir()
    
    output = {
        'timestamp': TIMESTAMP,
        'phase': '4E',
        'description': 'Reaction Delay Analysis',
        'question': 'How does edge degrade as reaction time increases?',
        'methodology': 'Same threshold detection as 4D, but simulate fills at crossing_time + delay for each delay level',
        'interval_pairs': [(s, e, l) for s, e, l in INTERVAL_PAIRS],
        'move_thresholds': MOVE_THRESHOLDS,
        'reaction_delays_seconds': REACTION_DELAYS_SECONDS,
        'delay_labels': DELAY_LABELS,
        'prob_buckets': PROB_BUCKETS,
        **summary,
    }
    
    json_path = os.path.join(OUTPUT_DIR, f'phase4e_delay_analysis_{TIMESTAMP}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    log(f"\nResults JSON saved: {json_path}")
    return json_path


def generate_report(summary):
    """Generate text report."""
    ensure_output_dir()
    
    report_path = os.path.join(OUTPUT_DIR, f'phase4e_delay_analysis_{TIMESTAMP}_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 4E: REACTION DELAY ANALYSIS - REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tokens Analyzed: {summary.get('n_tokens', 0):,}\n")
        f.write(f"Tokens with Data: {summary.get('n_tokens_with_data', 0):,}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("KEY QUESTION\n")
        f.write("-"*80 + "\n")
        f.write("If I detect a threshold crossing but can only react after a delay,\n")
        f.write("how much edge do I lose? Does this strategy require real-time\n")
        f.write("infrastructure or can it run as a periodic batch job?\n\n")
        
        f.write("-"*80 + "\n")
        f.write("REACTION DELAYS TESTED\n")
        f.write("-"*80 + "\n")
        for delay in REACTION_DELAYS_SECONDS:
            f.write(f"  {DELAY_LABELS.get(delay, f'{delay}s')}\n")
        
        f.write("\n")
        f.write("-"*80 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("-"*80 + "\n")
        f.write("Half-Life: The delay at which edge drops to 50% of baseline.\n")
        f.write("  - Half-life < 5min: Requires real-time infrastructure\n")
        f.write("  - Half-life 15-30min: Can run as frequent batch job\n")
        f.write("  - Half-life > 1hr: Can run as infrequent batch job\n")
        f.write("\nSee JSON file for full data.\n")
    
    log(f"Report saved: {report_path}")
    return report_path


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 4E: Reaction Delay Analysis (Memory-Safe)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnostic mode with small sample
  python phase4e_delay_analysis.py --diagnostic --sample 100
  
  # Full run
  python phase4e_delay_analysis.py
  
  # Resume interrupted run
  python phase4e_delay_analysis.py --resume
        """
    )
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--diagnostic', '-d', action='store_true',
                        help='Run in diagnostic mode with extra validation')
    
    args = parser.parse_args()
    
    log("Starting Phase 4E Reaction Delay Analysis...")
    
    ensure_output_dir()
    
    aggregator = run_analysis(
        sample_files=args.sample,
        resume=args.resume,
        diagnostic=args.diagnostic
    )
    
    if aggregator:
        summary = print_delay_analysis(aggregator)
        print_key_findings(summary)
        generate_report(summary)
        save_results_json(summary)
        
        log("\n" + "="*70)
        log("ANALYSIS COMPLETE")
        log("="*70)
        log(f"\nOutputs generated:")
        log(f"  - Report: phase4e_delay_analysis_{TIMESTAMP}_report.txt")
        log(f"  - JSON:   phase4e_delay_analysis_{TIMESTAMP}.json")
    else:
        log("\nAnalysis failed - no results generated")