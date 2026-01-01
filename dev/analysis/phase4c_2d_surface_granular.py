#!/usr/bin/env python3
"""
Phase 4C: Reaction Timing Analysis - MEMORY SAFE VERSION
Version: 1.0

OBJECTIVE:
  Test whether faster reaction to threshold crossings yields more edge.
  For each price drop that exceeds a threshold within an interval:
  - Record WHEN the threshold was first crossed (not just that it was crossed)
  - Bucket by "early" vs "mid" vs "late" crossing within the interval
  - Compute conditional edge by reaction timing bucket
  
HYPOTHESIS:
  If overreactions mean-revert, earlier reaction = more edge captured.
  The data showing 9h→6h > 12h→6h suggests faster moves have higher edge,
  but this tests whether reacting sooner to ANY threshold crossing matters.

METHODOLOGY:
  For each interval (e.g., 48h→24h):
    1. Find price at T-48h (start_price)
    2. Scan trades from T-48h to T-24h to find FIRST time price drops ≥threshold
    3. Record: crossing_time, fraction_of_interval, velocity
    4. Bucket into terciles: early (0-33%), mid (33-66%), late (66-100%)
    5. Simulate fill at crossing_time (not at end of interval)
    6. Compute edge by tercile
    
OUTPUT:
  - Distribution of time-to-threshold within intervals
  - Conditional edge by early/mid/late crossing
  - Velocity analysis: does move speed predict edge?

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
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'phase4c_checkpoint.pkl')

# ------------------------------------------------------------------------------
# ANALYSIS PARAMETERS
# ------------------------------------------------------------------------------

# Interval pairs for analysis
# Focus on intervals where we know fading works (≥4h end point)
INTERVAL_PAIRS = [
    (48, 24, '48h_to_24h'),
    (24, 12, '24h_to_12h'),
    (12, 6, '12h_to_6h'),
    (9, 6, '9h_to_6h'),
    (8, 4, '8h_to_4h'),
    (6, 4, '6h_to_4h'),
]

# Move thresholds (fractional)
MOVE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]

# Reaction timing terciles
TERCILE_LABELS = ['early', 'mid', 'late']
TERCILE_BOUNDS = [(0.0, 0.333), (0.333, 0.666), (0.666, 1.0)]

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
# TIMING AGGREGATOR
# ==============================================================================

def _make_tercile_cell():
    """Create a cell for one tercile bucket."""
    return {
        'n_samples': 0,
        'n_wins': 0,
        'n_fills': 0,
        'n_fill_wins': 0,
        'crossing_prices': StreamingStats(),
        'fill_prices': StreamingStats(),
        'velocities': StreamingStats(),
        'time_to_crossing_hours': StreamingStats(),
    }


def _make_timing_cell():
    """Create a cell that tracks all terciles for one (interval, threshold) pair."""
    return {
        'total_samples': 0,
        'total_with_crossing': 0,
        'terciles': {label: _make_tercile_cell() for label in TERCILE_LABELS},
        'fraction_distribution': StreamingStats(),  # Distribution of crossing fractions
    }


class TimingAggregator:
    """Aggregate reaction timing results incrementally."""
    
    def __init__(self):
        self.n_tokens = 0
        self.n_tokens_with_data = 0
        
        # Main structure: [interval_label][threshold] -> timing_cell
        self.surface = {}
        for _, _, interval_label in INTERVAL_PAIRS:
            self.surface[interval_label] = {}
            for threshold in MOVE_THRESHOLDS:
                self.surface[interval_label][threshold] = _make_timing_cell()
        
        self.interval_coverage = defaultdict(int)
    
    def add_result(self, result):
        """Add a single token result to aggregates."""
        self.n_tokens += 1
        
        if not result.get('interval_data'):
            return
        
        self.n_tokens_with_data += 1
        winner = result['winner']
        
        for interval_label, interval_data in result['interval_data'].items():
            self.interval_coverage[interval_label] += 1
            
            # Process each threshold
            for threshold in MOVE_THRESHOLDS:
                crossing_data = interval_data.get('crossings', {}).get(threshold)
                
                cell = self.surface[interval_label][threshold]
                cell['total_samples'] += 1
                
                if crossing_data is None or not crossing_data.get('crossed'):
                    continue
                
                cell['total_with_crossing'] += 1
                
                # Determine tercile
                fraction = crossing_data['fraction_of_interval']
                cell['fraction_distribution'].update(fraction)
                
                tercile = None
                for label, (lo, hi) in zip(TERCILE_LABELS, TERCILE_BOUNDS):
                    if lo <= fraction < hi or (label == 'late' and fraction == 1.0):
                        tercile = label
                        break
                
                if tercile is None:
                    continue
                
                tercile_cell = cell['terciles'][tercile]
                tercile_cell['n_samples'] += 1
                
                if winner:
                    tercile_cell['n_wins'] += 1
                
                tercile_cell['crossing_prices'].update(crossing_data['crossing_price'])
                tercile_cell['velocities'].update(crossing_data.get('velocity'))
                tercile_cell['time_to_crossing_hours'].update(crossing_data.get('time_to_crossing_hours'))
                
                # Fill simulation
                fill_data = crossing_data.get('fill_simulation')
                if fill_data and fill_data.get('filled'):
                    tercile_cell['n_fills'] += 1
                    if winner:
                        tercile_cell['n_fill_wins'] += 1
                    
                    fill_price = fill_data.get('fill_price')
                    if fill_price is not None:
                        tercile_cell['fill_prices'].update(fill_price)
    
    def get_summary(self):
        """Get aggregated summary statistics."""
        summary = {
            'n_tokens': self.n_tokens,
            'n_tokens_with_data': self.n_tokens_with_data,
            'interval_coverage': dict(self.interval_coverage),
            'surface': {},
        }
        
        for interval_label in [x[2] for x in INTERVAL_PAIRS]:
            summary['surface'][interval_label] = {}
            
            for threshold in MOVE_THRESHOLDS:
                cell = self.surface[interval_label][threshold]
                
                # Overall stats for this (interval, threshold)
                cell_summary = {
                    'total_samples': cell['total_samples'],
                    'total_with_crossing': cell['total_with_crossing'],
                    'crossing_rate': cell['total_with_crossing'] / cell['total_samples'] if cell['total_samples'] > 0 else 0,
                    'fraction_distribution': cell['fraction_distribution'].get_stats(),
                    'terciles': {},
                }
                
                # Per-tercile stats
                for tercile_label in TERCILE_LABELS:
                    tercile_cell = cell['terciles'][tercile_label]
                    
                    n = tercile_cell['n_samples']
                    n_wins = tercile_cell['n_wins']
                    n_fills = tercile_cell['n_fills']
                    n_fill_wins = tercile_cell['n_fill_wins']
                    
                    if n < MIN_SAMPLES_PER_CELL:
                        cell_summary['terciles'][tercile_label] = {
                            'n_samples': n,
                            'status': 'insufficient_data',
                        }
                        continue
                    
                    # Unconditional metrics
                    uncond_win_rate = n_wins / n if n > 0 else 0
                    crossing_stats = tercile_cell['crossing_prices'].get_stats()
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
                        fill_stats = tercile_cell['fill_prices'].get_stats()
                        avg_fill = fill_stats['mean'] if fill_stats else 0
                        
                        edge_after_fill_bps = (cond_win_rate - avg_fill) * 10000
                        se_cond = np.sqrt(cond_win_rate * (1 - cond_win_rate) / n_fills) * 10000 if n_fills > 0 else 0
                        
                        cond_result = {
                            'conditional_win_rate': cond_win_rate,
                            'avg_fill_price': avg_fill,
                            'edge_after_fill_bps': edge_after_fill_bps,
                            'se_edge_after_fill': se_cond,
                        }
                    
                    # Velocity stats
                    velocity_stats = tercile_cell['velocities'].get_stats()
                    timing_stats = tercile_cell['time_to_crossing_hours'].get_stats()
                    
                    cell_summary['terciles'][tercile_label] = {
                        'n_samples': n,
                        'n_fills': n_fills,
                        'unconditional_win_rate': uncond_win_rate,
                        'avg_crossing_price': avg_crossing_price,
                        'unconditional_edge_bps': uncond_edge_bps,
                        'se_unconditional_edge': se_uncond,
                        'fill_rate': fill_rate,
                        'velocity_stats': velocity_stats,
                        'timing_stats': timing_stats,
                        **cond_result,
                        'status': 'ok',
                    }
                
                summary['surface'][interval_label][threshold] = cell_summary
        
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
# PRICE EXTRACTION AND THRESHOLD CROSSING DETECTION
# ==============================================================================

def extract_price_at_horizon(trades, resolution_time, hours_before):
    """Extract price closest to specific time horizon before resolution."""
    target_time = resolution_time - (hours_before * 3600)
    
    # Tolerance based on horizon length
    if hours_before > 12:
        tolerance_hours = hours_before * 0.25
    elif hours_before >= 4:
        tolerance_hours = hours_before * 0.20
    else:
        tolerance_hours = max(0.5, hours_before * 0.15)
    
    tolerance_seconds = tolerance_hours * 3600
    
    best_trade = None
    best_distance = float('inf')
    
    for ts, price, size in trades:
        distance = abs(ts - target_time)
        if distance < best_distance and distance < tolerance_seconds:
            best_distance = distance
            best_trade = (ts, price)
    
    return best_trade if best_trade else (None, None)


def find_first_threshold_crossing(trades, start_time, end_time, start_price, threshold, direction='drop'):
    """
    Find the FIRST time the price crosses the threshold within the interval.
    
    For drops: looking for when price falls ≥threshold below start_price
    For rises: looking for when price rises ≥threshold above start_price
    
    Returns dict with:
    - crossed: bool
    - crossing_time: timestamp when threshold first crossed
    - crossing_price: price at crossing
    - fraction_of_interval: (crossing_time - start_time) / (end_time - start_time)
    - velocity: drop_size / time_elapsed (in %/hour)
    - time_to_crossing_hours: how long after start the crossing occurred
    """
    if start_time is None or end_time is None or start_price is None:
        return None
    
    interval_length = end_time - start_time
    if interval_length <= 0:
        return None
    
    # Get trades within the interval
    interval_trades = [(ts, price, size) for ts, price, size in trades 
                       if start_time <= ts <= end_time]
    
    if not interval_trades:
        return {'crossed': False}
    
    # Sort by timestamp
    interval_trades.sort(key=lambda x: x[0])
    
    # Scan for first threshold crossing
    for ts, price, size in interval_trades:
        if direction == 'drop':
            move = start_price - price
        else:
            move = price - start_price
        
        if move >= threshold:
            time_elapsed = ts - start_time
            time_elapsed_hours = time_elapsed / 3600
            fraction = time_elapsed / interval_length
            
            # Velocity: threshold size / time to reach it (in %/hour)
            velocity = (threshold / time_elapsed_hours) if time_elapsed_hours > 0 else float('inf')
            
            return {
                'crossed': True,
                'crossing_time': ts,
                'crossing_price': price,
                'fraction_of_interval': fraction,
                'time_to_crossing_hours': time_elapsed_hours,
                'velocity': velocity,  # threshold per hour (e.g., 0.10 / 2h = 0.05/hour = 5%/hour)
                'move_at_crossing': move,
            }
    
    # Check if threshold was crossed by end of interval
    # (price at end vs start)
    final_price = interval_trades[-1][1]
    if direction == 'drop':
        final_move = start_price - final_price
    else:
        final_move = final_price - start_price
    
    if final_move >= threshold:
        # Threshold crossed but we don't have the exact moment
        # This shouldn't happen if we have good coverage, but handle it
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


def compute_interval_data_with_timing(trades, resolution_time):
    """
    Compute interval data with threshold crossing timing.
    
    For each interval:
    - Find start and end prices
    - For each threshold, find WHEN it was first crossed
    - Simulate fill at crossing time (not end of interval)
    """
    results = {}
    
    for start_h, end_h, label in INTERVAL_PAIRS:
        start_time, start_price = extract_price_at_horizon(trades, resolution_time, start_h)
        end_time, end_price = extract_price_at_horizon(trades, resolution_time, end_h)
        
        if start_price is None or end_price is None:
            continue
        
        if start_time is None or end_time is None:
            continue
        
        # Overall move (for reference)
        overall_move = end_price - start_price
        
        # Find threshold crossings for DROPS
        # (We focus on drops since that's the fading strategy)
        crossings = {}
        
        for threshold in MOVE_THRESHOLDS:
            crossing = find_first_threshold_crossing(
                trades, start_time, end_time, start_price, threshold, direction='drop'
            )
            
            if crossing and crossing.get('crossed'):
                # Simulate fill at crossing time, at crossing price
                fill_result = simulate_limit_order_fill(
                    trades, crossing['crossing_time'], crossing['crossing_price'], is_buy=True
                )
                crossing['fill_simulation'] = fill_result
            
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

class TimingAccumulator:
    """Accumulates trade data for timing analysis."""
    
    def __init__(self, token_id, condition_id, resolution_time, winner_status):
        self.token_id = token_id
        self.condition_id = condition_id
        self.resolution_time = resolution_time
        self.winner_status = winner_status
        self.trades = []
    
    def add_trades(self, trades_batch):
        self.trades.extend(trades_batch)
    
    def compute_timing_data(self):
        """Compute timing data for this token."""
        if len(self.trades) < MIN_TRADES:
            return None
        
        self.trades.sort(key=lambda x: x[0])
        
        interval_data = compute_interval_data_with_timing(self.trades, self.resolution_time)
        
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
    for vc in VOLUME_COLUMNS:
        if vc in available:
            volume_col = vc
            columns_to_read.append(vc)
            break
    
    return columns_to_read, volume_col


# ==============================================================================
# CHECKPOINTING
# ==============================================================================

def save_checkpoint(aggregator, files_processed, total_files):
    """Save checkpoint to unique file."""
    temp_file = CHECKPOINT_FILE + '.tmp'
    
    checkpoint = {
        'aggregator': aggregator,
        'files_processed': files_processed,
        'total_files': total_files,
        'timestamp': datetime.now().isoformat(),
        'version': '4c_v1',
    }
    
    try:
        with open(temp_file, 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        os.replace(temp_file, CHECKPOINT_FILE)
        log(f"  Checkpoint saved: {files_processed}/{total_files} files")
    except Exception as e:
        log(f"  WARNING: Checkpoint save failed: {e}")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


def load_checkpoint():
    """Load checkpoint if available."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
        
        if checkpoint.get('version') != '4c_v1':
            log("  WARNING: Checkpoint version mismatch, ignoring")
            return None
        
        return checkpoint
    except Exception as e:
        log(f"  WARNING: Checkpoint load failed: {e}")
        return None


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def run_analysis(sample_files=None, resume=False, diagnostic=False):
    """Main analysis loop with streaming aggregation."""
    start_time = datetime.now()
    
    log("="*70)
    log("PHASE 4C: REACTION TIMING ANALYSIS (MEMORY-SAFE)")
    log("="*70)
    log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if diagnostic:
        log("MODE: DIAGNOSTIC (extra validation enabled)")
    log("")
    
    # -------------------------------------------------------------------------
    # LOAD BATCH FILES
    # -------------------------------------------------------------------------
    
    batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, '*.parquet')))
    if not batch_files:
        log(f"ERROR: No parquet files found in {BATCH_DIR}")
        return None
    
    log(f"Found {len(batch_files):,} batch files")
    
    # -------------------------------------------------------------------------
    # LOAD WINNER SIDECAR
    # -------------------------------------------------------------------------
    
    winner_lookup = load_winner_sidecar(SIDECAR_FILE)
    if not winner_lookup:
        log("ERROR: Failed to load winner sidecar")
        return None
    
    # -------------------------------------------------------------------------
    # LOAD MARKET INDEX (read-only)
    # -------------------------------------------------------------------------
    
    log("\nLoading market index from cache (read-only)...")
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
    aggregator = TimingAggregator()
    
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
                    
                    token_accumulators[token_id] = TimingAccumulator(
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
                            timing_result = acc.compute_timing_data()
                            
                            if timing_result is not None:
                                aggregator.add_result(timing_result)
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
        log(f"\nFinal flush: {len(remaining_conditions)} remaining conditions, "
            f"{len(token_accumulators):,} tokens...")
    
    for condition_id in remaining_conditions:
        tokens_to_flush = condition_tokens.get(condition_id, set())
        
        for token_id in tokens_to_flush:
            if token_id not in token_accumulators:
                continue
            
            acc = token_accumulators[token_id]
            timing_result = acc.compute_timing_data()
            
            if timing_result is not None:
                aggregator.add_result(timing_result)
            else:
                stats['tokens_filtered'] += 1
            
            del token_accumulators[token_id]
        
        stats['conditions_flushed'] += 1
    
    gc.collect()
    
    log(f"\nProcessed {stats['files_processed']:,} files, {stats['total_rows']:,} rows")
    log(f"Computed timing data for {aggregator.n_tokens_with_data:,} tokens")
    log(f"Conditions flushed: {stats['conditions_flushed']:,}")
    log(f"Skipped: {stats['tokens_no_winner']:,} (no winner), "
        f"{stats['tokens_filtered']:,} (insufficient trades)")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"\nTotal time: {format_duration(elapsed)}")
    log(f"Final {log_memory()}")
    
    # Remove checkpoint on successful completion
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            log("Checkpoint removed (run complete)")
        except:
            pass
    
    # -------------------------------------------------------------------------
    # DIAGNOSTIC VALIDATION
    # -------------------------------------------------------------------------
    
    if diagnostic and diagnostic_samples:
        log("\n" + "="*70)
        log("DIAGNOSTIC VALIDATION")
        log("="*70)
        
        for sample in diagnostic_samples:
            log(f"\nToken: {sample['token_id'][:20]}...")
            log(f"  Winner: {sample['winner']}")
            log(f"  Total trades: {sample['n_trades']}")
            
            trades = sample['trades']
            resolution_time = sample['resolution_time']
            
            interval_data = compute_interval_data_with_timing(trades, resolution_time)
            
            for label, data in interval_data.items():
                log(f"\n  {label}:")
                log(f"    Start: {data['start_price']:.4f} @ T-{data['start_hours']}h")
                log(f"    End:   {data['end_price']:.4f} @ T-{data['end_hours']}h")
                log(f"    Overall move: {data['overall_move']:+.4f}")
                
                crossings = data.get('crossings', {})
                for threshold, crossing in crossings.items():
                    if crossing and crossing.get('crossed'):
                        log(f"    Threshold {threshold*100:.0f}%: CROSSED")
                        log(f"      Time to crossing: {crossing['time_to_crossing_hours']:.2f}h")
                        log(f"      Fraction of interval: {crossing['fraction_of_interval']:.2f}")
                        log(f"      Velocity: {crossing['velocity']*100:.1f}%/hour")
                        log(f"      Price at crossing: {crossing['crossing_price']:.4f}")
                        
                        fill = crossing.get('fill_simulation', {})
                        if fill.get('filled'):
                            log(f"      Fill: YES @ {fill['fill_price']:.4f}")
                        else:
                            log(f"      Fill: NO")
                    else:
                        log(f"    Threshold {threshold*100:.0f}%: not crossed")
    
    return aggregator


# ==============================================================================
# REPORTING
# ==============================================================================

def print_timing_analysis(aggregator):
    """Print timing analysis from streaming aggregator."""
    summary = aggregator.get_summary()
    
    log("\n" + "="*70)
    log("REACTION TIMING ANALYSIS RESULTS")
    log("="*70)
    
    log(f"\nSample Size: {summary['n_tokens']:,} tokens")
    log(f"Tokens with interval data: {summary['n_tokens_with_data']:,}")
    
    log("\n" + "-"*50)
    log("INTERVAL COVERAGE")
    log("-"*50)
    
    for _, _, interval_label in INTERVAL_PAIRS:
        coverage = summary['interval_coverage'].get(interval_label, 0)
        log(f"  {interval_label}: {coverage:,} tokens")
    
    # For each interval and threshold, show edge by tercile
    log("\n" + "="*70)
    log("EDGE BY REACTION TIMING TERCILE")
    log("="*70)
    log("Does reacting earlier to threshold crossings yield more edge?")
    log("Terciles: early (0-33% of interval), mid (33-66%), late (66-100%)")
    
    for _, _, interval_label in INTERVAL_PAIRS:
        for threshold in MOVE_THRESHOLDS:
            cell = summary['surface'].get(interval_label, {}).get(threshold)
            if not cell:
                continue
            
            if cell['total_with_crossing'] < 50:
                continue
            
            log(f"\n{interval_label} | >= {threshold*100:.0f}% drop")
            log(f"  Crossing rate: {cell['crossing_rate']*100:.1f}%")
            
            # Distribution of crossing times
            frac_dist = cell.get('fraction_distribution')
            if frac_dist and frac_dist.get('n', 0) > 0:
                log(f"  Time-to-crossing distribution: median={frac_dist.get('p50', 0)*100:.0f}% of interval")
            
            log(f"  {'Tercile':<8} | {'n':>6} | {'Uncond Edge':>12} | {'Fill Rate':>10} | {'Edge After Fill':>15} | {'Velocity':>12}")
            log("  " + "-"*75)
            
            for tercile in TERCILE_LABELS:
                tercile_data = cell['terciles'].get(tercile, {})
                
                if tercile_data.get('status') != 'ok':
                    n = tercile_data.get('n_samples', 0)
                    log(f"  {tercile:<8} | {n:>6} | {'(insufficient data)':<45}")
                    continue
                
                n = tercile_data['n_samples']
                uncond = tercile_data['unconditional_edge_bps']
                se_uncond = tercile_data['se_unconditional_edge']
                fill_rate = tercile_data['fill_rate']
                
                edge_after = tercile_data.get('edge_after_fill_bps')
                se_after = tercile_data.get('se_edge_after_fill', 0)
                
                velocity_stats = tercile_data.get('velocity_stats', {})
                velocity_median = velocity_stats.get('p50', 0) * 100 if velocity_stats else 0
                
                if edge_after is not None:
                    edge_str = f"{edge_after:+.0f}±{se_after:.0f}"
                else:
                    edge_str = "n/a"
                
                log(f"  {tercile:<8} | {n:>6} | {uncond:+8.0f}±{se_uncond:.0f} bps | {fill_rate:>8.1%} | {edge_str:>15} | {velocity_median:>10.1f}%/hr")
    
    return summary


def print_key_findings(summary):
    """Print key findings and operational implications."""
    log("\n" + "="*70)
    log("KEY FINDINGS")
    log("="*70)
    
    # Collect edge differences across terciles
    timing_effects = []
    
    for interval_label in [x[2] for x in INTERVAL_PAIRS]:
        for threshold in MOVE_THRESHOLDS:
            cell = summary['surface'].get(interval_label, {}).get(threshold)
            if not cell or cell['total_with_crossing'] < 100:
                continue
            
            early = cell['terciles'].get('early', {})
            late = cell['terciles'].get('late', {})
            
            if early.get('status') != 'ok' or late.get('status') != 'ok':
                continue
            
            early_edge = early.get('edge_after_fill_bps')
            late_edge = late.get('edge_after_fill_bps')
            
            if early_edge is not None and late_edge is not None:
                diff = early_edge - late_edge
                timing_effects.append({
                    'interval': interval_label,
                    'threshold': threshold,
                    'early_edge': early_edge,
                    'late_edge': late_edge,
                    'diff': diff,
                    'early_n': early['n_samples'],
                    'late_n': late['n_samples'],
                })
    
    if timing_effects:
        log("\nEarly vs Late Reaction (edge_after_fill):")
        log(f"  {'Interval':<12} | {'Thresh':>6} | {'Early':>10} | {'Late':>10} | {'Diff':>10} | {'n (early/late)'}")
        log("  " + "-"*70)
        
        for effect in sorted(timing_effects, key=lambda x: -x['diff']):
            log(f"  {effect['interval']:<12} | {effect['threshold']*100:>5.0f}% | "
                f"{effect['early_edge']:>+8.0f}bp | {effect['late_edge']:>+8.0f}bp | "
                f"{effect['diff']:>+8.0f}bp | {effect['early_n']:>5}/{effect['late_n']:<5}")
        
        avg_diff = np.mean([e['diff'] for e in timing_effects])
        log(f"\n  Average early vs late difference: {avg_diff:+.0f} bps")
        
        if avg_diff > 20:
            log("\n  FINDING: Early reaction shows meaningfully higher edge than late reaction.")
            log("  This supports the hypothesis that reacting quickly to threshold crossings")
            log("  captures more of the overreaction before mean reversion occurs.")
        elif avg_diff < -20:
            log("\n  FINDING: Late reaction shows higher edge than early reaction.")
            log("  This suggests waiting for confirmation may be beneficial, or that")
            log("  early threshold crossings are more likely to be informative.")
        else:
            log("\n  FINDING: No clear timing effect detected.")
            log("  Edge is similar whether you react early or late within the interval.")


def save_results_json(summary):
    """Save results to JSON file."""
    ensure_output_dir()
    
    output = {
        'timestamp': TIMESTAMP,
        'phase': '4C',
        'description': 'Reaction Timing Analysis',
        'hypothesis': 'Does reacting earlier to threshold crossings yield more edge?',
        'methodology': 'For each threshold crossing, record time within interval and compute edge by tercile',
        'interval_pairs': [(s, e, l) for s, e, l in INTERVAL_PAIRS],
        'move_thresholds': MOVE_THRESHOLDS,
        'tercile_bounds': list(zip(TERCILE_LABELS, TERCILE_BOUNDS)),
        **summary,
    }
    
    json_path = os.path.join(OUTPUT_DIR, f'phase4c_timing_{TIMESTAMP}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    log(f"\nResults JSON saved: {json_path}")
    return json_path


def generate_report(summary):
    """Generate text report."""
    ensure_output_dir()
    
    report_path = os.path.join(OUTPUT_DIR, f'phase4c_timing_{TIMESTAMP}_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 4C: REACTION TIMING ANALYSIS - REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tokens Analyzed: {summary.get('n_tokens', 0):,}\n")
        f.write(f"Tokens with Data: {summary.get('n_tokens_with_data', 0):,}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("HYPOTHESIS\n")
        f.write("-"*80 + "\n")
        f.write("Does reacting earlier to threshold crossings yield more edge?\n\n")
        f.write("If overreactions mean-revert, earlier reaction should capture more edge.\n")
        f.write("We test this by bucketing threshold crossings into terciles:\n")
        f.write("  - Early (0-33% of interval): quick threshold crossings\n")
        f.write("  - Mid (33-66% of interval): moderate timing\n")
        f.write("  - Late (66-100% of interval): threshold barely crossed before interval end\n\n")
        
        f.write("-"*80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("-"*80 + "\n")
        f.write("For each interval (e.g., 48h→24h):\n")
        f.write("  1. Find start price at T-48h\n")
        f.write("  2. Scan trades forward to find FIRST time price drops ≥threshold\n")
        f.write("  3. Record crossing time as fraction of interval\n")
        f.write("  4. Simulate fill at crossing time (not end of interval)\n")
        f.write("  5. Compute edge by tercile bucket\n\n")
        
        f.write("This differs from Phase 4B which placed orders at interval END.\n")
        f.write("Here we test whether placing orders AT THE MOMENT of crossing is better.\n\n")
        
        f.write("-"*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n")
        
        # Summarize timing effects
        timing_effects = []
        for interval_label in [x[2] for x in INTERVAL_PAIRS]:
            for threshold in MOVE_THRESHOLDS:
                cell = summary['surface'].get(interval_label, {}).get(threshold)
                if not cell or cell['total_with_crossing'] < 100:
                    continue
                
                early = cell['terciles'].get('early', {})
                late = cell['terciles'].get('late', {})
                
                if early.get('status') != 'ok' or late.get('status') != 'ok':
                    continue
                
                early_edge = early.get('edge_after_fill_bps')
                late_edge = late.get('edge_after_fill_bps')
                
                if early_edge is not None and late_edge is not None:
                    timing_effects.append(early_edge - late_edge)
        
        if timing_effects:
            avg_diff = np.mean(timing_effects)
            f.write(f"Average early vs late edge difference: {avg_diff:+.0f} bps\n\n")
            
            if avg_diff > 20:
                f.write("CONCLUSION: Early reaction shows meaningfully higher edge.\n")
                f.write("Operational implication: Move to alert-based system that reacts\n")
                f.write("immediately when threshold is crossed, rather than waiting.\n")
            elif avg_diff < -20:
                f.write("CONCLUSION: Late reaction shows higher edge.\n")
                f.write("Operational implication: Wait for confirmation before acting.\n")
            else:
                f.write("CONCLUSION: No clear timing effect detected.\n")
                f.write("Reaction speed within the interval may not be critical.\n")
        
        f.write("\nSee JSON file for full data.\n")
    
    log(f"Report saved: {report_path}")
    return report_path


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 4C: Reaction Timing Analysis (Memory-Safe)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnostic mode with small sample
  python phase4c_reaction_timing.py --diagnostic --sample 100
  
  # Full run
  python phase4c_reaction_timing.py
  
  # Resume interrupted run
  python phase4c_reaction_timing.py --resume
        """
    )
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--diagnostic', '-d', action='store_true',
                        help='Run in diagnostic mode with extra validation')
    
    args = parser.parse_args()
    
    log("Starting Phase 4C Reaction Timing Analysis...")
    
    ensure_output_dir()
    
    aggregator = run_analysis(
        sample_files=args.sample,
        resume=args.resume,
        diagnostic=args.diagnostic
    )
    
    if aggregator:
        summary = print_timing_analysis(aggregator)
        print_key_findings(summary)
        generate_report(summary)
        save_results_json(summary)
        
        log("\n" + "="*70)
        log("ANALYSIS COMPLETE")
        log("="*70)
        log(f"\nOutputs generated:")
        log(f"  - Report: phase4c_timing_{TIMESTAMP}_report.txt")
        log(f"  - JSON:   phase4c_timing_{TIMESTAMP}.json")
    else:
        log("\nAnalysis failed - no results generated")