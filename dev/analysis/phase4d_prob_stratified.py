#!/usr/bin/env python3
"""
Phase 4D: Probability-Stratified Reaction Timing Analysis - MEMORY SAFE VERSION
Version: 1.0

OBJECTIVE:
  Extend Phase 4C analysis by stratifying edge by STARTING PROBABILITY.
  This answers: "In high-probability (bond-like) markets, does fading dips work,
  or is the edge concentrated in toss-up markets?"
  
HYPOTHESIS:
  Based on prior work suggesting dips in high-probability markets are more likely
  informed flow, we expect:
  - Edge concentrated in 51-75% probability range (toss-up markets)
  - Reduced or negative edge in 90%+ markets (informed flow dominates)
  - This would explain the drift from bond-like to toss-up strategy

NEW FEATURES vs 4C:
  1. Stratification by pre-dip probability bucket
  2. Fill rate sensitivity analysis (degraded fill rate scenarios)
  3. Preserved tercile/velocity analysis within each probability bucket

METHODOLOGY:
  Same as Phase 4C, with additional stratification:
  - For each (interval, threshold, prob_bucket, tercile): compute edge
  - Prob buckets: sub_51, 51_60, 60_75, 75_90, 90_99, all
  - Fill sensitivity: recompute edge assuming 80%, 70%, 60% fill rates

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
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'phase4d_checkpoint.pkl')

# ------------------------------------------------------------------------------
# ANALYSIS PARAMETERS
# ------------------------------------------------------------------------------

# Interval pairs for analysis
INTERVAL_PAIRS = [
    (48, 24, '48h_to_24h'),
    (24, 12, '24h_to_12h'),
    (12, 6, '12h_to_6h'),
    (9, 6, '9h_to_6h'),
    (8, 4, '8h_to_4h'),
    (6, 4, '6h_to_4h'),
]

# Move thresholds (fractional/absolute points)
MOVE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]

# Reaction timing terciles
TERCILE_LABELS = ['early', 'mid', 'late']
TERCILE_BOUNDS = [(0.0, 0.333), (0.333, 0.666), (0.666, 1.0)]

# NEW: Probability buckets for stratification (by start_price)
# These represent the pre-dip probability
PROB_BUCKETS = [
    ('sub_51', 0.0, 0.51),      # Longshots (for completeness)
    ('51_60', 0.51, 0.60),      # Toss-up favoring YES
    ('60_75', 0.60, 0.75),      # Moderate favorite
    ('75_90', 0.75, 0.90),      # Strong favorite  
    ('90_99', 0.90, 0.99),      # Heavy favorite (bond-adjacent)
    ('99_plus', 0.99, 1.01),    # Near-certain (bonds)
]
PROB_BUCKET_LABELS = [b[0] for b in PROB_BUCKETS] + ['all']

# Fill rate sensitivity scenarios
FILL_SENSITIVITY_RATES = [1.0, 0.9, 0.8, 0.7, 0.6]

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
# TIMING AGGREGATOR WITH PROBABILITY STRATIFICATION
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
    """Create a cell that tracks all terciles for one (interval, threshold, prob_bucket)."""
    return {
        'total_samples': 0,
        'total_with_crossing': 0,
        'terciles': {label: _make_tercile_cell() for label in TERCILE_LABELS},
        'fraction_distribution': StreamingStats(),
        'start_price_distribution': StreamingStats(),  # NEW: track start prices
    }


def get_prob_bucket(start_price):
    """Determine which probability bucket a start_price falls into."""
    for label, lo, hi in PROB_BUCKETS:
        if lo <= start_price < hi:
            return label
    return None


class TimingAggregator:
    """Aggregate reaction timing results with probability stratification."""
    
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
                        'fraction_distribution': cell['fraction_distribution'].get_stats(),
                        'start_price_distribution': cell['start_price_distribution'].get_stats(),
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
                                'n_fills': n_fills,
                                'n_wins': n_wins,
                                'n_fill_wins': n_fill_wins,
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
                            'n_wins': n_wins,
                            'n_fill_wins': n_fill_wins,
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
                    
                    summary['surface'][interval_label][threshold][prob_bucket] = cell_summary
        
        # Add fill rate sensitivity analysis
        summary['fill_sensitivity'] = compute_fill_sensitivity(summary)
        
        return summary


def compute_fill_sensitivity(summary):
    """
    Compute edge under degraded fill rate assumptions.
    
    If actual fill rate is 85% but we only achieve 70% in practice,
    what does edge look like? We assume the missed fills are random
    (not adversely selected), which is optimistic.
    """
    sensitivity = {}
    
    for interval_label in [x[2] for x in INTERVAL_PAIRS]:
        sensitivity[interval_label] = {}
        
        for threshold in MOVE_THRESHOLDS:
            sensitivity[interval_label][threshold] = {}
            
            # Only compute for 'all' bucket to keep output manageable
            cell = summary['surface'].get(interval_label, {}).get(threshold, {}).get('all', {})
            if not cell or cell['total_with_crossing'] < 50:
                continue
            
            # Get early tercile data (our primary trading signal)
            early = cell['terciles'].get('early', {})
            if early.get('status') != 'ok':
                continue
            
            n_samples = early['n_samples']
            n_fills = early['n_fills']
            n_fill_wins = early['n_fill_wins']
            actual_fill_rate = early['fill_rate']
            cond_win_rate = early.get('conditional_win_rate', 0)
            avg_fill_price = early.get('avg_fill_price', 0)
            
            if n_fills < 20:
                continue
            
            sensitivity[interval_label][threshold] = {
                'actual_fill_rate': actual_fill_rate,
                'actual_edge_bps': early.get('edge_after_fill_bps', 0),
                'scenarios': {}
            }
            
            for target_rate in FILL_SENSITIVITY_RATES:
                if target_rate > actual_fill_rate:
                    # Can't have higher fill rate than actual
                    continue
                
                # Scale down fills proportionally
                # Assumption: missed fills are random, not adversely selected
                scale = target_rate / actual_fill_rate if actual_fill_rate > 0 else 0
                
                degraded_fills = int(n_fills * scale)
                degraded_fill_wins = int(n_fill_wins * scale)
                
                if degraded_fills < 10:
                    continue
                
                degraded_win_rate = degraded_fill_wins / degraded_fills if degraded_fills > 0 else 0
                
                # Edge under degraded scenario
                # Note: avg_fill_price stays same (assumption: we still get same prices when filled)
                degraded_edge = (degraded_win_rate - avg_fill_price) * 10000
                
                sensitivity[interval_label][threshold]['scenarios'][f'{int(target_rate*100)}pct'] = {
                    'fill_rate': target_rate,
                    'edge_bps': degraded_edge,
                    'n_fills': degraded_fills,
                }
    
    return sensitivity


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
# CHECKPOINT HANDLING (read-only for cache, new file for phase4d)
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
    
    Returns crossing details including time, fraction of interval, velocity.
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
            
            velocity = threshold / time_elapsed_hours if time_elapsed_hours > 0 else float('inf')
            
            return {
                'crossed': True,
                'crossing_time': ts,
                'crossing_price': price,
                'fraction_of_interval': fraction,
                'time_to_crossing_hours': time_elapsed_hours,
                'velocity': velocity,
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
    """Run the probability-stratified timing analysis."""
    global start_time
    start_time = datetime.now()
    
    log("="*70)
    log("PHASE 4D: PROBABILITY-STRATIFIED TIMING ANALYSIS")
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
        log(f"\nFinal flush: {len(remaining_conditions)} remaining conditions...")
        
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
            
            interval_data = compute_interval_data_with_timing(trades, resolution_time)
            
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
    log("PROBABILITY-STRATIFIED TIMING ANALYSIS RESULTS")
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
    
    # For each probability bucket, show edge comparison
    log("\n" + "="*70)
    log("EDGE BY PROBABILITY BUCKET (Early Tercile Only)")
    log("="*70)
    log("This shows whether edge varies by pre-dip probability level.")
    log("Key question: Is edge concentrated in toss-up markets or available in high-prob too?")
    
    # Summary table: one row per prob bucket, columns for different thresholds
    log(f"\n{'Prob Bucket':>12} | {'10% Thresh':>12} | {'15% Thresh':>12} | {'20% Thresh':>12} | {'n (10%)':>10}")
    log("-"*70)
    
    for prob_bucket in PROB_BUCKET_LABELS:
        row = f"{prob_bucket:>12} |"
        n_10 = 0
        
        for threshold in [0.10, 0.15, 0.20]:
            # Use 8h_to_4h as reference interval (best edge from 4C)
            cell = summary['surface'].get('8h_to_4h', {}).get(threshold, {}).get(prob_bucket, {})
            early = cell.get('terciles', {}).get('early', {})
            
            if early.get('status') == 'ok':
                edge = early.get('edge_after_fill_bps', 0)
                se = early.get('se_edge_after_fill', 0)
                row += f" {edge:>+6.0f}±{se:<4.0f} |"
                if threshold == 0.10:
                    n_10 = early.get('n_samples', 0)
            else:
                row += f" {'n/a':>12} |"
        
        row += f" {n_10:>10,}"
        log(row)
    
    # Detailed view for 'all' bucket (backward compatible with 4C)
    log("\n" + "="*70)
    log("DETAILED TERCILE ANALYSIS (All Probability Buckets Combined)")
    log("="*70)
    
    for _, _, interval_label in INTERVAL_PAIRS:
        for threshold in MOVE_THRESHOLDS:
            cell = summary['surface'].get(interval_label, {}).get(threshold, {}).get('all', {})
            if not cell:
                continue
            
            if cell['total_with_crossing'] < 50:
                continue
            
            log(f"\n{interval_label} | >= {threshold*100:.0f}% drop")
            log(f"  Crossing rate: {cell['crossing_rate']*100:.1f}%")
            
            frac_dist = cell.get('fraction_distribution')
            if frac_dist and frac_dist.get('n', 0) > 0:
                log(f"  Time-to-crossing distribution: median={frac_dist.get('p50', 0)*100:.0f}% of interval")
            
            log(f"  {'Tercile':<8} | {'n':>6} | {'Uncond Edge':>12} | {'Fill Rate':>10} | {'Edge After Fill':>15}")
            log("  " + "-"*60)
            
            for tercile in TERCILE_LABELS:
                tercile_data = cell['terciles'].get(tercile, {})
                
                if tercile_data.get('status') != 'ok':
                    n = tercile_data.get('n_samples', 0)
                    log(f"  {tercile:<8} | {n:>6} | {'(insufficient data)':<40}")
                    continue
                
                n = tercile_data['n_samples']
                uncond = tercile_data['unconditional_edge_bps']
                se_uncond = tercile_data['se_unconditional_edge']
                fill_rate = tercile_data['fill_rate']
                
                edge_after = tercile_data.get('edge_after_fill_bps')
                se_after = tercile_data.get('se_edge_after_fill', 0)
                
                if edge_after is not None:
                    edge_str = f"{edge_after:+.0f}±{se_after:.0f}"
                else:
                    edge_str = "n/a"
                
                log(f"  {tercile:<8} | {n:>6} | {uncond:+8.0f}±{se_uncond:.0f} bps | {fill_rate:>8.1%} | {edge_str:>15}")
    
    return summary


def print_fill_sensitivity(summary):
    """Print fill rate sensitivity analysis."""
    log("\n" + "="*70)
    log("FILL RATE SENSITIVITY ANALYSIS")
    log("="*70)
    log("What happens to edge if actual fill rates are lower than simulated?")
    log("(Assumes missed fills are random, not adversely selected - optimistic)")
    
    sensitivity = summary.get('fill_sensitivity', {})
    
    for interval_label in [x[2] for x in INTERVAL_PAIRS]:
        for threshold in MOVE_THRESHOLDS:
            sens_data = sensitivity.get(interval_label, {}).get(threshold, {})
            if not sens_data:
                continue
            
            log(f"\n{interval_label} | {threshold*100:.0f}% threshold (early tercile):")
            log(f"  Actual fill rate: {sens_data['actual_fill_rate']*100:.1f}%")
            log(f"  Actual edge: {sens_data['actual_edge_bps']:+.0f} bps")
            
            scenarios = sens_data.get('scenarios', {})
            if scenarios:
                log(f"  Degraded scenarios:")
                for scenario_name, scenario_data in sorted(scenarios.items(), reverse=True):
                    log(f"    {scenario_name}: edge = {scenario_data['edge_bps']:+.0f} bps (n_fills={scenario_data['n_fills']})")


def print_key_findings(summary):
    """Print key findings and operational implications."""
    log("\n" + "="*70)
    log("KEY FINDINGS")
    log("="*70)
    
    # 1. Edge by probability bucket
    log("\n1. EDGE BY PROBABILITY BUCKET")
    log("-"*50)
    
    bucket_edges = {}
    for prob_bucket in [b[0] for b in PROB_BUCKETS]:
        # Aggregate edge across intervals for this bucket
        edges = []
        for interval_label in [x[2] for x in INTERVAL_PAIRS]:
            for threshold in [0.10, 0.15, 0.20]:
                cell = summary['surface'].get(interval_label, {}).get(threshold, {}).get(prob_bucket, {})
                early = cell.get('terciles', {}).get('early', {})
                if early.get('status') == 'ok' and early.get('n_samples', 0) >= 50:
                    edges.append(early.get('edge_after_fill_bps', 0))
        
        if edges:
            bucket_edges[prob_bucket] = np.mean(edges)
            log(f"  {prob_bucket}: avg edge = {bucket_edges[prob_bucket]:+.0f} bps (across {len(edges)} cells)")
    
    if bucket_edges:
        best_bucket = max(bucket_edges, key=bucket_edges.get)
        worst_bucket = min(bucket_edges, key=bucket_edges.get)
        log(f"\n  Best bucket: {best_bucket} ({bucket_edges[best_bucket]:+.0f} bps)")
        log(f"  Worst bucket: {worst_bucket} ({bucket_edges[worst_bucket]:+.0f} bps)")
        
        # Check if high-prob buckets have positive edge
        high_prob_edge = bucket_edges.get('90_99', None)
        toss_up_edge = bucket_edges.get('51_60', None)
        
        if high_prob_edge is not None and toss_up_edge is not None:
            if high_prob_edge > 50:
                log(f"\n  FINDING: High-probability markets (90-99%) show positive edge ({high_prob_edge:+.0f} bps)")
                log(f"           Bond-adjacent strategy may be viable.")
            elif high_prob_edge < -50:
                log(f"\n  FINDING: High-probability markets show NEGATIVE edge ({high_prob_edge:+.0f} bps)")
                log(f"           Confirms informed flow dominates in bond-like markets.")
            else:
                log(f"\n  FINDING: High-probability markets show near-zero edge ({high_prob_edge:+.0f} bps)")
                log(f"           Toss-up markets ({toss_up_edge:+.0f} bps) appear to be where edge concentrates.")
    
    # 2. Timing effect (early vs late)
    log("\n2. TIMING EFFECT (Early vs Late Reaction)")
    log("-"*50)
    
    timing_effects = []
    for interval_label in [x[2] for x in INTERVAL_PAIRS]:
        for threshold in MOVE_THRESHOLDS:
            cell = summary['surface'].get(interval_label, {}).get(threshold, {}).get('all', {})
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
        log(f"  Average early vs late difference: {avg_diff:+.0f} bps")
        
        if avg_diff > 100:
            log(f"  FINDING: Strong timing effect - react early to capture edge.")
        elif avg_diff > 20:
            log(f"  FINDING: Moderate timing effect - early reaction preferred.")
        else:
            log(f"  FINDING: Weak timing effect - reaction speed less critical.")
    
    # 3. Fill sensitivity
    log("\n3. FILL RATE VULNERABILITY")
    log("-"*50)
    
    sensitivity = summary.get('fill_sensitivity', {})
    critical_scenarios = []
    
    for interval_label in [x[2] for x in INTERVAL_PAIRS]:
        for threshold in MOVE_THRESHOLDS:
            sens_data = sensitivity.get(interval_label, {}).get(threshold, {})
            if not sens_data:
                continue
            
            actual_edge = sens_data.get('actual_edge_bps', 0)
            scenarios = sens_data.get('scenarios', {})
            
            # Check 70% fill rate scenario
            seventy_pct = scenarios.get('70pct', {})
            if seventy_pct:
                degraded_edge = seventy_pct.get('edge_bps', 0)
                if actual_edge > 100 and degraded_edge < 50:
                    critical_scenarios.append({
                        'interval': interval_label,
                        'threshold': threshold,
                        'actual': actual_edge,
                        'degraded': degraded_edge,
                    })
    
    if critical_scenarios:
        log(f"  WARNING: {len(critical_scenarios)} scenario(s) show significant edge degradation at 70% fill rate:")
        for scenario in critical_scenarios[:3]:
            log(f"    {scenario['interval']} {scenario['threshold']*100:.0f}%: {scenario['actual']:+.0f} -> {scenario['degraded']:+.0f} bps")
    else:
        log(f"  Edge appears robust to moderate fill rate degradation.")


def save_results_json(summary):
    """Save results to JSON file."""
    ensure_output_dir()
    
    output = {
        'timestamp': TIMESTAMP,
        'phase': '4D',
        'description': 'Probability-Stratified Timing Analysis',
        'hypothesis': 'Does edge vary by pre-dip probability level?',
        'methodology': 'Same as 4C with additional stratification by starting probability',
        'interval_pairs': [(s, e, l) for s, e, l in INTERVAL_PAIRS],
        'move_thresholds': MOVE_THRESHOLDS,
        'tercile_bounds': list(zip(TERCILE_LABELS, TERCILE_BOUNDS)),
        'prob_buckets': PROB_BUCKETS,
        **summary,
    }
    
    json_path = os.path.join(OUTPUT_DIR, f'phase4d_prob_stratified_{TIMESTAMP}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    log(f"\nResults JSON saved: {json_path}")
    return json_path


def generate_report(summary):
    """Generate text report."""
    ensure_output_dir()
    
    report_path = os.path.join(OUTPUT_DIR, f'phase4d_prob_stratified_{TIMESTAMP}_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 4D: PROBABILITY-STRATIFIED TIMING ANALYSIS - REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tokens Analyzed: {summary.get('n_tokens', 0):,}\n")
        f.write(f"Tokens with Data: {summary.get('n_tokens_with_data', 0):,}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("HYPOTHESIS\n")
        f.write("-"*80 + "\n")
        f.write("Does edge vary by pre-dip probability level?\n\n")
        f.write("Prior work suggested dips in high-probability markets are more likely\n")
        f.write("informed flow, while toss-up markets may have more noise to exploit.\n\n")
        
        f.write("-"*80 + "\n")
        f.write("PROBABILITY BUCKETS\n")
        f.write("-"*80 + "\n")
        for label, lo, hi in PROB_BUCKETS:
            f.write(f"  {label}: {lo*100:.0f}% - {hi*100:.0f}%\n")
        
        f.write("\n")
        f.write("-"*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n")
        
        # Summarize edge by bucket
        f.write("\nEdge by Probability Bucket (Early Tercile, 8h->4h interval):\n\n")
        
        for prob_bucket in PROB_BUCKET_LABELS:
            cell = summary['surface'].get('8h_to_4h', {}).get(0.15, {}).get(prob_bucket, {})
            early = cell.get('terciles', {}).get('early', {})
            
            if early.get('status') == 'ok':
                edge = early.get('edge_after_fill_bps', 0)
                n = early.get('n_samples', 0)
                f.write(f"  {prob_bucket:>12}: {edge:+6.0f} bps (n={n:,})\n")
            else:
                f.write(f"  {prob_bucket:>12}: insufficient data\n")
        
        f.write("\nSee JSON file for full data.\n")
    
    log(f"Report saved: {report_path}")
    return report_path


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 4D: Probability-Stratified Timing Analysis (Memory-Safe)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnostic mode with small sample
  python phase4d_prob_stratified.py --diagnostic --sample 100
  
  # Full run
  python phase4d_prob_stratified.py
  
  # Resume interrupted run
  python phase4d_prob_stratified.py --resume
        """
    )
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--diagnostic', '-d', action='store_true',
                        help='Run in diagnostic mode with extra validation')
    
    args = parser.parse_args()
    
    log("Starting Phase 4D Probability-Stratified Timing Analysis...")
    
    ensure_output_dir()
    
    aggregator = run_analysis(
        sample_files=args.sample,
        resume=args.resume,
        diagnostic=args.diagnostic
    )
    
    if aggregator:
        summary = print_timing_analysis(aggregator)
        print_fill_sensitivity(summary)
        print_key_findings(summary)
        generate_report(summary)
        save_results_json(summary)
        
        log("\n" + "="*70)
        log("ANALYSIS COMPLETE")
        log("="*70)
        log(f"\nOutputs generated:")
        log(f"  - Report: phase4d_prob_stratified_{TIMESTAMP}_report.txt")
        log(f"  - JSON:   phase4d_prob_stratified_{TIMESTAMP}.json")
    else:
        log("\nAnalysis failed - no results generated")