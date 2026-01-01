#!/usr/bin/env python3
"""
Phase 3C: 2D Temporal Edge Surface Analysis - MEMORY SAFE VERSION
Version: 1.0

OBJECTIVE:
  For a given (interval, move_size) pair, compute the expected edge after fills.
  This extends Phase 2/3B findings by mapping the full parameter surface.

STRUCTURE:
  For intervals in [(48h,24h), (24h,12h), (12h,6h), (6h,3h), (3h,1h)]:
      For move_thresholds in [±2%, ±5%, ±10%, ±15%, ±20%]:
          Compute:
          - n_samples (capacity constraint)
          - unconditional_edge (signal quality)
          - fill_rate (execution probability)  
          - edge_after_fill (net expected return)
          - standard_error (confidence)

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
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'phase3c_checkpoint.pkl')

# ------------------------------------------------------------------------------
# ANALYSIS PARAMETERS
# ------------------------------------------------------------------------------

# Interval pairs: (start_hours_before_resolution, end_hours_before_resolution)
# e.g., (48, 24) means "price at T-48h to T-24h"
#
# Extended set includes finer granularity near the critical 6h→3h transition
# to map where "noise trading" flips to "informed flow"
INTERVAL_PAIRS = [
    # Original coarse intervals
    (48, 24, '48h_to_24h'),
    (24, 12, '24h_to_12h'),
    (12, 6, '12h_to_6h'),
    (6, 3, '6h_to_3h'),
    (3, 1, '3h_to_1h'),
    
    # Fine-grained intervals to map the transition
    (9, 6, '9h_to_6h'),    # Does edge persist above 6h?
    (6, 4, '6h_to_4h'),    # Where exactly does it flip?
    (4, 2, '4h_to_2h'),    # Is 6h→3h the boundary or gradual?
    
    # Additional useful intervals
    (8, 4, '8h_to_4h'),    # 4-hour window starting at 8h
    (5, 2, '5h_to_2h'),    # 3-hour window in the critical zone
]

# Move thresholds (fractional, not percentage)
MOVE_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20]

# Direction labels
DIRECTIONS = ['drop', 'rise']  # negative move = drop, positive move = rise

# Order placement strategies for fill simulation
ORDER_OFFSETS = [
    (0.00, 'at_mid'),
    (0.01, 'aggressive_1pct'),
    (0.02, 'aggressive_2pct'),
]

MIN_TRADES = 10
MIN_SAMPLES_PER_CELL = 20  # Minimum samples for statistical validity

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
        self.max_percentile_samples = 10000
    
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
                'p10': np.percentile(arr, 10),
                'p25': np.percentile(arr, 25),
                'p50': np.percentile(arr, 50),
                'p75': np.percentile(arr, 75),
                'p90': np.percentile(arr, 90),
            }
        
        return {
            'n': self.n,
            'mean': self.mean,
            'std': std,
            'min': self.min_val if self.min_val != float('inf') else None,
            'max': self.max_val if self.max_val != float('-inf') else None,
            **percentiles,
        }


# ==============================================================================
# 2D SURFACE AGGREGATOR
# ==============================================================================

def _make_cell_stats():
    """Create a new cell statistics structure."""
    return {
        'n_samples': 0,
        'n_wins': 0,
        'n_fills': 0,
        'n_fill_wins': 0,
        'snapshot_prices': StreamingStats(),
        'post_prices': StreamingStats(),
        'fill_prices': StreamingStats(),
        'move_sizes': StreamingStats(),
    }


class SurfaceAggregator:
    """Aggregate 2D surface results incrementally."""
    
    def __init__(self):
        self.n_tokens = 0
        self.n_tokens_with_data = 0
        
        # Main surface: [interval_label][direction][threshold] -> cell_stats
        # e.g., surface['48h_to_24h']['drop'][0.10] -> {...}
        self.surface = {}
        for _, _, interval_label in INTERVAL_PAIRS:
            self.surface[interval_label] = {}
            for direction in DIRECTIONS:
                self.surface[interval_label][direction] = {}
                for threshold in MOVE_THRESHOLDS:
                    self.surface[interval_label][direction][threshold] = _make_cell_stats()
        
        # Summary stats by interval
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
            
            start_price = interval_data.get('start_price')
            end_price = interval_data.get('end_price')
            
            if start_price is None or end_price is None:
                continue
            
            move = end_price - start_price
            move_pct = move  # Already in absolute terms (0.1 = 10%)
            
            # Determine direction
            if move < 0:
                direction = 'drop'
                abs_move = abs(move_pct)
            else:
                direction = 'rise'
                abs_move = abs(move_pct)
            
            # Populate all applicable threshold buckets
            for threshold in MOVE_THRESHOLDS:
                if abs_move >= threshold:
                    cell = self.surface[interval_label][direction][threshold]
                    
                    cell['n_samples'] += 1
                    if winner:
                        cell['n_wins'] += 1
                    
                    cell['snapshot_prices'].update(end_price)
                    cell['post_prices'].update(end_price)  # Using end_price as entry reference
                    cell['move_sizes'].update(abs_move)
                    
                    # Fill simulation data
                    fill_data = interval_data.get('fill_simulation')
                    if fill_data and fill_data.get('filled'):
                        cell['n_fills'] += 1
                        if winner:
                            cell['n_fill_wins'] += 1
                        
                        fill_price = fill_data.get('fill_price')
                        if fill_price is not None:
                            cell['fill_prices'].update(fill_price)
    
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
            
            for direction in DIRECTIONS:
                summary['surface'][interval_label][direction] = {}
                
                for threshold in MOVE_THRESHOLDS:
                    cell = self.surface[interval_label][direction][threshold]
                    
                    if cell['n_samples'] < MIN_SAMPLES_PER_CELL:
                        summary['surface'][interval_label][direction][threshold] = {
                            'n_samples': cell['n_samples'],
                            'status': 'insufficient_data',
                        }
                        continue
                    
                    n = cell['n_samples']
                    n_wins = cell['n_wins']
                    n_fills = cell['n_fills']
                    n_fill_wins = cell['n_fill_wins']
                    
                    # Unconditional metrics
                    uncond_win_rate = n_wins / n if n > 0 else 0
                    snapshot_stats = cell['snapshot_prices'].get_stats()
                    avg_snapshot = snapshot_stats['mean'] if snapshot_stats else 0
                    
                    uncond_edge_bps = (uncond_win_rate - avg_snapshot) * 10000
                    
                    # Standard error for win rate (binomial)
                    se_uncond_wr = np.sqrt(uncond_win_rate * (1 - uncond_win_rate) / n) if n > 0 else 0
                    se_uncond_edge = se_uncond_wr * 10000
                    
                    # Fill rate
                    fill_rate = n_fills / n if n > 0 else 0
                    
                    # Conditional metrics (among filled orders)
                    cond_result = {}
                    if n_fills >= 10:
                        cond_win_rate = n_fill_wins / n_fills
                        fill_stats = cell['fill_prices'].get_stats()
                        avg_fill = fill_stats['mean'] if fill_stats else 0
                        
                        edge_after_fill_bps = (cond_win_rate - avg_fill) * 10000
                        
                        # Standard error for conditional win rate
                        se_cond_wr = np.sqrt(cond_win_rate * (1 - cond_win_rate) / n_fills) if n_fills > 0 else 0
                        se_cond_edge = se_cond_wr * 10000
                        
                        cond_result = {
                            'conditional_win_rate': cond_win_rate,
                            'avg_fill_price': avg_fill,
                            'edge_after_fill_bps': edge_after_fill_bps,
                            'se_edge_after_fill': se_cond_edge,
                        }
                    
                    # Move size stats
                    move_stats = cell['move_sizes'].get_stats()
                    
                    summary['surface'][interval_label][direction][threshold] = {
                        'n_samples': n,
                        'n_fills': n_fills,
                        'unconditional_win_rate': uncond_win_rate,
                        'avg_snapshot_price': avg_snapshot,
                        'unconditional_edge_bps': uncond_edge_bps,
                        'se_unconditional_edge': se_uncond_edge,
                        'fill_rate': fill_rate,
                        'avg_move_size': move_stats['mean'] if move_stats else None,
                        **cond_result,
                        'status': 'ok',
                    }
        
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
# PRICE EXTRACTION AND FILL SIMULATION
# ==============================================================================

def extract_price_at_horizon(trades, resolution_time, hours_before):
    """Extract price closest to specific time horizon before resolution."""
    target_time = resolution_time - (hours_before * 3600)
    
    # Tighter tolerance for shorter horizons
    # Long horizons (>12h): 25% tolerance
    # Medium horizons (4-12h): 20% tolerance  
    # Short horizons (<4h): 15% tolerance, min 30 minutes
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
                'fill_price': price,
                'time_to_fill': ts - placement_time,
            }
        elif not is_buy and price >= limit_price:
            return {
                'filled': True,
                'fill_time': ts,
                'fill_price': price,
                'time_to_fill': ts - placement_time,
            }
    
    return {
        'filled': False,
        'fill_time': None,
        'fill_price': None,
        'time_to_fill': None,
    }


def compute_interval_data(trades, resolution_time):
    """Compute price changes across all interval pairs with fill simulation."""
    results = {}
    
    for start_h, end_h, label in INTERVAL_PAIRS:
        start_time, start_price = extract_price_at_horizon(trades, resolution_time, start_h)
        end_time, end_price = extract_price_at_horizon(trades, resolution_time, end_h)
        
        if start_price is None or end_price is None:
            continue
        
        # Simulate fill at end_time (entry point) using at_mid strategy
        fill_result = None
        if end_time is not None:
            fill_result = simulate_limit_order_fill(
                trades, end_time, end_price, is_buy=True
            )
        
        results[label] = {
            'start_hours': start_h,
            'end_hours': end_h,
            'start_time': start_time,
            'end_time': end_time,
            'start_price': start_price,
            'end_price': end_price,
            'move': end_price - start_price,
            'fill_simulation': fill_result,
        }
    
    return results


# ==============================================================================
# DATA ACCUMULATOR CLASS
# ==============================================================================

class SurfaceAccumulator:
    """Accumulates trade data for 2D surface analysis."""
    
    def __init__(self, token_id, condition_id, resolution_time, winner_status):
        self.token_id = token_id
        self.condition_id = condition_id
        self.resolution_time = resolution_time
        self.winner_status = winner_status
        self.trades = []
    
    def add_trades(self, trades_batch):
        self.trades.extend(trades_batch)
    
    def compute_surface_data(self):
        """Compute 2D surface data for this token."""
        if len(self.trades) < MIN_TRADES:
            return None
        
        self.trades.sort(key=lambda x: x[0])
        
        # Compute interval data
        interval_data = compute_interval_data(self.trades, self.resolution_time)
        
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
# MARKET INDEX LOADING (read-only, no overwrite)
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
            # Still proceed, just warn
        
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
# CHECKPOINTING (safe pkl handling)
# ==============================================================================

def save_checkpoint(aggregator, files_processed, total_files):
    """Save checkpoint to unique file. Uses HIGHEST_PROTOCOL for safety."""
    temp_file = CHECKPOINT_FILE + '.tmp'
    
    checkpoint = {
        'aggregator': aggregator,
        'files_processed': files_processed,
        'total_files': total_files,
        'timestamp': datetime.now().isoformat(),
        'version': '3c_v1',
    }
    
    try:
        with open(temp_file, 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Atomic rename
        os.replace(temp_file, CHECKPOINT_FILE)
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
        
        # Version check
        if checkpoint.get('version') != '3c_v1':
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
    log("PHASE 3C: 2D TEMPORAL EDGE SURFACE (MEMORY-SAFE)")
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
    aggregator = SurfaceAggregator()
    
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
            
            # Checkpoint every 10k files
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
                    
                    token_accumulators[token_id] = SurfaceAccumulator(
                        token_id, condition_id, resolution_time, winner_status
                    )
                    condition_tokens[condition_id].add(token_id)
                
                timestamps = group['timestamp'].values
                prices = group['price'].values
                sizes = group['size_tokens'].values
                
                trades_batch = list(zip(timestamps, prices, sizes))
                token_accumulators[token_id].add_trades(trades_batch)
                
                # Capture diagnostic samples
                if diagnostic and len(diagnostic_samples) < 5 and len(trades_batch) > 50:
                    diagnostic_samples.append({
                        'token_id': token_id,
                        'winner': winner_status,
                        'trades_sample': trades_batch[:100],
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
                            surface_result = acc.compute_surface_data()
                            
                            if surface_result is not None:
                                aggregator.add_result(surface_result)
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
            surface_result = acc.compute_surface_data()
            
            if surface_result is not None:
                aggregator.add_result(surface_result)
            else:
                stats['tokens_filtered'] += 1
            
            del token_accumulators[token_id]
        
        stats['conditions_flushed'] += 1
    
    gc.collect()
    
    log(f"\nProcessed {stats['files_processed']:,} files, {stats['total_rows']:,} rows")
    log(f"Computed surface data for {aggregator.n_tokens_with_data:,} tokens")
    log(f"Conditions flushed: {stats['conditions_flushed']:,}")
    log(f"Skipped: {stats['tokens_no_winner']:,} (no winner), "
        f"{stats['tokens_filtered']:,} (insufficient trades)")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"\nTotal time: {format_duration(elapsed)}")
    log(f"Final {log_memory()}")
    
    # Remove checkpoint on successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log("Checkpoint removed (run complete)")
    
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
            
            trades = sample['trades_sample']
            resolution_time = sample['resolution_time']
            
            # Compute interval data for this sample
            interval_data = compute_interval_data(trades, resolution_time)
            
            for label, data in interval_data.items():
                log(f"  {label}:")
                log(f"    Start: {data['start_price']:.4f} @ T-{data['start_hours']}h")
                log(f"    End:   {data['end_price']:.4f} @ T-{data['end_hours']}h")
                log(f"    Move:  {data['move']:+.4f} ({data['move']*100:+.1f}%)")
                
                fill = data.get('fill_simulation')
                if fill:
                    fill_str = "FILLED" if fill['filled'] else "NO FILL"
                    if fill['filled']:
                        fill_str += f" @ {fill['fill_price']:.4f}"
                    log(f"    Fill:  {fill_str}")
    
    return aggregator


# ==============================================================================
# REPORTING
# ==============================================================================

def print_surface_analysis(aggregator):
    """Print 2D surface analysis from streaming aggregator."""
    summary = aggregator.get_summary()
    
    log("\n" + "="*70)
    log("2D SURFACE ANALYSIS RESULTS")
    log("="*70)
    
    log(f"\nSample Size: {summary['n_tokens']:,} tokens")
    log(f"Tokens with interval data: {summary['n_tokens_with_data']:,}")
    
    log("\n" + "-"*50)
    log("INTERVAL COVERAGE")
    log("-"*50)
    
    # Sort intervals by start hour descending for logical ordering
    sorted_intervals = sorted(INTERVAL_PAIRS, key=lambda x: -x[0])
    
    for start_h, end_h, interval_label in sorted_intervals:
        coverage = summary['interval_coverage'].get(interval_label, 0)
        log(f"  {interval_label}: {coverage:,} tokens")
    
    # Group intervals for display
    coarse_intervals = ['48h_to_24h', '24h_to_12h', '12h_to_6h', '6h_to_3h', '3h_to_1h']
    fine_intervals = ['9h_to_6h', '8h_to_4h', '6h_to_4h', '5h_to_2h', '4h_to_2h']
    
    for direction in DIRECTIONS:
        dir_label = "DROPS (Fading drops = buying after price fell)" if direction == 'drop' else "RISES (Following rises = buying after price rose)"
        
        log("\n" + "-"*70)
        log(f"2D EDGE SURFACE: {dir_label}")
        log("-"*70)
        log("Format: edge_after_fill (n_samples) | -- if insufficient data")
        
        # Print coarse intervals first
        log("\n  COARSE INTERVALS:")
        _print_interval_table(summary, coarse_intervals, direction)
        
        # Print fine intervals
        log("\n  FINE INTERVALS (transition zone mapping):")
        _print_interval_table(summary, fine_intervals, direction)
    
    return summary


def _print_interval_table(summary, interval_labels, direction):
    """Helper to print a table for a subset of intervals."""
    # Header
    header = f"  {'Interval':<12}"
    for thresh in MOVE_THRESHOLDS:
        header += f" | >={thresh*100:>2.0f}%".center(14)
    log(header)
    log("  " + "-" * (12 + 15 * len(MOVE_THRESHOLDS)))
    
    for interval_label in interval_labels:
        if interval_label not in summary.get('surface', {}):
            continue
            
        row = f"  {interval_label:<12}"
        for thresh in MOVE_THRESHOLDS:
            cell = summary['surface'][interval_label][direction].get(thresh, {})
            if cell.get('status') == 'insufficient_data':
                row += f" |    n={cell.get('n_samples', 0):>3}   "
            elif cell.get('status') == 'ok':
                n = cell['n_samples']
                edge_after = cell.get('edge_after_fill_bps')
                
                if edge_after is not None:
                    # Color coding: bold positive, normal negative
                    edge_str = f"{edge_after:+.0f}"
                    n_str = f"n={n//1000}k" if n >= 1000 else f"n={n}"
                    row += f" | {edge_str:>5} ({n_str:>4})"
                else:
                    row += f" |      --     "
            else:
                row += f" |      --     "
        log(row)


def find_transition_points(summary, threshold=0.10):
    """Find where edge flips from positive to negative for each direction.
    
    Returns dict mapping direction -> list of (interval, edge, status)
    where status is 'positive', 'negative', or 'transition'
    """
    # Sort intervals by start hour descending (further from resolution first)
    all_intervals = [(s, e, l) for s, e, l in INTERVAL_PAIRS]
    sorted_intervals = sorted(all_intervals, key=lambda x: -x[0])
    
    results = {}
    
    for direction in DIRECTIONS:
        results[direction] = []
        prev_edge = None
        
        for start_h, end_h, label in sorted_intervals:
            if label not in summary.get('surface', {}):
                continue
            
            cell = summary['surface'][label].get(direction, {}).get(threshold, {})
            if cell.get('status') != 'ok':
                continue
            
            edge = cell.get('edge_after_fill_bps')
            if edge is None:
                continue
            
            if prev_edge is not None and prev_edge > 0 and edge < 0:
                status = 'TRANSITION'
            elif edge > 0:
                status = 'positive'
            else:
                status = 'negative'
            
            results[direction].append({
                'interval': label,
                'start_h': start_h,
                'end_h': end_h,
                'edge_bps': edge,
                'n_samples': cell.get('n_samples', 0),
                'status': status,
            })
            
            prev_edge = edge
    
    return results


def print_transition_analysis(summary):
    """Print focused analysis of the transition zone."""
    log("\n" + "="*70)
    log("TRANSITION ZONE ANALYSIS")
    log("="*70)
    log("Mapping where edge flips from positive (noise) to negative (informed)")
    
    transitions = find_transition_points(summary, threshold=0.10)
    
    for direction in DIRECTIONS:
        dir_label = "DROPS" if direction == 'drop' else "RISES"
        log(f"\n{dir_label} (>=10% threshold):")
        log(f"  {'Interval':<12} | {'Start':<6} | {'Edge':<10} | {'Status':<12}")
        log("  " + "-"*50)
        
        for item in transitions[direction]:
            edge_str = f"{item['edge_bps']:+.0f} bps"
            status = item['status']
            if status == 'TRANSITION':
                status = ">>> FLIP <<<"
            log(f"  {item['interval']:<12} | T-{item['start_h']:<4}h | {edge_str:<10} | {status}")
        
        # Find the transition point
        flip_points = [x for x in transitions[direction] if x['status'] == 'TRANSITION']
        if flip_points:
            fp = flip_points[0]
            log(f"\n  TRANSITION DETECTED: Edge flips negative at {fp['interval']}")
            log(f"  Operational cutoff: Stop fading {dir_label.lower()} ~{fp['start_h']}h before resolution")


def save_results_json(summary):
    """Save results to JSON file."""
    ensure_output_dir()
    
    output = {
        'timestamp': TIMESTAMP,
        'phase': '3C',
        'description': '2D Temporal Edge Surface Analysis',
        'methodology': 'For each (interval, move_threshold) pair, compute edge before/after fills',
        'interval_pairs': [(s, e, l) for s, e, l in INTERVAL_PAIRS],
        'move_thresholds': MOVE_THRESHOLDS,
        **summary,
    }
    
    json_path = os.path.join(OUTPUT_DIR, f'phase3c_2d_surface_{TIMESTAMP}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    log(f"Results JSON saved: {json_path}")
    return json_path


def generate_report(summary):
    """Generate text report."""
    ensure_output_dir()
    
    report_path = os.path.join(OUTPUT_DIR, f'phase3c_2d_surface_{TIMESTAMP}_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 3C: 2D TEMPORAL EDGE SURFACE - REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tokens Analyzed: {summary.get('n_tokens', 0):,}\n")
        f.write(f"Tokens with Interval Data: {summary.get('n_tokens_with_data', 0):,}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("-"*80 + "\n")
        f.write("For each interval pair (e.g., 48h->24h), measure price move.\n")
        f.write("Categorize by direction (drop/rise) and magnitude threshold.\n")
        f.write("Compute unconditional edge and edge after simulated fill.\n\n")
        
        f.write("-"*80 + "\n")
        f.write("TRANSITION ZONE ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write("The original analysis showed edge flipping negative around 6h->3h.\n")
        f.write("Fine-grained intervals map exactly where noise trading gives way\n")
        f.write("to informed flow:\n\n")
        
        transition_intervals = ['12h_to_6h', '9h_to_6h', '8h_to_4h', '6h_to_4h', '6h_to_3h', '5h_to_2h', '4h_to_2h', '3h_to_1h']
        
        f.write("DROPS (fading strategy):\n")
        f.write(f"{'Interval':<12} | {'>=10% edge':<12} | {'n_samples':<10}\n")
        f.write("-"*40 + "\n")
        
        for interval_label in transition_intervals:
            if interval_label not in summary.get('surface', {}):
                continue
            cell = summary['surface'][interval_label]['drop'].get(0.10, {})
            if cell.get('status') == 'ok':
                edge = cell.get('edge_after_fill_bps', 0)
                n = cell.get('n_samples', 0)
                marker = " <-- FLIP" if edge < 0 and interval_label in ['6h_to_3h', '6h_to_4h', '5h_to_2h', '4h_to_2h'] else ""
                f.write(f"{interval_label:<12} | {edge:>+8.0f} bps | {n:>8,}{marker}\n")
        
        f.write("\n")
        
        f.write("-"*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n")
        
        # Find best cells and transition points
        best_drop = None
        worst_drop = None
        
        for interval_label in summary.get('surface', {}):
            cell = summary['surface'][interval_label].get('drop', {}).get(0.10, {})
            if cell.get('status') != 'ok':
                continue
            edge = cell.get('edge_after_fill_bps')
            if edge is None:
                continue
            
            entry = (interval_label, edge, cell)
            
            if best_drop is None or edge > best_drop[1]:
                best_drop = entry
            if worst_drop is None or edge < worst_drop[1]:
                worst_drop = entry
        
        if best_drop:
            f.write(f"\nBest DROPS (>=10% threshold):\n")
            f.write(f"  Interval: {best_drop[0]}\n")
            f.write(f"  Edge after fill: {best_drop[1]:+.0f} bps\n")
            f.write(f"  Sample size: {best_drop[2]['n_samples']:,}\n")
        
        if worst_drop:
            f.write(f"\nWorst DROPS (>=10% threshold):\n")
            f.write(f"  Interval: {worst_drop[0]}\n")
            f.write(f"  Edge after fill: {worst_drop[1]:+.0f} bps\n")
            f.write(f"  Sample size: {worst_drop[2]['n_samples']:,}\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("OPERATIONAL IMPLICATIONS\n")
        f.write("-"*80 + "\n")
        f.write("1. Fading drops works until ~6h before resolution\n")
        f.write("2. After 6h, informed flow dominates - drops predict losses\n")
        f.write("3. The transition zone data helps set the cutoff precisely\n")
        f.write("4. Strategy: fade large drops, but exit/stop new entries <6h\n")
        
        f.write("\n" + "-"*80 + "\n")
        f.write("SURFACE DATA (see JSON for full data)\n")
        f.write("-"*80 + "\n")
    
    log(f"Report saved: {report_path}")
    return report_path


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 3C: 2D Temporal Edge Surface (Memory-Safe)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnostic mode with small sample
  python phase3c_2d_surface.py --diagnostic --sample 100
  
  # Full run
  python phase3c_2d_surface.py
  
  # Resume interrupted run
  python phase3c_2d_surface.py --resume
        """
    )
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--diagnostic', '-d', action='store_true',
                        help='Run in diagnostic mode with extra validation')
    
    args = parser.parse_args()
    
    log("Starting Phase 3C 2D Surface Analysis (Memory-Safe)...")
    
    ensure_output_dir()
    
    aggregator = run_analysis(
        sample_files=args.sample,
        resume=args.resume,
        diagnostic=args.diagnostic
    )
    
    if aggregator:
        summary = print_surface_analysis(aggregator)
        print_transition_analysis(summary)
        generate_report(summary)
        save_results_json(summary)
        
        log("\n" + "="*70)
        log("ANALYSIS COMPLETE")
        log("="*70)
        log(f"\nOutputs generated:")
        log(f"  - Report: phase3c_2d_surface_{TIMESTAMP}_report.txt")
        log(f"  - JSON:   phase3c_2d_surface_{TIMESTAMP}.json")
    else:
        log("\nAnalysis failed - no results generated")