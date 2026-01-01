#!/usr/bin/env python3
"""
Phase 3A: Roll Measure Spread Estimation - MEMORY SAFE VERSION
Version: 2.0

CHANGES FROM v1.0:
  - Streaming aggregation: no longer stores individual results in memory
  - Running statistics computed incrementally via Welford's algorithm
  - Results written to disk in batches for post-hoc analysis if needed
  - Memory should stay flat throughout run

OBJECTIVE:
  Estimate effective bid-ask spreads using the Roll (1984) measure.
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

warnings.filterwarnings('ignore', message='.*constant.*correlation.*')
warnings.filterwarnings('ignore', message='.*invalid value.*')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = "../../"

BATCH_DIR = os.path.join(BASE_DIR, 'order_history_batches')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis/outputs')
INDEX_CACHE_FILE = os.path.join(BASE_DIR, 'analysis/market_index_cache_full.pkl')
SIDECAR_FILE = os.path.join(BASE_DIR, 'data/repair/api_derived_winners.parquet')

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'phase3a_checkpoint.pkl')

# For streaming results to disk
RESULTS_BATCH_FILE = os.path.join(OUTPUT_DIR, f'phase3a_results_batch_{TIMESTAMP}.pkl')
RESULTS_BATCH_SIZE = 5000  # Write to disk every N results

# ------------------------------------------------------------------------------
# ANALYSIS PARAMETERS
# ------------------------------------------------------------------------------

MIN_TRADES_ROLL = 20
TIME_HORIZONS = [720, 168, 72, 24, 12, 6, 1]

PRICE_BUCKETS = [
    (0.00, 0.10, 'longshot'),
    (0.10, 0.25, 'underdog'),
    (0.25, 0.40, 'toss-up-'),
    (0.40, 0.51, 'toss-up+'),
    (0.51, 0.60, 'mild-fav'),
    (0.60, 0.75, 'moderate-fav'),
    (0.75, 0.90, 'strong-fav'),
    (0.90, 0.99, 'heavy-fav'),
    (0.99, 1.00, 'near-certain'),
]

VOLUME_BUCKETS = [
    (0, 50, 'micro'),
    (50, 200, 'small'),
    (200, 1000, 'medium'),
    (1000, 5000, 'large'),
    (5000, float('inf'), 'very_large'),
]

TTR_BUCKETS = [
    (0, 6, '0-6h'),
    (6, 24, '6-24h'),
    (24, 72, '1-3d'),
    (72, 168, '3-7d'),
    (168, 720, '1-4w'),
    (720, float('inf'), '>4w'),
]

REQUIRED_COLUMNS = ['timestamp', 'price', 'token_id', 'condition_id', 'resolution_time']
VOLUME_COLUMNS = ['size_tokens', 'maker_amount', 'size']

PROGRESS_INTERVAL = 1000

# ==============================================================================
# STREAMING STATISTICS CLASS (Welford's Algorithm)
# ==============================================================================

class StreamingStats:
    """
    Compute mean, variance, min, max incrementally without storing all values.
    Uses Welford's online algorithm for numerical stability.
    """
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from mean
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.values_for_percentiles = []  # Keep limited sample for percentiles
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
        
        # Reservoir sampling for percentiles
        if len(self.values_for_percentiles) < self.max_percentile_samples:
            self.values_for_percentiles.append(value)
        else:
            # Randomly replace with decreasing probability
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

def _make_price_bucket_entry():
    return {'spread_bps': StreamingStats(), 'valid': 0, 'invalid': 0}

def _make_simple_bucket_entry():
    return {'spread_bps': StreamingStats(), 'count': 0}

class StreamingAggregator:
    """
    Aggregate Roll measure results incrementally by various dimensions.
    """
    def __init__(self):
        # Overall stats
        self.overall_spread_bps = StreamingStats()
        self.overall_cov = StreamingStats()
        
        # By price bucket (using pickle-friendly factory)
        self.by_price_bucket = defaultdict(_make_price_bucket_entry)
        
        # By volume bucket
        self.by_volume_bucket = defaultdict(_make_simple_bucket_entry)
        
        # By TTR bucket
        self.by_ttr_bucket = defaultdict(_make_simple_bucket_entry)
        
        # By horizon
        self.by_horizon = defaultdict(_make_simple_bucket_entry)
        
        # Counts
        self.n_tokens = 0
        self.n_valid = 0
        self.n_invalid = 0
        self.n_winners = 0
        self.n_losers = 0
    
    def add_result(self, result):
        """Add a single token result to aggregates."""
        self.n_tokens += 1
        
        if result['winner']:
            self.n_winners += 1
        else:
            self.n_losers += 1
        
        overall_roll = result.get('overall_roll', {})
        
        # Overall stats
        if overall_roll.get('valid'):
            self.n_valid += 1
            spread_bps = overall_roll.get('spread_bps')
            if spread_bps is not None:
                self.overall_spread_bps.update(spread_bps)
        else:
            self.n_invalid += 1
        
        cov = overall_roll.get('cov')
        if cov is not None:
            self.overall_cov.update(cov)
        
        # By price bucket
        price_bucket = result.get('price_bucket')
        if price_bucket:
            if overall_roll.get('valid'):
                self.by_price_bucket[price_bucket]['valid'] += 1
                spread_bps = overall_roll.get('spread_bps')
                if spread_bps is not None:
                    self.by_price_bucket[price_bucket]['spread_bps'].update(spread_bps)
            else:
                self.by_price_bucket[price_bucket]['invalid'] += 1
        
        # By volume bucket
        volume_bucket = result.get('volume_bucket')
        if volume_bucket and overall_roll.get('valid'):
            self.by_volume_bucket[volume_bucket]['count'] += 1
            spread_bps = overall_roll.get('spread_bps')
            if spread_bps is not None:
                self.by_volume_bucket[volume_bucket]['spread_bps'].update(spread_bps)
        
        # By TTR bucket
        ttr_bucket = result.get('ttr_bucket')
        if ttr_bucket and overall_roll.get('valid'):
            self.by_ttr_bucket[ttr_bucket]['count'] += 1
            spread_bps = overall_roll.get('spread_bps')
            if spread_bps is not None:
                self.by_ttr_bucket[ttr_bucket]['spread_bps'].update(spread_bps)
        
        # By horizon
        for horizon_key, horizon_data in result.get('horizon_rolls', {}).items():
            if horizon_data.get('valid'):
                self.by_horizon[horizon_key]['count'] += 1
                spread_bps = horizon_data.get('spread_bps')
                if spread_bps is not None:
                    self.by_horizon[horizon_key]['spread_bps'].update(spread_bps)
    
    def get_summary(self):
        """Get aggregated summary statistics."""
        summary = {
            'n_tokens': self.n_tokens,
            'n_valid': self.n_valid,
            'n_invalid': self.n_invalid,
            'valid_pct': self.n_valid / self.n_tokens * 100 if self.n_tokens > 0 else 0,
            'n_winners': self.n_winners,
            'n_losers': self.n_losers,
        }
        
        # Overall spread
        overall_stats = self.overall_spread_bps.get_stats()
        if overall_stats:
            summary['overall_spread'] = {
                'mean_bps': overall_stats['mean'],
                'median_bps': overall_stats.get('p50'),
                'std_bps': overall_stats['std'],
                'p10_bps': overall_stats.get('p10'),
                'p25_bps': overall_stats.get('p25'),
                'p75_bps': overall_stats.get('p75'),
                'p90_bps': overall_stats.get('p90'),
            }
        
        # By price bucket
        summary['spread_by_price_bucket'] = {}
        for bucket_info in PRICE_BUCKETS:
            label = bucket_info[2]
            bucket_data = self.by_price_bucket[label]
            stats = bucket_data['spread_bps'].get_stats()
            if stats and stats['n'] > 0:
                total = bucket_data['valid'] + bucket_data['invalid']
                summary['spread_by_price_bucket'][label] = {
                    'n': stats['n'],
                    'median_bps': stats.get('p50'),
                    'mean_bps': stats['mean'],
                    'valid_pct': bucket_data['valid'] / total * 100 if total > 0 else 0,
                }
        
        # By volume bucket
        summary['spread_by_volume'] = {}
        for bucket_info in VOLUME_BUCKETS:
            label = bucket_info[2]
            bucket_data = self.by_volume_bucket[label]
            stats = bucket_data['spread_bps'].get_stats()
            if stats and stats['n'] > 0:
                summary['spread_by_volume'][label] = {
                    'n': stats['n'],
                    'median_bps': stats.get('p50'),
                    'mean_bps': stats['mean'],
                }
        
        # By TTR bucket
        summary['spread_by_ttr'] = {}
        for bucket_info in TTR_BUCKETS:
            label = bucket_info[2]
            bucket_data = self.by_ttr_bucket[label]
            stats = bucket_data['spread_bps'].get_stats()
            if stats and stats['n'] > 0:
                summary['spread_by_ttr'][label] = {
                    'n': stats['n'],
                    'median_bps': stats.get('p50'),
                    'mean_bps': stats['mean'],
                }
        
        # By horizon
        summary['spread_by_horizon'] = {}
        for horizon in TIME_HORIZONS:
            key = f'horizon_{horizon}h'
            bucket_data = self.by_horizon[key]
            stats = bucket_data['spread_bps'].get_stats()
            if stats and stats['n'] > 0:
                summary['spread_by_horizon'][key] = {
                    'n': stats['n'],
                    'median_bps': stats.get('p50'),
                    'mean_bps': stats['mean'],
                }
        
        # Covariance distribution summary
        cov_stats = self.overall_cov.get_stats()
        if cov_stats:
            summary['covariance_stats'] = cov_stats
        
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

def assign_bucket(value, buckets):
    if value is None or not np.isfinite(value):
        return None
    for lower, upper, label in buckets:
        if lower <= value < upper:
            return label
    if value >= buckets[-1][1] and buckets[-1][1] != float('inf'):
        return buckets[-1][2]
    return None

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
# ROLL MEASURE COMPUTATION
# ==============================================================================

def compute_roll_measure(trades, min_trades=MIN_TRADES_ROLL):
    if len(trades) < min_trades:
        return None
    
    prices = np.array([t[1] for t in trades])
    delta_p = np.diff(prices)
    
    if len(delta_p) < 2:
        return None
    
    delta_p_t = delta_p[1:]
    delta_p_lag = delta_p[:-1]
    
    n_pairs = len(delta_p_t)
    
    mean_t = np.mean(delta_p_t)
    mean_lag = np.mean(delta_p_lag)
    cov = np.mean((delta_p_t - mean_t) * (delta_p_lag - mean_lag))
    
    if cov < 0:
        spread = 2 * np.sqrt(-cov)
        valid = True
    else:
        spread = None
        valid = False
    
    mean_price = np.mean(prices)
    spread_bps = (spread / mean_price * 10000) if spread and mean_price > 0 else None
    
    return {
        'spread': spread,
        'spread_bps': spread_bps,
        'cov': cov,
        'valid': valid,
        'n_trades': len(trades),
        'n_pairs': n_pairs,
        'mean_price': mean_price,
    }

def compute_roll_by_horizon(trades, resolution_time, horizons=TIME_HORIZONS):
    results = {}
    
    for horizon in horizons:
        cutoff_time = resolution_time - (horizon * 3600)
        window_trades = [(ts, p, s) for ts, p, s in trades if ts >= cutoff_time]
        
        if len(window_trades) >= MIN_TRADES_ROLL:
            roll_result = compute_roll_measure(window_trades)
            if roll_result:
                results[f'horizon_{horizon}h'] = roll_result
    
    return results

# ==============================================================================
# DATA ACCUMULATOR CLASS
# ==============================================================================

class RollMeasureAccumulator:
    def __init__(self, token_id, condition_id, resolution_time, winner_status):
        self.token_id = token_id
        self.condition_id = condition_id
        self.resolution_time = resolution_time
        self.winner_status = winner_status
        self.trades = []
    
    def add_trades(self, trades_batch):
        self.trades.extend(trades_batch)
    
    def compute_roll_analysis(self):
        if len(self.trades) < MIN_TRADES_ROLL:
            return None
        
        self.trades.sort(key=lambda x: x[0])
        
        earliest_trade = self.trades[0][0]
        hours_of_data = (self.resolution_time - earliest_trade) / 3600
        
        overall_roll = compute_roll_measure(self.trades)
        if overall_roll is None:
            return None
        
        horizon_rolls = compute_roll_by_horizon(self.trades, self.resolution_time)
        
        prices = [t[1] for t in self.trades]
        median_price = np.median(prices)
        price_bucket = assign_bucket(median_price, PRICE_BUCKETS)
        volume_bucket = assign_bucket(len(self.trades), VOLUME_BUCKETS)
        
        ttr_hours = (self.resolution_time - earliest_trade) / 3600
        ttr_bucket = assign_bucket(ttr_hours, TTR_BUCKETS)
        
        price_volatility = np.std(prices)
        
        return {
            'token_id': self.token_id,
            'condition_id': self.condition_id,
            'winner': self.winner_status,
            'n_trades': len(self.trades),
            'hours_of_data': hours_of_data,
            'median_price': median_price,
            'price_bucket': price_bucket,
            'volume_bucket': volume_bucket,
            'ttr_bucket': ttr_bucket,
            'price_volatility': price_volatility,
            'overall_roll': overall_roll,
            'horizon_rolls': horizon_rolls,
        }

# ==============================================================================
# MARKET INDEX LOADING
# ==============================================================================

def load_market_index(cache_file, batch_files):
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        cached_num_files = cache_data.get('num_files', 0)
        current_num_files = len(batch_files)
        
        file_diff_pct = abs(cached_num_files - current_num_files) / max(cached_num_files, 1) * 100
        
        if file_diff_pct > 10:
            return None
        
        return cache_data['market_index']
        
    except Exception as e:
        log(f"  Cache load failed: {e}")
        return None

def get_available_columns(filepath):
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
    checkpoint = {
        'aggregator': aggregator,
        'files_processed': files_processed,
        'total_files': total_files,
        'timestamp': datetime.now().isoformat(),
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  Checkpoint saved: {files_processed}/{total_files} files")

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    except:
        return None

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def run_analysis(sample_files=None, resume=False, diagnostic=False):
    start_time = datetime.now()
    
    log("="*70)
    log("PHASE 3A: ROLL MEASURE SPREAD ESTIMATION (MEMORY-SAFE)")
    log("="*70)
    log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if diagnostic:
        log("MODE: DIAGNOSTIC (extra validation enabled)")
    log("")
    
    batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, '*.parquet')))
    if not batch_files:
        log(f"ERROR: No parquet files found in {BATCH_DIR}")
        return None
    
    log(f"Found {len(batch_files):,} batch files")
    
    winner_lookup = load_winner_sidecar(SIDECAR_FILE)
    if not winner_lookup:
        log("ERROR: Failed to load winner sidecar")
        return None
    
    log("\nLoading market index from cache...")
    market_index = load_market_index(INDEX_CACHE_FILE, batch_files)
    
    if market_index is None:
        log("ERROR: Market index cache not found or invalid.")
        log(f"       Expected: {INDEX_CACHE_FILE}")
        return None
    
    log(f"  Loaded market index: {len(market_index):,} conditions")
    
    # ==========================================================================
    # CONDITION COMPLETION TRACKING
    # ==========================================================================
    
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
    
    # ==========================================================================
    # STREAMING AGGREGATOR (instead of results list)
    # ==========================================================================
    
    start_file_idx = 0
    aggregator = StreamingAggregator()
    
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
                gc.collect()  # Periodic GC
        
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
                    
                    token_accumulators[token_id] = RollMeasureAccumulator(
                        token_id, condition_id, resolution_time, winner_status
                    )
                    condition_tokens[condition_id].add(token_id)
                
                timestamps = group['timestamp'].values
                prices = group['price'].values
                sizes = group['size_tokens'].values
                
                trades_batch = list(zip(timestamps, prices, sizes))
                token_accumulators[token_id].add_trades(trades_batch)
                
                if diagnostic and len(diagnostic_samples) < 5 and len(trades_batch) > 30:
                    diagnostic_samples.append({
                        'token_id': token_id,
                        'trades_sample': trades_batch[:50],
                        'n_trades': len(trades_batch),
                    })
            
            del df
            
            # ==================================================================
            # STREAMING FLUSH - aggregate immediately, don't store results
            # ==================================================================
            
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
                            roll_result = acc.compute_roll_analysis()
                            
                            if roll_result is not None:
                                # STREAMING: aggregate immediately instead of storing
                                aggregator.add_result(roll_result)
                                # Don't append to results list!
                            else:
                                stats['tokens_filtered'] += 1
                            
                            # FREE THE MEMORY
                            del token_accumulators[token_id]
                        
                        if condition_id in condition_tokens:
                            del condition_tokens[condition_id]
                        del condition_remaining_files[condition_id]
                        
                        stats['conditions_flushed'] += 1
            
        except Exception as e:
            log(f"  Error processing {filepath}: {e}")
            continue
    
    # ==========================================================================
    # FINAL FLUSH
    # ==========================================================================
    
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
            roll_result = acc.compute_roll_analysis()
            
            if roll_result is not None:
                aggregator.add_result(roll_result)
            else:
                stats['tokens_filtered'] += 1
            
            del token_accumulators[token_id]
        
        stats['conditions_flushed'] += 1
    
    gc.collect()
    
    log(f"\nProcessed {stats['files_processed']:,} files, {stats['total_rows']:,} rows")
    log(f"Computed Roll measures for {aggregator.n_tokens:,} tokens")
    log(f"Conditions flushed: {stats['conditions_flushed']:,}")
    log(f"Skipped: {stats['tokens_no_winner']:,} (no winner), "
        f"{stats['tokens_filtered']:,} (insufficient trades)")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"\nTotal time: {format_duration(elapsed)}")
    log(f"Final {log_memory()}")
    
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log("Checkpoint removed (run complete)")
    
    # Diagnostic validation
    if diagnostic and diagnostic_samples:
        log("\n" + "="*70)
        log("DIAGNOSTIC VALIDATION")
        log("="*70)
        for sample in diagnostic_samples:
            log(f"\nToken: {sample['token_id']}")
            log(f"  Total trades: {sample['n_trades']}")
            trades = sample['trades_sample'][:10]
            log(f"  First 10 trades:")
            for ts, p, s in trades:
                dt = datetime.fromtimestamp(ts)
                log(f"    {dt}: price={p:.4f}, size={s:.2f}")
            
            test_roll = compute_roll_measure(sample['trades_sample'])
            if test_roll:
                log(f"  Roll measure on sample:")
                log(f"    Spread: {test_roll['spread']:.6f}" if test_roll['spread'] else "    Spread: UNDEFINED (cov > 0)")
                log(f"    Spread (bps): {test_roll['spread_bps']:.1f}" if test_roll['spread_bps'] else "    Spread (bps): N/A")
                log(f"    Covariance: {test_roll['cov']:.8f}")
                log(f"    Valid: {test_roll['valid']}")
    
    return aggregator

# ==============================================================================
# REPORTING (uses aggregated stats, not individual results)
# ==============================================================================

def print_analysis(aggregator):
    """Print analysis from streaming aggregator."""
    summary = aggregator.get_summary()
    
    log("\n" + "="*70)
    log("ANALYSIS RESULTS")
    log("="*70)
    
    log(f"\nSample Size: {summary['n_tokens']:,} tokens")
    log(f"  Valid Roll (cov < 0): {summary['n_valid']:,} ({summary['valid_pct']:.1f}%)")
    log(f"  Invalid (momentum):   {summary['n_invalid']:,}")
    
    log("\n" + "-"*50)
    log("1. OVERALL SPREAD DISTRIBUTION")
    log("-"*50)
    
    overall = summary.get('overall_spread', {})
    if overall:
        log(f"\n  Spread (basis points):")
        log(f"    Mean:   {overall.get('mean_bps', 0):8.1f} bps")
        log(f"    Median: {overall.get('median_bps', 0):8.1f} bps")
        log(f"    Std:    {overall.get('std_bps', 0):8.1f} bps")
        log(f"    P10:    {overall.get('p10_bps', 0):8.1f} bps")
        log(f"    P25:    {overall.get('p25_bps', 0):8.1f} bps")
        log(f"    P75:    {overall.get('p75_bps', 0):8.1f} bps")
        log(f"    P90:    {overall.get('p90_bps', 0):8.1f} bps")
    
    log("\n" + "-"*50)
    log("2. SPREAD BY PRICE BUCKET")
    log("-"*50)
    
    for bucket_info in PRICE_BUCKETS:
        label = bucket_info[2]
        data = summary.get('spread_by_price_bucket', {}).get(label)
        if data:
            log(f"\n  {label:12s}: n={data['n']:5,}, "
                f"median={data['median_bps']:6.1f}bps, mean={data['mean_bps']:6.1f}bps, "
                f"valid={data['valid_pct']:.0f}%")
    
    log("\n" + "-"*50)
    log("3. SPREAD BY VOLUME (trade count)")
    log("-"*50)
    
    for bucket_info in VOLUME_BUCKETS:
        label = bucket_info[2]
        data = summary.get('spread_by_volume', {}).get(label)
        if data:
            log(f"  {label:10s}: n={data['n']:5,}, "
                f"median={data['median_bps']:6.1f}bps, mean={data['mean_bps']:6.1f}bps")
    
    log("\n" + "-"*50)
    log("4. SPREAD BY TIME-TO-RESOLUTION")
    log("-"*50)
    
    for bucket_info in TTR_BUCKETS:
        label = bucket_info[2]
        data = summary.get('spread_by_ttr', {}).get(label)
        if data:
            log(f"  {label:8s}: n={data['n']:5,}, "
                f"median={data['median_bps']:6.1f}bps, mean={data['mean_bps']:6.1f}bps")
    
    log("\n" + "-"*50)
    log("5. SPREAD BY ANALYSIS HORIZON")
    log("-"*50)
    
    for horizon in TIME_HORIZONS:
        key = f'horizon_{horizon}h'
        data = summary.get('spread_by_horizon', {}).get(key)
        if data:
            log(f"  {key:15s}: n={data['n']:5,}, "
                f"median={data['median_bps']:6.1f}bps, mean={data['mean_bps']:6.1f}bps")
    
    return summary

def save_results_json(summary):
    ensure_output_dir()
    
    output = {
        'timestamp': TIMESTAMP,
        'phase': '3A',
        'description': 'Roll Measure Spread Estimation (Memory-Safe)',
        'methodology': 'Roll (1984): Spread = 2*sqrt(-Cov(dP_t, dP_{t-1}))',
        **summary,
    }
    
    json_path = os.path.join(OUTPUT_DIR, f'phase3a_roll_spread_{TIMESTAMP}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    log(f"Results JSON saved: {json_path}")
    return json_path

def generate_report(summary):
    ensure_output_dir()
    
    report_path = os.path.join(OUTPUT_DIR, f'phase3a_roll_spread_{TIMESTAMP}_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 3A: ROLL MEASURE SPREAD ESTIMATION - REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tokens Analyzed: {summary.get('n_tokens', 0):,}\n")
        f.write(f"Valid Roll Measures: {summary.get('n_valid', 0):,} "
                f"({summary.get('valid_pct', 0):.1f}%)\n\n")
        
        f.write("-"*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n\n")
        
        overall = summary.get('overall_spread', {})
        if overall:
            f.write(f"Overall Spread (valid tokens only):\n")
            f.write(f"  Median: {overall.get('median_bps', 0):.1f} bps\n")
            f.write(f"  Mean:   {overall.get('mean_bps', 0):.1f} bps\n")
            f.write(f"  P10:    {overall.get('p10_bps', 0):.1f} bps\n")
            f.write(f"  P90:    {overall.get('p90_bps', 0):.1f} bps\n\n")
        
        f.write("-"*80 + "\n")
        f.write("SPREAD BY PRICE BUCKET\n")
        f.write("-"*80 + "\n\n")
        
        for bucket, data in summary.get('spread_by_price_bucket', {}).items():
            f.write(f"  {bucket:12s}: n={data['n']:5,}, "
                    f"median={data['median_bps']:6.1f}bps, "
                    f"mean={data['mean_bps']:6.1f}bps\n")
    
    log(f"Report saved: {report_path}")
    return report_path

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 3A: Roll Measure Spread Estimation (Memory-Safe)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnostic mode (small sample with validation)
  python phase3a_roll_spread_memsafe.py --diagnostic --sample 100
  
  # Full run
  python phase3a_roll_spread_memsafe.py
  
  # Resume interrupted run
  python phase3a_roll_spread_memsafe.py --resume
        """
    )
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--diagnostic', '-d', action='store_true',
                        help='Run in diagnostic mode with extra validation')
    
    args = parser.parse_args()
    
    log("Starting Phase 3A Roll Measure Analysis (Memory-Safe)...")
    
    ensure_output_dir()
    
    aggregator = run_analysis(
        sample_files=args.sample,
        resume=args.resume,
        diagnostic=args.diagnostic
    )
    
    if aggregator:
        summary = print_analysis(aggregator)
        generate_report(summary)
        save_results_json(summary)
        
        log("\n" + "="*70)
        log("ANALYSIS COMPLETE")
        log("="*70)
        log(f"\nOutputs generated:")
        log(f"  - Report: phase3a_roll_spread_{TIMESTAMP}_report.txt")
        log(f"  - JSON:   phase3a_roll_spread_{TIMESTAMP}.json")
    else:
        log("\nAnalysis failed - no results generated")