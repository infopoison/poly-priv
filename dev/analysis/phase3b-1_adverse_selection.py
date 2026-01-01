#!/usr/bin/env python3
"""
Phase 3B: Adverse Selection Simulation - MEMORY SAFE VERSION
Version: 2.0

CHANGES FROM v1.0:
  - Streaming aggregation: no longer stores individual results in memory
  - Running statistics computed incrementally via Welford's algorithm
  - Memory should stay flat throughout run

OBJECTIVE:
  Quantify adverse selection costs by simulating limit order fills.
  This determines whether edge identified in Phase 2 survives execution.
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

CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'phase3b_checkpoint.pkl')

# ------------------------------------------------------------------------------
# ANALYSIS PARAMETERS
# ------------------------------------------------------------------------------

PLACEMENT_HORIZONS = [48, 24, 12, 6, 1]

ORDER_OFFSETS = [
    (-0.02, 'passive_2pct'),
    (-0.01, 'passive_1pct'),
    (0.00, 'at_mid'),
    (0.01, 'aggressive_1pct'),
    (0.02, 'aggressive_2pct'),
]

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

MIN_TRADES = 10

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
# STREAMING AGGREGATOR FOR ADVERSE SELECTION
# ==============================================================================

def _make_fill_rate_entry():
    return {'filled': 0, 'total': 0}

def _make_conditional_entry():
    return {'wins': 0, 'fills': 0, 'fill_prices': StreamingStats()}

def _make_bucket_stats_entry():
    return {
        'unconditional_wins': 0,
        'unconditional_total': 0,
        'conditional_wins': 0,
        'conditional_fills': 0,
        'snapshot_prices': StreamingStats(),
        'fill_prices': StreamingStats(),
    }

def _make_nested_fill_rate_dict():
    return defaultdict(_make_fill_rate_entry)

def _make_nested_conditional_dict():
    return defaultdict(_make_conditional_entry)

class AdverseSelectionAggregator:
    """Aggregate adverse selection results incrementally."""
    
    def __init__(self):
        # Overall counts
        self.n_tokens = 0
        self.n_winners = 0
        self.n_losers = 0
        
        # Fill rates: [horizon][strategy] -> {filled, total}
        self.fill_rates = defaultdict(_make_nested_fill_rate_dict)
        
        # Conditional stats: [horizon][strategy] -> {wins, fills, fill_prices}
        self.conditional_stats = defaultdict(_make_nested_conditional_dict)
        
        # By price bucket (for at_mid strategy at H-24h)
        self.bucket_stats = defaultdict(_make_bucket_stats_entry)
        
        # Time to fill: [horizon] -> StreamingStats
        self.time_to_fill = defaultdict(StreamingStats)
    
    def add_result(self, result):
        """Add a single token result to aggregates."""
        self.n_tokens += 1
        
        winner = result['winner']
        if winner:
            self.n_winners += 1
        else:
            self.n_losers += 1
        
        price_bucket = result.get('price_bucket')
        
        # Process each horizon
        for horizon_key, horizon_data in result.get('horizon_results', {}).items():
            snapshot_price = horizon_data.get('snapshot_price')
            
            # Update bucket unconditional stats (use H_24h as reference)
            if horizon_key == 'H_24h' and price_bucket:
                self.bucket_stats[price_bucket]['unconditional_total'] += 1
                if winner:
                    self.bucket_stats[price_bucket]['unconditional_wins'] += 1
                if snapshot_price is not None:
                    self.bucket_stats[price_bucket]['snapshot_prices'].update(snapshot_price)
            
            # Process each strategy
            for strat_name, strat_data in horizon_data.get('strategies', {}).items():
                # Fill rate tracking
                self.fill_rates[horizon_key][strat_name]['total'] += 1
                
                if strat_data['filled']:
                    self.fill_rates[horizon_key][strat_name]['filled'] += 1
                    
                    # Conditional stats
                    self.conditional_stats[horizon_key][strat_name]['fills'] += 1
                    if winner:
                        self.conditional_stats[horizon_key][strat_name]['wins'] += 1
                    
                    fill_price = strat_data.get('fill_price')
                    if fill_price is not None:
                        self.conditional_stats[horizon_key][strat_name]['fill_prices'].update(fill_price)
                    
                    # Time to fill (at_mid strategy only)
                    if strat_name == 'at_mid':
                        ttf = strat_data.get('time_to_fill')
                        if ttf is not None:
                            self.time_to_fill[horizon_key].update(ttf / 3600)  # Convert to hours
                        
                        # Bucket conditional stats (H_24h, at_mid only)
                        if horizon_key == 'H_24h' and price_bucket:
                            self.bucket_stats[price_bucket]['conditional_fills'] += 1
                            if winner:
                                self.bucket_stats[price_bucket]['conditional_wins'] += 1
                            if fill_price is not None:
                                self.bucket_stats[price_bucket]['fill_prices'].update(fill_price)
    
    def get_summary(self):
        """Get aggregated summary statistics."""
        unconditional_win_rate = self.n_winners / self.n_tokens if self.n_tokens > 0 else 0
        
        summary = {
            'n_tokens': self.n_tokens,
            'n_winners': self.n_winners,
            'n_losers': self.n_losers,
            'unconditional_win_rate': unconditional_win_rate,
        }
        
        # Fill rates
        summary['fill_rates'] = {}
        for horizon in PLACEMENT_HORIZONS:
            horizon_key = f'H_{horizon}h'
            if horizon_key in self.fill_rates:
                summary['fill_rates'][horizon_key] = {}
                for offset, strat_name in ORDER_OFFSETS:
                    data = self.fill_rates[horizon_key][strat_name]
                    if data['total'] > 0:
                        summary['fill_rates'][horizon_key][strat_name] = {
                            'fill_rate': data['filled'] / data['total'],
                            'n_filled': data['filled'],
                            'n_total': data['total'],
                        }
        
        # Conditional analysis
        summary['conditional_analysis'] = {}
        for horizon in PLACEMENT_HORIZONS:
            horizon_key = f'H_{horizon}h'
            if horizon_key in self.conditional_stats:
                summary['conditional_analysis'][horizon_key] = {}
                for offset, strat_name in ORDER_OFFSETS:
                    data = self.conditional_stats[horizon_key][strat_name]
                    if data['fills'] >= 10:
                        cond_win_rate = data['wins'] / data['fills']
                        
                        fill_price_stats = data['fill_prices'].get_stats()
                        avg_fill_price = fill_price_stats['mean'] if fill_price_stats else 0
                        
                        adv_sel_tax = (unconditional_win_rate - cond_win_rate) * 10000
                        edge_after_fill = (cond_win_rate - avg_fill_price) * 10000
                        
                        summary['conditional_analysis'][horizon_key][strat_name] = {
                            'conditional_win_rate': cond_win_rate,
                            'avg_fill_price': avg_fill_price,
                            'adverse_selection_bps': adv_sel_tax,
                            'edge_after_fill_bps': edge_after_fill,
                            'n_fills': data['fills'],
                        }
        
        # Bucket analysis
        summary['bucket_analysis'] = {}
        for bucket_info in PRICE_BUCKETS:
            label = bucket_info[2]
            bs = self.bucket_stats[label]
            
            if bs['unconditional_total'] >= 20 and bs['conditional_fills'] >= 10:
                n_uncond = bs['unconditional_total']
                n_cond = bs['conditional_fills']
                
                uncond_wr = bs['unconditional_wins'] / n_uncond
                cond_wr = bs['conditional_wins'] / n_cond
                
                # Standard errors for win rates (binomial approximation)
                se_uncond_wr = np.sqrt(uncond_wr * (1 - uncond_wr) / n_uncond)
                se_cond_wr = np.sqrt(cond_wr * (1 - cond_wr) / n_cond)
                
                snapshot_stats = bs['snapshot_prices'].get_stats()
                fill_stats = bs['fill_prices'].get_stats()
                
                avg_snapshot = snapshot_stats['mean'] if snapshot_stats else 0
                avg_fill = fill_stats['mean'] if fill_stats else 0
                
                # Standard errors for prices (SEM)
                se_snapshot = (snapshot_stats['std'] / np.sqrt(snapshot_stats['n']) 
                              if snapshot_stats and snapshot_stats['n'] > 0 else 0)
                se_fill = (fill_stats['std'] / np.sqrt(fill_stats['n']) 
                          if fill_stats and fill_stats['n'] > 0 else 0)
                
                uncond_edge = (uncond_wr - avg_snapshot) * 10000
                cond_edge = (cond_wr - avg_fill) * 10000
                adv_sel = uncond_edge - cond_edge
                
                # Standard errors for edges (propagated, in bps)
                se_uncond_edge = np.sqrt(se_uncond_wr**2 + se_snapshot**2) * 10000
                se_cond_edge = np.sqrt(se_cond_wr**2 + se_fill**2) * 10000
                se_adv_sel = np.sqrt(se_uncond_edge**2 + se_cond_edge**2)
                
                # 95% confidence intervals
                z = 1.96
                
                summary['bucket_analysis'][label] = {
                    'unconditional_win_rate': uncond_wr,
                    'conditional_win_rate': cond_wr,
                    'unconditional_edge_bps': uncond_edge,
                    'conditional_edge_bps': cond_edge,
                    'adverse_selection_bps': adv_sel,
                    'fill_rate': n_cond / n_uncond,
                    'n_unconditional': n_uncond,
                    'n_conditional': n_cond,
                    # Confidence intervals (95%)
                    'uncond_edge_ci_low': uncond_edge - z * se_uncond_edge,
                    'uncond_edge_ci_high': uncond_edge + z * se_uncond_edge,
                    'cond_edge_ci_low': cond_edge - z * se_cond_edge,
                    'cond_edge_ci_high': cond_edge + z * se_cond_edge,
                    'adv_sel_ci_low': adv_sel - z * se_adv_sel,
                    'adv_sel_ci_high': adv_sel + z * se_adv_sel,
                    'se_uncond_edge': se_uncond_edge,
                    'se_cond_edge': se_cond_edge,
                    'se_adv_sel': se_adv_sel,
                }
        
        # Time to fill
        summary['time_to_fill'] = {}
        for horizon in PLACEMENT_HORIZONS:
            horizon_key = f'H_{horizon}h'
            ttf_stats = self.time_to_fill[horizon_key].get_stats()
            if ttf_stats and ttf_stats['n'] > 0:
                summary['time_to_fill'][horizon_key] = {
                    'median_hours': ttf_stats.get('p50'),
                    'mean_hours': ttf_stats['mean'],
                    'p25_hours': ttf_stats.get('p25'),
                    'p75_hours': ttf_stats.get('p75'),
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
# LIMIT ORDER SIMULATION
# ==============================================================================

def extract_price_at_horizon(trades, resolution_time, hours_before):
    """Extract price closest to specific time horizon before resolution."""
    target_time = resolution_time - (hours_before * 3600)
    
    tolerance_hours = max(1, hours_before * 0.25)
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
            'trades_until_fill': None,
        }
    
    for i, (ts, price, size) in enumerate(future_trades):
        if is_buy and price <= limit_price:
            return {
                'filled': True,
                'fill_time': ts,
                'fill_price': price,
                'time_to_fill': ts - placement_time,
                'trades_until_fill': i + 1,
            }
        elif not is_buy and price >= limit_price:
            return {
                'filled': True,
                'fill_time': ts,
                'fill_price': price,
                'time_to_fill': ts - placement_time,
                'trades_until_fill': i + 1,
            }
    
    return {
        'filled': False,
        'fill_time': None,
        'fill_price': None,
        'time_to_fill': None,
        'trades_until_fill': None,
    }

def simulate_orders_at_horizon(trades, resolution_time, horizon_hours, offsets=ORDER_OFFSETS):
    """Simulate limit orders at a specific horizon with various offsets."""
    placement_time, snapshot_price = extract_price_at_horizon(trades, resolution_time, horizon_hours)
    
    if placement_time is None or snapshot_price is None:
        return None
    
    results = {
        'horizon_hours': horizon_hours,
        'placement_time': placement_time,
        'snapshot_price': snapshot_price,
        'strategies': {},
    }
    
    for offset, strategy_name in offsets:
        limit_price = snapshot_price + offset
        limit_price = max(0.001, min(0.999, limit_price))
        
        fill_result = simulate_limit_order_fill(
            trades, placement_time, limit_price, is_buy=True
        )
        
        results['strategies'][strategy_name] = {
            'limit_price': limit_price,
            'offset': offset,
            **fill_result,
        }
    
    return results

# ==============================================================================
# DATA ACCUMULATOR CLASS
# ==============================================================================

class AdverseSelectionAccumulator:
    """Accumulates trade data for adverse selection simulation."""
    
    def __init__(self, token_id, condition_id, resolution_time, winner_status):
        self.token_id = token_id
        self.condition_id = condition_id
        self.resolution_time = resolution_time
        self.winner_status = winner_status
        self.trades = []
    
    def add_trades(self, trades_batch):
        self.trades.extend(trades_batch)
    
    def compute_adverse_selection(self):
        """Run adverse selection simulation for this token."""
        if len(self.trades) < MIN_TRADES:
            return None
        
        self.trades.sort(key=lambda x: x[0])
        
        earliest_trade = self.trades[0][0]
        hours_of_data = (self.resolution_time - earliest_trade) / 3600
        
        # Compute horizon results first (needed for snapshot-based bucketing)
        horizon_results = {}
        for horizon in PLACEMENT_HORIZONS:
            result = simulate_orders_at_horizon(
                self.trades, self.resolution_time, horizon
            )
            if result:
                horizon_results[f'H_{horizon}h'] = result
        
        if not horizon_results:
            return None
        
        # Use snapshot price at H_24h for bucket assignment (removes lookahead bias)
        # Fallback to first available horizon if H_24h not present
        snapshot_price_for_bucket = None
        if 'H_24h' in horizon_results:
            snapshot_price_for_bucket = horizon_results['H_24h']['snapshot_price']
        else:
            for horizon in PLACEMENT_HORIZONS:
                horizon_key = f'H_{horizon}h'
                if horizon_key in horizon_results:
                    snapshot_price_for_bucket = horizon_results[horizon_key]['snapshot_price']
                    break
        
        price_bucket = assign_bucket(snapshot_price_for_bucket, PRICE_BUCKETS)
        
        # Keep median_price for reference/backward compatibility
        prices = [t[1] for t in self.trades]
        median_price = np.median(prices)
        
        return {
            'token_id': self.token_id,
            'condition_id': self.condition_id,
            'winner': self.winner_status,
            'n_trades': len(self.trades),
            'hours_of_data': hours_of_data,
            'median_price': median_price,
            'snapshot_price_for_bucket': snapshot_price_for_bucket,
            'price_bucket': price_bucket,
            'horizon_results': horizon_results,
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
    log("PHASE 3B: ADVERSE SELECTION SIMULATION (MEMORY-SAFE)")
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
    # STREAMING AGGREGATOR
    # ==========================================================================
    
    start_file_idx = 0
    aggregator = AdverseSelectionAggregator()
    
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
                    
                    token_accumulators[token_id] = AdverseSelectionAccumulator(
                        token_id, condition_id, resolution_time, winner_status
                    )
                    condition_tokens[condition_id].add(token_id)
                
                timestamps = group['timestamp'].values
                prices = group['price'].values
                sizes = group['size_tokens'].values
                
                trades_batch = list(zip(timestamps, prices, sizes))
                token_accumulators[token_id].add_trades(trades_batch)
                
                if diagnostic and len(diagnostic_samples) < 3 and len(trades_batch) > 50:
                    diagnostic_samples.append({
                        'token_id': token_id,
                        'winner': winner_status,
                        'trades_sample': trades_batch[:100],
                        'n_trades': len(trades_batch),
                        'resolution_time': float(group['resolution_time'].iloc[0]),
                    })
            
            del df
            
            # ==================================================================
            # STREAMING FLUSH
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
                            adv_sel_result = acc.compute_adverse_selection()
                            
                            if adv_sel_result is not None:
                                # STREAMING: aggregate immediately
                                aggregator.add_result(adv_sel_result)
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
            adv_sel_result = acc.compute_adverse_selection()
            
            if adv_sel_result is not None:
                aggregator.add_result(adv_sel_result)
            else:
                stats['tokens_filtered'] += 1
            
            del token_accumulators[token_id]
        
        stats['conditions_flushed'] += 1
    
    gc.collect()
    
    log(f"\nProcessed {stats['files_processed']:,} files, {stats['total_rows']:,} rows")
    log(f"Computed adverse selection for {aggregator.n_tokens:,} tokens")
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
            log(f"\nToken: {sample['token_id'][:20]}...")
            log(f"  Winner: {sample['winner']}")
            log(f"  Total trades: {sample['n_trades']}")
            
            trades = sample['trades_sample']
            resolution_time = sample['resolution_time']
            
            result = simulate_orders_at_horizon(trades, resolution_time, 24)
            if result:
                log(f"  Simulation at H-24h:")
                log(f"    Snapshot price: {result['snapshot_price']:.4f}")
                for strat_name, strat_data in result['strategies'].items():
                    fill_str = "FILLED" if strat_data['filled'] else "NO FILL"
                    if strat_data['filled']:
                        fill_str += f" @ {strat_data['fill_price']:.4f}"
                        fill_str += f" after {strat_data['trades_until_fill']} trades"
                    log(f"    {strat_name:20s}: limit={strat_data['limit_price']:.4f} -> {fill_str}")
            else:
                log(f"  No snapshot available at H-24h")
    
    return aggregator

# ==============================================================================
# REPORTING
# ==============================================================================

def print_analysis(aggregator):
    """Print analysis from streaming aggregator."""
    summary = aggregator.get_summary()
    
    log("\n" + "="*70)
    log("ANALYSIS RESULTS")
    log("="*70)
    
    log(f"\nSample Size: {summary['n_tokens']:,} tokens")
    log(f"Unconditional Win Rate: {summary['unconditional_win_rate']:.4f} "
        f"({summary['unconditional_win_rate']*100:.2f}%)")
    
    # Fill rates
    log("\n" + "-"*50)
    log("1. FILL RATES BY HORIZON AND STRATEGY")
    log("-"*50)
    
    for horizon in PLACEMENT_HORIZONS:
        horizon_key = f'H_{horizon}h'
        if horizon_key in summary.get('fill_rates', {}):
            log(f"\n  {horizon_key}:")
            for offset, strat_name in ORDER_OFFSETS:
                data = summary['fill_rates'][horizon_key].get(strat_name)
                if data:
                    log(f"    {strat_name:20s}: {data['fill_rate']*100:5.1f}% "
                        f"({data['n_filled']:,}/{data['n_total']:,})")
    
    # Conditional win rate
    log("\n" + "-"*50)
    log("2. CONDITIONAL WIN RATE (among filled orders)")
    log("-"*50)
    log(f"If conditional < unconditional, there is adverse selection")
    log(f"Unconditional win rate: {summary['unconditional_win_rate']:.4f}")
    
    for horizon in PLACEMENT_HORIZONS:
        horizon_key = f'H_{horizon}h'
        if horizon_key in summary.get('conditional_analysis', {}):
            log(f"\n  {horizon_key}:")
            for offset, strat_name in ORDER_OFFSETS:
                data = summary['conditional_analysis'][horizon_key].get(strat_name)
                if data:
                    log(f"    {strat_name:20s}: win_rate={data['conditional_win_rate']:.4f}, "
                        f"adv_sel={data['adverse_selection_bps']:+.0f}bps, "
                        f"edge={data['edge_after_fill_bps']:+.0f}bps "
                        f"(n={data['n_fills']:,})")
    
    # Bucket analysis
    log("\n" + "-"*50)
    log("3. ADVERSE SELECTION BY PRICE BUCKET (at_mid strategy, H-24h)")
    log("-"*50)
    log("NOTE: Buckets assigned by snapshot price at entry (no lookahead)")
    
    for bucket_info in PRICE_BUCKETS:
        label = bucket_info[2]
        data = summary.get('bucket_analysis', {}).get(label)
        if data:
            log(f"\n  {label:12s}:")
            log(f"    Unconditional: win_rate={data['unconditional_win_rate']:.4f}, "
                f"edge={data['unconditional_edge_bps']:+.0f}bps "
                f"[{data['uncond_edge_ci_low']:+.0f}, {data['uncond_edge_ci_high']:+.0f}] "
                f"(n={data['n_unconditional']:,})")
            log(f"    Conditional:   win_rate={data['conditional_win_rate']:.4f}, "
                f"edge={data['conditional_edge_bps']:+.0f}bps "
                f"[{data['cond_edge_ci_low']:+.0f}, {data['cond_edge_ci_high']:+.0f}] "
                f"(n={data['n_conditional']:,})")
            log(f"    Adverse Selection Tax: {data['adverse_selection_bps']:+.0f}bps "
                f"[{data['adv_sel_ci_low']:+.0f}, {data['adv_sel_ci_high']:+.0f}]")
    
    # Time to fill
    log("\n" + "-"*50)
    log("4. TIME TO FILL DISTRIBUTION (at_mid strategy)")
    log("-"*50)
    
    for horizon in PLACEMENT_HORIZONS:
        horizon_key = f'H_{horizon}h'
        data = summary.get('time_to_fill', {}).get(horizon_key)
        if data:
            log(f"  {horizon_key}: median={data['median_hours']:.1f}h, "
                f"mean={data['mean_hours']:.1f}h, "
                f"P25-P75=[{data['p25_hours']:.1f}, {data['p75_hours']:.1f}]h")
    
    return summary

def save_results_json(summary):
    ensure_output_dir()
    
    output = {
        'timestamp': TIMESTAMP,
        'phase': '3B',
        'description': 'Adverse Selection Simulation (Memory-Safe)',
        'methodology': 'Simulate limit order fills, compare conditional vs unconditional outcomes',
        **summary,
    }
    
    json_path = os.path.join(OUTPUT_DIR, f'phase3b_adverse_selection_{TIMESTAMP}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    log(f"Results JSON saved: {json_path}")
    return json_path

def generate_report(summary):
    ensure_output_dir()
    
    report_path = os.path.join(OUTPUT_DIR, f'phase3b_adverse_selection_{TIMESTAMP}_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 3B: ADVERSE SELECTION SIMULATION - REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tokens Analyzed: {summary.get('n_tokens', 0):,}\n")
        f.write(f"Unconditional Win Rate: {summary.get('unconditional_win_rate', 0):.4f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("ADVERSE SELECTION BY PRICE BUCKET\n")
        f.write("-"*80 + "\n")
        f.write("NOTE: Buckets assigned by snapshot price at entry (no lookahead)\n")
        f.write("95% confidence intervals shown in brackets\n\n")
        
        for label, data in summary.get('bucket_analysis', {}).items():
            f.write(f"  {label:12s}:\n")
            f.write(f"    Unconditional Edge: {data['unconditional_edge_bps']:+.0f} bps "
                    f"[{data['uncond_edge_ci_low']:+.0f}, {data['uncond_edge_ci_high']:+.0f}]\n")
            f.write(f"    Edge After Fill:    {data['conditional_edge_bps']:+.0f} bps "
                    f"[{data['cond_edge_ci_low']:+.0f}, {data['cond_edge_ci_high']:+.0f}]\n")
            f.write(f"    Adverse Selection:  {data['adverse_selection_bps']:+.0f} bps "
                    f"[{data['adv_sel_ci_low']:+.0f}, {data['adv_sel_ci_high']:+.0f}]\n")
            f.write(f"    Fill Rate:          {data['fill_rate']*100:.1f}%\n")
            f.write(f"    Sample Size:        n_uncond={data['n_unconditional']:,}, "
                    f"n_cond={data['n_conditional']:,}\n\n")
    
    log(f"Report saved: {report_path}")
    return report_path

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 3B: Adverse Selection Simulation (Memory-Safe)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnostic mode
  python phase3b_adverse_selection_memsafe.py --diagnostic --sample 100
  
  # Full run
  python phase3b_adverse_selection_memsafe.py
  
  # Resume interrupted run
  python phase3b_adverse_selection_memsafe.py --resume
        """
    )
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--diagnostic', '-d', action='store_true',
                        help='Run in diagnostic mode with extra validation')
    
    args = parser.parse_args()
    
    log("Starting Phase 3B Adverse Selection Analysis (Memory-Safe)...")
    
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
        log(f"  - Report: phase3b_adverse_selection_{TIMESTAMP}_report.txt")
        log(f"  - JSON:   phase3b_adverse_selection_{TIMESTAMP}.json")
    else:
        log("\nAnalysis failed - no results generated")