#!/usr/bin/env python3
"""
Phase 4D Volume Percentile Stratification - MEMORY SAFE VERSION
Version: 1.0

OBJECTIVE:
  Stratify existing dip-fade analysis by volume at the dip crossing to determine
  if high-volume dips show systematically worse outcomes.
  
  From the framing document:
  "Before Phase 4e (rolling window), run a quick pass on your existing data 
   stratifying by volume percentile at the dip. If high-volume dips show 
   systematically worse outcomes, you have your discriminator."

APPROACH - TWO STAGE:
  Stage 1: Stream through data and write individual returns WITH VOLUME to disk
           - Captures volume in a window around the dip crossing
           - Memory safe: flush to disk after each batch
           - Append-only: never overwrite existing output files
  
  Stage 2: Read parquet chunks, compute volume percentiles, and generate 
           distribution analysis stratified by volume quintile
           - Stratified by: window, threshold, tercile, prob_bucket, volume_quintile
           - Box plots, histograms, percentile tables
           - Compare high vs low volume dip outcomes

MEMORY SAFETY:
  - Streaming write to disk (no accumulation of large arrays in memory)
  - Chunked reading for plotting
  - Progressive flush after each file batch
  - Separate diagnostic mode data paths

CACHE SAFETY:
  - NEVER overwrites existing cache or checkpoint files
  - Uses timestamped output directories
  - Checks for existing files before writing
"""

import pyarrow.parquet as pq
import pyarrow as pa
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
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, f'phase4d_volume_checkpoint_{TIMESTAMP}.pkl')

# Returns data output (timestamped to never overwrite)
RETURNS_DATA_DIR = os.path.join(OUTPUT_DIR, f'phase4d_volume_data_{TIMESTAMP}')
DIAGNOSTIC_RETURNS_DIR = os.path.join(OUTPUT_DIR, f'phase4d_volume_data_DIAGNOSTIC_{TIMESTAMP}')

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
PROB_BUCKET_LABELS = [b[0] for b in PROB_BUCKETS] + ['all']

# Volume quintile labels
VOLUME_QUINTILE_LABELS = ['Q1_low', 'Q2', 'Q3', 'Q4', 'Q5_high']

# Window around crossing for volume calculation (seconds)
VOLUME_WINDOW_BEFORE = 1800  # 30 minutes before crossing
VOLUME_WINDOW_AFTER = 1800   # 30 minutes after crossing

MIN_TRADES = 10
MIN_SAMPLES_PER_CELL = 20

REQUIRED_COLUMNS = ['timestamp', 'price', 'token_id', 'condition_id', 'resolution_time']
VOLUME_COLUMNS = ['size_tokens', 'maker_amount', 'size']

PROGRESS_INTERVAL = 1000
CHUNK_FLUSH_INTERVAL = 5000  # Flush returns to disk every N tokens


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

def ensure_returns_dir(diagnostic=False):
    """Create returns data directory - NEVER overwrites existing."""
    target_dir = DIAGNOSTIC_RETURNS_DIR if diagnostic else RETURNS_DATA_DIR
    
    # Safety check - never overwrite
    if os.path.exists(target_dir):
        log(f"  WARNING: Returns directory already exists: {target_dir}")
        log(f"           Using timestamped suffix to avoid overwrite")
        target_dir = target_dir + f"_{datetime.now().strftime('%H%M%S')}"
    
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

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
    if quintile_thresholds is None:
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
# SIDECAR AND CACHE LOADING (READ-ONLY)
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
    if time_diff > 3600:
        return None, None
    
    return closest_trade[0], closest_trade[1]


def find_first_threshold_crossing(trades, start_time, end_time, start_price, threshold, direction='drop'):
    """Find the first time price crosses threshold within interval."""
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


def calculate_volume_at_crossing(trades, crossing_time, window_before=1800, window_after=1800):
    """
    Calculate volume in a window around the crossing time.
    
    Args:
        trades: List of (timestamp, price, size) tuples
        crossing_time: Unix timestamp of the crossing
        window_before: Seconds before crossing to include
        window_after: Seconds after crossing to include
    
    Returns:
        Total volume in the window (sum of trade sizes)
    """
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


# ==============================================================================
# RETURNS WRITER - STREAMING PARQUET OUTPUT (WITH VOLUME)
# ==============================================================================

class VolumeReturnsWriter:
    """
    Memory-safe streaming writer for returns data with volume information.
    
    Accumulates returns in memory up to a threshold, then flushes to parquet.
    Never overwrites existing files - uses incremental chunk naming.
    """
    
    def __init__(self, output_dir, chunk_size=10000):
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.buffer = []
        self.chunk_counter = 0
        self.total_written = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for existing chunks to continue numbering
        existing = glob.glob(os.path.join(output_dir, 'volume_returns_chunk_*.parquet'))
        if existing:
            max_idx = max(int(f.split('_')[-1].replace('.parquet', '')) for f in existing)
            self.chunk_counter = max_idx + 1
            log(f"  Found {len(existing)} existing chunks, starting at {self.chunk_counter}")
    
    def add_return(self, interval_label, threshold, tercile, prob_bucket,
                   entry_price, fill_price, is_winner, return_bps, volume_at_dip):
        """Add a single return observation to buffer."""
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
        """Write buffer to parquet file."""
        if not self.buffer:
            return
        
        df = pd.DataFrame(self.buffer)
        
        # Ensure unique filename (never overwrite)
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
        """Flush any remaining data and return summary."""
        self.flush()
        return {
            'total_written': self.total_written,
            'num_chunks': self.chunk_counter,
            'output_dir': self.output_dir,
        }


# ==============================================================================
# TOKEN ACCUMULATOR (WITH VOLUME CAPTURE)
# ==============================================================================

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
    
    def compute_returns_with_volume(self, returns_writer):
        """
        Compute returns for all cells WITH VOLUME AT DIP and write to returns writer.
        
        Returns are computed as: (win - fill_price) * 10000 bps
        where win = 1 if winner, 0 otherwise
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
                
                # Determine tercile
                fraction = crossing['fraction_of_interval']
                tercile = get_tercile(fraction)
                if tercile is None:
                    continue
                
                # Calculate volume at dip
                crossing_time = crossing['crossing_time']
                volume_at_dip = calculate_volume_at_crossing(
                    self.trades, 
                    crossing_time,
                    window_before=VOLUME_WINDOW_BEFORE,
                    window_after=VOLUME_WINDOW_AFTER
                )
                
                # Simulate fill
                fill_result = simulate_limit_order_fill(
                    self.trades, crossing['crossing_time'], crossing['crossing_price'], is_buy=True
                )
                
                if not fill_result.get('filled'):
                    continue
                
                fill_price = fill_result['fill_price']
                
                # Compute return (in basis points)
                outcome = 1.0 if self.winner_status else 0.0
                return_bps = (outcome - fill_price) * 10000
                
                # Write to both specific bucket and 'all' bucket
                for bucket in [prob_bucket, 'all']:
                    returns_writer.add_return(
                        interval_label=interval_label,
                        threshold=threshold,
                        tercile=tercile,
                        prob_bucket=bucket,
                        entry_price=crossing['crossing_price'],
                        fill_price=fill_price,
                        is_winner=self.winner_status,
                        return_bps=return_bps,
                        volume_at_dip=volume_at_dip,
                    )
                    n_returns += 1
        
        return n_returns


# ==============================================================================
# STAGE 1: DATA COLLECTION (WITH VOLUME)
# ==============================================================================

def run_stage1_collection(sample_files=None, diagnostic=False):
    """
    Stage 1: Stream through data and collect individual returns WITH VOLUME to disk.
    
    Memory safe: writes to parquet chunks incrementally.
    """
    global start_time
    start_time = datetime.now()
    
    log("="*70)
    log("PHASE 4D VOLUME STRATIFICATION - STAGE 1: DATA COLLECTION")
    log("="*70)
    
    if diagnostic:
        log("\n*** DIAGNOSTIC MODE ***")
    
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
    # SETUP RETURNS WRITER (WITH VOLUME)
    # -------------------------------------------------------------------------
    
    returns_dir = ensure_returns_dir(diagnostic)
    log(f"\nReturns data will be written to: {returns_dir}")
    
    returns_writer = VolumeReturnsWriter(returns_dir, chunk_size=10000)
    
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
    # STREAMING ACCUMULATION
    # -------------------------------------------------------------------------
    
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
    
    log(f"\nProcessing {len(files_to_process_indices)} files...")
    log(f"  {log_memory()}")
    
    # -------------------------------------------------------------------------
    # MAIN PROCESSING LOOP
    # -------------------------------------------------------------------------
    
    for file_idx in files_to_process_indices:
        stats['files_processed'] += 1
        filepath = batch_files[file_idx]
        
        if stats['files_processed'] % PROGRESS_INTERVAL == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = stats['files_processed'] / elapsed if elapsed > 0 else 0
            eta = (len(files_to_process_indices) - stats['files_processed']) / rate if rate > 0 else 0
            
            log(f"  [{stats['files_processed']:,}/{len(files_to_process_indices):,}] "
                f"Active: {len(token_accumulators):,} | "
                f"Flushed: {stats['conditions_flushed']:,} | "
                f"Returns: {returns_writer.total_written:,} | "
                f"Rate: {rate:.1f}/s | "
                f"ETA: {format_duration(eta)} | "
                f"{log_memory()}")
            
            if stats['files_processed'] % (PROGRESS_INTERVAL * 5) == 0:
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
                    
                    token_accumulators[token_id] = VolumeTimingAccumulator(
                        token_id, condition_id, resolution_time, winner_status
                    )
                    condition_tokens[condition_id].add(token_id)
                
                timestamps = group['timestamp'].values
                prices = group['price'].values
                sizes = group['size_tokens'].values
                
                trades_batch = list(zip(timestamps, prices, sizes))
                token_accumulators[token_id].add_trades(trades_batch)
            
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
                            n_returns = acc.compute_returns_with_volume(returns_writer)
                            
                            if n_returns == 0:
                                stats['tokens_filtered'] += 1
                            else:
                                stats['returns_written'] += n_returns
                            
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
                n_returns = acc.compute_returns_with_volume(returns_writer)
                
                if n_returns == 0:
                    stats['tokens_filtered'] += 1
                else:
                    stats['returns_written'] += n_returns
                
                del token_accumulators[token_id]
            
            stats['conditions_flushed'] += 1
        
        token_accumulators.clear()
        condition_tokens.clear()
    
    # Finalize returns writer
    writer_summary = returns_writer.finalize()
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    log("\n" + "="*70)
    log("STAGE 1 COMPLETE")
    log("="*70)
    log(f"\nFiles processed: {stats['files_processed']:,}")
    log(f"Total rows: {stats['total_rows']:,}")
    log(f"Conditions flushed: {stats['conditions_flushed']:,}")
    log(f"Tokens filtered: {stats['tokens_filtered']:,}")
    log(f"Tokens no winner: {stats['tokens_no_winner']:,}")
    log(f"\nReturns data (with volume):")
    log(f"  Total returns written: {writer_summary['total_written']:,}")
    log(f"  Number of chunks: {writer_summary['num_chunks']}")
    log(f"  Output directory: {writer_summary['output_dir']}")
    log(f"\nElapsed: {format_duration(elapsed)}")
    
    # Save metadata
    metadata = {
        'timestamp': TIMESTAMP,
        'diagnostic': diagnostic,
        'stats': stats,
        'writer_summary': writer_summary,
        'interval_pairs': [(s, e, l) for s, e, l in INTERVAL_PAIRS],
        'move_thresholds': MOVE_THRESHOLDS,
        'tercile_bounds': list(zip(TERCILE_LABELS, TERCILE_BOUNDS)),
        'prob_buckets': PROB_BUCKETS,
        'volume_window_before': VOLUME_WINDOW_BEFORE,
        'volume_window_after': VOLUME_WINDOW_AFTER,
    }
    
    metadata_path = os.path.join(returns_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    log(f"\nMetadata saved: {metadata_path}")
    
    return returns_dir


# ==============================================================================
# STAGE 2: ANALYSIS AND PLOTTING (WITH VOLUME STRATIFICATION)
# ==============================================================================

def load_returns_data(returns_dir):
    """Load all returns data from parquet chunks."""
    chunk_files = sorted(glob.glob(os.path.join(returns_dir, 'volume_returns_chunk_*.parquet')))
    
    if not chunk_files:
        log(f"ERROR: No chunk files found in {returns_dir}")
        return None
    
    log(f"  Loading {len(chunk_files)} chunk files...")
    
    dfs = []
    for chunk_file in chunk_files:
        df = pd.read_parquet(chunk_file)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    log(f"  Loaded {len(combined):,} returns with volume data")
    
    return combined


def compute_volume_quintiles(returns_df):
    """
    Compute volume quintile thresholds and assign quintile labels.
    
    Quintiles are computed globally across all observations.
    Returns the modified dataframe with 'volume_quintile' column.
    """
    log("\nComputing volume quintiles...")
    
    # Filter out zero volume (markets with no trades in window)
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


def compute_percentiles_by_volume(returns_df):
    """
    Compute percentile statistics by cell, stratified by volume quintile.
    
    Returns DataFrame with columns:
      interval, threshold, tercile, prob_bucket, volume_quintile,
      n, p10, p25, p50, p75, p90, mean, std, min, max, win_rate
    """
    results = []
    
    # Filter to valid volume quintiles
    valid_df = returns_df[returns_df['volume_quintile'].notna()]
    
    group_cols = ['interval', 'threshold', 'tercile', 'prob_bucket', 'volume_quintile']
    
    for keys, group in valid_df.groupby(group_cols):
        interval, threshold, tercile, prob_bucket, volume_quintile = keys
        returns = group['return_bps'].values
        
        if len(returns) < MIN_SAMPLES_PER_CELL:
            continue
        
        results.append({
            'interval': interval,
            'threshold': threshold,
            'tercile': tercile,
            'prob_bucket': prob_bucket,
            'volume_quintile': volume_quintile,
            'n': len(returns),
            'p10': np.percentile(returns, 10),
            'p25': np.percentile(returns, 25),
            'p50': np.percentile(returns, 50),
            'p75': np.percentile(returns, 75),
            'p90': np.percentile(returns, 90),
            'mean': np.mean(returns),
            'std': np.std(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'win_rate': group['is_winner'].mean(),
            'mean_volume': group['volume_at_dip'].mean(),
        })
    
    return pd.DataFrame(results)


def generate_volume_plots(returns_df, output_dir):
    """
    Generate distribution plots stratified by volume quintile.
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    log("\nGenerating volume stratification plots...")
    
    valid_df = returns_df[returns_df['volume_quintile'].notna()]
    
    # -------------------------------------------------------------------------
    # 1. Box plots: Returns by Volume Quintile (10% threshold, early tercile)
    # -------------------------------------------------------------------------
    
    log("  1. Box plots by volume quintile (per prob bucket)...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, prob_bucket in enumerate(PROB_BUCKET_LABELS[:6]):
        subset = valid_df[
            (valid_df['threshold'] == 0.10) & 
            (valid_df['tercile'] == 'early') &
            (valid_df['prob_bucket'] == prob_bucket)
        ]
        
        if len(subset) == 0:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(f'{prob_bucket}')
            continue
        
        data_by_quintile = [subset[subset['volume_quintile'] == q]['return_bps'].values 
                           for q in VOLUME_QUINTILE_LABELS]
        
        valid_data = [(d, q) for d, q in zip(data_by_quintile, VOLUME_QUINTILE_LABELS) if len(d) > 0]
        
        if valid_data:
            bp = axes[idx].boxplot([d[0] for d in valid_data], labels=[d[1] for d in valid_data])
            axes[idx].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # Add sample sizes
            for i, (d, _) in enumerate(valid_data):
                axes[idx].text(i+1, axes[idx].get_ylim()[1], f'n={len(d)}', 
                              ha='center', va='bottom', fontsize=8)
        
        axes[idx].set_title(f'{prob_bucket}')
        axes[idx].set_ylabel('Return (bps)')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Returns by Volume Quintile at Dip\n(10% threshold, early tercile)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'boxplot_by_volume_quintile.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 2. CRITICAL: 24h_to_12h by Volume Quintile (the anomaly)
    # -------------------------------------------------------------------------
    
    log("  2. 24h_to_12h anomaly by volume quintile...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 2a. 90_99 bucket: Returns by volume quintile
    subset_90_99 = valid_df[
        (valid_df['interval'] == '24h_to_12h') &
        (valid_df['threshold'] == 0.10) & 
        (valid_df['tercile'] == 'early') &
        (valid_df['prob_bucket'] == '90_99')
    ]
    
    data_by_quintile = [subset_90_99[subset_90_99['volume_quintile'] == q]['return_bps'].values 
                       for q in VOLUME_QUINTILE_LABELS]
    valid_data = [(d, q) for d, q in zip(data_by_quintile, VOLUME_QUINTILE_LABELS) if len(d) > 0]
    
    if valid_data:
        bp = axes[0,0].boxplot([d[0] for d in valid_data], labels=[d[1] for d in valid_data], patch_artist=True)
        colors = ['lightgreen', 'lightgreen', 'lightyellow', 'lightsalmon', 'salmon']
        for patch, color in zip(bp['boxes'], colors[:len(valid_data)]):
            patch.set_facecolor(color)
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        for i, (d, _) in enumerate(valid_data):
            axes[0,0].text(i+1, axes[0,0].get_ylim()[1]*0.95, f'n={len(d)}\nmean={np.mean(d):.0f}', 
                          ha='center', va='top', fontsize=8)
    
    axes[0,0].set_title('24h_to_12h, 90_99, Early: Returns by Volume Quintile')
    axes[0,0].set_ylabel('Return (bps)')
    axes[0,0].set_xlabel('Volume Quintile')
    
    # 2b. Compare low vs high volume across all prob buckets for 24h_to_12h
    subset_24h = valid_df[
        (valid_df['interval'] == '24h_to_12h') &
        (valid_df['threshold'] == 0.10) & 
        (valid_df['tercile'] == 'early')
    ]
    
    low_vol_data = []
    high_vol_data = []
    bucket_labels = []
    
    for pb in PROB_BUCKET_LABELS[:6]:
        low_vol = subset_24h[(subset_24h['prob_bucket'] == pb) & 
                            (subset_24h['volume_quintile'].isin(['Q1_low', 'Q2']))]['return_bps'].values
        high_vol = subset_24h[(subset_24h['prob_bucket'] == pb) & 
                             (subset_24h['volume_quintile'].isin(['Q4', 'Q5_high']))]['return_bps'].values
        
        if len(low_vol) >= MIN_SAMPLES_PER_CELL and len(high_vol) >= MIN_SAMPLES_PER_CELL:
            low_vol_data.append(np.mean(low_vol))
            high_vol_data.append(np.mean(high_vol))
            bucket_labels.append(pb)
    
    if bucket_labels:
        x = np.arange(len(bucket_labels))
        width = 0.35
        
        axes[0,1].bar(x - width/2, low_vol_data, width, label='Low Volume (Q1-Q2)', color='lightgreen')
        axes[0,1].bar(x + width/2, high_vol_data, width, label='High Volume (Q4-Q5)', color='salmon')
        axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(bucket_labels, rotation=45)
        axes[0,1].legend()
    
    axes[0,1].set_title('24h_to_12h: Mean Edge by Volume (Low vs High)')
    axes[0,1].set_ylabel('Mean Return (bps)')
    
    # 2c. Volume discrimination across ALL intervals for 90_99
    subset_90_99_all = valid_df[
        (valid_df['threshold'] == 0.10) & 
        (valid_df['tercile'] == 'early') &
        (valid_df['prob_bucket'] == '90_99')
    ]
    
    intervals = [x[2] for x in INTERVAL_PAIRS]
    low_vol_means = []
    high_vol_means = []
    valid_intervals = []
    
    for iv in intervals:
        low_vol = subset_90_99_all[(subset_90_99_all['interval'] == iv) & 
                                   (subset_90_99_all['volume_quintile'].isin(['Q1_low', 'Q2']))]['return_bps'].values
        high_vol = subset_90_99_all[(subset_90_99_all['interval'] == iv) & 
                                    (subset_90_99_all['volume_quintile'].isin(['Q4', 'Q5_high']))]['return_bps'].values
        
        if len(low_vol) >= MIN_SAMPLES_PER_CELL and len(high_vol) >= MIN_SAMPLES_PER_CELL:
            low_vol_means.append(np.mean(low_vol))
            high_vol_means.append(np.mean(high_vol))
            valid_intervals.append(iv)
    
    if valid_intervals:
        x = np.arange(len(valid_intervals))
        width = 0.35
        
        bars1 = axes[1,0].bar(x - width/2, low_vol_means, width, label='Low Volume (Q1-Q2)', color='lightgreen')
        bars2 = axes[1,0].bar(x + width/2, high_vol_means, width, label='High Volume (Q4-Q5)', color='salmon')
        axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(valid_intervals, rotation=45)
        axes[1,0].legend()
        
        # Highlight 24h_to_12h
        if '24h_to_12h' in valid_intervals:
            idx_24h = valid_intervals.index('24h_to_12h')
            axes[1,0].axvspan(idx_24h - 0.5, idx_24h + 0.5, alpha=0.2, color='red')
    
    axes[1,0].set_title('90_99 Bucket: Volume Discrimination by Interval')
    axes[1,0].set_ylabel('Mean Return (bps)')
    
    # 2d. Scatter: Volume vs Returns for 24h_to_12h, 90_99
    if len(subset_90_99) > 0:
        axes[1,1].scatter(subset_90_99['volume_at_dip'], subset_90_99['return_bps'], 
                         alpha=0.3, s=20, c=subset_90_99['is_winner'].map({True: 'green', False: 'red'}))
        axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add trend line
        if len(subset_90_99) > 10:
            z = np.polyfit(subset_90_99['volume_at_dip'], subset_90_99['return_bps'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset_90_99['volume_at_dip'].min(), subset_90_99['volume_at_dip'].max(), 100)
            axes[1,1].plot(x_line, p(x_line), 'b--', alpha=0.7, label=f'Trend: {z[0]:.2f}x + {z[1]:.0f}')
            axes[1,1].legend()
        
        # Compute correlation
        corr = subset_90_99['volume_at_dip'].corr(subset_90_99['return_bps'])
        axes[1,1].set_title(f'24h_to_12h, 90_99: Volume vs Returns\n(corr = {corr:.3f})')
    else:
        axes[1,1].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[1,1].set_title('24h_to_12h, 90_99: Volume vs Returns')
    
    axes[1,1].set_xlabel('Volume at Dip')
    axes[1,1].set_ylabel('Return (bps)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'volume_24h_to_12h_anomaly.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 3. Volume discrimination heatmap
    # -------------------------------------------------------------------------
    
    log("  3. Volume discrimination heatmap...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Compute mean edge by interval x prob_bucket for low vs high volume
    intervals = [x[2] for x in INTERVAL_PAIRS]
    buckets = PROB_BUCKET_LABELS[:6]
    
    low_vol_matrix = np.full((len(intervals), len(buckets)), np.nan)
    high_vol_matrix = np.full((len(intervals), len(buckets)), np.nan)
    
    subset_early = valid_df[(valid_df['threshold'] == 0.10) & (valid_df['tercile'] == 'early')]
    
    for i, iv in enumerate(intervals):
        for j, pb in enumerate(buckets):
            low_vol = subset_early[(subset_early['interval'] == iv) & 
                                   (subset_early['prob_bucket'] == pb) &
                                   (subset_early['volume_quintile'].isin(['Q1_low', 'Q2']))]['return_bps'].values
            high_vol = subset_early[(subset_early['interval'] == iv) & 
                                    (subset_early['prob_bucket'] == pb) &
                                    (subset_early['volume_quintile'].isin(['Q4', 'Q5_high']))]['return_bps'].values
            
            if len(low_vol) >= MIN_SAMPLES_PER_CELL:
                low_vol_matrix[i, j] = np.mean(low_vol)
            if len(high_vol) >= MIN_SAMPLES_PER_CELL:
                high_vol_matrix[i, j] = np.mean(high_vol)
    
    # Low volume heatmap
    im1 = axes[0].imshow(low_vol_matrix, cmap='RdYlGn', aspect='auto', 
                         vmin=-500, vmax=500)
    axes[0].set_xticks(range(len(buckets)))
    axes[0].set_xticklabels(buckets, rotation=45)
    axes[0].set_yticks(range(len(intervals)))
    axes[0].set_yticklabels(intervals)
    axes[0].set_title('Low Volume (Q1-Q2): Mean Edge (bps)')
    plt.colorbar(im1, ax=axes[0])
    
    # Add values
    for i in range(len(intervals)):
        for j in range(len(buckets)):
            val = low_vol_matrix[i, j]
            if not np.isnan(val):
                axes[0].text(j, i, f'{val:.0f}', ha='center', va='center', 
                           fontsize=8, color='black')
    
    # High volume heatmap
    im2 = axes[1].imshow(high_vol_matrix, cmap='RdYlGn', aspect='auto',
                         vmin=-500, vmax=500)
    axes[1].set_xticks(range(len(buckets)))
    axes[1].set_xticklabels(buckets, rotation=45)
    axes[1].set_yticks(range(len(intervals)))
    axes[1].set_yticklabels(intervals)
    axes[1].set_title('High Volume (Q4-Q5): Mean Edge (bps)')
    plt.colorbar(im2, ax=axes[1])
    
    # Add values
    for i in range(len(intervals)):
        for j in range(len(buckets)):
            val = high_vol_matrix[i, j]
            if not np.isnan(val):
                axes[1].text(j, i, f'{val:.0f}', ha='center', va='center', 
                           fontsize=8, color='black')
    
    plt.suptitle('Volume Discrimination: Low vs High Volume Mean Edge\n(10% threshold, early tercile)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'volume_discrimination_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 4. Volume discrimination difference heatmap
    # -------------------------------------------------------------------------
    
    log("  4. Volume discrimination difference heatmap...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    diff_matrix = low_vol_matrix - high_vol_matrix
    
    im = ax.imshow(diff_matrix, cmap='RdYlGn', aspect='auto', vmin=-500, vmax=500)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets, rotation=45)
    ax.set_yticks(range(len(intervals)))
    ax.set_yticklabels(intervals)
    ax.set_title('Volume Discrimination: Low Vol - High Vol (bps)\nPositive = Low volume better, Negative = High volume better')
    plt.colorbar(im, ax=ax)
    
    # Add values and significance markers
    for i in range(len(intervals)):
        for j in range(len(buckets)):
            val = diff_matrix[i, j]
            if not np.isnan(val):
                color = 'black'
                marker = ''
                if abs(val) > 200:
                    marker = '*'
                ax.text(j, i, f'{val:.0f}{marker}', ha='center', va='center', 
                       fontsize=8, color=color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'volume_discrimination_difference.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"  Plots saved to: {plots_dir}")
    return plots_dir


def generate_volume_report(percentiles_df, returns_df, output_dir, quintile_thresholds):
    """Generate text report with volume stratification analysis."""
    
    report_path = os.path.join(output_dir, f'volume_stratification_report_{TIMESTAMP}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 4D: VOLUME PERCENTILE STRATIFICATION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("OBJECTIVE\n")
        f.write("-"*80 + "\n\n")
        
        f.write("Stratify dip-fade analysis by volume at the dip crossing to determine\n")
        f.write("if high-volume dips show systematically worse outcomes.\n\n")
        f.write("From the framing document:\n")
        f.write('"Before Phase 4e (rolling window), run a quick pass on your existing data\n')
        f.write(' stratifying by volume percentile at the dip. If high-volume dips show\n')
        f.write(' systematically worse outcomes, you have your discriminator."\n\n')
        
        f.write("-"*80 + "\n")
        f.write("VOLUME QUINTILE THRESHOLDS\n")
        f.write("-"*80 + "\n\n")
        
        if quintile_thresholds:
            q20, q40, q60, q80 = quintile_thresholds
            f.write(f"Q1 (low):    volume <= {q20:.2f}\n")
            f.write(f"Q2:          {q20:.2f} < volume <= {q40:.2f}\n")
            f.write(f"Q3:          {q40:.2f} < volume <= {q60:.2f}\n")
            f.write(f"Q4:          {q60:.2f} < volume <= {q80:.2f}\n")
            f.write(f"Q5 (high):   volume > {q80:.2f}\n\n")
        
        f.write("Volume calculated as: sum of trade sizes in 30-min window around dip crossing\n\n")
        
        # Key findings section
        f.write("="*80 + "\n")
        f.write("KEY FINDINGS: VOLUME DISCRIMINATION\n")
        f.write("="*80 + "\n\n")
        
        valid_df = returns_df[returns_df['volume_quintile'].notna()]
        
        # 24h_to_12h anomaly analysis
        f.write("1. 24h_to_12h INTERVAL ANALYSIS (The Anomaly)\n")
        f.write("-"*60 + "\n\n")
        
        subset_24h = valid_df[
            (valid_df['interval'] == '24h_to_12h') &
            (valid_df['threshold'] == 0.10) &
            (valid_df['tercile'] == 'early')
        ]
        
        f.write("Mean Returns by Volume Quintile (10% threshold, early tercile):\n\n")
        f.write(f"{'Prob Bucket':<12} {'Q1_low':>10} {'Q2':>10} {'Q3':>10} {'Q4':>10} {'Q5_high':>10} {'Diff(Q1-Q5)':>12}\n")
        f.write("-"*76 + "\n")
        
        for pb in PROB_BUCKET_LABELS[:6]:
            row = f"{pb:<12}"
            values = []
            for q in VOLUME_QUINTILE_LABELS:
                data = subset_24h[(subset_24h['prob_bucket'] == pb) & 
                                 (subset_24h['volume_quintile'] == q)]['return_bps']
                if len(data) >= MIN_SAMPLES_PER_CELL:
                    mean_val = data.mean()
                    values.append(mean_val)
                    row += f" {mean_val:>9.0f}"
                else:
                    values.append(None)
                    row += f" {'N/A':>9}"
            
            # Difference
            if values[0] is not None and values[4] is not None:
                diff = values[0] - values[4]
                row += f" {diff:>11.0f}"
                if diff > 200:
                    row += " *LOW BETTER"
                elif diff < -200:
                    row += " *HIGH BETTER"
            else:
                row += f" {'N/A':>11}"
            
            f.write(row + "\n")
        
        f.write("\n* indicates >200 bps difference\n\n")
        
        # Statistical test
        f.write("\n2. STATISTICAL TESTS: LOW VS HIGH VOLUME\n")
        f.write("-"*60 + "\n\n")
        
        f.write(f"{'Interval':<12} {'Bucket':<10} {'Low Mean':>10} {'High Mean':>10} {'Diff':>10} {'t-stat':>8} {'p-value':>10}\n")
        f.write("-"*80 + "\n")
        
        critical_tests = []
        
        for iv in [x[2] for x in INTERVAL_PAIRS]:
            for pb in ['90_99', '75_90']:
                subset = valid_df[
                    (valid_df['interval'] == iv) &
                    (valid_df['threshold'] == 0.10) &
                    (valid_df['tercile'] == 'early') &
                    (valid_df['prob_bucket'] == pb)
                ]
                
                low_vol = subset[subset['volume_quintile'].isin(['Q1_low', 'Q2'])]['return_bps'].values
                high_vol = subset[subset['volume_quintile'].isin(['Q4', 'Q5_high'])]['return_bps'].values
                
                if len(low_vol) >= MIN_SAMPLES_PER_CELL and len(high_vol) >= MIN_SAMPLES_PER_CELL:
                    t_stat, p_val = stats.ttest_ind(low_vol, high_vol)
                    diff = np.mean(low_vol) - np.mean(high_vol)
                    
                    sig = ''
                    if p_val < 0.01:
                        sig = '***'
                    elif p_val < 0.05:
                        sig = '**'
                    elif p_val < 0.10:
                        sig = '*'
                    
                    f.write(f"{iv:<12} {pb:<10} {np.mean(low_vol):>10.0f} {np.mean(high_vol):>10.0f} "
                           f"{diff:>10.0f} {t_stat:>8.2f} {p_val:>9.4f} {sig}\n")
                    
                    if p_val < 0.05 and abs(diff) > 100:
                        critical_tests.append({
                            'interval': iv,
                            'bucket': pb,
                            'low_mean': np.mean(low_vol),
                            'high_mean': np.mean(high_vol),
                            'diff': diff,
                            'p_val': p_val
                        })
        
        f.write("\n*** p<0.01, ** p<0.05, * p<0.10\n")
        
        # Recommendations
        f.write("\n\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        if critical_tests:
            f.write("1. SIGNIFICANT VOLUME DISCRIMINATION FOUND:\n\n")
            for test in critical_tests:
                direction = "Low volume better" if test['diff'] > 0 else "High volume better"
                f.write(f"   - {test['interval']}, {test['bucket']}: {direction} by {abs(test['diff']):.0f} bps (p={test['p_val']:.4f})\n")
            f.write("\n")
        else:
            f.write("1. NO SIGNIFICANT VOLUME DISCRIMINATION FOUND\n")
            f.write("   Volume at dip does not appear to be a reliable discriminator.\n\n")
        
        # 24h_to_12h specific recommendation
        f.write("2. 24h_to_12h INTERVAL RECOMMENDATION:\n\n")
        
        subset_90_99 = subset_24h[subset_24h['prob_bucket'] == '90_99']
        low_vol_90 = subset_90_99[subset_90_99['volume_quintile'].isin(['Q1_low', 'Q2'])]['return_bps'].values
        high_vol_90 = subset_90_99[subset_90_99['volume_quintile'].isin(['Q4', 'Q5_high'])]['return_bps'].values
        
        if len(low_vol_90) >= MIN_SAMPLES_PER_CELL and len(high_vol_90) >= MIN_SAMPLES_PER_CELL:
            t_stat, p_val = stats.ttest_ind(low_vol_90, high_vol_90)
            diff = np.mean(low_vol_90) - np.mean(high_vol_90)
            
            if p_val < 0.05 and diff > 100:
                f.write("   VOLUME DISCRIMINATION IS VIABLE for 24h_to_12h, 90_99:\n")
                f.write(f"   - Low volume dips: {np.mean(low_vol_90):.0f} bps mean edge\n")
                f.write(f"   - High volume dips: {np.mean(high_vol_90):.0f} bps mean edge\n")
                f.write(f"   - Difference: {diff:.0f} bps (p={p_val:.4f})\n")
                f.write("   - RECOMMENDATION: Filter to low-volume dips only in this cell.\n")
            else:
                f.write("   Volume discrimination NOT significant for 24h_to_12h, 90_99.\n")
                f.write("   The anomaly in this cell may have other causes (e.g., information barrier).\n")
        else:
            f.write("   Insufficient data for 24h_to_12h, 90_99 analysis.\n")
        
        # Full percentiles table
        f.write("\n\n" + "="*80 + "\n")
        f.write("DETAILED PERCENTILES BY VOLUME QUINTILE\n")
        f.write("="*80 + "\n\n")
        
        f.write("10% threshold, early tercile:\n\n")
        
        for pb in ['90_99', '75_90', '60_75']:
            f.write(f"\n{pb.upper()}\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Interval':<12} {'VolQ':>8} {'n':>6} {'P10':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8} {'Mean':>8} {'WinRate':>8}\n")
            f.write("-"*100 + "\n")
            
            subset = percentiles_df[
                (percentiles_df['threshold'] == 0.10) &
                (percentiles_df['tercile'] == 'early') &
                (percentiles_df['prob_bucket'] == pb)
            ].sort_values(['interval', 'volume_quintile'])
            
            for _, row in subset.iterrows():
                f.write(f"{row['interval']:<12} {row['volume_quintile']:>8} {row['n']:>6.0f} "
                       f"{row['p10']:>8.0f} {row['p25']:>8.0f} {row['p50']:>8.0f} "
                       f"{row['p75']:>8.0f} {row['p90']:>8.0f} {row['mean']:>8.0f} {row['win_rate']:>8.2%}\n")
    
    log(f"  Report saved: {report_path}")
    return report_path


def run_stage2_analysis(returns_dir):
    """
    Stage 2: Read returns data and generate volume stratification analysis.
    """
    log("="*70)
    log("PHASE 4D VOLUME STRATIFICATION - STAGE 2: ANALYSIS")
    log("="*70)
    
    # Load data
    log("\nLoading returns data...")
    returns_df = load_returns_data(returns_dir)
    
    if returns_df is None:
        return
    
    # Compute volume quintiles
    returns_df, quintile_thresholds = compute_volume_quintiles(returns_df)
    
    # Compute percentiles by volume
    log("\nComputing percentiles by volume quintile...")
    percentiles_df = compute_percentiles_by_volume(returns_df)
    log(f"  Computed stats for {len(percentiles_df)} cells")
    
    # Save percentiles
    percentiles_path = os.path.join(returns_dir, f'volume_percentiles_{TIMESTAMP}.csv')
    percentiles_df.to_csv(percentiles_path, index=False)
    log(f"  Percentiles saved: {percentiles_path}")
    
    # Generate plots
    plots_dir = generate_volume_plots(returns_df, returns_dir)
    
    # Generate report
    report_path = generate_volume_report(percentiles_df, returns_df, returns_dir, quintile_thresholds)
    
    log("\n" + "="*70)
    log("STAGE 2 COMPLETE")
    log("="*70)
    log(f"\nOutputs:")
    log(f"  - Percentiles CSV: {percentiles_path}")
    log(f"  - Plots directory: {plots_dir}")
    log(f"  - Report: {report_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 4D: Volume Percentile Stratification (Memory-Safe)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnostic mode - Stage 1 only (collect data)
  python phase4d_volume_stratification.py --diagnostic --sample 100 --stage1
  
  # Diagnostic mode - Stage 2 only (analyze existing data)
  python phase4d_volume_stratification.py --stage2 --returns-dir /path/to/volume_data
  
  # Diagnostic mode - Both stages
  python phase4d_volume_stratification.py --diagnostic --sample 100
  
  # Full run - Stage 1 only
  python phase4d_volume_stratification.py --stage1
  
  # Full run - Both stages
  python phase4d_volume_stratification.py
        """
    )
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files')
    parser.add_argument('--diagnostic', '-d', action='store_true',
                        help='Run in diagnostic mode with extra validation')
    parser.add_argument('--stage1', action='store_true',
                        help='Run only Stage 1 (data collection)')
    parser.add_argument('--stage2', action='store_true',
                        help='Run only Stage 2 (analysis)')
    parser.add_argument('--returns-dir', type=str, default=None,
                        help='Path to existing returns data directory (for Stage 2 only)')
    
    args = parser.parse_args()
    
    ensure_output_dir()
    
    # Determine what to run
    run_s1 = args.stage1 or (not args.stage1 and not args.stage2)
    run_s2 = args.stage2 or (not args.stage1 and not args.stage2)
    
    returns_dir = args.returns_dir
    
    if run_s1:
        log("Starting Stage 1: Data Collection with Volume...")
        returns_dir = run_stage1_collection(
            sample_files=args.sample,
            diagnostic=args.diagnostic
        )
    
    if run_s2:
        if returns_dir is None:
            log("ERROR: Stage 2 requires returns directory. Use --returns-dir or run Stage 1 first.")
            sys.exit(1)
        
        log("\nStarting Stage 2: Volume Stratification Analysis...")
        run_stage2_analysis(returns_dir)
    
    log("\n" + "="*70)
    log("ALL COMPLETE")
    log("="*70)