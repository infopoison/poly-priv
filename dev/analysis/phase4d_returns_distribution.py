#!/usr/bin/env python3
"""
Phase 4D Returns Distribution Analysis - MEMORY SAFE VERSION
Version: 1.0

OBJECTIVE:
  Generate loss/returns distribution analysis as recommended in the framing document:
  "Show the 10th, 25th, 50th, 75th, 90th percentile outcomes by cell. 
   If the losses are fat-tailed, you need stop-loss rules."

APPROACH - TWO STAGE:
  Stage 1: Stream through data and write individual returns to disk (parquet chunks)
           - Memory safe: flush to disk after each batch
           - Append-only: never overwrite existing output files
  
  Stage 2: Read parquet chunks and generate distribution plots
           - Stratified by: window, threshold, tercile, prob_bucket
           - Box plots, histograms, percentile tables

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
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, f'phase4d_returns_checkpoint_{TIMESTAMP}.pkl')

# Returns data output (timestamped to never overwrite)
RETURNS_DATA_DIR = os.path.join(OUTPUT_DIR, f'phase4d_returns_data_{TIMESTAMP}')
DIAGNOSTIC_RETURNS_DIR = os.path.join(OUTPUT_DIR, f'phase4d_returns_data_DIAGNOSTIC_{TIMESTAMP}')

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
# RETURNS WRITER - STREAMING PARQUET OUTPUT
# ==============================================================================

class ReturnsWriter:
    """
    Memory-safe streaming writer for individual returns data.
    
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
        existing = glob.glob(os.path.join(output_dir, 'returns_chunk_*.parquet'))
        if existing:
            max_idx = max(int(f.split('_')[-1].replace('.parquet', '')) for f in existing)
            self.chunk_counter = max_idx + 1
            log(f"  Found {len(existing)} existing chunks, starting at {self.chunk_counter}")
    
    def add_return(self, interval_label, threshold, tercile, prob_bucket,
                   entry_price, fill_price, is_winner, return_bps):
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
        })
        
        if len(self.buffer) >= self.chunk_size:
            self.flush()
    
    def flush(self):
        """Write buffer to parquet file."""
        if not self.buffer:
            return
        
        df = pd.DataFrame(self.buffer)
        
        # Ensure unique filename (never overwrite)
        chunk_path = os.path.join(self.output_dir, f'returns_chunk_{self.chunk_counter:06d}.parquet')
        while os.path.exists(chunk_path):
            self.chunk_counter += 1
            chunk_path = os.path.join(self.output_dir, f'returns_chunk_{self.chunk_counter:06d}.parquet')
        
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
# TOKEN ACCUMULATOR
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
    
    def compute_returns(self, returns_writer):
        """
        Compute returns for all cells and write to returns writer.
        
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
                
                # Simulate fill
                fill_result = simulate_limit_order_fill(
                    self.trades, crossing['crossing_time'], crossing['crossing_price'], is_buy=True
                )
                
                if not fill_result.get('filled'):
                    continue
                
                fill_price = fill_result['fill_price']
                
                # Compute return (in basis points)
                # Return = (outcome - entry_price) * 10000
                # If winner: outcome = 1.0, so return = (1 - fill_price) * 10000
                # If loser: outcome = 0.0, so return = (0 - fill_price) * 10000 = -fill_price * 10000
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
                    )
                    n_returns += 1
        
        return n_returns


# ==============================================================================
# STAGE 1: DATA COLLECTION
# ==============================================================================

def run_stage1_collection(sample_files=None, diagnostic=False):
    """
    Stage 1: Stream through data and collect individual returns to disk.
    
    Memory safe: writes to parquet chunks incrementally.
    """
    global start_time
    start_time = datetime.now()
    
    log("="*70)
    log("PHASE 4D RETURNS DISTRIBUTION - STAGE 1: DATA COLLECTION")
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
    # SETUP RETURNS WRITER
    # -------------------------------------------------------------------------
    
    returns_dir = ensure_returns_dir(diagnostic)
    log(f"\nReturns data will be written to: {returns_dir}")
    
    returns_writer = ReturnsWriter(returns_dir, chunk_size=10000)
    
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
                    
                    token_accumulators[token_id] = TimingAccumulator(
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
                            n_returns = acc.compute_returns(returns_writer)
                            
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
                n_returns = acc.compute_returns(returns_writer)
                
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
    log(f"\nReturns data:")
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
    }
    
    metadata_path = os.path.join(returns_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    log(f"\nMetadata saved: {metadata_path}")
    
    return returns_dir


# ==============================================================================
# STAGE 2: ANALYSIS AND PLOTTING
# ==============================================================================

def load_returns_data(returns_dir):
    """Load all returns data from parquet chunks."""
    chunk_files = sorted(glob.glob(os.path.join(returns_dir, 'returns_chunk_*.parquet')))
    
    if not chunk_files:
        log(f"ERROR: No chunk files found in {returns_dir}")
        return None
    
    log(f"  Loading {len(chunk_files)} chunk files...")
    
    dfs = []
    for chunk_file in chunk_files:
        df = pd.read_parquet(chunk_file)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    log(f"  Loaded {len(combined):,} returns")
    
    return combined


def compute_percentiles(returns_df):
    """
    Compute percentile statistics by cell.
    
    Returns DataFrame with columns:
      interval, threshold, tercile, prob_bucket, 
      n, p10, p25, p50, p75, p90, mean, std, min, max
    """
    results = []
    
    group_cols = ['interval', 'threshold', 'tercile', 'prob_bucket']
    
    for keys, group in returns_df.groupby(group_cols):
        interval, threshold, tercile, prob_bucket = keys
        returns = group['return_bps'].values
        
        if len(returns) < MIN_SAMPLES_PER_CELL:
            continue
        
        results.append({
            'interval': interval,
            'threshold': threshold,
            'tercile': tercile,
            'prob_bucket': prob_bucket,
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
        })
    
    return pd.DataFrame(results)


def generate_distribution_plots(returns_df, output_dir):
    """
    Generate distribution plots stratified by different dimensions.
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    log("\nGenerating distribution plots...")
    
    # -------------------------------------------------------------------------
    # 1. Box plots by interval (for 10% threshold, early tercile)
    # -------------------------------------------------------------------------
    
    log("  1. Box plots by interval...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, prob_bucket in enumerate(PROB_BUCKET_LABELS[:6]):
        subset = returns_df[
            (returns_df['threshold'] == 0.10) & 
            (returns_df['tercile'] == 'early') &
            (returns_df['prob_bucket'] == prob_bucket)
        ]
        
        if len(subset) == 0:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(f'{prob_bucket}')
            continue
        
        intervals = [x[2] for x in INTERVAL_PAIRS]
        data_by_interval = [subset[subset['interval'] == iv]['return_bps'].values 
                           for iv in intervals]
        
        # Filter out empty arrays
        valid_data = [(d, iv) for d, iv in zip(data_by_interval, intervals) if len(d) > 0]
        
        if valid_data:
            axes[idx].boxplot([d[0] for d in valid_data], labels=[d[1] for d in valid_data])
            axes[idx].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        axes[idx].set_title(f'{prob_bucket}')
        axes[idx].set_ylabel('Return (bps)')
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Returns Distribution by Interval\n(10% threshold, early tercile)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'boxplot_by_interval.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 2. Box plots by threshold (for 8h_to_4h interval, early tercile)
    # -------------------------------------------------------------------------
    
    log("  2. Box plots by threshold...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, prob_bucket in enumerate(PROB_BUCKET_LABELS[:6]):
        subset = returns_df[
            (returns_df['interval'] == '8h_to_4h') & 
            (returns_df['tercile'] == 'early') &
            (returns_df['prob_bucket'] == prob_bucket)
        ]
        
        if len(subset) == 0:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(f'{prob_bucket}')
            continue
        
        data_by_threshold = [subset[subset['threshold'] == th]['return_bps'].values 
                            for th in MOVE_THRESHOLDS]
        
        valid_data = [(d, th) for d, th in zip(data_by_threshold, MOVE_THRESHOLDS) if len(d) > 0]
        
        if valid_data:
            axes[idx].boxplot([d[0] for d in valid_data], 
                             labels=[f'{int(d[1]*100)}%' for d in valid_data])
            axes[idx].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        axes[idx].set_title(f'{prob_bucket}')
        axes[idx].set_ylabel('Return (bps)')
    
    plt.suptitle('Returns Distribution by Threshold\n(8h_to_4h interval, early tercile)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'boxplot_by_threshold.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 3. Box plots by tercile (for 8h_to_4h interval, 10% threshold)
    # -------------------------------------------------------------------------
    
    log("  3. Box plots by tercile...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, prob_bucket in enumerate(PROB_BUCKET_LABELS[:6]):
        subset = returns_df[
            (returns_df['interval'] == '8h_to_4h') & 
            (returns_df['threshold'] == 0.10) &
            (returns_df['prob_bucket'] == prob_bucket)
        ]
        
        if len(subset) == 0:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(f'{prob_bucket}')
            continue
        
        data_by_tercile = [subset[subset['tercile'] == t]['return_bps'].values 
                          for t in TERCILE_LABELS]
        
        valid_data = [(d, t) for d, t in zip(data_by_tercile, TERCILE_LABELS) if len(d) > 0]
        
        if valid_data:
            axes[idx].boxplot([d[0] for d in valid_data], labels=[d[1] for d in valid_data])
            axes[idx].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        axes[idx].set_title(f'{prob_bucket}')
        axes[idx].set_ylabel('Return (bps)')
    
    plt.suptitle('Returns Distribution by Tercile\n(8h_to_4h interval, 10% threshold)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'boxplot_by_tercile.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 4. CRITICAL: 24h_to_12h Anomaly Deep Dive
    # -------------------------------------------------------------------------
    
    log("  4. 24h_to_12h anomaly analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 4a. 24h_to_12h returns by prob bucket (10% threshold, early tercile)
    subset = returns_df[
        (returns_df['interval'] == '24h_to_12h') & 
        (returns_df['threshold'] == 0.10) &
        (returns_df['tercile'] == 'early')
    ]
    
    data_by_bucket = [subset[subset['prob_bucket'] == pb]['return_bps'].values 
                     for pb in PROB_BUCKET_LABELS[:6]]
    valid_data = [(d, pb) for d, pb in zip(data_by_bucket, PROB_BUCKET_LABELS[:6]) if len(d) > 0]
    
    if valid_data:
        axes[0,0].boxplot([d[0] for d in valid_data], labels=[d[1] for d in valid_data])
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0,0].set_title('24h_to_12h: Returns by Prob Bucket\n(10% threshold, early tercile)')
    axes[0,0].set_ylabel('Return (bps)')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 4b. 90_99 bucket: 24h_to_12h vs other intervals
    subset_90_99 = returns_df[
        (returns_df['threshold'] == 0.10) & 
        (returns_df['tercile'] == 'early') &
        (returns_df['prob_bucket'] == '90_99')
    ]
    
    intervals = [x[2] for x in INTERVAL_PAIRS]
    data_by_interval = [subset_90_99[subset_90_99['interval'] == iv]['return_bps'].values 
                       for iv in intervals]
    valid_data = [(d, iv) for d, iv in zip(data_by_interval, intervals) if len(d) > 0]
    
    if valid_data:
        colors = ['red' if d[1] == '24h_to_12h' else 'blue' for d in valid_data]
        bp = axes[0,1].boxplot([d[0] for d in valid_data], labels=[d[1] for d in valid_data], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color if color == 'red' else 'lightblue')
            patch.set_alpha(0.5)
        axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0,1].set_title('90_99 Bucket: Returns by Interval\n(10% threshold, early tercile)')
    axes[0,1].set_ylabel('Return (bps)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 4c. Histogram of 24h_to_12h losses for 90_99
    subset_anomaly = returns_df[
        (returns_df['interval'] == '24h_to_12h') & 
        (returns_df['threshold'] == 0.10) &
        (returns_df['tercile'] == 'early') &
        (returns_df['prob_bucket'] == '90_99')
    ]
    
    if len(subset_anomaly) > 0:
        returns = subset_anomaly['return_bps'].values
        axes[1,0].hist(returns, bins=30, edgecolor='black', alpha=0.7)
        axes[1,0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1,0].axvline(x=np.mean(returns), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(returns):.0f}')
        axes[1,0].axvline(x=np.median(returns), color='orange', linestyle='-', linewidth=2, label=f'Median: {np.median(returns):.0f}')
        axes[1,0].legend()
    axes[1,0].set_title('24h_to_12h, 90_99, Early Tercile\nReturns Distribution')
    axes[1,0].set_xlabel('Return (bps)')
    axes[1,0].set_ylabel('Frequency')
    
    # 4d. Cumulative distribution
    if len(subset_anomaly) > 0:
        returns_sorted = np.sort(returns)
        cdf = np.arange(1, len(returns_sorted)+1) / len(returns_sorted)
        axes[1,1].plot(returns_sorted, cdf, linewidth=2)
        axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1,1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Mark percentiles
        for p, color in [(10, 'blue'), (25, 'green'), (50, 'orange'), (75, 'purple'), (90, 'red')]:
            pval = np.percentile(returns, p)
            axes[1,1].scatter([pval], [p/100], s=100, color=color, zorder=5)
            axes[1,1].annotate(f'P{p}: {pval:.0f}', (pval, p/100), textcoords='offset points', 
                              xytext=(5, 5), fontsize=9)
    axes[1,1].set_title('24h_to_12h, 90_99, Early Tercile\nCumulative Distribution')
    axes[1,1].set_xlabel('Return (bps)')
    axes[1,1].set_ylabel('Cumulative Probability')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'anomaly_24h_to_12h_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # -------------------------------------------------------------------------
    # 5. Summary heatmap of median returns
    # -------------------------------------------------------------------------
    
    log("  5. Summary heatmaps...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, prob_bucket in enumerate(PROB_BUCKET_LABELS[:6]):
        subset = returns_df[
            (returns_df['tercile'] == 'early') &
            (returns_df['prob_bucket'] == prob_bucket)
        ]
        
        # Create pivot table
        pivot_data = []
        for interval in [x[2] for x in INTERVAL_PAIRS]:
            row = []
            for threshold in MOVE_THRESHOLDS:
                cell = subset[
                    (subset['interval'] == interval) &
                    (subset['threshold'] == threshold)
                ]['return_bps']
                
                if len(cell) >= MIN_SAMPLES_PER_CELL:
                    row.append(np.median(cell))
                else:
                    row.append(np.nan)
            pivot_data.append(row)
        
        pivot_df = pd.DataFrame(
            pivot_data,
            index=[x[2] for x in INTERVAL_PAIRS],
            columns=[f'{int(t*100)}%' for t in MOVE_THRESHOLDS]
        )
        
        # Plot heatmap
        im = axes[idx].imshow(pivot_df.values, cmap='RdYlGn', aspect='auto',
                              vmin=-500, vmax=500)
        
        axes[idx].set_xticks(range(len(MOVE_THRESHOLDS)))
        axes[idx].set_xticklabels([f'{int(t*100)}%' for t in MOVE_THRESHOLDS])
        axes[idx].set_yticks(range(len(INTERVAL_PAIRS)))
        axes[idx].set_yticklabels([x[2] for x in INTERVAL_PAIRS])
        
        # Add text annotations
        for i in range(len(INTERVAL_PAIRS)):
            for j in range(len(MOVE_THRESHOLDS)):
                val = pivot_df.values[i, j]
                if not np.isnan(val):
                    axes[idx].text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=9)
        
        axes[idx].set_title(f'{prob_bucket}')
        axes[idx].set_xlabel('Threshold')
        axes[idx].set_ylabel('Interval')
    
    plt.suptitle('Median Returns (bps) by Cell\n(Early Tercile)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'heatmap_median_returns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"  Plots saved to: {plots_dir}")
    return plots_dir


def generate_percentile_report(percentiles_df, output_dir):
    """Generate text report with percentile tables."""
    
    report_path = os.path.join(output_dir, f'returns_distribution_report_{TIMESTAMP}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 4D: RETURNS DISTRIBUTION ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("LOSS DISTRIBUTION ANALYSIS\n")
        f.write("-"*80 + "\n\n")
        
        f.write("This report shows the 10th, 25th, 50th, 75th, 90th percentile outcomes by cell.\n")
        f.write("If losses are fat-tailed (large negative P10), stop-loss rules may be needed.\n\n")
        
        # Table by interval for 10% threshold, early tercile
        f.write("="*80 + "\n")
        f.write("PERCENTILES BY INTERVAL (10% threshold, early tercile)\n")
        f.write("="*80 + "\n\n")
        
        for prob_bucket in PROB_BUCKET_LABELS[:6]:
            subset = percentiles_df[
                (percentiles_df['threshold'] == 0.10) & 
                (percentiles_df['tercile'] == 'early') &
                (percentiles_df['prob_bucket'] == prob_bucket)
            ].sort_values('interval')
            
            if len(subset) == 0:
                continue
            
            f.write(f"\n{prob_bucket.upper()}\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Interval':<12} {'n':>6} {'P10':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8} {'Mean':>8}\n")
            f.write("-"*70 + "\n")
            
            for _, row in subset.iterrows():
                f.write(f"{row['interval']:<12} {row['n']:>6.0f} {row['p10']:>8.0f} {row['p25']:>8.0f} "
                       f"{row['p50']:>8.0f} {row['p75']:>8.0f} {row['p90']:>8.0f} {row['mean']:>8.0f}\n")
        
        # CRITICAL: 24h_to_12h Anomaly Table
        f.write("\n\n" + "="*80 + "\n")
        f.write("CRITICAL: 24h_to_12h INTERVAL ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("This interval shows anomalous negative edge in 90_99 bucket.\n")
        f.write("The framing document recommends stratifying and analyzing this specifically.\n\n")
        
        subset_24h = percentiles_df[percentiles_df['interval'] == '24h_to_12h']
        
        f.write(f"{'Threshold':<10} {'Tercile':<8} {'Bucket':<10} {'n':>6} {'P10':>8} {'P25':>8} {'P50':>8} {'Mean':>8} {'StdDev':>8}\n")
        f.write("-"*80 + "\n")
        
        for _, row in subset_24h.sort_values(['threshold', 'tercile', 'prob_bucket']).iterrows():
            f.write(f"{row['threshold']:<10.2f} {row['tercile']:<8} {row['prob_bucket']:<10} "
                   f"{row['n']:>6.0f} {row['p10']:>8.0f} {row['p25']:>8.0f} {row['p50']:>8.0f} "
                   f"{row['mean']:>8.0f} {row['std']:>8.0f}\n")
        
        # Fat tails identification
        f.write("\n\n" + "="*80 + "\n")
        f.write("FAT TAIL IDENTIFICATION\n")
        f.write("="*80 + "\n\n")
        
        f.write("Cells where P10 < -5000 bps (potential fat-tailed losses):\n\n")
        
        fat_tail = percentiles_df[percentiles_df['p10'] < -5000].sort_values('p10')
        
        if len(fat_tail) > 0:
            f.write(f"{'Interval':<12} {'Thresh':<8} {'Tercile':<8} {'Bucket':<10} {'n':>6} {'P10':>8} {'Mean':>8}\n")
            f.write("-"*70 + "\n")
            
            for _, row in fat_tail.iterrows():
                f.write(f"{row['interval']:<12} {row['threshold']:<8.2f} {row['tercile']:<8} "
                       f"{row['prob_bucket']:<10} {row['n']:>6.0f} {row['p10']:>8.0f} {row['mean']:>8.0f}\n")
        else:
            f.write("No cells with P10 < -5000 bps identified.\n")
        
        # Recommendations
        f.write("\n\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n\n")
        
        # Check for negative mean cells
        neg_mean = percentiles_df[percentiles_df['mean'] < -100].sort_values('mean')
        if len(neg_mean) > 0:
            f.write("1. AVOID: Cells with negative mean edge:\n")
            for _, row in neg_mean.head(10).iterrows():
                f.write(f"   - {row['interval']}, {row['threshold']:.0%}, {row['tercile']}, "
                       f"{row['prob_bucket']}: {row['mean']:.0f} bps\n")
        
        f.write("\n2. STOP-LOSS CONSIDERATION:\n")
        high_spread = percentiles_df[(percentiles_df['p90'] - percentiles_df['p10']) > 8000]
        if len(high_spread) > 0:
            f.write(f"   {len(high_spread)} cells have P90-P10 spread > 8000 bps\n")
            f.write("   These may benefit from stop-loss rules.\n")
        
    log(f"  Report saved: {report_path}")
    return report_path


def run_stage2_analysis(returns_dir):
    """
    Stage 2: Read returns data and generate analysis.
    """
    log("="*70)
    log("PHASE 4D RETURNS DISTRIBUTION - STAGE 2: ANALYSIS")
    log("="*70)
    
    # Load data
    log("\nLoading returns data...")
    returns_df = load_returns_data(returns_dir)
    
    if returns_df is None:
        return
    
    # Compute percentiles
    log("\nComputing percentiles...")
    percentiles_df = compute_percentiles(returns_df)
    log(f"  Computed stats for {len(percentiles_df)} cells")
    
    # Save percentiles
    percentiles_path = os.path.join(returns_dir, f'percentiles_{TIMESTAMP}.csv')
    percentiles_df.to_csv(percentiles_path, index=False)
    log(f"  Percentiles saved: {percentiles_path}")
    
    # Generate plots
    plots_dir = generate_distribution_plots(returns_df, returns_dir)
    
    # Generate report
    report_path = generate_percentile_report(percentiles_df, returns_dir)
    
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
        description='Phase 4D: Returns Distribution Analysis (Memory-Safe)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnostic mode - Stage 1 only (collect data)
  python phase4d_returns_distribution.py --diagnostic --sample 100 --stage1
  
  # Diagnostic mode - Stage 2 only (analyze existing data)
  python phase4d_returns_distribution.py --stage2 --returns-dir /path/to/returns_data
  
  # Diagnostic mode - Both stages
  python phase4d_returns_distribution.py --diagnostic --sample 100
  
  # Full run - Stage 1 only
  python phase4d_returns_distribution.py --stage1
  
  # Full run - Both stages
  python phase4d_returns_distribution.py
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
        log("Starting Stage 1: Data Collection...")
        returns_dir = run_stage1_collection(
            sample_files=args.sample,
            diagnostic=args.diagnostic
        )
    
    if run_s2:
        if returns_dir is None:
            log("ERROR: Stage 2 requires returns directory. Use --returns-dir or run Stage 1 first.")
            sys.exit(1)
        
        log("\nStarting Stage 2: Analysis...")
        run_stage2_analysis(returns_dir)
    
    log("\n" + "="*70)
    log("ALL COMPLETE")
    log("="*70)