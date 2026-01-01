#!/usr/bin/env python3
"""
Phase 2: Price Dynamics Analysis - FULL PRODUCTION VERSION
Version: 1.0

OBJECTIVE:
  Characterize the unconditional dynamics of price changes in Polymarket.
  This is Step 1 from the research agenda: understand whether there's
  mean reversion or momentum in price movements before resolution.

FEATURES ANALYZED (price-conditional only):
  1. Price snapshots at multiple horizons: T-720h to T-1h (30 days to 1 hour)
  2. ΔP (price changes) between adjacent horizons
  3. Autocorrelation of price changes at various lags (1h, 6h, 12h, 24h, 48h, 168h)
  4. Conditional distribution of outcomes given:
     - Past price levels (snapshot buckets, aligned with Phase 1)
     - Recent price changes (ΔP buckets)
  5. Entropy-stratified calibration analysis

INTERPRETATION FRAMEWORK:
  - Negative autocorrelation → Mean reversion (non-resolving moves dominate)
  - Positive autocorrelation → Momentum (information arrival is gradual)
  - If P(win | ΔP > 0) > E[P | post-move], moves are typically underreactions
  - If P(win | ΔP > 0) < E[P | post-move], moves are typically overreactions

PHASE 1 CONNECTION:
  - Uses same price buckets as Phase 1 calibration for direct comparison
  - Adds entropy analysis to test Phase 1 finding of high-entropy mispricing
  - Investigates whether price dynamics explain calibration patterns

ARCHITECTURE:
  - Reuses sidecar-based winner lookup from Phase 1
  - File-first accumulator pattern with streaming aggregation
  - Memory efficient: O(active_conditions)
  - Checkpointed for resumability
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

# Suppress specific warnings
warnings.filterwarnings('ignore', message='.*constant.*correlation.*')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = "../../"  # Adjust based on execution location

BATCH_DIR = os.path.join(BASE_DIR, 'order_history_batches')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis/outputs')
INDEX_CACHE_FILE = os.path.join(BASE_DIR, 'analysis/market_index_cache_full.pkl')
SIDECAR_FILE = os.path.join(BASE_DIR, 'data/repair/api_derived_winners.parquet')

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Checkpoint file for resumability
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, 'phase2_checkpoint.pkl')

# ------------------------------------------------------------------------------
# ANALYSIS PARAMETERS
# ------------------------------------------------------------------------------

# Time horizons for price snapshots (hours before resolution)
# Extended to capture longer-term market dynamics (many markets open for weeks/months)
SNAPSHOT_HORIZONS = [
    720,   # 30 days - long-term baseline
    336,   # 14 days - medium-term
    168,   # 7 days  - weekly
    72,    # 3 days  - short-term
    48,    # 2 days
    24,    # 1 day   - key reference (matches Phase 1 calibration window)
    12,    # 12 hours
    6,     # 6 hours
    1,     # 1 hour  - near resolution
]

# Lags for autocorrelation analysis (computed from price changes between horizons)
# Extended to capture longer-term reversion/momentum patterns
AUTOCORR_LAGS = [1, 6, 12, 24, 48, 168]  # hours (up to 1 week)

# Price change buckets for conditional analysis
# Refined to capture smaller moves that might be more common at longer horizons
DELTA_P_BUCKETS = [
    (-1.0, -0.20, 'large_drop'),
    (-0.20, -0.10, 'moderate_drop'),
    (-0.10, -0.05, 'small_drop'),
    (-0.05, -0.02, 'minor_drop'),
    (-0.02, 0.02, 'stable'),
    (0.02, 0.05, 'minor_rise'),
    (0.05, 0.10, 'small_rise'),
    (0.10, 0.20, 'moderate_rise'),
    (0.20, 1.0, 'large_rise'),
]

# Price level buckets for conditional analysis
# Aligned with Phase 1 calibration buckets for interpretive consistency
PRICE_BUCKETS = [
    (0.00, 0.10, 'longshot'),       # Phase 1: 'longshot'
    (0.10, 0.25, 'underdog'),       # Phase 1: 'underdog'
    (0.25, 0.40, 'toss-up-'),       # Phase 1: 'toss-up-'
    (0.40, 0.51, 'toss-up+'),       # Phase 1: 'toss-up+'
    (0.51, 0.60, 'mild-fav'),       # Phase 1: 'mild-fav'
    (0.60, 0.75, 'moderate-fav'),   # Phase 1: 'moderate-fav'
    (0.75, 0.90, 'strong-fav'),     # Phase 1: 'strong-fav'
    (0.90, 0.99, 'heavy-fav'),      # Phase 1: 'heavy-fav'
    (0.99, 1.00, 'near-certain'),   # Phase 1: 'near-certain' + 'extreme'
]

# Minimum requirements for inclusion - RELAXED for flexibility
# We'll track data coverage and analyze what we have rather than hard filtering
MIN_TRADES_IN_WINDOW = 5   # Reduced - we'll report coverage statistics
MIN_HORIZON_HOURS = 24     # Minimum 24h of data (but we'll analyze all available)

# Column requirements
REQUIRED_COLUMNS = ['timestamp', 'price', 'token_id', 'condition_id', 'resolution_time']
VOLUME_COLUMNS = ['size_tokens', 'maker_amount', 'size']

# Progress reporting interval
PROGRESS_INTERVAL = 1000  # Report every N files

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

def assign_bucket(value, buckets):
    """Assign value to a bucket"""
    if value is None:
        return None
    for lower, upper, label in buckets:
        if lower <= value < upper:
            return label
    # Edge cases
    if value >= buckets[-1][1]:
        return buckets[-1][2]
    if value < buckets[0][0]:
        return buckets[0][2]
    return None

def format_duration(seconds):
    """Format seconds into human-readable duration"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# ==============================================================================
# SIDECAR LOADING (from Phase 1)
# ==============================================================================

def load_winner_sidecar(sidecar_path):
    """
    Load pre-computed winner data from sidecar file.
    Returns dict: token_id -> bool (is_winner)
    """
    log(f"Loading winner sidecar from {sidecar_path}...")
    
    if not os.path.exists(sidecar_path):
        log(f"  ERROR: Sidecar file not found: {sidecar_path}")
        return None
    
    try:
        df = pd.read_parquet(sidecar_path)
        log(f"  Raw sidecar: {len(df):,} records")
        
        # Filter to SUCCESS records only
        success_df = df[df['repair_status'] == 'SUCCESS']
        log(f"  SUCCESS records: {len(success_df):,}")
        
        # Build lookup dict
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
        
        return winner_lookup
        
    except Exception as e:
        log(f"  ERROR loading sidecar: {e}")
        return None

# ==============================================================================
# PRICE DYNAMICS EXTRACTION
# ==============================================================================

def extract_price_at_horizon(trades, resolution_time, hours_before):
    """
    Extract the price closest to a specific time horizon before resolution.
    
    Args:
        trades: list of (timestamp, price, size) tuples, sorted by timestamp
        resolution_time: Unix timestamp of resolution
        hours_before: hours before resolution to sample
    
    Returns:
        float or None: price at that horizon, or None if no trade within tolerance
    """
    target_time = resolution_time - (hours_before * 3600)
    
    # Find trade closest to target time
    # Accept trades within a window around the target
    tolerance_hours = max(1, hours_before * 0.25)  # 25% tolerance or 1 hour minimum
    tolerance_seconds = tolerance_hours * 3600
    
    best_trade = None
    best_distance = float('inf')
    
    for ts, price, size in trades:
        distance = abs(ts - target_time)
        if distance < best_distance and distance < tolerance_seconds:
            best_distance = distance
            best_trade = (ts, price, size)
    
    if best_trade:
        return best_trade[1]  # Return price
    return None

def compute_hourly_price_series(trades, resolution_time, hours_back=720):
    """
    Compute hourly price series for autocorrelation analysis.
    Extended to support longer horizons (up to 30 days).
    
    Returns:
        dict: {hour: price} where hour is hours before resolution
    """
    hourly_prices = {}
    
    for hour in range(hours_back, 0, -1):
        price = extract_price_at_horizon(trades, resolution_time, hour)
        if price is not None:
            hourly_prices[hour] = price
    
    return hourly_prices

def compute_price_changes(hourly_prices):
    """
    Compute price changes between consecutive hours.
    
    Returns:
        dict: {hour: delta_p} where delta_p is price[hour] - price[hour+1]
              (positive = price increased as we approach resolution)
    """
    hours = sorted(hourly_prices.keys(), reverse=True)  # From T-N to T-1
    changes = {}
    
    for i in range(len(hours) - 1):
        hour_earlier = hours[i]
        hour_later = hours[i + 1]
        
        delta_p = hourly_prices[hour_later] - hourly_prices[hour_earlier]
        changes[hour_later] = delta_p
    
    return changes

def compute_autocorrelation(price_changes, lag):
    """
    Compute autocorrelation of price changes at a given lag.
    
    Args:
        price_changes: dict {hour: delta_p}
        lag: int, lag in hours
    
    Returns:
        float or None: correlation coefficient, or None if insufficient data
    """
    hours = sorted(price_changes.keys())
    
    pairs = []
    for h in hours:
        h_lagged = h + lag
        if h_lagged in price_changes:
            pairs.append((price_changes[h], price_changes[h_lagged]))
    
    if len(pairs) < 5:  # Need at least 5 pairs for meaningful correlation
        return None
    
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    
    # Check for constant arrays (zero variance) - can't compute correlation
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    
    # Pearson correlation with warning suppression
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = stats.pearsonr(x, y)
        return corr if np.isfinite(corr) else None
    except:
        return None

# ==============================================================================
# DATA ACCUMULATOR CLASS
# ==============================================================================

class PriceDynamicsAccumulator:
    """
    Accumulates price dynamics data for a single token.
    """
    def __init__(self, token_id, condition_id, resolution_time, winner_status):
        self.token_id = token_id
        self.condition_id = condition_id
        self.resolution_time = resolution_time
        self.winner_status = winner_status
        self.trades = []  # List of (timestamp, price, size)
    
    def add_trades(self, trades_batch):
        self.trades.extend(trades_batch)
    
    def compute_dynamics(self):
        """
        Compute all price dynamics features for this token.
        
        Returns:
            dict with all computed features, or None if insufficient data
        """
        if not self.trades or len(self.trades) < MIN_TRADES_IN_WINDOW:
            return None
        
        # Sort trades by timestamp
        self.trades.sort(key=lambda x: x[0])
        
        # Check data coverage
        earliest_trade = self.trades[0][0]
        latest_trade = self.trades[-1][0]
        hours_of_data = (self.resolution_time - earliest_trade) / 3600
        
        if hours_of_data < MIN_HORIZON_HOURS:
            return None
        
        # Extract price snapshots at key horizons
        # Track which horizons we have data for (coverage)
        snapshots = {}
        horizons_available = []
        for horizon in SNAPSHOT_HORIZONS:
            price = extract_price_at_horizon(self.trades, self.resolution_time, horizon)
            snapshots[f'P_T-{horizon}h'] = price
            if price is not None:
                horizons_available.append(horizon)
        
        # Compute ΔP between adjacent AVAILABLE horizons
        delta_ps = {}
        horizons_sorted = sorted(horizons_available, reverse=True)  # From oldest to newest
        for i in range(len(horizons_sorted) - 1):
            h_earlier = horizons_sorted[i]
            h_later = horizons_sorted[i + 1]
            
            p_earlier = snapshots.get(f'P_T-{h_earlier}h')
            p_later = snapshots.get(f'P_T-{h_later}h')
            
            if p_earlier is not None and p_later is not None:
                delta_ps[f'dP_T-{h_earlier}h_to_T-{h_later}h'] = p_later - p_earlier
        
        # Compute hourly price series for autocorrelation
        # Use a sparser sampling for longer horizons to avoid excessive computation
        max_hours = min(int(hours_of_data), 720)  # Cap at 30 days
        hourly_prices = compute_hourly_price_series(self.trades, self.resolution_time, max_hours)
        price_changes = compute_price_changes(hourly_prices)
        
        autocorrs = {}
        for lag in AUTOCORR_LAGS:
            ac = compute_autocorrelation(price_changes, lag)
            autocorrs[f'autocorr_lag_{lag}h'] = ac
        
        # Determine bucket for the T-24h price (primary reference)
        p_24h = snapshots.get('P_T-24h')
        price_bucket = assign_bucket(p_24h, PRICE_BUCKETS) if p_24h else None
        
        # Compute binary entropy at T-24h price
        # H(p) = -p*log2(p) - (1-p)*log2(1-p)
        # Higher entropy = more uncertainty = potentially more mispricing (per Phase 1 findings)
        entropy_24h = None
        if p_24h is not None and 0 < p_24h < 1:
            entropy_24h = -p_24h * np.log2(p_24h) - (1 - p_24h) * np.log2(1 - p_24h)
        
        # Determine ΔP bucket for key horizon pairs
        # Primary: T-48h to T-24h (if available)
        # Secondary: T-168h to T-24h (weekly move into final day)
        dp_48_24 = delta_ps.get('dP_T-48h_to_T-24h')
        delta_bucket_48_24 = assign_bucket(dp_48_24, DELTA_P_BUCKETS) if dp_48_24 else None
        
        dp_168_24 = delta_ps.get('dP_T-168h_to_T-24h')
        delta_bucket_168_24 = assign_bucket(dp_168_24, DELTA_P_BUCKETS) if dp_168_24 else None
        
        # Compute total price move from earliest available to T-24h
        total_move = None
        if horizons_available and p_24h is not None:
            earliest_horizon = max(horizons_available)  # Largest = earliest in time
            p_earliest = snapshots.get(f'P_T-{earliest_horizon}h')
            if p_earliest is not None:
                total_move = p_24h - p_earliest
        
        # Return compact result (exclude large intermediate data)
        return {
            'token_id': self.token_id,
            'condition_id': self.condition_id,
            'winner': self.winner_status,
            'trade_count': len(self.trades),
            'hours_of_data': hours_of_data,
            'horizons_available': horizons_available,
            'coverage_pct': len(horizons_available) / len(SNAPSHOT_HORIZONS) * 100,
            'snapshots': snapshots,
            'delta_ps': delta_ps,
            'autocorrs': autocorrs,
            'price_bucket_24h': price_bucket,
            'entropy_24h': entropy_24h,
            'delta_bucket_48_24': delta_bucket_48_24,
            'delta_bucket_168_24': delta_bucket_168_24,
            'total_move': total_move,
        }

# ==============================================================================
# MAIN PROCESSING
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
    
    return columns_to_read, volume_col

def load_market_index(cache_file, batch_files):
    """Load market index from cache if valid"""
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

def save_checkpoint(results, files_processed, total_files):
    """Save checkpoint for resumability"""
    checkpoint = {
        'results': results,
        'files_processed': files_processed,
        'total_files': total_files,
        'timestamp': datetime.now().isoformat(),
    }
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  Checkpoint saved: {files_processed}/{total_files} files")

def load_checkpoint():
    """Load checkpoint if exists"""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def run_full_analysis(sample_files=None, resume=False):
    """
    Run full price dynamics analysis on all data.
    Uses STREAMING FLUSH pattern from Phase 1 for memory efficiency.
    
    Args:
        sample_files: int or None, limit number of files for testing
        resume: bool, attempt to resume from checkpoint
    """
    start_time = datetime.now()
    
    log("="*70)
    log("PHASE 2: PRICE DYNAMICS ANALYSIS - FULL RUN")
    log("="*70)
    log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")
    
    # Load batch files
    batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, '*.parquet')))
    if not batch_files:
        log(f"ERROR: No parquet files found in {BATCH_DIR}")
        return None
    
    log(f"Found {len(batch_files):,} batch files")
    
    # Load winner sidecar
    winner_lookup = load_winner_sidecar(SIDECAR_FILE)
    if not winner_lookup:
        log("ERROR: Failed to load winner sidecar")
        return None
    
    # ==========================================================================
    # LOAD MARKET INDEX FROM CACHE (same cache as Phase 1)
    # ==========================================================================
    
    log("\nLoading market index from cache...")
    market_index = load_market_index(INDEX_CACHE_FILE, batch_files)
    
    if market_index is None:
        log("ERROR: Market index cache not found or invalid.")
        log("       Run Phase 1 first to build the cache, or ensure cache file exists.")
        log(f"       Expected: {INDEX_CACHE_FILE}")
        return None
    
    log(f"  Loaded market index: {len(market_index):,} conditions")
    
    # ==========================================================================
    # CONDITION COMPLETION TRACKING (from Phase 1 pattern)
    # ==========================================================================
    
    log("\nSetting up streaming flush tracking...")
    
    # Determine files to process
    if sample_files:
        files_to_process_indices = list(range(min(sample_files, len(batch_files))))
        log(f"  SAMPLE MODE: Processing {len(files_to_process_indices)} files")
    else:
        files_to_process_indices = list(range(len(batch_files)))
    
    unique_file_set = set(files_to_process_indices)
    
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
    
    single_file_conditions = sum(1 for c in condition_remaining_files.values() if c == 1)
    log(f"  Single-file conditions (flush immediately): {single_file_conditions:,}")
    
    # ==========================================================================
    # ACCUMULATORS
    # ==========================================================================
    
    # Check for checkpoint
    start_file_idx = 0
    results = []  # Stores computed dynamics (compact)
    
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            results = checkpoint['results']
            start_file_idx = checkpoint['files_processed']
            log(f"Resuming from checkpoint: {start_file_idx}/{checkpoint['total_files']} files")
            log(f"  Existing results: {len(results)} tokens")
    
    # Token accumulator (will be flushed as conditions complete)
    token_accumulators = {}  # token_id -> PriceDynamicsAccumulator
    
    # Track which tokens belong to which condition (for flushing)
    condition_tokens = defaultdict(set)
    
    # Statistics
    stats = {
        'files_processed': start_file_idx,
        'total_rows': 0,
        'tokens_no_winner': 0,
        'tokens_filtered': 0,
        'conditions_flushed': 0,
    }
    
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
                f"Results: {len(results):,} | "
                f"Rate: {rate:.1f}/s | "
                f"ETA: {format_duration(eta)} | "
                f"{log_memory()}")
            
            # Save checkpoint periodically
            if stats['files_processed'] % (PROGRESS_INTERVAL * 10) == 0:
                save_checkpoint(results, stats['files_processed'], len(files_to_process_indices))
        
        try:
            columns_to_read, volume_col = get_available_columns(filepath)
            
            if not volume_col:
                continue
            
            df = pq.read_table(filepath, columns=columns_to_read).to_pandas()
            
            if len(df) == 0:
                continue
            
            stats['total_rows'] += len(df)
            
            # Normalize volume column
            if volume_col != 'size_tokens':
                df.rename(columns={volume_col: 'size_tokens'}, inplace=True)
            
            # Type safety
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            if len(df) == 0:
                continue
            
            # Detect milliseconds
            if df['timestamp'].iloc[0] > 3e9:
                df['timestamp'] = df['timestamp'] / 1000.0
                df['resolution_time'] = df['resolution_time'] / 1000.0
            
            # Process each token
            df['token_id'] = df['token_id'].astype(str)
            
            for token_id, group in df.groupby('token_id', sort=False):
                condition_id = group['condition_id'].iloc[0]
                
                # Initialize accumulator if needed
                if token_id not in token_accumulators:
                    winner_status = winner_lookup.get(token_id, None)
                    
                    if winner_status is None:
                        stats['tokens_no_winner'] += 1
                        continue
                    
                    resolution_time = float(group['resolution_time'].iloc[0])
                    
                    token_accumulators[token_id] = PriceDynamicsAccumulator(
                        token_id, condition_id, resolution_time, winner_status
                    )
                    # Track token -> condition mapping for flush
                    condition_tokens[condition_id].add(token_id)
                
                # Extract trades
                timestamps = group['timestamp'].values
                prices = group['price'].values
                sizes = group['size_tokens'].values
                
                trades_batch = list(zip(timestamps, prices, sizes))
                token_accumulators[token_id].add_trades(trades_batch)
            
            del df
            
            # ==================================================================
            # STREAMING FLUSH: Check for completed conditions (Phase 1 pattern)
            # ==================================================================
            
            conditions_in_this_file = file_to_conditions.get(file_idx, set())
            
            for condition_id in conditions_in_this_file:
                if condition_id in condition_remaining_files:
                    condition_remaining_files[condition_id] -= 1
                    
                    # Condition complete! Flush it.
                    if condition_remaining_files[condition_id] == 0:
                        tokens_to_flush = condition_tokens.get(condition_id, set())
                        
                        for token_id in tokens_to_flush:
                            if token_id not in token_accumulators:
                                continue
                            
                            acc = token_accumulators[token_id]
                            dynamics = acc.compute_dynamics()
                            
                            if dynamics is not None:
                                results.append(dynamics)
                            else:
                                stats['tokens_filtered'] += 1
                            
                            # FREE THE MEMORY
                            del token_accumulators[token_id]
                        
                        # Clean up condition tracking
                        if condition_id in condition_tokens:
                            del condition_tokens[condition_id]
                        del condition_remaining_files[condition_id]
                        
                        stats['conditions_flushed'] += 1
            
        except Exception as e:
            log(f"  Error processing {filepath}: {e}")
            continue
    
    # ==========================================================================
    # FINAL FLUSH: Process any remaining conditions
    # ==========================================================================
    
    remaining_conditions = list(condition_tokens.keys())
    if remaining_conditions:
        log(f"\nFinal flush: {len(remaining_conditions)} remaining conditions, {len(token_accumulators):,} tokens...")
    
    for condition_id in remaining_conditions:
        tokens_to_flush = condition_tokens.get(condition_id, set())
        
        for token_id in tokens_to_flush:
            if token_id not in token_accumulators:
                continue
            
            acc = token_accumulators[token_id]
            dynamics = acc.compute_dynamics()
            
            if dynamics is not None:
                results.append(dynamics)
            else:
                stats['tokens_filtered'] += 1
            
            del token_accumulators[token_id]
        
        stats['conditions_flushed'] += 1
    
    gc.collect()
    
    log(f"\nProcessed {stats['files_processed']:,} files, {stats['total_rows']:,} rows")
    log(f"Computed dynamics for {len(results):,} tokens")
    log(f"Conditions flushed: {stats['conditions_flushed']:,}")
    log(f"Skipped: {stats['tokens_no_winner']:,} (no winner), {stats['tokens_filtered']:,} (filtered)")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"\nTotal time: {format_duration(elapsed)}")
    log(f"Final {log_memory()}")
    
    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log("Checkpoint removed (run complete)")
    
    return results

# ==============================================================================
# ANALYSIS AND REPORTING
# ==============================================================================

def analyze_results(results):
    """
    Analyze the collected price dynamics data.
    Interpretations aligned with Phase 1 calibration findings.
    """
    if not results:
        log("No results to analyze")
        return
    
    log("\n" + "="*70)
    log("ANALYSIS RESULTS")
    log("="*70)
    
    n_tokens = len(results)
    n_winners = sum(1 for r in results if r['winner'])
    n_losers = n_tokens - n_winners
    
    log(f"\nSample Size: {n_tokens:,} tokens ({n_winners:,} winners, {n_losers:,} losers)")
    
    # -------------------------------------------------------------------------
    # 0. Data Coverage Statistics
    # -------------------------------------------------------------------------
    log("\n" + "-"*50)
    log("0. DATA COVERAGE")
    log("-"*50)
    
    coverages = [r['coverage_pct'] for r in results]
    hours_data = [r['hours_of_data'] for r in results]
    
    log(f"  Horizon coverage: mean={np.mean(coverages):.1f}%, median={np.median(coverages):.1f}%")
    log(f"  Hours of data:    mean={np.mean(hours_data):.1f}h, median={np.median(hours_data):.1f}h")
    log(f"                    min={np.min(hours_data):.1f}h, max={np.max(hours_data):.1f}h")
    
    # Count tokens with data at each horizon
    log(f"\n  Tokens with data at each horizon:")
    for horizon in SNAPSHOT_HORIZONS:
        count = sum(1 for r in results if horizon in r['horizons_available'])
        pct = count / n_tokens * 100
        log(f"    T-{horizon:3d}h: {count:6,} tokens ({pct:5.1f}%)")
    
    # -------------------------------------------------------------------------
    # 1. Aggregate Autocorrelation
    # -------------------------------------------------------------------------
    log("\n" + "-"*50)
    log("1. AUTOCORRELATION OF PRICE CHANGES")
    log("-"*50)
    log("Interpretation:")
    log("  Negative = mean reversion (non-resolving moves, noise)")
    log("  Positive = momentum (gradual information incorporation)")
    log("  Phase 1 context: Tight calibration suggests efficient info processing")
    
    autocorr_summary = {}
    
    for lag in AUTOCORR_LAGS:
        key = f'autocorr_lag_{lag}h'
        values = [r['autocorrs'].get(key) for r in results if r['autocorrs'].get(key) is not None]
        
        if values:
            mean_ac = np.mean(values)
            std_ac = np.std(values)
            n_obs = len(values)
            
            # T-test against zero
            if n_obs > 1 and std_ac > 0:
                t_stat = mean_ac / (std_ac / np.sqrt(n_obs))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_obs-1))
            else:
                t_stat, p_val = 0, 1
            
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
            
            # Interpretive label
            if mean_ac < -0.05 and p_val < 0.05:
                interp = "[REVERSION]"
            elif mean_ac > 0.05 and p_val < 0.05:
                interp = "[MOMENTUM]"
            else:
                interp = "[~NEUTRAL]"
            
            autocorr_summary[lag] = {
                'mean': mean_ac, 'std': std_ac, 'n': n_obs,
                't': t_stat, 'p': p_val, 'interp': interp
            }
            
            log(f"  Lag {lag:3d}h: mean={mean_ac:+.4f}, std={std_ac:.4f}, n={n_obs:,} "
                f"(t={t_stat:+.2f}, p={p_val:.4f}) {sig} {interp}")
        else:
            log(f"  Lag {lag:3d}h: insufficient data")
    
    # -------------------------------------------------------------------------
    # 2. Price Change Analysis (ΔP) at Multiple Horizons
    # -------------------------------------------------------------------------
    log("\n" + "-"*50)
    log("2. PRICE CHANGES BETWEEN HORIZONS")
    log("-"*50)
    log("Interpretation: Mean ΔP shows directional drift as resolution approaches")
    
    # Dynamically find all available delta_p keys
    all_dp_keys = set()
    for r in results:
        all_dp_keys.update(r['delta_ps'].keys())
    
    # Sort by horizon (extract numbers from key)
    def extract_horizon(key):
        try:
            return int(key.split('-')[1].replace('h_to_T', ''))
        except:
            return 0
    
    dp_keys_sorted = sorted(all_dp_keys, key=extract_horizon, reverse=True)
    
    for key in dp_keys_sorted:
        values = [r['delta_ps'].get(key) for r in results if r['delta_ps'].get(key) is not None]
        
        if len(values) >= 10:
            mean_dp = np.mean(values)
            std_dp = np.std(values)
            n_obs = len(values)
            
            log(f"  {key:30s}: mean={mean_dp:+.4f}, std={std_dp:.4f}, n={n_obs:,}")
    
    # -------------------------------------------------------------------------
    # 3. Conditional Outcome Analysis by Price Bucket
    # -------------------------------------------------------------------------
    log("\n" + "-"*50)
    log("3. CONDITIONAL WIN RATE BY PRICE BUCKET (at T-24h)")
    log("-"*50)
    log("Interpretation: Compare to Phase 1 calibration results")
    log("  Phase 1 found tight calibration overall, but:")
    log("  - Subtle favorite-longshot bias")
    log("  - Higher entropy buckets slightly more mispriced")
    
    bucket_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'prices': [], 'entropies': []})
    
    for r in results:
        bucket = r.get('price_bucket_24h')
        p_24h = r['snapshots'].get('P_T-24h')
        entropy = r.get('entropy_24h')
        
        if bucket and p_24h is not None:
            bucket_stats[bucket]['total'] += 1
            bucket_stats[bucket]['prices'].append(p_24h)
            if entropy is not None:
                bucket_stats[bucket]['entropies'].append(entropy)
            if r['winner']:
                bucket_stats[bucket]['wins'] += 1
    
    calibration_results = []
    for bucket_info in PRICE_BUCKETS:
        lower, upper, label = bucket_info
        bs = bucket_stats[label]
        
        if bs['total'] > 0:
            win_rate = bs['wins'] / bs['total']
            avg_price = np.mean(bs['prices'])
            edge_bps = (win_rate - avg_price) * 10000
            avg_entropy = np.mean(bs['entropies']) if bs['entropies'] else None
            
            calibration_results.append({
                'bucket': label, 'n': bs['total'], 'win_rate': win_rate,
                'implied': avg_price, 'edge_bps': edge_bps, 'entropy': avg_entropy
            })
            
            entropy_str = f"H={avg_entropy:.3f}" if avg_entropy else "H=N/A"
            
            log(f"  {label:12s} ({lower:.0%}-{upper:.0%}): "
                f"n={bs['total']:6,}, win_rate={win_rate:.3f}, "
                f"implied={avg_price:.3f}, edge={edge_bps:+.1f}bps, {entropy_str}")
    
    # -------------------------------------------------------------------------
    # 4. Entropy-Stratified Analysis
    # -------------------------------------------------------------------------
    log("\n" + "-"*50)
    log("4. ENTROPY-STRATIFIED CALIBRATION")
    log("-"*50)
    log("Interpretation: Phase 1 found high-entropy markets more mispriced")
    log("  Low entropy  = confident markets (prices near 0 or 1)")
    log("  High entropy = uncertain markets (prices near 0.5)")
    
    # Stratify by entropy quartiles
    entropies = [(r['entropy_24h'], r['winner'], r['snapshots'].get('P_T-24h')) 
                 for r in results if r.get('entropy_24h') is not None]
    
    if len(entropies) >= 20:
        entropies_sorted = sorted(entropies, key=lambda x: x[0])
        quartile_size = len(entropies_sorted) // 4
        
        quartiles = [
            ('Q1 (lowest H)', entropies_sorted[:quartile_size]),
            ('Q2', entropies_sorted[quartile_size:2*quartile_size]),
            ('Q3', entropies_sorted[2*quartile_size:3*quartile_size]),
            ('Q4 (highest H)', entropies_sorted[3*quartile_size:]),
        ]
        
        for q_name, q_data in quartiles:
            if q_data:
                wins = sum(1 for _, w, _ in q_data if w)
                total = len(q_data)
                win_rate = wins / total
                avg_entropy = np.mean([e for e, _, _ in q_data])
                avg_price = np.mean([p for _, _, p in q_data if p is not None])
                edge_bps = (win_rate - avg_price) * 10000 if avg_price else 0
                
                log(f"  {q_name:15s}: n={total:6,}, H={avg_entropy:.3f}, "
                    f"win_rate={win_rate:.3f}, implied={avg_price:.3f}, edge={edge_bps:+.1f}bps")
    else:
        log("  Insufficient data for entropy stratification")
    
    # -------------------------------------------------------------------------
    # 5. Conditional on Price CHANGE (Short-term: T-48h to T-24h)
    # -------------------------------------------------------------------------
    log("\n" + "-"*50)
    log("5. CONDITIONAL WIN RATE BY PRICE CHANGE (T-48h to T-24h)")
    log("-"*50)
    log("Interpretation: Does recent price movement predict outcome?")
    log("  If win_rate > post_price: move was UNDERREACTION (momentum)")
    log("  If win_rate < post_price: move was OVERREACTION (reversion)")
    
    dp_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'dps': [], 'post_prices': []})
    
    for r in results:
        bucket = r.get('delta_bucket_48_24')
        dp = r['delta_ps'].get('dP_T-48h_to_T-24h')
        p_24h = r['snapshots'].get('P_T-24h')
        
        if bucket and dp is not None and p_24h is not None:
            dp_stats[bucket]['total'] += 1
            dp_stats[bucket]['dps'].append(dp)
            dp_stats[bucket]['post_prices'].append(p_24h)
            if r['winner']:
                dp_stats[bucket]['wins'] += 1
    
    for bucket_info in DELTA_P_BUCKETS:
        lower, upper, label = bucket_info
        ds = dp_stats[label]
        
        if ds['total'] >= 10:
            win_rate = ds['wins'] / ds['total']
            avg_dp = np.mean(ds['dps'])
            avg_post_price = np.mean(ds['post_prices'])
            
            efficiency = win_rate - avg_post_price
            eff_bps = efficiency * 10000
            
            if efficiency > 0.02:
                reaction_type = "UNDERREACTION"
            elif efficiency < -0.02:
                reaction_type = "OVERREACTION"
            else:
                reaction_type = "efficient"
            
            log(f"  {label:15s} ({lower:+.0%} to {upper:+.0%}): "
                f"n={ds['total']:6,}, win_rate={win_rate:.3f}, "
                f"post_P={avg_post_price:.3f}, eff={eff_bps:+.1f}bps ({reaction_type})")
    
    # -------------------------------------------------------------------------
    # 6. Conditional on Price CHANGE (Longer-term: T-168h to T-24h)
    # -------------------------------------------------------------------------
    log("\n" + "-"*50)
    log("6. CONDITIONAL WIN RATE BY WEEKLY MOVE (T-168h to T-24h)")
    log("-"*50)
    log("Interpretation: Does the week-long price trajectory predict outcome?")
    
    dp_stats_weekly = defaultdict(lambda: {'wins': 0, 'total': 0, 'dps': [], 'post_prices': []})
    
    for r in results:
        bucket = r.get('delta_bucket_168_24')
        dp = r['delta_ps'].get('dP_T-168h_to_T-24h')
        p_24h = r['snapshots'].get('P_T-24h')
        
        if bucket and dp is not None and p_24h is not None:
            dp_stats_weekly[bucket]['total'] += 1
            dp_stats_weekly[bucket]['dps'].append(dp)
            dp_stats_weekly[bucket]['post_prices'].append(p_24h)
            if r['winner']:
                dp_stats_weekly[bucket]['wins'] += 1
    
    for bucket_info in DELTA_P_BUCKETS:
        lower, upper, label = bucket_info
        ds = dp_stats_weekly[label]
        
        if ds['total'] >= 10:
            win_rate = ds['wins'] / ds['total']
            avg_dp = np.mean(ds['dps'])
            avg_post_price = np.mean(ds['post_prices'])
            
            efficiency = win_rate - avg_post_price
            eff_bps = efficiency * 10000
            
            if efficiency > 0.02:
                reaction_type = "UNDERREACTION"
            elif efficiency < -0.02:
                reaction_type = "OVERREACTION"
            else:
                reaction_type = "efficient"
            
            log(f"  {label:15s} ({lower:+.0%} to {upper:+.0%}): "
                f"n={ds['total']:6,}, win_rate={win_rate:.3f}, "
                f"post_P={avg_post_price:.3f}, eff={eff_bps:+.1f}bps ({reaction_type})")
    
    # -------------------------------------------------------------------------
    # 7. Summary and Phase 1 Connection
    # -------------------------------------------------------------------------
    log("\n" + "-"*50)
    log("7. SUMMARY & PHASE 1 ALIGNMENT")
    log("-"*50)
    
    # Overall price change direction and outcome relationship
    positive_moves = [r for r in results 
                      if r['delta_ps'].get('dP_T-48h_to_T-24h', 0) > 0.05]
    negative_moves = [r for r in results 
                      if r['delta_ps'].get('dP_T-48h_to_T-24h', 0) < -0.05]
    
    if positive_moves:
        pos_win_rate = sum(1 for r in positive_moves if r['winner']) / len(positive_moves)
        pos_avg_price = np.mean([r['snapshots'].get('P_T-24h', 0.5) for r in positive_moves])
        log(f"  Tokens with positive move (>5%):  n={len(positive_moves):,}, "
            f"win_rate={pos_win_rate:.3f}, implied={pos_avg_price:.3f}")
    
    if negative_moves:
        neg_win_rate = sum(1 for r in negative_moves if r['winner']) / len(negative_moves)
        neg_avg_price = np.mean([r['snapshots'].get('P_T-24h', 0.5) for r in negative_moves])
        log(f"  Tokens with negative move (<-5%): n={len(negative_moves):,}, "
            f"win_rate={neg_win_rate:.3f}, implied={neg_avg_price:.3f}")
    
    log("\n  Phase 1 Calibration Context:")
    log("  - Aggregate calibration was tight (null result for baseline edge)")
    log("  - TWAP 48h-24h and Snapshot 24h showed sign-flipping across buckets")
    log("  - Subtle favorite-longshot bias detected")
    log("  - High-entropy markets showed more mispricing")
    log("\n  This analysis reveals:")
    log("  - Time scales at which information gets incorporated")
    log("  - Whether moves are systematically over/under-reactions")
    log("  - Entropy's role in market efficiency")
    
    return {
        'n_tokens': n_tokens,
        'n_winners': n_winners,
        'autocorr_summary': autocorr_summary,
        'calibration_results': calibration_results,
    }

def generate_report(results, analysis_summary):
    """Generate comprehensive text report"""
    ensure_output_dir()
    
    report_path = os.path.join(OUTPUT_DIR, f'phase2_price_dynamics_{TIMESTAMP}_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 2: PRICE DYNAMICS ANALYSIS - FULL REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tokens Analyzed: {len(results):,}\n")
        f.write(f"Time Horizons: {SNAPSHOT_HORIZONS}\n")
        f.write(f"Autocorr Lags: {AUTOCORR_LAGS}\n\n")
        
        # Key findings
        f.write("-"*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n\n")
        
        if analysis_summary and 'autocorr_summary' in analysis_summary:
            f.write("Autocorrelation Results:\n")
            for lag, data in analysis_summary['autocorr_summary'].items():
                f.write(f"  Lag {lag}h: {data['mean']:+.4f} (p={data['p']:.4f}) {data['interp']}\n")
            f.write("\n")
        
        if analysis_summary and 'calibration_results' in analysis_summary:
            f.write("Calibration by Price Bucket:\n")
            for cr in analysis_summary['calibration_results']:
                f.write(f"  {cr['bucket']:12s}: n={cr['n']:6,}, edge={cr['edge_bps']:+.1f}bps\n")
            f.write("\n")
        
        # Phase 1 connection
        f.write("-"*80 + "\n")
        f.write("PHASE 1 CONNECTION\n")
        f.write("-"*80 + "\n\n")
        
        f.write("This analysis complements Phase 1 calibration findings:\n\n")
        f.write("Phase 1 Key Results:\n")
        f.write("  - Aggregate calibration was tight (null result for baseline edge)\n")
        f.write("  - TWAP 48h-24h and Snapshot 24h showed sign-flipping across buckets\n")
        f.write("  - Subtle favorite-longshot bias detected\n")
        f.write("  - High-entropy markets showed more mispricing\n\n")
        
        f.write("Phase 2 Findings:\n")
        f.write("  - [See KEY FINDINGS above]\n")
        f.write("  - Price dynamics reveal information incorporation patterns\n")
        f.write("  - Autocorrelation structure indicates market efficiency regime\n\n")
        
        f.write("="*80 + "\n")
    
    log(f"Report saved: {report_path}")
    return report_path

def generate_visualization(results):
    """Generate visualization of price dynamics"""
    ensure_output_dir()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Autocorrelation bar chart
    ax1 = axes[0, 0]
    autocorr_means = []
    autocorr_stds = []
    valid_lags = []
    for lag in AUTOCORR_LAGS:
        key = f'autocorr_lag_{lag}h'
        values = [r['autocorrs'].get(key) for r in results if r['autocorrs'].get(key) is not None]
        if values and len(values) >= 10:
            autocorr_means.append(np.mean(values))
            autocorr_stds.append(np.std(values) / np.sqrt(len(values)))
            valid_lags.append(lag)
    
    if valid_lags:
        x = np.arange(len(valid_lags))
        bars = ax1.bar(x, autocorr_means, yerr=autocorr_stds, capsize=5, alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{lag}h' for lag in valid_lags])
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Mean Autocorrelation')
        ax1.set_title('Price Change Autocorrelation by Lag')
        
        for bar, mean in zip(bars, autocorr_means):
            bar.set_color('indianred' if mean < 0 else 'seagreen')
    
    # 2. Calibration by price bucket
    ax2 = axes[0, 1]
    bucket_labels = []
    win_rates = []
    implied_probs = []
    
    bucket_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'prices': []})
    for r in results:
        bucket = r.get('price_bucket_24h')
        p_24h = r['snapshots'].get('P_T-24h')
        if bucket and p_24h is not None:
            bucket_stats[bucket]['total'] += 1
            bucket_stats[bucket]['prices'].append(p_24h)
            if r['winner']:
                bucket_stats[bucket]['wins'] += 1
    
    for lower, upper, label in PRICE_BUCKETS:
        bs = bucket_stats[label]
        if bs['total'] >= 10:
            bucket_labels.append(label)
            win_rates.append(bs['wins'] / bs['total'])
            implied_probs.append(np.mean(bs['prices']))
    
    if bucket_labels:
        x = np.arange(len(bucket_labels))
        width = 0.35
        ax2.bar(x - width/2, implied_probs, width, label='Implied Prob', alpha=0.7, color='steelblue')
        ax2.bar(x + width/2, win_rates, width, label='Realized Win Rate', alpha=0.7, color='coral')
        ax2.set_xticks(x)
        ax2.set_xticklabels(bucket_labels, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Probability')
        ax2.set_title('Calibration by Price Bucket (T-24h)')
        ax2.legend(fontsize=8)
    
    # 3. Reaction efficiency
    ax3 = axes[0, 2]
    dp_labels = []
    dp_win_rates = []
    dp_implied = []
    
    dp_stats = defaultdict(lambda: {'wins': 0, 'total': 0, 'post_prices': []})
    for r in results:
        bucket = r.get('delta_bucket_48_24')
        p_24h = r['snapshots'].get('P_T-24h')
        if bucket and p_24h is not None:
            dp_stats[bucket]['total'] += 1
            dp_stats[bucket]['post_prices'].append(p_24h)
            if r['winner']:
                dp_stats[bucket]['wins'] += 1
    
    for lower, upper, label in DELTA_P_BUCKETS:
        ds = dp_stats[label]
        if ds['total'] >= 10:
            dp_labels.append(label)
            dp_win_rates.append(ds['wins'] / ds['total'])
            dp_implied.append(np.mean(ds['post_prices']))
    
    if dp_labels:
        x = np.arange(len(dp_labels))
        width = 0.35
        ax3.bar(x - width/2, dp_implied, width, label='Post-Move Price', alpha=0.7, color='steelblue')
        ax3.bar(x + width/2, dp_win_rates, width, label='Win Rate', alpha=0.7, color='coral')
        ax3.set_xticks(x)
        ax3.set_xticklabels(dp_labels, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('Probability')
        ax3.set_title('Reaction Efficiency (T-48h to T-24h)')
        ax3.legend(fontsize=8)
    
    # 4. Entropy vs Edge
    ax4 = axes[1, 0]
    entropies = []
    edges = []
    
    for r in results:
        entropy = r.get('entropy_24h')
        p_24h = r['snapshots'].get('P_T-24h')
        if entropy is not None and p_24h is not None:
            outcome = 1 if r['winner'] else 0
            edge = outcome - p_24h
            entropies.append(entropy)
            edges.append(edge)
    
    if entropies:
        colors = ['seagreen' if e > 0 else 'indianred' for e in edges]
        ax4.scatter(entropies, edges, c=colors, alpha=0.2, s=10)
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax4.set_xlabel('Binary Entropy at T-24h')
        ax4.set_ylabel('Edge (Outcome - Price)')
        ax4.set_title('Entropy vs Edge')
    
    # 5. Data coverage
    ax5 = axes[1, 1]
    hours_data = [r['hours_of_data'] for r in results]
    if hours_data:
        ax5.hist(hours_data, bins=50, alpha=0.7, color='steelblue', edgecolor='white')
        ax5.axvline(x=np.median(hours_data), color='indianred', linestyle='--', 
                    label=f'Median: {np.median(hours_data):.0f}h')
        ax5.set_xlabel('Hours of Trading Data')
        ax5.set_ylabel('Count')
        ax5.set_title('Data Coverage Distribution')
        ax5.legend()
    
    # 6. Price trajectory scatter
    ax6 = axes[1, 2]
    dps = []
    post_prices = []
    outcomes = []
    
    for r in results:
        dp = r['delta_ps'].get('dP_T-48h_to_T-24h')
        p_24h = r['snapshots'].get('P_T-24h')
        if dp is not None and p_24h is not None:
            dps.append(dp)
            post_prices.append(p_24h)
            outcomes.append(1 if r['winner'] else 0)
    
    if dps:
        colors = ['seagreen' if o == 1 else 'indianred' for o in outcomes]
        ax6.scatter(dps, post_prices, c=colors, alpha=0.2, s=10)
        ax6.axhline(y=0.5, color='black', linestyle='--', linewidth=0.5)
        ax6.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax6.set_xlabel('ΔP (T-48h to T-24h)')
        ax6.set_ylabel('Price at T-24h')
        ax6.set_title('Price Change vs Post-Change Price')
    
    plt.tight_layout()
    
    viz_path = os.path.join(OUTPUT_DIR, f'phase2_price_dynamics_{TIMESTAMP}.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"Visualization saved: {viz_path}")
    return viz_path

def save_results_json(results, analysis_summary):
    """Save results as JSON for further analysis"""
    ensure_output_dir()
    
    # Create summary-only version (full results too large for JSON)
    summary = {
        'timestamp': TIMESTAMP,
        'n_tokens': len(results),
        'n_winners': sum(1 for r in results if r['winner']),
        'horizons': SNAPSHOT_HORIZONS,
        'autocorr_lags': AUTOCORR_LAGS,
        'analysis_summary': analysis_summary,
    }
    
    json_path = os.path.join(OUTPUT_DIR, f'phase2_price_dynamics_{TIMESTAMP}_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    log(f"Summary JSON saved: {json_path}")
    return json_path

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phase 2 Price Dynamics Analysis - Full Production Run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (all files)
  python phase2_price_dynamics_full.py
  
  # Sample run (subset of files)
  python phase2_price_dynamics_full.py --sample 1000
  
  # Resume interrupted run
  python phase2_price_dynamics_full.py --resume
        """
    )
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files (for testing)')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint if available')
    
    args = parser.parse_args()
    
    log("Starting Phase 2 Price Dynamics Analysis...")
    
    ensure_output_dir()
    
    # Run analysis
    results = run_full_analysis(sample_files=args.sample, resume=args.resume)
    
    if results:
        # Analyze results
        analysis_summary = analyze_results(results)
        
        # Generate outputs
        generate_report(results, analysis_summary)
        generate_visualization(results)
        save_results_json(results, analysis_summary)
        
        log("\n" + "="*70)
        log("ANALYSIS COMPLETE")
        log("="*70)
        log(f"\nOutputs generated:")
        log(f"  - Report:        phase2_price_dynamics_{TIMESTAMP}_report.txt")
        log(f"  - Visualization: phase2_price_dynamics_{TIMESTAMP}.png")
        log(f"  - Summary JSON:  phase2_price_dynamics_{TIMESTAMP}_summary.json")
    else:
        log("\nAnalysis failed - no results generated")