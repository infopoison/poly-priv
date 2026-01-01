#!/usr/bin/env python3
"""
Phase 2: Price Dynamics Analysis - Unconditional Features
Version: 1.0 - Diagnostic Mode

OBJECTIVE:
  Characterize the unconditional dynamics of price changes in Polymarket.
  This is Step 1 from the research agenda: understand whether there's
  mean reversion or momentum in price movements before resolution.

FEATURES ANALYZED (price-conditional only):
  1. Price snapshots at multiple horizons: T-72h, T-48h, T-24h, T-12h, T-6h, T-1h
  2. ΔP (price changes) between adjacent horizons
  3. Autocorrelation of price changes at various lags (1h, 6h, 12h, 24h)
  4. Conditional distribution of outcomes given:
     - Past price levels (snapshot buckets)
     - Recent price changes (ΔP buckets)

INTERPRETATION FRAMEWORK:
  - Negative autocorrelation → Mean reversion (non-resolving moves dominate)
  - Positive autocorrelation → Momentum (information arrival is gradual)
  - If P(win | ΔP > 0) > E[P | post-move], moves are typically underreactions
  - If P(win | ΔP > 0) < E[P | post-move], moves are typically overreactions

DIAGNOSTIC MODE:
  - Processes ~100 conditions with verbose output
  - Validates data quality and pipeline correctness
  - Outputs detailed per-condition diagnostics

ARCHITECTURE:
  - Reuses sidecar-based winner lookup from Phase 1
  - File-first accumulator pattern with streaming aggregation
  - Memory efficient: O(active_conditions)
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

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = "../../"  # Adjust based on execution location

BATCH_DIR = os.path.join(BASE_DIR, 'order_history_batches')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis/outputs')
INDEX_CACHE_FILE = os.path.join(BASE_DIR, 'analysis/market_index_cache.pkl')
SIDECAR_FILE = os.path.join(BASE_DIR, 'data/repair/api_derived_winners.parquet')

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

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
    for lower, upper, label in buckets:
        if lower <= value < upper:
            return label
    # Edge cases
    if value >= buckets[-1][1]:
        return buckets[-1][2]
    if value < buckets[0][0]:
        return buckets[0][2]
    return None

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
    hours = sorted(hourly_prices.keys(), reverse=True)  # From T-72 to T-1
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
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = stats.pearsonr(x, y)
        return corr if np.isfinite(corr) else None
    except:
        return None

# ==============================================================================
# DIAGNOSTIC DATA COLLECTION
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
            'hourly_prices': hourly_prices,
            'price_changes': price_changes,
        }

# ==============================================================================
# MAIN DIAGNOSTIC PROCESSING
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

def run_diagnostic(sample_files=50, max_conditions=100):
    """
    Run diagnostic mode: process a small sample and output detailed results.
    """
    log("="*70)
    log("PRICE DYNAMICS ANALYSIS - DIAGNOSTIC MODE")
    log("="*70)
    log(f"Sample files: {sample_files}")
    log(f"Max conditions: {max_conditions}")
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
    
    # Load market index from cache
    market_index = load_market_index(INDEX_CACHE_FILE, batch_files)
    if market_index:
        log(f"Loaded market index from cache: {len(market_index):,} conditions")
    else:
        log("No valid cache found - will process files directly")
        market_index = None
    
    # Sample files
    sample_batch_files = batch_files[:sample_files]
    log(f"Processing {len(sample_batch_files)} sample files...")
    
    # Accumulators
    token_accumulators = {}  # token_id -> PriceDynamicsAccumulator
    conditions_seen = set()
    
    files_processed = 0
    total_rows = 0
    
    for filepath in sample_batch_files:
        files_processed += 1
        
        if files_processed % 10 == 0:
            log(f"  [{files_processed}/{len(sample_batch_files)}] "
                f"Tokens: {len(token_accumulators):,} | "
                f"Conditions: {len(conditions_seen):,} | "
                f"{log_memory()}")
        
        try:
            columns_to_read, volume_col = get_available_columns(filepath)
            
            if not volume_col:
                continue
            
            df = pq.read_table(filepath, columns=columns_to_read).to_pandas()
            
            if len(df) == 0:
                continue
            
            total_rows += len(df)
            
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
                
                # Limit conditions for diagnostic
                if len(conditions_seen) >= max_conditions and condition_id not in conditions_seen:
                    continue
                
                conditions_seen.add(condition_id)
                
                # Initialize accumulator if needed
                if token_id not in token_accumulators:
                    winner_status = winner_lookup.get(token_id, None)
                    
                    if winner_status is None:
                        continue  # Skip tokens not in sidecar
                    
                    resolution_time = float(group['resolution_time'].iloc[0])
                    
                    token_accumulators[token_id] = PriceDynamicsAccumulator(
                        token_id, condition_id, resolution_time, winner_status
                    )
                
                # Extract trades
                timestamps = group['timestamp'].values
                prices = group['price'].values
                sizes = group['size_tokens'].values
                
                trades_batch = list(zip(timestamps, prices, sizes))
                token_accumulators[token_id].add_trades(trades_batch)
            
            del df
            gc.collect()
            
        except Exception as e:
            log(f"  Error processing {filepath}: {e}")
            continue
    
    log(f"\nProcessed {files_processed} files, {total_rows:,} rows")
    log(f"Accumulated {len(token_accumulators):,} tokens from {len(conditions_seen):,} conditions")
    
    # Compute dynamics for all tokens
    log("\nComputing price dynamics...")
    
    results = []
    skipped_insufficient_data = 0
    
    for token_id, acc in token_accumulators.items():
        dynamics = acc.compute_dynamics()
        
        if dynamics is None:
            skipped_insufficient_data += 1
            continue
        
        results.append(dynamics)
    
    log(f"Computed dynamics for {len(results):,} tokens")
    log(f"Skipped {skipped_insufficient_data:,} tokens (insufficient data)")
    
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
    
    log(f"\nSample Size: {n_tokens} tokens ({n_winners} winners, {n_losers} losers)")
    
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
        log(f"    T-{horizon:3d}h: {count:4d} tokens ({pct:5.1f}%)")
    
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
            if mean_ac < -0.05:
                interp = "[REVERSION]"
            elif mean_ac > 0.05:
                interp = "[MOMENTUM]"
            else:
                interp = "[~NEUTRAL]"
            
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
        # e.g., 'dP_T-720h_to_T-336h' -> 720
        try:
            return int(key.split('-')[1].replace('h_to_T', ''))
        except:
            return 0
    
    dp_keys_sorted = sorted(all_dp_keys, key=extract_horizon, reverse=True)
    
    for key in dp_keys_sorted:
        values = [r['delta_ps'].get(key) for r in results if r['delta_ps'].get(key) is not None]
        
        if len(values) >= 3:  # Need at least 3 for meaningful stats
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
    
    for bucket_info in PRICE_BUCKETS:
        lower, upper, label = bucket_info
        bs = bucket_stats[label]
        
        if bs['total'] > 0:
            win_rate = bs['wins'] / bs['total']
            avg_price = np.mean(bs['prices'])
            edge_bps = (win_rate - avg_price) * 10000
            avg_entropy = np.mean(bs['entropies']) if bs['entropies'] else None
            
            entropy_str = f"H={avg_entropy:.3f}" if avg_entropy else "H=N/A"
            
            log(f"  {label:12s} ({lower:.0%}-{upper:.0%}): "
                f"n={bs['total']:4d}, win_rate={win_rate:.3f}, "
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
    
    if len(entropies) >= 4:
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
                
                log(f"  {q_name:15s}: n={total:4d}, H={avg_entropy:.3f}, "
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
        
        if ds['total'] > 0:
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
                f"n={ds['total']:4d}, win_rate={win_rate:.3f}, "
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
        
        if ds['total'] > 0:
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
                f"n={ds['total']:4d}, win_rate={win_rate:.3f}, "
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
    log("\n  This analysis should reveal:")
    log("  - Whether price dynamics explain the calibration pattern")
    log("  - Time scales at which information gets incorporated")
    log("  - Whether moves are systematically over/under-reactions")
    
    return results

def generate_diagnostic_report(results):
    """
    Generate a detailed diagnostic report file.
    """
    ensure_output_dir()
    
    report_path = os.path.join(OUTPUT_DIR, f'phase2_price_dynamics_diagnostic_{TIMESTAMP}.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 2: PRICE DYNAMICS ANALYSIS - DIAGNOSTIC REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tokens Analyzed: {len(results)}\n")
        f.write(f"Time Horizons: {SNAPSHOT_HORIZONS}\n")
        f.write(f"Autocorr Lags: {AUTOCORR_LAGS}\n\n")
        
        # Data coverage summary
        f.write("-"*80 + "\n")
        f.write("DATA COVERAGE SUMMARY\n")
        f.write("-"*80 + "\n\n")
        
        coverages = [r['coverage_pct'] for r in results]
        hours_data = [r['hours_of_data'] for r in results]
        
        f.write(f"Horizon coverage: mean={np.mean(coverages):.1f}%, median={np.median(coverages):.1f}%\n")
        f.write(f"Hours of data:    mean={np.mean(hours_data):.1f}h, median={np.median(hours_data):.1f}h\n")
        f.write(f"                  min={np.min(hours_data):.1f}h, max={np.max(hours_data):.1f}h\n\n")
        
        # Sample of individual token results
        f.write("-"*80 + "\n")
        f.write("SAMPLE TOKEN DETAILS (first 10)\n")
        f.write("-"*80 + "\n\n")
        
        for r in results[:10]:
            f.write(f"Token: {r['token_id'][:40]}...\n")
            f.write(f"  Condition: {r['condition_id'][:40]}...\n")
            f.write(f"  Winner: {r['winner']}\n")
            f.write(f"  Trade Count: {r['trade_count']}\n")
            f.write(f"  Hours of Data: {r['hours_of_data']:.1f}\n")
            f.write(f"  Coverage: {r['coverage_pct']:.1f}% ({len(r['horizons_available'])} horizons)\n")
            f.write(f"  Entropy at T-24h: {r.get('entropy_24h', 'N/A')}\n")
            
            f.write(f"  Snapshots:\n")
            for k, v in sorted(r['snapshots'].items(), key=lambda x: -int(x[0].split('-')[1].replace('h', ''))):
                if v is not None:
                    f.write(f"    {k}: {v:.4f}\n")
            
            f.write(f"  Delta Ps:\n")
            for k, v in sorted(r['delta_ps'].items()):
                if v is not None:
                    f.write(f"    {k}: {v:+.4f}\n")
            
            f.write(f"  Autocorrelations:\n")
            for k, v in sorted(r['autocorrs'].items()):
                if v is not None:
                    f.write(f"    {k}: {v:+.4f}\n")
            
            f.write("\n")
        
        # Phase 1 connection notes
        f.write("-"*80 + "\n")
        f.write("PHASE 1 CALIBRATION CONNECTION\n")
        f.write("-"*80 + "\n\n")
        
        f.write("This analysis complements Phase 1 calibration findings:\n\n")
        f.write("Phase 1 Key Results:\n")
        f.write("  - Aggregate calibration was tight (null result for baseline edge)\n")
        f.write("  - TWAP 48h-24h and Snapshot 24h showed sign-flipping across buckets\n")
        f.write("  - Subtle favorite-longshot bias detected\n")
        f.write("  - High-entropy markets showed more mispricing\n\n")
        
        f.write("Phase 2 Investigates:\n")
        f.write("  - Whether price dynamics explain the calibration pattern\n")
        f.write("  - Time scales at which information gets incorporated\n")
        f.write("  - Whether moves are systematically over/under-reactions\n")
        f.write("  - Entropy's role in market efficiency\n\n")
        
        f.write("="*80 + "\n")
    
    log(f"\nDiagnostic report saved: {report_path}")
    return report_path

def generate_visualization(results):
    """
    Generate visualization of price dynamics.
    Updated to match wider time horizons and entropy analysis.
    """
    ensure_output_dir()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Autocorrelation bar chart (extended lags)
    ax1 = axes[0, 0]
    autocorr_means = []
    autocorr_stds = []
    valid_lags = []
    for lag in AUTOCORR_LAGS:
        key = f'autocorr_lag_{lag}h'
        values = [r['autocorrs'].get(key) for r in results if r['autocorrs'].get(key) is not None]
        if values and len(values) >= 3:
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
        
        # Color bars by sign
        for bar, mean in zip(bars, autocorr_means):
            bar.set_color('indianred' if mean < 0 else 'seagreen')
    else:
        ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Price Change Autocorrelation by Lag')
    
    # 2. Win rate by price bucket (aligned with Phase 1 buckets)
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
        if bs['total'] >= 3:
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
    else:
        ax2.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Calibration by Price Bucket (T-24h)')
    
    # 3. Win rate by price change direction (T-48h to T-24h)
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
        if ds['total'] >= 3:
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
    else:
        ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Reaction Efficiency (T-48h to T-24h)')
    
    # 4. Entropy vs Edge scatter
    ax4 = axes[1, 0]
    entropies = []
    edges = []
    
    for r in results:
        entropy = r.get('entropy_24h')
        p_24h = r['snapshots'].get('P_T-24h')
        if entropy is not None and p_24h is not None:
            outcome = 1 if r['winner'] else 0
            edge = outcome - p_24h  # Individual token "edge"
            entropies.append(entropy)
            edges.append(edge)
    
    if entropies:
        colors = ['seagreen' if e > 0 else 'indianred' for e in edges]
        ax4.scatter(entropies, edges, c=colors, alpha=0.4, s=20)
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax4.set_xlabel('Binary Entropy at T-24h')
        ax4.set_ylabel('Edge (Outcome - Price)')
        ax4.set_title('Entropy vs Edge (green=won, red=lost)')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Entropy vs Edge')
    
    # 5. Data coverage histogram
    ax5 = axes[1, 1]
    hours_data = [r['hours_of_data'] for r in results]
    if hours_data:
        ax5.hist(hours_data, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
        ax5.axvline(x=np.median(hours_data), color='indianred', linestyle='--', 
                    label=f'Median: {np.median(hours_data):.0f}h')
        ax5.set_xlabel('Hours of Trading Data')
        ax5.set_ylabel('Count')
        ax5.set_title('Data Coverage Distribution')
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Data Coverage Distribution')
    
    # 6. Price trajectory scatter (ΔP vs post-price)
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
        ax6.scatter(dps, post_prices, c=colors, alpha=0.4, s=20)
        ax6.axhline(y=0.5, color='black', linestyle='--', linewidth=0.5)
        ax6.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax6.set_xlabel('ΔP (T-48h to T-24h)')
        ax6.set_ylabel('Price at T-24h')
        ax6.set_title('Price Change vs Post-Change Price (green=winner)')
    else:
        ax6.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Price Change vs Post-Change Price')
    
    plt.tight_layout()
    
    viz_path = os.path.join(OUTPUT_DIR, f'phase2_price_dynamics_diagnostic_{TIMESTAMP}.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"Visualization saved: {viz_path}")
    return viz_path

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Phase 2 Price Dynamics Analysis - Diagnostic Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default diagnostic (50 files, 100 conditions)
  python phase2_price_dynamics_diagnostic.py
  
  # Larger sample
  python phase2_price_dynamics_diagnostic.py --files 200 --conditions 500
  
  # Quick validation
  python phase2_price_dynamics_diagnostic.py --files 20 --conditions 50
        """
    )
    parser.add_argument('--files', '-f', type=int, default=50,
                        help='Number of batch files to process (default: 50)')
    parser.add_argument('--conditions', '-c', type=int, default=100,
                        help='Maximum conditions to analyze (default: 100)')
    
    args = parser.parse_args()
    
    log("Starting Price Dynamics Diagnostic...")
    log(f"Parameters: files={args.files}, conditions={args.conditions}")
    
    ensure_output_dir()
    
    # Run diagnostic
    results = run_diagnostic(sample_files=args.files, max_conditions=args.conditions)
    
    if results:
        # Analyze results
        analyze_results(results)
        
        # Generate outputs
        generate_diagnostic_report(results)
        generate_visualization(results)
        
        log("\nDiagnostic complete!")
    else:
        log("\nDiagnostic failed - no results generated")