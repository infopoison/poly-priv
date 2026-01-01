#!/usr/bin/env python3
"""
Phase 4E: Event-Based Dip Analysis - MEMORY SAFE VERSION
Version: 1.0

OBJECTIVE:
  Shift from window-based tercile analysis (4D) to event-based dip detection
  with continuous parameterization. This answers: "If I deploy a rolling detector
  that triggers the moment price drops from a trailing maximum, what is the 
  relationship between reaction latency, velocity, time-to-resolution, and edge?"

KEY CONCEPTUAL SHIFT FROM 4D:
  4D: For each (start_hour, end_hour) window, did price cross threshold?
      If yes, bucket into tercile → categorical edge estimates
  
  4E: For each token, identify ALL dip events (local max → threshold crossing)
      For each event, record continuous features → edge curves

METHODOLOGY:
  1. Identify dip events using rolling maximum with configurable lookback
  2. Record continuous features:
     - t_max: when local maximum occurred
     - t_cross: when threshold was crossed
     - t_resolution: market resolution time
     - reaction_latency = t_cross - t_max (how long the dip took)
     - time_to_resolution = t_resolution - t_cross
     - velocity = threshold / reaction_latency
  3. Stratify by probability bucket (as in 4D)
  4. Output edge curves (edge as function of continuous variables)

KEY PARAMETERS TO VARY:
  - Lookback window for local max detection (3h, 6h, 12h)
  - Threshold for dip detection (5%, 10%, 15%, 20%)
  - Minimum time-to-resolution filter (2h, 3h, 4h, etc.)

KEY OUTPUTS:
  - Edge as a function of reaction latency (0-15min, 15-30min, 30min-1h, 1-2h, 2-4h, etc.)
  - Edge as a function of velocity
  - Recommended minimum latency (if any)
  - Confirmation or refinement of the ≥4h boundary

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

# Lookback windows for local max detection (hours)
# "How far back do we look for the start of a dip?"
LOOKBACK_WINDOWS_HOURS = [3, 6, 12]

# Move thresholds (fractional/absolute points)
MOVE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]

# Minimum time-to-resolution filters (hours)
# Events closer to resolution than this are filtered out
MIN_TIME_TO_RESOLUTION_HOURS = [2, 3, 4, 6]

# Reaction latency bins (hours) - for continuous edge curves
# These define the buckets for "how long after dip started did we enter"
LATENCY_BINS_HOURS = [
    (0, 0.25, '0-15min'),
    (0.25, 0.5, '15-30min'),
    (0.5, 1.0, '30min-1h'),
    (1.0, 2.0, '1-2h'),
    (2.0, 4.0, '2-4h'),
    (4.0, 8.0, '4-8h'),
    (8.0, float('inf'), '8h+'),
]

# Velocity bins (threshold_pts per hour)
# High velocity = fast dip (potentially informed flow)
# Low velocity = slow dip (potentially noise)
VELOCITY_BINS_PTS_PER_HOUR = [
    (0, 0.05, 'very_slow'),      # <5% per hour
    (0.05, 0.10, 'slow'),        # 5-10% per hour
    (0.10, 0.20, 'moderate'),    # 10-20% per hour
    (0.20, 0.50, 'fast'),        # 20-50% per hour
    (0.50, float('inf'), 'very_fast'),  # >50% per hour
]

# Probability buckets for stratification (by start_price)
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
# EDGE CELL FOR STREAMING AGGREGATION
# ==============================================================================

def _make_edge_cell():
    """Create a cell for one (latency_bin x velocity_bin x prob_bucket) combination."""
    return {
        'n_samples': 0,
        'n_wins': 0,
        'n_fills': 0,
        'n_fill_wins': 0,
        'crossing_prices': StreamingStats(),
        'fill_prices': StreamingStats(),
        'latencies_hours': StreamingStats(),
        'velocities': StreamingStats(),
        'time_to_resolution_hours': StreamingStats(),
        'local_max_prices': StreamingStats(),
    }


def get_prob_bucket(price):
    """Determine which probability bucket a price falls into."""
    for label, lo, hi in PROB_BUCKETS:
        if lo <= price < hi:
            return label
    return None


def get_latency_bin(latency_hours):
    """Determine which latency bin a reaction latency falls into."""
    for lo, hi, label in LATENCY_BINS_HOURS:
        if lo <= latency_hours < hi:
            return label
    return None


def get_velocity_bin(velocity_pts_per_hour):
    """Determine which velocity bin a dip velocity falls into."""
    for lo, hi, label in VELOCITY_BINS_PTS_PER_HOUR:
        if lo <= velocity_pts_per_hour < hi:
            return label
    return None


# ==============================================================================
# DIP EVENT AGGREGATOR
# ==============================================================================

class DipEventAggregator:
    """Aggregate dip events with continuous feature tracking."""
    
    def __init__(self):
        self.n_tokens = 0
        self.n_tokens_with_events = 0
        self.n_total_events = 0
        
        # Main structure: 
        # [lookback][threshold][min_ttr][latency_bin][prob_bucket] -> edge_cell
        # Also track velocity separately:
        # [lookback][threshold][min_ttr][velocity_bin][prob_bucket] -> edge_cell
        
        self.latency_surface = {}
        self.velocity_surface = {}
        
        for lookback in LOOKBACK_WINDOWS_HOURS:
            self.latency_surface[lookback] = {}
            self.velocity_surface[lookback] = {}
            
            for threshold in MOVE_THRESHOLDS:
                self.latency_surface[lookback][threshold] = {}
                self.velocity_surface[lookback][threshold] = {}
                
                for min_ttr in MIN_TIME_TO_RESOLUTION_HOURS:
                    self.latency_surface[lookback][threshold][min_ttr] = {}
                    self.velocity_surface[lookback][threshold][min_ttr] = {}
                    
                    # Latency bins
                    for _, _, lat_label in LATENCY_BINS_HOURS:
                        self.latency_surface[lookback][threshold][min_ttr][lat_label] = {}
                        for prob_label in PROB_BUCKET_LABELS:
                            self.latency_surface[lookback][threshold][min_ttr][lat_label][prob_label] = _make_edge_cell()
                    
                    # Velocity bins
                    for _, _, vel_label in VELOCITY_BINS_PTS_PER_HOUR:
                        self.velocity_surface[lookback][threshold][min_ttr][vel_label] = {}
                        for prob_label in PROB_BUCKET_LABELS:
                            self.velocity_surface[lookback][threshold][min_ttr][vel_label][prob_label] = _make_edge_cell()
        
        # Track overall statistics
        self.event_counts_by_config = defaultdict(int)
        self.prob_bucket_counts = defaultdict(int)
    
    def add_token_events(self, events_by_config, winner_status):
        """
        Add all dip events for a single token.
        
        events_by_config: dict of (lookback, threshold) -> list of event dicts
        winner_status: bool, whether the token won
        """
        self.n_tokens += 1
        
        if not events_by_config:
            return
        
        has_any_events = False
        
        for (lookback, threshold), events in events_by_config.items():
            for event in events:
                has_any_events = True
                self.n_total_events += 1
                
                # Extract event features
                latency_hours = event['reaction_latency_hours']
                velocity = event['velocity_pts_per_hour']
                time_to_resolution_hours = event['time_to_resolution_hours']
                crossing_price = event['crossing_price']
                local_max_price = event['local_max_price']
                fill_data = event.get('fill_simulation', {})
                
                # Determine bins
                latency_bin = get_latency_bin(latency_hours)
                velocity_bin = get_velocity_bin(velocity)
                prob_bucket = get_prob_bucket(local_max_price)
                
                if latency_bin is None or velocity_bin is None:
                    continue
                
                # Update for each min_ttr filter
                for min_ttr in MIN_TIME_TO_RESOLUTION_HOURS:
                    if time_to_resolution_hours < min_ttr:
                        continue  # Filter out events too close to resolution
                    
                    self.event_counts_by_config[(lookback, threshold, min_ttr)] += 1
                    
                    # Update latency surface
                    buckets_to_update = ['all']
                    if prob_bucket:
                        buckets_to_update.append(prob_bucket)
                        self.prob_bucket_counts[prob_bucket] += 1
                    
                    for bucket in buckets_to_update:
                        cell = self.latency_surface[lookback][threshold][min_ttr][latency_bin][bucket]
                        self._update_cell(cell, winner_status, crossing_price, local_max_price,
                                         latency_hours, velocity, time_to_resolution_hours, fill_data)
                        
                        vel_cell = self.velocity_surface[lookback][threshold][min_ttr][velocity_bin][bucket]
                        self._update_cell(vel_cell, winner_status, crossing_price, local_max_price,
                                         latency_hours, velocity, time_to_resolution_hours, fill_data)
        
        if has_any_events:
            self.n_tokens_with_events += 1
    
    def _update_cell(self, cell, winner, crossing_price, local_max_price,
                     latency_hours, velocity, time_to_resolution_hours, fill_data):
        """Update a single edge cell with event data."""
        cell['n_samples'] += 1
        
        if winner:
            cell['n_wins'] += 1
        
        cell['crossing_prices'].update(crossing_price)
        cell['local_max_prices'].update(local_max_price)
        cell['latencies_hours'].update(latency_hours)
        cell['velocities'].update(velocity)
        cell['time_to_resolution_hours'].update(time_to_resolution_hours)
        
        if fill_data and fill_data.get('filled'):
            cell['n_fills'] += 1
            if winner:
                cell['n_fill_wins'] += 1
            fill_price = fill_data.get('fill_price')
            if fill_price is not None:
                cell['fill_prices'].update(fill_price)
    
    def get_summary(self):
        """Get aggregated summary statistics."""
        # Convert tuple keys to strings for JSON serialization
        event_counts_str_keys = {
            f"{lookback}h_{threshold}_{min_ttr}h": count
            for (lookback, threshold, min_ttr), count in self.event_counts_by_config.items()
        }
        
        summary = {
            'n_tokens': self.n_tokens,
            'n_tokens_with_events': self.n_tokens_with_events,
            'n_total_events': self.n_total_events,
            'event_counts_by_config': event_counts_str_keys,
            'prob_bucket_distribution': dict(self.prob_bucket_counts),
            'latency_surface': {},
            'velocity_surface': {},
        }
        
        # Process latency surface
        for lookback in LOOKBACK_WINDOWS_HOURS:
            summary['latency_surface'][lookback] = {}
            
            for threshold in MOVE_THRESHOLDS:
                summary['latency_surface'][lookback][threshold] = {}
                
                for min_ttr in MIN_TIME_TO_RESOLUTION_HOURS:
                    summary['latency_surface'][lookback][threshold][min_ttr] = {}
                    
                    for _, _, lat_label in LATENCY_BINS_HOURS:
                        summary['latency_surface'][lookback][threshold][min_ttr][lat_label] = {}
                        
                        for prob_bucket in PROB_BUCKET_LABELS:
                            cell = self.latency_surface[lookback][threshold][min_ttr][lat_label][prob_bucket]
                            summary['latency_surface'][lookback][threshold][min_ttr][lat_label][prob_bucket] = \
                                self._summarize_cell(cell)
        
        # Process velocity surface
        for lookback in LOOKBACK_WINDOWS_HOURS:
            summary['velocity_surface'][lookback] = {}
            
            for threshold in MOVE_THRESHOLDS:
                summary['velocity_surface'][lookback][threshold] = {}
                
                for min_ttr in MIN_TIME_TO_RESOLUTION_HOURS:
                    summary['velocity_surface'][lookback][threshold][min_ttr] = {}
                    
                    for _, _, vel_label in VELOCITY_BINS_PTS_PER_HOUR:
                        summary['velocity_surface'][lookback][threshold][min_ttr][vel_label] = {}
                        
                        for prob_bucket in PROB_BUCKET_LABELS:
                            cell = self.velocity_surface[lookback][threshold][min_ttr][vel_label][prob_bucket]
                            summary['velocity_surface'][lookback][threshold][min_ttr][vel_label][prob_bucket] = \
                                self._summarize_cell(cell)
        
        return summary
    
    def _summarize_cell(self, cell):
        """Summarize a single edge cell."""
        n = cell['n_samples']
        n_wins = cell['n_wins']
        n_fills = cell['n_fills']
        n_fill_wins = cell['n_fill_wins']
        
        if n < MIN_SAMPLES_PER_CELL:
            return {
                'n_samples': n,
                'n_fills': n_fills,
                'n_wins': n_wins,
                'n_fill_wins': n_fill_wins,
                'status': 'insufficient_data',
            }
        
        # Unconditional metrics
        uncond_win_rate = n_wins / n if n > 0 else 0
        crossing_stats = cell['crossing_prices'].get_stats()
        avg_crossing_price = crossing_stats['mean'] if crossing_stats else 0
        
        uncond_edge_bps = (uncond_win_rate - avg_crossing_price) * 10000
        se_uncond = np.sqrt(uncond_win_rate * (1 - uncond_win_rate) / n) * 10000 if n > 0 else 0
        
        # Fill rate
        fill_rate = n_fills / n if n > 0 else 0
        
        # Conditional metrics (among filled)
        cond_result = {}
        if n_fills >= 10:
            cond_win_rate = n_fill_wins / n_fills
            fill_stats = cell['fill_prices'].get_stats()
            avg_fill = fill_stats['mean'] if fill_stats else 0
            
            edge_after_fill_bps = (cond_win_rate - avg_fill) * 10000
            se_cond = np.sqrt(cond_win_rate * (1 - cond_win_rate) / n_fills) * 10000 if n_fills > 0 else 0
            
            cond_result = {
                'conditional_win_rate': cond_win_rate,
                'avg_fill_price': avg_fill,
                'edge_after_fill_bps': edge_after_fill_bps,
                'se_edge_after_fill': se_cond,
            }
        
        # Feature statistics
        latency_stats = cell['latencies_hours'].get_stats()
        velocity_stats = cell['velocities'].get_stats()
        ttr_stats = cell['time_to_resolution_hours'].get_stats()
        local_max_stats = cell['local_max_prices'].get_stats()
        
        return {
            'n_samples': n,
            'n_fills': n_fills,
            'n_wins': n_wins,
            'n_fill_wins': n_fill_wins,
            'unconditional_win_rate': uncond_win_rate,
            'avg_crossing_price': avg_crossing_price,
            'unconditional_edge_bps': uncond_edge_bps,
            'se_unconditional_edge': se_uncond,
            'fill_rate': fill_rate,
            'latency_stats': latency_stats,
            'velocity_stats': velocity_stats,
            'time_to_resolution_stats': ttr_stats,
            'local_max_price_stats': local_max_stats,
            **cond_result,
            'status': 'ok',
        }


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
# CHECKPOINT HANDLING
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
# DIP EVENT DETECTION (Core 4E Logic)
# ==============================================================================

def find_dip_events(trades, resolution_time, lookback_hours, threshold):
    """
    Identify dip events using a rolling maximum with lookback window.
    
    A dip event occurs when:
    1. Price was at local maximum within the lookback window
    2. Price subsequently drops by at least 'threshold' points
    
    Returns list of dip event dicts with continuous features.
    """
    if len(trades) < 3:
        return []
    
    # Sort by timestamp
    trades = sorted(trades, key=lambda x: x[0])
    
    lookback_seconds = lookback_hours * 3600
    events = []
    
    # Track which events we've already recorded to avoid duplicates
    recorded_crossings = set()
    
    # Iterate through trades looking for threshold crossings
    for i, (ts, price, size) in enumerate(trades):
        # Only consider trades at least 2h before resolution for meaningful events
        time_to_resolution = resolution_time - ts
        if time_to_resolution < 2 * 3600:  # Skip last 2 hours entirely
            continue
        
        # Find the local maximum within the lookback window ending at this trade
        lookback_start = ts - lookback_seconds
        
        # Get trades in lookback window
        lookback_trades = [(t, p, s) for t, p, s in trades if lookback_start <= t < ts]
        
        if not lookback_trades:
            continue
        
        # Find max price and when it occurred in lookback window
        max_price = max(p for _, p, _ in lookback_trades)
        max_time = max(t for t, p, _ in lookback_trades if p == max_price)
        
        # Check if current price represents a threshold crossing from the local max
        drop = max_price - price
        
        if drop >= threshold:
            # This is a potential dip event
            # But we need to check if this is the FIRST crossing of this threshold
            # from this particular local maximum
            
            # Create a unique key for this local max + threshold combination
            event_key = (max_time, threshold)
            
            if event_key in recorded_crossings:
                continue
            
            recorded_crossings.add(event_key)
            
            # Calculate continuous features
            reaction_latency_seconds = ts - max_time
            reaction_latency_hours = reaction_latency_seconds / 3600
            
            time_to_resolution_hours = time_to_resolution / 3600
            
            # Velocity: how fast the dip happened (points per hour)
            if reaction_latency_hours > 0:
                velocity_pts_per_hour = drop / reaction_latency_hours
            else:
                velocity_pts_per_hour = float('inf')
            
            events.append({
                't_max': max_time,
                't_cross': ts,
                't_resolution': resolution_time,
                'local_max_price': max_price,
                'crossing_price': price,
                'drop_amount': drop,
                'reaction_latency_hours': reaction_latency_hours,
                'time_to_resolution_hours': time_to_resolution_hours,
                'velocity_pts_per_hour': velocity_pts_per_hour,
            })
    
    return events


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


def compute_token_dip_events(trades, resolution_time):
    """
    Compute all dip events for a single token across all lookback/threshold configurations.
    
    Returns dict of (lookback, threshold) -> list of event dicts
    """
    events_by_config = {}
    
    for lookback in LOOKBACK_WINDOWS_HOURS:
        for threshold in MOVE_THRESHOLDS:
            events = find_dip_events(trades, resolution_time, lookback, threshold)
            
            if not events:
                continue
            
            # Add fill simulation to each event
            for event in events:
                fill_result = simulate_limit_order_fill(
                    trades, event['t_cross'], event['crossing_price'], is_buy=True
                )
                event['fill_simulation'] = fill_result
            
            events_by_config[(lookback, threshold)] = events
    
    return events_by_config


# ==============================================================================
# DATA ACCUMULATOR CLASS
# ==============================================================================

class DipEventAccumulator:
    """Accumulates trade data for dip event analysis."""
    
    def __init__(self, token_id, condition_id, resolution_time, winner_status):
        self.token_id = token_id
        self.condition_id = condition_id
        self.resolution_time = resolution_time
        self.winner_status = winner_status
        self.trades = []
    
    def add_trades(self, trades_batch):
        self.trades.extend(trades_batch)
    
    def compute_dip_events(self):
        """Compute dip events for this token."""
        if len(self.trades) < MIN_TRADES:
            return None
        
        self.trades.sort(key=lambda x: x[0])
        
        events_by_config = compute_token_dip_events(self.trades, self.resolution_time)
        
        if not events_by_config:
            return None
        
        return {
            'token_id': self.token_id,
            'condition_id': self.condition_id,
            'winner': self.winner_status,
            'n_trades': len(self.trades),
            'events_by_config': events_by_config,
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
    """Run the event-based dip analysis."""
    global start_time
    start_time = datetime.now()
    
    log("="*70)
    log("PHASE 4E: EVENT-BASED DIP ANALYSIS")
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
    aggregator = DipEventAggregator()
    
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
                f"Tokens: {aggregator.n_tokens:,} | "
                f"Events: {aggregator.n_total_events:,} | "
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
                    
                    token_accumulators[token_id] = DipEventAccumulator(
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
                        'trades': trades_batch[:500],
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
                            dip_result = acc.compute_dip_events()
                            
                            if dip_result is not None:
                                aggregator.add_token_events(
                                    dip_result['events_by_config'],
                                    dip_result['winner']
                                )
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
                dip_result = acc.compute_dip_events()
                
                if dip_result is not None:
                    aggregator.add_token_events(
                        dip_result['events_by_config'],
                        dip_result['winner']
                    )
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
    log(f"  Tokens filtered (low trades/no events): {stats['tokens_filtered']:,}")
    log(f"  Final token count: {aggregator.n_tokens:,}")
    log(f"  Tokens with events: {aggregator.n_tokens_with_events:,}")
    log(f"  Total dip events: {aggregator.n_total_events:,}")
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
            
            events_by_config = compute_token_dip_events(trades, resolution_time)
            
            total_events = sum(len(events) for events in events_by_config.values())
            log(f"  Total dip events detected: {total_events}")
            
            for (lookback, threshold), events in sorted(events_by_config.items()):
                log(f"\n  Lookback={lookback}h, Threshold={threshold*100:.0f}%: {len(events)} events")
                
                for i, event in enumerate(events[:3]):  # Show first 3 events
                    log(f"    Event {i+1}:")
                    log(f"      Local max price: {event['local_max_price']:.4f}")
                    log(f"      Crossing price: {event['crossing_price']:.4f}")
                    log(f"      Drop: {event['drop_amount']*100:.1f}%")
                    log(f"      Reaction latency: {event['reaction_latency_hours']:.2f}h")
                    log(f"      Time to resolution: {event['time_to_resolution_hours']:.1f}h")
                    log(f"      Velocity: {event['velocity_pts_per_hour']*100:.1f}%/hour")
                    
                    fill = event.get('fill_simulation', {})
                    if fill.get('filled'):
                        log(f"      Fill: YES @ {fill['fill_price']:.4f}")
                    else:
                        log(f"      Fill: NO")
    
    return aggregator


# ==============================================================================
# REPORTING
# ==============================================================================

def print_latency_analysis(summary):
    """Print edge as a function of reaction latency."""
    log("\n" + "="*70)
    log("EDGE BY REACTION LATENCY (Core Finding)")
    log("="*70)
    log("This shows whether reacting faster to dips improves edge.")
    log("Key question: Is there a minimum latency below which edge degrades?")
    
    # Use reference configuration: 6h lookback, 15% threshold, 4h min TTR
    ref_lookback = 6
    ref_threshold = 0.15
    ref_min_ttr = 4
    
    log(f"\nReference config: lookback={ref_lookback}h, threshold={ref_threshold*100:.0f}%, min_TTR={ref_min_ttr}h")
    log(f"\n{'Latency Bin':<15} | {'n':<8} | {'Win Rate':<10} | {'Edge (bps)':<15} | {'Avg Velocity':<15}")
    log("-"*75)
    
    for _, _, lat_label in LATENCY_BINS_HOURS:
        cell = summary['latency_surface'].get(ref_lookback, {}).get(ref_threshold, {}).get(ref_min_ttr, {}).get(lat_label, {}).get('all', {})
        
        if cell.get('status') != 'ok':
            n = cell.get('n_samples', 0)
            log(f"{lat_label:<15} | {n:<8} | {'(insufficient data)':<45}")
            continue
        
        n = cell['n_samples']
        win_rate = cell.get('conditional_win_rate', cell.get('unconditional_win_rate', 0))
        edge = cell.get('edge_after_fill_bps', cell.get('unconditional_edge_bps', 0))
        se = cell.get('se_edge_after_fill', cell.get('se_unconditional_edge', 0))
        
        vel_stats = cell.get('velocity_stats', {})
        avg_vel = vel_stats.get('mean', 0) * 100 if vel_stats else 0
        
        log(f"{lat_label:<15} | {n:<8,} | {win_rate:.1%}      | {edge:+.0f}±{se:.0f}         | {avg_vel:.1f}%/hour")
    
    return summary


def print_velocity_analysis(summary):
    """Print edge as a function of dip velocity."""
    log("\n" + "="*70)
    log("EDGE BY DIP VELOCITY")
    log("="*70)
    log("This shows whether fast dips (potentially informed flow) differ from slow dips.")
    log("Key question: Does velocity discriminate between noise and informed trading?")
    
    ref_lookback = 6
    ref_threshold = 0.15
    ref_min_ttr = 4
    
    log(f"\nReference config: lookback={ref_lookback}h, threshold={ref_threshold*100:.0f}%, min_TTR={ref_min_ttr}h")
    log(f"\n{'Velocity Bin':<15} | {'n':<8} | {'Win Rate':<10} | {'Edge (bps)':<15} | {'Avg Latency':<15}")
    log("-"*75)
    
    for _, _, vel_label in VELOCITY_BINS_PTS_PER_HOUR:
        cell = summary['velocity_surface'].get(ref_lookback, {}).get(ref_threshold, {}).get(ref_min_ttr, {}).get(vel_label, {}).get('all', {})
        
        if cell.get('status') != 'ok':
            n = cell.get('n_samples', 0)
            log(f"{vel_label:<15} | {n:<8} | {'(insufficient data)':<45}")
            continue
        
        n = cell['n_samples']
        win_rate = cell.get('conditional_win_rate', cell.get('unconditional_win_rate', 0))
        edge = cell.get('edge_after_fill_bps', cell.get('unconditional_edge_bps', 0))
        se = cell.get('se_edge_after_fill', cell.get('se_unconditional_edge', 0))
        
        lat_stats = cell.get('latency_stats', {})
        avg_lat = lat_stats.get('mean', 0) if lat_stats else 0
        
        log(f"{vel_label:<15} | {n:<8,} | {win_rate:.1%}      | {edge:+.0f}±{se:.0f}         | {avg_lat:.2f}h")
    
    return summary


def print_ttr_boundary_analysis(summary):
    """Analyze edge across different time-to-resolution boundaries."""
    log("\n" + "="*70)
    log("TIME-TO-RESOLUTION BOUNDARY ANALYSIS")
    log("="*70)
    log("Confirming/refining the ≥4h boundary from 4D analysis.")
    log("Key question: What's the safe minimum time-to-resolution?")
    
    ref_lookback = 6
    ref_threshold = 0.15
    
    log(f"\nReference config: lookback={ref_lookback}h, threshold={ref_threshold*100:.0f}%")
    log(f"\n{'Min TTR':<10} | {'Total Events':<15} | {'Avg Edge (early latency)':<25}")
    log("-"*60)
    
    for min_ttr in MIN_TIME_TO_RESOLUTION_HOURS:
        # Get edge for early latency bin (0-15min)
        cell = summary['latency_surface'].get(ref_lookback, {}).get(ref_threshold, {}).get(min_ttr, {}).get('0-15min', {}).get('all', {})
        
        total_events = sum(
            summary['latency_surface'].get(ref_lookback, {}).get(ref_threshold, {}).get(min_ttr, {}).get(lat, {}).get('all', {}).get('n_samples', 0)
            for _, _, lat in LATENCY_BINS_HOURS
        )
        
        if cell.get('status') != 'ok':
            log(f"{min_ttr}h        | {total_events:<15,} | insufficient data")
            continue
        
        edge = cell.get('edge_after_fill_bps', cell.get('unconditional_edge_bps', 0))
        se = cell.get('se_edge_after_fill', cell.get('se_unconditional_edge', 0))
        
        log(f"{min_ttr}h        | {total_events:<15,} | {edge:+.0f}±{se:.0f} bps")


def print_prob_bucket_analysis(summary):
    """Print edge stratified by probability bucket."""
    log("\n" + "="*70)
    log("EDGE BY PROBABILITY BUCKET (Early Latency)")
    log("="*70)
    log("Where does edge concentrate? Toss-up markets or high-probability?")
    
    ref_lookback = 6
    ref_threshold = 0.15
    ref_min_ttr = 4
    
    log(f"\nReference config: lookback={ref_lookback}h, threshold={ref_threshold*100:.0f}%, min_TTR={ref_min_ttr}h")
    log(f"\n{'Prob Bucket':<12} | {'n':<8} | {'Edge (bps)':<15}")
    log("-"*45)
    
    for prob_bucket in PROB_BUCKET_LABELS:
        # Aggregate across early latency bins
        edges = []
        total_n = 0
        
        for lat_lo, lat_hi, lat_label in LATENCY_BINS_HOURS[:3]:  # 0-15min, 15-30min, 30min-1h
            cell = summary['latency_surface'].get(ref_lookback, {}).get(ref_threshold, {}).get(ref_min_ttr, {}).get(lat_label, {}).get(prob_bucket, {})
            
            if cell.get('status') == 'ok':
                n = cell.get('n_samples', 0)
                edge = cell.get('edge_after_fill_bps', cell.get('unconditional_edge_bps', 0))
                total_n += n
                if n >= 10:
                    edges.append((edge, n))
        
        if edges:
            # Weighted average
            total_weight = sum(n for _, n in edges)
            avg_edge = sum(e * n for e, n in edges) / total_weight if total_weight > 0 else 0
            log(f"{prob_bucket:<12} | {total_n:<8,} | {avg_edge:+.0f}")
        else:
            log(f"{prob_bucket:<12} | {total_n:<8,} | insufficient data")


def print_key_findings(summary):
    """Print key findings and operational implications."""
    log("\n" + "="*70)
    log("KEY FINDINGS & OPERATIONAL IMPLICATIONS")
    log("="*70)
    
    ref_lookback = 6
    ref_threshold = 0.15
    ref_min_ttr = 4
    
    # 1. Optimal reaction latency
    log("\n1. OPTIMAL REACTION LATENCY")
    log("-"*50)
    
    latency_edges = []
    for _, _, lat_label in LATENCY_BINS_HOURS:
        cell = summary['latency_surface'].get(ref_lookback, {}).get(ref_threshold, {}).get(ref_min_ttr, {}).get(lat_label, {}).get('all', {})
        if cell.get('status') == 'ok':
            edge = cell.get('edge_after_fill_bps', 0)
            n = cell.get('n_samples', 0)
            latency_edges.append((lat_label, edge, n))
    
    if latency_edges:
        best = max(latency_edges, key=lambda x: x[1])
        worst = min(latency_edges, key=lambda x: x[1])
        
        log(f"  Best latency bin: {best[0]} ({best[1]:+.0f} bps, n={best[2]:,})")
        log(f"  Worst latency bin: {worst[0]} ({worst[1]:+.0f} bps, n={worst[2]:,})")
        
        # Check if early is better
        early_bins = [x for x in latency_edges if x[0] in ['0-15min', '15-30min', '30min-1h']]
        late_bins = [x for x in latency_edges if x[0] in ['2-4h', '4-8h', '8h+']]
        
        if early_bins and late_bins:
            early_avg = np.mean([x[1] for x in early_bins])
            late_avg = np.mean([x[1] for x in late_bins])
            
            if early_avg > late_avg + 100:
                log(f"\n  FINDING: Strong advantage to reacting early ({early_avg:+.0f} vs {late_avg:+.0f} bps)")
            elif early_avg > late_avg + 20:
                log(f"\n  FINDING: Moderate advantage to reacting early ({early_avg:+.0f} vs {late_avg:+.0f} bps)")
            elif late_avg > early_avg + 20:
                log(f"\n  FINDING: UNEXPECTED - Late reaction shows better edge ({late_avg:+.0f} vs {early_avg:+.0f} bps)")
            else:
                log(f"\n  FINDING: Reaction speed has minimal impact ({early_avg:+.0f} vs {late_avg:+.0f} bps)")
    
    # 2. Velocity discrimination
    log("\n2. VELOCITY DISCRIMINATION")
    log("-"*50)
    
    velocity_edges = []
    for _, _, vel_label in VELOCITY_BINS_PTS_PER_HOUR:
        cell = summary['velocity_surface'].get(ref_lookback, {}).get(ref_threshold, {}).get(ref_min_ttr, {}).get(vel_label, {}).get('all', {})
        if cell.get('status') == 'ok':
            edge = cell.get('edge_after_fill_bps', 0)
            n = cell.get('n_samples', 0)
            velocity_edges.append((vel_label, edge, n))
    
    if velocity_edges:
        slow_bins = [x for x in velocity_edges if 'slow' in x[0]]
        fast_bins = [x for x in velocity_edges if 'fast' in x[0]]
        
        if slow_bins and fast_bins:
            slow_avg = np.mean([x[1] for x in slow_bins])
            fast_avg = np.mean([x[1] for x in fast_bins])
            
            if slow_avg > fast_avg + 100:
                log(f"  FINDING: Slow dips show better edge ({slow_avg:+.0f} vs {fast_avg:+.0f} bps)")
                log(f"           Fast dips may be informed flow - consider filtering.")
            elif fast_avg > slow_avg + 100:
                log(f"  FINDING: Fast dips show better edge ({fast_avg:+.0f} vs {slow_avg:+.0f} bps)")
                log(f"           Velocity does not discriminate informed flow.")
            else:
                log(f"  FINDING: Velocity has minimal discriminatory power ({slow_avg:+.0f} vs {fast_avg:+.0f} bps)")
    
    # 3. Time-to-resolution boundary
    log("\n3. TIME-TO-RESOLUTION BOUNDARY")
    log("-"*50)
    
    ttr_edges = []
    for min_ttr in MIN_TIME_TO_RESOLUTION_HOURS:
        cell = summary['latency_surface'].get(ref_lookback, {}).get(ref_threshold, {}).get(min_ttr, {}).get('0-15min', {}).get('all', {})
        if cell.get('status') == 'ok':
            edge = cell.get('edge_after_fill_bps', 0)
            n = cell.get('n_samples', 0)
            ttr_edges.append((min_ttr, edge, n))
    
    if ttr_edges:
        log(f"  Edge by min time-to-resolution:")
        for ttr, edge, n in ttr_edges:
            marker = "←" if edge > 0 and n > 100 else ""
            log(f"    ≥{ttr}h: {edge:+.0f} bps (n={n:,}) {marker}")
        
        positive_edges = [(ttr, e) for ttr, e, n in ttr_edges if e > 50 and n > 100]
        if positive_edges:
            min_safe_ttr = min(ttr for ttr, _ in positive_edges)
            log(f"\n  FINDING: Safe minimum TTR appears to be ≥{min_safe_ttr}h")


def save_results_json(summary):
    """Save results to JSON file."""
    ensure_output_dir()
    
    output = {
        'timestamp': TIMESTAMP,
        'phase': '4E',
        'description': 'Event-Based Dip Analysis',
        'methodology': 'Rolling maximum dip detection with continuous features',
        'lookback_windows_hours': LOOKBACK_WINDOWS_HOURS,
        'move_thresholds': MOVE_THRESHOLDS,
        'min_time_to_resolution_hours': MIN_TIME_TO_RESOLUTION_HOURS,
        'latency_bins': [(lo, hi, label) for lo, hi, label in LATENCY_BINS_HOURS],
        'velocity_bins': [(lo, hi, label) for lo, hi, label in VELOCITY_BINS_PTS_PER_HOUR],
        'prob_buckets': PROB_BUCKETS,
        **summary,
    }
    
    json_path = os.path.join(OUTPUT_DIR, f'phase4e_event_based_{TIMESTAMP}.json')
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    log(f"\nResults JSON saved: {json_path}")
    return json_path


def generate_report(summary):
    """Generate text report."""
    ensure_output_dir()
    
    report_path = os.path.join(OUTPUT_DIR, f'phase4e_event_based_{TIMESTAMP}_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 4E: EVENT-BASED DIP ANALYSIS - REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Tokens Analyzed: {summary.get('n_tokens', 0):,}\n")
        f.write(f"Tokens with Events: {summary.get('n_tokens_with_events', 0):,}\n")
        f.write(f"Total Dip Events: {summary.get('n_total_events', 0):,}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("-"*80 + "\n")
        f.write("This analysis shifts from fixed-window tercile categorization to\n")
        f.write("event-based dip detection with continuous parameterization.\n\n")
        f.write("A 'dip event' is identified when price drops from a rolling maximum\n")
        f.write("by at least the threshold amount.\n\n")
        
        f.write("Key parameters:\n")
        f.write(f"  Lookback windows: {LOOKBACK_WINDOWS_HOURS} hours\n")
        f.write(f"  Thresholds: {[t*100 for t in MOVE_THRESHOLDS]}%\n")
        f.write(f"  Min TTR filters: {MIN_TIME_TO_RESOLUTION_HOURS} hours\n\n")
        
        f.write("-"*80 + "\n")
        f.write("See JSON file for full data.\n")
    
    log(f"Report saved: {report_path}")
    return report_path


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 4E: Event-Based Dip Analysis (Memory-Safe)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnostic mode with small sample
  python phase4e_event_based_dips.py --diagnostic --sample 100
  
  # Full run
  python phase4e_event_based_dips.py
  
  # Resume interrupted run
  python phase4e_event_based_dips.py --resume
        """
    )
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Process only first N batch files')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint if available')
    parser.add_argument('--diagnostic', '-d', action='store_true',
                        help='Run in diagnostic mode with extra validation')
    
    args = parser.parse_args()
    
    log("Starting Phase 4E Event-Based Dip Analysis...")
    
    ensure_output_dir()
    
    aggregator = run_analysis(
        sample_files=args.sample,
        resume=args.resume,
        diagnostic=args.diagnostic
    )
    
    if aggregator:
        summary = aggregator.get_summary()
        
        print_latency_analysis(summary)
        print_velocity_analysis(summary)
        print_ttr_boundary_analysis(summary)
        print_prob_bucket_analysis(summary)
        print_key_findings(summary)
        
        generate_report(summary)
        save_results_json(summary)
        
        log("\n" + "="*70)
        log("ANALYSIS COMPLETE")
        log("="*70)
        log(f"\nOutputs generated:")
        log(f"  - Report: phase4e_event_based_{TIMESTAMP}_report.txt")
        log(f"  - JSON:   phase4e_event_based_{TIMESTAMP}.json")
    else:
        log("\nAnalysis failed - no results generated")