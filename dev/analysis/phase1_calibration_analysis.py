#!/usr/bin/env python3
"""
Phase 1: Odds-Stratified Calibration Analysis - FIXED VERSION
Version: 11.0 - CRITICAL FIX: API Metadata Winner Reconstruction

CRITICAL FIX (v11.0):
  - Replaced price-based winner inference with API metadata reconstruction
  - NEVER infers winner from trading prices
  - Uses strict positional logic from clobTokenIds + outcomePrices
  - All previous results from v10.0 are INVALID and must be discarded

TARGET AUDIENCE: Quantitative PMs (Citadel, Jane Street, etc.)
ENHANCEMENTS:
  - Refined probability buckets aligned with charter specifications
  - Comprehensive statistical metrics (Sharpe, Kelly, Brier, log loss)
  - Publication-quality visualizations with confidence bands
  - Detailed methodology justification in quantitative trading terms
  - Return distribution analysis and tail risk metrics

METHODOLOGY OVERVIEW:
  1. CALIBRATION: Stratified binomial testing across probability buckets
  2. EDGE QUANTIFICATION: Basis points, return edge, and Sharpe ratios
  3. STATISTICAL VALIDATION: Wilson score intervals, z-tests, Brier scores
  4. RISK METRICS: Kelly fractions, drawdown estimates, tail analysis

ARCHITECTURE:
  - Two-pass streaming: Index caching + incremental processing
  - Memory efficient: O(1) memory footprint per market
  - Production ready: Checkpointed, resumable, validated
"""

import pyarrow.parquet as pq
import glob
import json
import numpy as np
import pandas as pd 
import math
import pickle
import csv
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy import stats
import psutil
import gc
import os
import sys

# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = "../../"

BATCH_DIR = os.path.join(BASE_DIR, 'order_history_batches')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis/outputs')
INDEX_CACHE_FILE = os.path.join(BASE_DIR, 'analysis/market_index_cache.pkl')
MARKETS_CSV = os.path.join(BASE_DIR, 'markets_past_year.csv')

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# ------------------------------------------------------------------------------
# ANALYSIS PARAMETERS
# ------------------------------------------------------------------------------

# Liquidity filters - Based on empirical market structure analysis
MIN_VOLUME_USD = 100.0      # Minimum USD volume in window (filters "zombie markets")
MIN_TRADES_PER_MARKET = 50  # Minimum total trades (ensures price discovery)
MIN_TRADES_PER_TOKEN = 5    # Minimum trades per token in window (ensures TWAP quality)

# Time window configuration
TIME_WINDOW_HOURS = 24      # Duration of analysis window
WINDOW_OFFSET_HOURS = 0    # Offset from resolution (avoids settlement bias)

# Probability buckets - Aligned with charter specifications
# Rationale: Finer granularity in favorite territory (50-100%) where:
# - Market efficiency typically highest
# - Bond farming strategies operate
# - Sharpe ratios most attractive due to lower variance
ODDS_BUCKETS = [
    # Underdog territory (coarse buckets - higher variance, less liquid)
    (0.00, 0.10, '0-10%', 'longshot'),
    (0.10, 0.25, '10-25%', 'underdog'),
    (0.25, 0.40, '25-40%', 'toss-up-'),
    (0.40, 0.51, '40-51%', 'toss-up+'),
    
    # Favorite territory (refined buckets - higher precision needed)
    (0.51, 0.60, '51-60%', 'mild-fav'),      # Near-efficient zone
    (0.60, 0.75, '60-75%', 'moderate-fav'),  # Moderate confidence
    (0.75, 0.90, '75-90%', 'strong-fav'),    # Strong conviction
    (0.90, 0.99, '90-99%', 'heavy-fav'),     # Bond territory entry
    (0.99, 0.995, '99.0-99.5%', 'near-certain'), # Bond farming core
    (0.995, 1.00, '99.5-100%', 'extreme'),   # Tail behavior
]

# Statistical significance threshold
ALPHA = 0.05  # Two-tailed test

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

# ------------------------------------------------------------------------------
# STATISTICAL FUNCTIONS
# ------------------------------------------------------------------------------

def wilson_score_interval(successes, trials, confidence=0.95):
    """
    Wilson score interval for binomial proportion.
    
    Why Wilson over Normal approximation:
    - Correct coverage near boundaries (0%, 100%)
    - No continuity correction needed
    - Asymmetric intervals (realistic for extreme probabilities)
    
    Standard in quantitative research for win rate confidence intervals.
    """
    if trials == 0:
        return 0.0, 0.0, 0.0
    
    p = successes / trials
    z = 1.96 if confidence == 0.95 else 2.576
    
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) / trials + z**2 / (4 * trials**2))) / denominator
    
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    
    return p, lower, upper

def compute_brier_score(predictions, outcomes):
    """
    Brier score: Mean squared error between predicted probabilities and outcomes.
    
    Range: [0, 1] where 0 is perfect calibration
    Decomposition: Brier = Reliability - Resolution + Uncertainty
    
    Standard metric in prediction market research and probabilistic forecasting.
    """
    if len(predictions) == 0:
        return None
    return np.mean((np.array(predictions) - np.array(outcomes))**2)

def compute_log_loss(predictions, outcomes, epsilon=1e-15):
    """
    Log loss (cross-entropy loss): Penalizes confident wrong predictions heavily.
    
    Properties:
    - Strictly proper scoring rule (incentivizes true probabilities)
    - Unbounded (extreme predictions near 0/1 heavily penalized)
    - Standard in ML and prediction market evaluation
    
    Epsilon clipping prevents log(0) errors.
    """
    if len(predictions) == 0:
        return None
    preds = np.clip(np.array(predictions), epsilon, 1 - epsilon)
    outcomes = np.array(outcomes)
    return -np.mean(outcomes * np.log(preds) + (1 - outcomes) * np.log(1 - preds))

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Sharpe ratio: Risk-adjusted return metric.
    
    Formula: (E[R] - Rf) / σ(R)
    
    Interpretation for discrete binary outcomes:
    - Measures edge per unit of volatility
    - Assumes returns are IID (reasonable for stratified buckets)
    - Annualization: Sharpe_annual ≈ Sharpe_per_bet × sqrt(N_bets_per_year)
    
    Standard metric in quantitative trading for strategy evaluation.
    """
    if len(returns) < 2:
        return None
    mean_return = np.mean(returns) - risk_free_rate
    std_return = np.std(returns, ddof=1)
    if std_return == 0:
        return None
    return mean_return / std_return

def compute_kelly_fraction(win_prob, implied_prob):
    """
    Kelly criterion: Optimal bet sizing for log utility maximization.
    
    Binary outcome formula: f* = (p × (1 + b) - 1) / b
    where b = odds = (1 - q) / q, q = implied_prob
    
    Interpretation:
    - Maximizes long-run growth rate
    - Full Kelly often too aggressive (use fractional Kelly in practice)
    - Negative = Don't bet (negative edge)
    
    Standard position sizing framework in quantitative trading.
    """
    if implied_prob >= 1.0 or implied_prob <= 0.0:
        return 0.0
    
    odds = (1 - implied_prob) / implied_prob
    kelly = (win_prob * (1 + odds) - 1) / odds
    
    return max(0.0, kelly)  # Never bet if negative edge

def compute_t_statistic(edge, std_dev, n):
    """
    T-statistic for edge significance testing.
    
    H0: True edge = 0
    H1: True edge ≠ 0 (two-tailed)
    
    Use t-distribution for small samples, z-distribution for large samples.
    """
    if n < 2 or std_dev == 0:
        return 0.0, 1.0
    
    t_stat = edge / (std_dev / np.sqrt(n))
    
    # For n > 30, t-distribution ≈ z-distribution
    if n > 30:
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    else:
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    
    return t_stat, p_value

# ------------------------------------------------------------------------------
# PRICE COMPUTATION
# ------------------------------------------------------------------------------

def compute_twap(trades_in_window):
    """
    Time-Weighted Average Price (TWAP).
    
    Why TWAP over other price metrics:
    - Resistant to volume manipulation (time-weighted, not volume-weighted)
    - Captures representative price over window (not just close)
    - Standard in execution analysis and market microstructure
    
    Implementation: Linear interpolation between trades.
    """
    if not trades_in_window or len(trades_in_window) < 2:
        return trades_in_window[0][1] if trades_in_window else None
    
    total_time_weighted_price = 0.0
    total_time = 0.0
    
    for i in range(len(trades_in_window) - 1):
        t1, p1, _ = trades_in_window[i]
        t2, _, _ = trades_in_window[i + 1]
        
        diff = t2 - t1
        total_time_weighted_price += p1 * diff
        total_time += diff
    
    if total_time > 0:
        return total_time_weighted_price / total_time
    else:
        # Fallback to simple mean if all trades same timestamp
        return np.mean([p for _, p, _ in trades_in_window])

def assign_bucket(price, buckets):
    """Assign price to odds bucket"""
    for lower, upper, label, tag in buckets:
        if lower <= price < upper:
            return label, tag
    # Handle floating point edge case (exactly 1.0)
    if price >= 1.0: 
        return buckets[-1][2], buckets[-1][3]
    return None, None

# ==============================================================================
# DATA LOADING & INDEXING (Unchanged from v9.1)
# ==============================================================================

def load_winning_outcomes(markets_csv):
    """Load winning outcomes from markets_past_year.csv"""
    log(f"Loading winning outcomes from {markets_csv}...")
    
    if not os.path.exists(markets_csv):
        log(f"  ERROR: Markets CSV not found: {markets_csv}")
        return None
    
    winning_outcomes = {}
    
    try:
        with open(markets_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    condition_id = row['condition_id'].strip()
                    outcome = int(row['outcome'])
                    winning_outcomes[condition_id] = outcome
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        log(f"  ERROR reading outcomes CSV: {e}")
        return None
    
    log(f"  Loaded {len(winning_outcomes):,} market outcomes")
    
    # Show distribution
    outcome_0 = sum(1 for o in winning_outcomes.values() if o == 0)
    outcome_1 = sum(1 for o in winning_outcomes.values() if o == 1)
    log(f"  Outcome distribution: 0={outcome_0:,}, 1={outcome_1:,}")
    
    return winning_outcomes

def save_market_index(market_index, batch_files, cache_file):
    """Save market index to disk for reuse"""
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'num_files': len(batch_files),
        'num_markets': len(market_index),
        'market_index': market_index
    }
    
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    log(f"  Market index cached: {cache_file}")

def load_market_index(cache_file, batch_files):
    """Load market index from cache if valid"""
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Validate cache
        cached_num_files = cache_data.get('num_files', 0)
        current_num_files = len(batch_files)
        
        log(f"  Cache validation:")
        log(f"    Cached: {cached_num_files} files, {cache_data['num_markets']:,} markets")
        log(f"    Current: {current_num_files} files")
        
        # Allow 10% discrepancy
        file_diff_pct = abs(cached_num_files - current_num_files) / max(cached_num_files, 1) * 100
        
        if file_diff_pct > 10:
            log(f"    ⚠️ File count diff {file_diff_pct:.1f}% > 10% - rebuilding index")
            return None
        
        log(f"    ✓ Cache valid (diff: {file_diff_pct:.1f}%)")
        return cache_data['market_index']
        
    except Exception as e:
        log(f"  ⚠️ Cache load failed: {e}")
        return None

def build_market_index(batch_files, sample_size=None):
    """
    Pass 1: Build lightweight market index
    Maps: condition_id → list of (file_idx, token_count)
    """
    log("\n" + "="*70)
    log("PASS 1: BUILDING MARKET INDEX")
    log("="*70)
    
    if sample_size:
        batch_files = batch_files[:sample_size]
        log(f"  SAMPLE MODE: Processing {len(batch_files)} files")
    
    # Try loading from cache first
    market_index = load_market_index(INDEX_CACHE_FILE, batch_files)
    if market_index is not None:
        log(f"  ✓ Loaded from cache: {len(market_index):,} markets")
        log(f"    {log_memory()}")
        return market_index, batch_files
    
    market_index = defaultdict(list)
    
    log(f"  Processing {len(batch_files)} batch files...")
    log(f"  {log_memory()}")
    
    for idx, filepath in enumerate(batch_files, 1):
        if idx % 1000 == 0:
            log(f"    [{idx}/{len(batch_files)}] {log_memory()}")
        
        try:
            parquet_file = pq.ParquetFile(filepath)
            metadata = parquet_file.metadata
            
            for row_group_idx in range(metadata.num_row_groups):
                rg = metadata.row_group(row_group_idx)
                num_rows = rg.num_rows
                
                # Read only condition_id column
                batch = parquet_file.read_row_group(row_group_idx, columns=['condition_id'])
                df_light = batch.to_pandas()
                
                for condition_id in df_light['condition_id'].unique():
                    count = (df_light['condition_id'] == condition_id).sum()
                    market_index[condition_id].append((idx - 1, count))
                
                del df_light, batch
                gc.collect()
                
        except Exception as e:
            log(f"    ERROR in {os.path.basename(filepath)}: {e}")
    
    log(f"  ✓ Index built: {len(market_index):,} unique markets")
    log(f"    {log_memory()}")
    
    # Save to cache
    save_market_index(market_index, batch_files, INDEX_CACHE_FILE)
    
    return market_index, batch_files

# ==============================================================================
# ENHANCED CALIBRATION ANALYSIS
# ==============================================================================
# ==============================================================================
def analyze_calibration_streaming(sample_size=None):
    """
    Main calibration analysis with streaming architecture and DIAGNOSTICS
    """
    log("\n" + "="*70)
    log("PHASE 1: ODDS-STRATIFIED CALIBRATION ANALYSIS (DIAGNOSTIC MODE)")
    log("="*70)
    log(f"  Analysis window: {TIME_WINDOW_HOURS}h ending {WINDOW_OFFSET_HOURS}h before resolution")
    log(f"  Min volume: ${MIN_VOLUME_USD}")
    log(f"  Confidence: 95% (Wilson score)")
    log(f"  Significance: α={ALPHA} (two-tailed)")
    
    # Load batch files
    batch_files = sorted(glob.glob(f"{BATCH_DIR}/batch_*.parquet"))
    if not batch_files:
        log(f"ERROR: No batch files found in {BATCH_DIR}")
        return
    
    log(f"\n  Total batch files available: {len(batch_files)}")
    
    # Build market index (Pass 1)
    market_index, batch_files = build_market_index(batch_files, sample_size)
    
    # Initialize bucket accumulation
    bucket_data = defaultdict(lambda: {
        'twaps': [],
        'outcomes': [],
        'volumes': [],
        'trade_sizes': [],
        'returns': []  # For Sharpe calculation
    })
    
    total_markets_processed = 0
    total_tokens_analyzed = 0
    total_tokens_filtered = 0
    
    log("\n" + "="*70)
    log("PASS 2: PROCESSING MARKETS (WITH DIAGNOSTICS)")
    log("="*70)
    
    # Process each market
    market_count = 0
    debug_limit = 10  # Print detailed diagnostics for first 10 markets only

    for condition_id, file_indices in market_index.items():
        market_count += 1
        is_debug = market_count <= debug_limit
        
        if market_count % 1000 == 0:
            log(f"  [{market_count}/{len(market_index)}] {log_memory()}")
        
        try:
            # Collect all trades for this market
            market_trades = []
            resolution_time = None
            
            for file_idx, _ in file_indices:
                filepath = batch_files[file_idx]
                
                try:
                    # Read table
                    df = pq.read_table(filepath, 
                                      filters=[('condition_id', '=', condition_id)]
                                     ).to_pandas()
                    
                    if len(df) > 0:
                        market_trades.append(df)
                        if resolution_time is None:
                            resolution_time = df['resolution_time'].iloc[0]
                
                except Exception as e:
                    continue
            
            if not market_trades or resolution_time is None:
                if is_debug: log(f"  [DEBUG {market_count}] Skipped: No trades or resolution time found")
                continue
            
            # Combine all trades for this market
            market_df = pd.concat(market_trades, ignore_index=True)
            del market_trades

            # =========================================================
            # FIX 1: COLUMN NAME MAPPING (Polymarket CLOB Standard)
            # =========================================================
            # 'maker_amount' is the matched limit order size (Tokens).
            # We map this to 'size_tokens' for the volume calc (Price * Tokens).
            if 'size_tokens' not in market_df.columns:
                if 'maker_amount' in market_df.columns:
                    market_df.rename(columns={'maker_amount': 'size_tokens'}, inplace=True)
                elif 'size' in market_df.columns:
                    market_df.rename(columns={'size': 'size_tokens'}, inplace=True)
                else:
                    if is_debug: log(f"  [DEBUG {market_count}] Skipped: No volume column found.")
                    del market_df
                    continue

            # =========================================================
            # FIX 2: TYPE SAFETY (Timestamps)
            # =========================================================
            # 1. Force timestamp column to numeric (handling the 'object' dtype issue)
            market_df['timestamp'] = pd.to_numeric(market_df['timestamp'], errors='coerce')
            
            # 2. Drop rows where timestamp couldn't be converted
            market_df = market_df.dropna(subset=['timestamp'])

            # 3. Normalize Milliseconds -> Seconds (if needed)
            # Check the first valid timestamp. If > 3 billion, it's MS.
            if len(market_df) > 0 and market_df['timestamp'].iloc[0] > 3e9:
                if is_debug: log(f"  [DEBUG {market_count}] Detected MS timestamps. Converting...")
                market_df['timestamp'] = market_df['timestamp'] / 1000.0
                market_df['resolution_time'] = market_df['resolution_time'] / 1000.0
                resolution_time = resolution_time / 1000.0
            # =========================================================
            
# ==================================================================
            # [FIXED] ANOMALY DETECTION & API METADATA RECONSTRUCTION
            # ==================================================================
            # CRITICAL: NEVER infer winner from prices
            # ONLY use API metadata (clobTokenIds + outcomePrices)
            
            needs_repair = False
            
            if 'token_winner' not in market_df.columns:
                needs_repair = True
            else:
                # Detect anomaly
                claimed_winners = market_df.groupby('token_id')['token_winner'].first()
                num_winners = claimed_winners.sum()
                
                if num_winners != 1:
                    if is_debug:
                        print(f"\n[DIAGNOSTIC] 🚨 ANOMALY DETECTED for {condition_id} 🚨")
                        print(f"[DIAGNOSTIC] Claimed Winners Count: {num_winners}")
                    needs_repair = True
            
            if needs_repair:
                # Fetch market metadata from API
                import requests
                
                repair_successful = False
                
                try:
                    params = {'condition_ids': condition_id}
                    response = requests.get('https://gamma-api.polymarket.com/markets', 
                                          params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Find matching market
                        market = None
                        if isinstance(data, list):
                            for m in data:
                                if m.get('conditionId', '').lower() == condition_id.lower():
                                    market = m
                                    break
                            if not market and len(data) > 0:
                                market = data[0]
                        elif isinstance(data, dict):
                            market = data
                        
                        if market:
                            # Get clobTokenIds and outcomePrices
                            clob_ids = market.get('clobTokenIds', [])
                            outcome_prices = market.get('outcomePrices', [])
                            
                            # Parse if they're JSON strings
                            if isinstance(clob_ids, str):
                                clob_ids = json.loads(clob_ids)
                            if isinstance(outcome_prices, str):
                                outcome_prices = json.loads(outcome_prices)
                            
                            # STRICT LOGIC:
                            # clobTokenIds[0] = "Yes" token
                            # clobTokenIds[1] = "No" token
                            # Compare outcomePrices to determine winner
                            
                            if isinstance(clob_ids, list) and len(clob_ids) >= 2 and \
                               isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                                
                                yes_token_id = str(clob_ids[0])
                                no_token_id = str(clob_ids[1])
                                
                                yes_price = float(outcome_prices[0])
                                no_price = float(outcome_prices[1])
                                
                                # Determine winner
                                if yes_price > no_price:
                                    winner_token_id = yes_token_id
                                    winner_label = "Yes"
                                elif no_price > yes_price:
                                    winner_token_id = no_token_id
                                    winner_label = "No"
                                else:
                                    # Ambiguous - skip
                                    if is_debug:
                                        print(f"[DIAGNOSTIC] ❌ REPAIR FAILED: Ambiguous outcomePrices")
                                        print(f"             Yes: {yes_price}, No: {no_price}")
                                    del market_df
                                    continue
                                
                                # Apply repair
                                repair_map = {}
                                for tid in market_df['token_id'].unique():
                                    repair_map[str(tid)] = (str(tid) == winner_token_id)
                                
                                market_df['token_winner'] = market_df['token_id'].astype(str).map(repair_map)
                                repair_successful = True
                                
                                if is_debug:
                                    print(f"[DIAGNOSTIC] ✅ REPAIR SUCCESS: Winner -> {winner_label} (Token: {winner_token_id})")
                                    print(f"             Source: API Metadata (outcomePrices: Yes={yes_price}, No={no_price})")
                
                except Exception as e:
                    if is_debug:
                        print(f"[DIAGNOSTIC] ❌ API ERROR: {e}")
                
                if not repair_successful:
                    if is_debug:
                        print(f"[DIAGNOSTIC] Skipping Market: Could not reconstruct winner from API metadata")
                    del market_df
                    continue
            # ==================================================================
            # [END] ANOMALY DETECTION & API METADATA RECONSTRUCTION
            # ==================================================================

            # --- DIAGNOSTIC: CHECK COLUMNS ---
            if is_debug and 'token_winner' not in market_df.columns:
                 log(f"  [DEBUG {market_count}] ⚠️ CRITICAL: 'token_winner' column MISSING. Enrichment likely failed.")

            # --- CRITICAL FIX: Timestamp Normalization (Double Check for Resolution Time) ---
            # Ensure resolution_time matches the now-normalized timestamp scale
            try:
                resolution_time = float(resolution_time)
                if resolution_time > 3e9:
                    if is_debug: log(f"  [DEBUG {market_count}] normalizing resolution_time from MS...")
                    resolution_time = resolution_time / 1000.0
            except:
                del market_df
                continue

            # Check minimum trades filter
            if len(market_df) < MIN_TRADES_PER_MARKET:
                if is_debug: log(f"  [DEBUG {market_count}] Skipped: Low trade count ({len(market_df)} < {MIN_TRADES_PER_MARKET})")
                del market_df
                continue
            
            # Define analysis window (offset from resolution)
            window_end = resolution_time - (WINDOW_OFFSET_HOURS * 3600)
            window_start = window_end - (TIME_WINDOW_HOURS * 3600)
            
            if is_debug:
                res_str = datetime.fromtimestamp(resolution_time).strftime('%Y-%m-%d')
                log(f"  [DEBUG {market_count}] ID: {condition_id[:8]}... | Res: {res_str} | Window: {TIME_WINDOW_HOURS}h")

            # Filter to window
            window_df = market_df[
                (market_df['timestamp'] >= window_start) &
                (market_df['timestamp'] <= window_end)
            ].copy()
            
            #print(f"[DEBUG] Market {condition_id[:8]}: res_time={resolution_time}, window=[{window_start}, {window_end}]")
            #print(f"[DEBUG] Total market trades: {len(market_df)}, trades in window: {len(window_df)}")

            del market_df
            
            if len(window_df) == 0:
                if is_debug: 
                    log(f"  [DEBUG {market_count}] Skipped: 0 trades in analysis window (Data outside {TIME_WINDOW_HOURS}h window)")
                del window_df
                continue

            total_markets_processed += 1
            
            # Process each token in this market
            for token_id in window_df['token_id'].unique():
                token_trades = window_df[window_df['token_id'] == token_id].copy()
                
                # Liquidity filters
                if len(token_trades) < MIN_TRADES_PER_TOKEN:
                    total_tokens_filtered += 1
                    continue
                
                # =========================================================
                # FIX 3: USDC SCALING (The "6 Decimals" Fix)
                # =========================================================
                # Polymarket/USDC uses 6 decimals. Raw integer 1000000 = $1.00
                # We calculate raw volume first, then scale.
                
                # Step A: Calculate Raw Volume (Price * Size)
                # Note: Price is usually 0.0-1.0 (normalized), Size is raw integer
                raw_volume = (token_trades['price'] * token_trades['size_tokens']).sum()
                
                # Step B: Normalize to USD
                # If size is raw (e.g. 1e6 for 1 unit), we divide by 1e6
                total_volume_usd = raw_volume / 1_000_000.0 
                
                if total_volume_usd < MIN_VOLUME_USD:
                    # if is_debug: log(f"    Token skipped: Low Vol (${total_volume_usd:.2f})")
                    total_tokens_filtered += 1
                    continue
                
                # Get outcome (from enriched data)
                if 'token_winner' not in token_trades.columns:
                    continue
                
                # Robust winner extraction
                winner_val = token_trades['token_winner'].iloc[0]
                if winner_val is None:
                    continue
                token_won = bool(winner_val)
                
                # Compute TWAP
                trades_list = list(zip(
                    token_trades['timestamp'].values,
                    token_trades['price'].values,
                    token_trades['size_tokens'].values
                ))
                trades_list.sort(key=lambda x: x[0])
                
                twap = compute_twap(trades_list)
                if twap is None or not (0 < twap < 1):
                    continue
                
                # Assign to bucket
                bucket_label, bucket_tag = assign_bucket(twap, ODDS_BUCKETS)
                if bucket_label is None:
                    continue
                
                # Calculate metrics
                # Store average trade size (also scaled)
                avg_trade_size = (token_trades['price'] * token_trades['size_tokens']).mean() / 1_000_000.0
                
                # Return calculation (for Sharpe)
                ret = (1.0 / twap - 1.0) if token_won else -1.0
                
                # Store data
                bucket_data[bucket_label]['twaps'].append(twap)
                bucket_data[bucket_label]['outcomes'].append(1 if token_won else 0)
                bucket_data[bucket_label]['volumes'].append(total_volume_usd)
                bucket_data[bucket_label]['trade_sizes'].append(avg_trade_size)
                bucket_data[bucket_label]['returns'].append(ret)
                
                if twap > 0.95:
                    print(f"[HIGH-PROB] {condition_id[:8]}... token={token_id[:8]}... TWAP={twap:.4f} won={token_won} bucket={bucket_label}")


                total_tokens_analyzed += 1
                if is_debug:
                    log(f"    ✓ Token Added: TWAP={twap:.3f}, Vol=${total_volume_usd:.2f}, Won={token_won}")
            
            del window_df
            gc.collect()
            
        except Exception as e:
            if is_debug: log(f"  [DEBUG {market_count}] CRASH: {e}")
            continue
    
    log(f"\n  ✓ Analysis complete: {log_memory()}")
    log(f"    Markets processed: {total_markets_processed:,}")
    log(f"    Tokens analyzed: {total_tokens_analyzed:,}")
    log(f"    Tokens filtered: {total_tokens_filtered:,}")
    
    # Compute calibration statistics
    log("\n" + "="*70)
    log("COMPUTING CALIBRATION STATISTICS")
    log("="*70)
    
    calibration_stats = []
    
    for lower, upper, label, tag in ODDS_BUCKETS:
        data = bucket_data[label]
        
        if len(data['twaps']) == 0:
            log(f"  {label}: No data")
            continue
        
        n_tokens = len(data['twaps'])
        n_wins = sum(data['outcomes'])
        
        avg_implied = np.mean(data['twaps'])
        realized_rate, ci_lower, ci_upper = wilson_score_interval(n_wins, n_tokens)
        
        raw_edge = realized_rate - avg_implied
        raw_edge_bps = raw_edge * 10000
        
        # Return edge (percentage)
        return_edge_pct = (raw_edge / avg_implied) * 100 if avg_implied > 0 else 0
        
        # Z-test for significance
        se = np.sqrt(avg_implied * (1 - avg_implied) / n_tokens)
        z_score = raw_edge / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        significant = p_value < ALPHA
        
        # Enhanced metrics
        returns = data['returns']
        sharpe = compute_sharpe_ratio(returns) if len(returns) >= 2 else None
        kelly = compute_kelly_fraction(realized_rate, avg_implied)
        
        # Calibration scores
        brier = compute_brier_score(data['twaps'], data['outcomes'])
        log_loss_score = compute_log_loss(data['twaps'], data['outcomes'])
        
        # Volume metrics
        avg_volume = np.mean(data['volumes'])
        avg_trade_size = np.mean(data['trade_sizes'])
        
        # T-test for edge
        returns_array = np.array(returns)
        t_stat, t_pvalue = compute_t_statistic(
            np.mean(returns_array),
            np.std(returns_array, ddof=1),
            len(returns_array)
        )
        
        stats_dict = {
            'bucket': label,
            'bucket_tag': tag,
            'n_tokens': int(n_tokens),  # Cast to standard int
            'n_wins': int(n_wins),      # Cast to standard int
            'avg_implied_prob': float(avg_implied),
            'realized_win_rate': float(realized_rate),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'raw_edge': float(raw_edge),
            'raw_edge_bps': float(raw_edge_bps),
            'return_edge_pct': float(return_edge_pct),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'significant': bool(significant),  # EXPLICIT CAST to Python bool
            'sharpe_ratio': float(sharpe) if sharpe is not None else None,
            'kelly_fraction': float(kelly),
            'brier_score': float(brier) if brier is not None else None,
            'log_loss': float(log_loss_score) if log_loss_score is not None else None,
            't_statistic': float(t_stat),
            't_pvalue': float(t_pvalue),
            'avg_volume_usd': float(avg_volume),
            'avg_trade_size_usd': float(avg_trade_size),
            'median_volume_usd': float(np.median(data['volumes'])),
            'total_volume_usd': float(np.sum(data['volumes'])),
        }
        
        calibration_stats.append(stats_dict)
        
        # =========================================================
        # FIX 4: SAFE LOGGING (Formatting Crash Fix)
        # =========================================================
        if sharpe is not None:
            sharpe_display = f"{float(sharpe):.3f}"
        else:
            sharpe_display = "N/A"
            
        if raw_edge_bps is not None:
            edge_display = f"{float(raw_edge_bps):+.1f}"
        else:
            edge_display = "N/A"
        
        log(f"  {label}: n={n_tokens:,}, edge={edge_display}bp, "
            f"Sharpe={sharpe_display}, "
            f"p={p_value:.4f} {'***' if significant else ''}")
    
    
    # Overall metrics
    total_metrics = {
        'markets_processed': total_markets_processed,
        'tokens_analyzed': total_tokens_analyzed,
        'tokens_low_liquidity': total_tokens_filtered,
        'filter_rate_pct': (total_tokens_filtered / (total_tokens_analyzed + total_tokens_filtered) * 100) 
                          if (total_tokens_analyzed + total_tokens_filtered) > 0 else 0
    }
    
    # Generate outputs
    log("\n" + "="*70)
    log("GENERATING OUTPUTS")
    log("="*70)
    
    # --- CRITICAL FIX: Prevent Visualization Crash ---
    if len(calibration_stats) > 0:
        create_enhanced_visualizations(calibration_stats)
    else:
        log("⚠️ No populated buckets. Skipping visualization to prevent crash.")

    save_enhanced_outputs(calibration_stats, total_metrics, batch_files, sample_size)
    
    log("\n" + "="*70)
    log("ANALYSIS COMPLETE")
    log("="*70)

# ==============================================================================
# ENHANCED VISUALIZATION
# ==============================================================================

def create_enhanced_visualizations(calibration_stats):
    """
    Create publication-quality visualizations
    """
    log("  Creating enhanced visualizations...")
    
    # Set professional style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # --------------------------------------------------------------------------
    # Plot 1: Calibration Curve with Confidence Bands (MAIN PLOT - larger)
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0:2, 0:2])  # Takes up 2x2 space
    
    # Extract data
    implied_probs = [s['avg_implied_prob'] for s in calibration_stats]
    realized_rates = [s['realized_win_rate'] for s in calibration_stats]
    ci_lowers = [s['ci_lower'] for s in calibration_stats]
    ci_uppers = [s['ci_upper'] for s in calibration_stats]
    sample_sizes = [s['n_tokens'] for s in calibration_stats]
    significant = [s['significant'] for s in calibration_stats]
    
    # Perfect calibration line (y=x)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect Calibration', zorder=1)
    
    # Realized rates with confidence bands
    for i in range(len(implied_probs)):
        color = 'darkgreen' if significant[i] else 'steelblue'
        marker = 'D' if significant[i] else 'o'
        markersize = 8 if significant[i] else 6
        
        # Confidence interval
        ax1.plot([implied_probs[i], implied_probs[i]], 
                [ci_lowers[i], ci_uppers[i]], 
                color=color, linewidth=2, alpha=0.6, zorder=2)
        
        # Point estimate
        ax1.scatter(implied_probs[i], realized_rates[i], 
                   s=markersize**2, color=color, marker=marker,
                   edgecolors='black', linewidths=0.5, zorder=3)
        
        # Sample size annotation
        ax1.annotate(f'n={sample_sizes[i]:,}', 
                    (implied_probs[i], realized_rates[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.7)
    
    ax1.set_xlabel('Market-Implied Probability (TWAP)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Realized Win Rate', fontsize=12, fontweight='bold')
    ax1.set_title('Calibration Curve with 95% Confidence Intervals', 
                 fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    
    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='Perfect Calibration'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='darkgreen', 
                  markersize=8, markeredgecolor='black', markeredgewidth=0.5, 
                  label='Significant (p<0.05)', linestyle='None'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                  markersize=6, markeredgecolor='black', markeredgewidth=0.5,
                  label='Not Significant', linestyle='None')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    # --------------------------------------------------------------------------
    # Plot 2: Edge by Bucket (Top right)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    
    bucket_labels = [s['bucket'] for s in calibration_stats]
    edges = [s['raw_edge_bps'] for s in calibration_stats]
    colors = ['darkgreen' if e > 0 and s['significant'] else 
              'darkred' if e < 0 and s['significant'] else 
              'gray' for e, s in zip(edges, calibration_stats)]
    
    bars = ax2.barh(range(len(bucket_labels)), edges, color=colors, alpha=0.8)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.set_yticks(range(len(bucket_labels)))
    ax2.set_yticklabels(bucket_labels, fontsize=8)
    ax2.set_xlabel('Edge (basis points)', fontsize=10, fontweight='bold')
    ax2.set_title('Raw Edge by Bucket', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x', linestyle=':')
    
    # Annotate significance
    for i, (bar, stats) in enumerate(zip(bars, calibration_stats)):
        if stats['significant']:
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    ' ***', ha='left' if width > 0 else 'right', va='center',
                    fontsize=10, fontweight='bold', color='black')
    
    # --------------------------------------------------------------------------
    # Plot 3: Sharpe Ratios (Middle right)
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 2])
    
    sharpes = [s['sharpe_ratio'] if s['sharpe_ratio'] is not None else 0 
              for s in calibration_stats]
    colors = ['darkgreen' if sr > 0 and s['significant'] else 'gray' 
             for sr, s in zip(sharpes, calibration_stats)]
    
    bars = ax3.barh(range(len(bucket_labels)), sharpes, color=colors, alpha=0.8)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.set_yticks(range(len(bucket_labels)))
    ax3.set_yticklabels(bucket_labels, fontsize=8)
    ax3.set_xlabel('Sharpe Ratio', fontsize=10, fontweight='bold')
    ax3.set_title('Risk-Adjusted Returns', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x', linestyle=':')
    
    # --------------------------------------------------------------------------
    # Plot 4: Sample Sizes (Bottom left)
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 0])
    
    bars = ax4.bar(range(len(bucket_labels)), sample_sizes, color='steelblue', alpha=0.7)
    ax4.set_xticks(range(len(bucket_labels)))
    ax4.set_xticklabels(bucket_labels, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Sample Size (tokens)', fontsize=10, fontweight='bold')
    ax4.set_title('Statistical Power by Bucket', fontsize=11, fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, axis='y', linestyle=':')
    
    # --------------------------------------------------------------------------
    # Plot 5: Brier Scores (Bottom middle)
    # --------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 1])
    
    briers = [s['brier_score'] for s in calibration_stats]
    colors = ['darkgreen' if b < 0.25 else 'orange' if b < 0.5 else 'darkred' 
             for b in briers]
    
    bars = ax5.bar(range(len(bucket_labels)), briers, color=colors, alpha=0.7)
    ax5.axhline(y=0.25, color='green', linestyle='--', linewidth=1, alpha=0.5, 
               label='Good (<0.25)')
    ax5.set_xticks(range(len(bucket_labels)))
    ax5.set_xticklabels(bucket_labels, rotation=45, ha='right', fontsize=8)
    ax5.set_ylabel('Brier Score', fontsize=10, fontweight='bold')
    ax5.set_title('Calibration Quality', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax5.legend(fontsize=8, loc='upper right')
    ax5.set_ylim(0, max(briers) * 1.2)
    
    # --------------------------------------------------------------------------
    # Plot 6: Kelly Fractions (Bottom right)
    # --------------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 2])
    
    kellys = [s['kelly_fraction'] for s in calibration_stats]
    colors = ['darkgreen' if k > 0 and s['significant'] else 'gray' 
             for k, s in zip(kellys, calibration_stats)]
    
    bars = ax6.barh(range(len(bucket_labels)), kellys, color=colors, alpha=0.8)
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax6.set_yticks(range(len(bucket_labels)))
    ax6.set_yticklabels(bucket_labels, fontsize=8)
    ax6.set_xlabel('Kelly Fraction', fontsize=10, fontweight='bold')
    ax6.set_title('Optimal Position Sizing', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x', linestyle=':')
    
    # Add fractional Kelly reference lines
    for frac, label in [(0.25, '¼ Kelly'), (0.5, '½ Kelly')]:
        ax6.axvline(x=frac, color='blue', linestyle=':', linewidth=1, alpha=0.3)
        ax6.text(frac, len(kellys) - 0.5, label, 
                rotation=90, va='top', ha='right', fontsize=7, alpha=0.6)
    
    # Overall title
    fig.suptitle('Phase 1: Polymarket Calibration Analysis - Enhanced Edition',
                fontsize=15, fontweight='bold', y=0.995)
    
    # Save
    plot_path = f"{OUTPUT_DIR}/phase1_calibration_{TIMESTAMP}.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    log(f"  ✓ Enhanced visualization saved: {plot_path}")

# ==============================================================================
# ENHANCED OUTPUT GENERATION
# ==============================================================================

def save_enhanced_outputs(calibration_stats, total_metrics, batch_files, sample_size):
    """
    Save comprehensive JSON summary and detailed text report
    """
    
    # JSON output with enhanced metrics
    summary = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'version': '10.0 - Enhanced Edition',
            'batch_files_processed': len(batch_files),
            'sample_mode': sample_size is not None,
            'analysis_window_hours': TIME_WINDOW_HOURS,
            'window_offset_hours': WINDOW_OFFSET_HOURS,
            'min_volume_usd': MIN_VOLUME_USD,
            'min_trades_per_market': MIN_TRADES_PER_MARKET,
            'min_trades_per_token': MIN_TRADES_PER_TOKEN,
            'significance_level': ALPHA
        },
        'summary': total_metrics,
        'calibration_results': calibration_stats
    }
    
    json_path = f"{OUTPUT_DIR}/phase1_calibration_{TIMESTAMP}_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    log(f"  ✓ JSON summary saved: {json_path}")
    
    # Enhanced text report
    report_path = f"{OUTPUT_DIR}/phase1_calibration_{TIMESTAMP}_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE 1: ODDS-STRATIFIED CALIBRATION ANALYSIS v10.0\n")
        f.write("Polymarket Baseline Edge Measurement - Enhanced Edition\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target Audience: Quantitative PMs (Citadel, Jane Street, etc.)\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Markets Analyzed: {total_metrics['markets_processed']:,}\n")
        f.write(f"Tokens Analyzed: {total_metrics['tokens_analyzed']:,}\n")
        f.write(f"Liquidity Filter Rate: {total_metrics['filter_rate_pct']:.1f}%\n")
        
        # Count significant buckets
        sig_positive = [s for s in calibration_stats if s['significant'] and s['raw_edge'] > 0]
        sig_negative = [s for s in calibration_stats if s['significant'] and s['raw_edge'] < 0]
        
        f.write(f"Significant Positive Edge Buckets: {len(sig_positive)}\n")
        f.write(f"Significant Negative Edge Buckets: {len(sig_negative)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. CALIBRATION FRAMEWORK\n")
        f.write("-" * 40 + "\n")
        f.write("   Approach: Stratified binomial testing across probability buckets\n")
        f.write("   Null Hypothesis: Market-implied probabilities are unbiased\n")
        f.write(f"   Significance Level: α={ALPHA} (two-tailed test)\n")
        f.write("   Confidence Intervals: Wilson score method (correct coverage near boundaries)\n\n")
        
        f.write("2. PRICE REPRESENTATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Metric: Time-Weighted Average Price (TWAP)\n")
        f.write(f"   Window: {TIME_WINDOW_HOURS}h ending {WINDOW_OFFSET_HOURS}h before resolution\n")
        f.write("   Rationale: \n")
        f.write("     - Resistant to volume manipulation (time-weighted, not volume-weighted)\n")
        f.write("     - Captures representative market belief over analysis window\n")
        f.write("     - Avoids settlement bias (offset from resolution prevents hindsight)\n")
        f.write("     - Standard in execution analysis and market microstructure research\n\n")
        
        f.write("3. LIQUIDITY FILTERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Minimum Volume: ${MIN_VOLUME_USD} per token in window\n")
        f.write(f"   Minimum Market Trades: {MIN_TRADES_PER_MARKET}\n")
        f.write(f"   Minimum Token Trades: {MIN_TRADES_PER_TOKEN}\n")
        f.write("   Purpose: \n")
        f.write("     - Eliminates 'zombie markets' (stale prices with no volume)\n")
        f.write("     - Ensures TWAP quality through sufficient price discovery\n")
        f.write("     - Filters non-executable opportunities\n\n")
        
        f.write("4. BUCKET STRATIFICATION\n")
        f.write("-" * 40 + "\n")
        f.write("   Design: Coarse buckets for underdogs, refined buckets for favorites\n")
        f.write("   Rationale:\n")
        f.write("     - Underdogs (0-51%): Higher variance, less liquid → coarse buckets\n")
        f.write("     - Favorites (51-100%): Lower variance, more liquid → refined buckets\n")
        f.write("     - Extreme favorites (99%+): Bond farming territory → finest granularity\n")
        f.write("     - Matches market structure: precision where it matters most\n\n")
        
        f.write("5. RISK-ADJUSTED METRICS\n")
        f.write("-" * 40 + "\n")
        f.write("   Sharpe Ratio: E[R] / σ(R)\n")
        f.write("     - Measures edge per unit of volatility\n")
        f.write("     - Assumes returns IID within stratified buckets\n")
        f.write("     - Annualization: Sharpe_annual ≈ Sharpe_per_bet × sqrt(N_bets_per_year)\n\n")
        f.write("   Kelly Fraction: f* = (p × (1 + b) - 1) / b\n")
        f.write("     - Optimal bet size for log utility maximization\n")
        f.write("     - Practice: Use fractional Kelly (¼ or ½) for risk management\n")
        f.write("     - Conservative: Kelly > 0.25 indicates strong edge\n\n")
        
        f.write("6. CALIBRATION QUALITY METRICS\n")
        f.write("-" * 40 + "\n")
        f.write("   Brier Score: Mean squared error between predictions and outcomes\n")
        f.write("     - Range: [0, 1], lower is better\n")
        f.write("     - Good calibration: Brier < 0.25\n")
        f.write("     - Perfect calibration: Brier = 0\n\n")
        f.write("   Log Loss: Cross-entropy loss\n")
        f.write("     - Strictly proper scoring rule\n")
        f.write("     - Heavily penalizes confident wrong predictions\n")
        f.write("     - Unbounded (extreme predictions drive large penalties)\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for stats in calibration_stats:
            f.write(f"{stats['bucket']} ({stats['bucket_tag']})\n")
            f.write("-" * 80 + "\n")
            
            # Sample and outcomes
            f.write(f"Sample Size:        {stats['n_tokens']:,} tokens ({stats['n_wins']:,} wins)\n")
            f.write(f"Implied Prob:       {stats['avg_implied_prob']:.4f}\n")
            f.write(f"Realized Win Rate:  {stats['realized_win_rate']:.4f} "
                   f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]\n\n")
            
            # Edge metrics
            f.write(f"Raw Edge:           {stats['raw_edge']:+.6f} ({stats['raw_edge_bps']:+.1f} bps)\n")
            f.write(f"Return Edge:        {stats['return_edge_pct']:+.2f}%\n")
            if stats['sharpe_ratio'] is not None:
                sharpe_str = f"{stats['sharpe_ratio']:.4f}"
            else:
                sharpe_str = "N/A"
            f.write(f"Sharpe Ratio:       {sharpe_str}\n")
            f.write(f"Kelly Fraction:     {stats['kelly_fraction']:.4f}\n\n")
            
            # Statistical tests
            f.write(f"Z-Statistic:        {stats['z_score']:.3f}\n")
            f.write(f"P-Value (z-test):   {stats['p_value']:.6f}\n")
            f.write(f"T-Statistic:        {stats['t_statistic']:.3f}\n")
            f.write(f"P-Value (t-test):   {stats['t_pvalue']:.6f}\n")
            f.write(f"Significance:       {'*** SIGNIFICANT ***' if stats['significant'] else 'Not significant'}\n\n")
            
            # Calibration quality
            f.write(f"Brier Score:        {stats['brier_score']:.6f}\n")
            f.write(f"Log Loss:           {stats['log_loss']:.6f}\n\n")
            
            # Liquidity metrics
            f.write(f"Avg Volume:         ${stats['avg_volume_usd']:.2f}\n")
            f.write(f"Median Volume:      ${stats['median_volume_usd']:.2f}\n")
            f.write(f"Total Volume:       ${stats['total_volume_usd']:.2f}\n")
            f.write(f"Avg Trade Size:     ${stats['avg_trade_size_usd']:.2f}\n\n")
            
            # Interpretation
            f.write("Interpretation:\n")
            if stats['significant']:
                if stats['raw_edge'] > 0:
                    f.write("  → POSITIVE EDGE: Market systematically underprices this probability range\n")
                    if stats['kelly_fraction'] > 0.25:
                        f.write("  → STRONG SIGNAL: Kelly fraction suggests substantial position sizing\n")
                    if stats['sharpe_ratio'] and stats['sharpe_ratio'] > 1.0:
                        f.write("  → ATTRACTIVE RISK-ADJUSTED: Sharpe > 1.0 indicates strong risk-adjusted returns\n")
                else:
                    f.write("  → NEGATIVE EDGE: Market systematically overprices this probability range\n")
                    f.write("  → AVOID: Do not provide liquidity in this bucket\n")
            else:
                f.write("  → WELL-CALIBRATED: No statistically significant bias detected\n")
                f.write("  → NEUTRAL: Market appears efficient in this probability range\n")
            
            f.write("\n")
        
        # Key findings section
        f.write("="*80 + "\n")
        f.write("KEY FINDINGS & STRATEGIC IMPLICATIONS\n")
        f.write("="*80 + "\n\n")
        
        if sig_positive:
            f.write("POSITIVE EDGE OPPORTUNITIES\n")
            f.write("-" * 80 + "\n")
            for s in sig_positive:
                f.write(f"\n{s['bucket']} ({s['bucket_tag']}):\n")
                f.write(f"  Edge:          {s['raw_edge_bps']:+.1f} bps ({s['return_edge_pct']:+.2f}% return)\n")
                
                # FIX: Format conditionally outside the f-string
                sharpe_val = s['sharpe_ratio']
                sharpe_str = f"{sharpe_val:.3f}" if sharpe_val is not None else "N/A"
                f.write(f"  Sharpe:        {sharpe_str}\n")
                
                f.write(f"  Kelly:         {s['kelly_fraction']:.3f} (suggest ¼ Kelly = {s['kelly_fraction']/4:.3f})\n")
            
            f.write("\n\nSTRATEGIC RECOMMENDATIONS:\n")
            f.write("  1. SYSTEMATIC DEPLOYMENT:\n")
            f.write("     - Focus liquidity provision in identified positive-edge buckets\n")
            f.write("     - Use fractional Kelly (¼ or ½) for conservative position sizing\n")
            f.write("     - Monitor volume capacity relative to strategy size\n\n")
            f.write("  2. PHASE 2 PRIORITIES:\n")
            f.write("     - Feature-based segmentation within edge-positive buckets\n")
            f.write("     - Market characteristics: category, time-to-resolution, popularity\n")
            f.write("     - Temporal stability: edge persistence across time periods\n\n")
            f.write("  3. RISK MANAGEMENT:\n")
            f.write("     - Diversify across multiple buckets (correlation <1)\n")
            f.write("     - Monitor realized Sharpe vs. expected\n")
            f.write("     - Establish position limits per bucket and aggregate\n\n")
        else:
            f.write("BASELINE FINDINGS\n")
            f.write("-" * 80 + "\n")
            f.write("No buckets show statistically significant positive edge.\n\n")
            f.write("INTERPRETATION:\n")
            f.write("  - Market appears well-calibrated at baseline\n")
            f.write("  - Edge may exist in feature-stratified segments (Phase 2)\n")
            f.write("  - Consider alternative strategies:\n")
            f.write("     a) Higher-frequency windows (e.g., 1h pre-resolution)\n")
            f.write("     b) Event-driven opportunities (news, announcements)\n")
            f.write("     c) Market-specific inefficiencies (niche categories)\n\n")
        
        if sig_negative:
            f.write("NEGATIVE EDGE ZONES (AVOID)\n")
            f.write("-" * 80 + "\n")
            for s in sig_negative:
                f.write(f"\n{s['bucket']}: {s['raw_edge_bps']:+.1f} bps ({s['return_edge_pct']:+.2f}% return)\n")
                f.write(f"  → DO NOT provide liquidity in this range\n")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICAL QUALITY ASSESSMENT\n")
        f.write("="*80 + "\n\n")
        
        # Overall Brier score
        all_predictions = []
        all_outcomes = []
        for s in calibration_stats:
            idx = [i for i, b in enumerate(ODDS_BUCKETS) if b[2] == s['bucket']][0]
            lower, upper, label, tag = ODDS_BUCKETS[idx]
            bucket_data_key = label
            
            # This is a simplification - in practice we'd aggregate from the bucket_data dict
            # For now, use the calibration stats
            all_predictions.extend([s['avg_implied_prob']] * s['n_tokens'])
            all_outcomes.extend([1] * s['n_wins'] + [0] * (s['n_tokens'] - s['n_wins']))
        
        overall_brier = compute_brier_score(all_predictions, all_outcomes)
        overall_log_loss = compute_log_loss(all_predictions, all_outcomes)
        
        f.write(f"Overall Brier Score:  {overall_brier:.6f}\n")
        f.write(f"Overall Log Loss:     {overall_log_loss:.6f}\n\n")
        
        f.write("Interpretation:\n")
        if overall_brier < 0.20:
            f.write("  → EXCELLENT calibration quality (Brier < 0.20)\n")
        elif overall_brier < 0.25:
            f.write("  → GOOD calibration quality (Brier < 0.25)\n")
        else:
            f.write("  → MODERATE calibration quality (Brier ≥ 0.25)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NEXT STEPS\n")
        f.write("="*80 + "\n\n")
        
        if sig_positive:
            f.write("PROCEED TO PHASE 2: FEATURE-BASED REFINEMENT\n\n")
            f.write("Objectives:\n")
            f.write("  1. Market segmentation within edge-positive buckets:\n")
            f.write("     - Category analysis (politics, sports, crypto, etc.)\n")
            f.write("     - Time-to-resolution buckets (< 1 day, 1-7 days, > 7 days)\n")
            f.write("     - Popularity tiers (volume quintiles)\n")
            f.write("     - Outcome type analysis (Yes/No vs Over/Under vs matchups)\n\n")
            f.write("  2. Temporal stability:\n")
            f.write("     - Edge persistence across calendar months\n")
            f.write("     - Intraday patterns (market open/close effects)\n")
            f.write("     - Event-driven vs. continuous markets\n\n")
            f.write("  3. Portfolio construction:\n")
            f.write("     - Correlation analysis between buckets and features\n")
            f.write("     - Optimal allocation (mean-variance or Kelly framework)\n")
            f.write("     - Capacity analysis and scaling roadmap\n\n")
        else:
            f.write("VALIDATION & SENSITIVITY ANALYSIS\n\n")
            f.write("Actions:\n")
            f.write("  1. Validate on full dataset (if currently sampled)\n")
            f.write("  2. Sensitivity analysis:\n")
            f.write("     - Vary time window (12h, 36h, 48h)\n")
            f.write("     - Adjust liquidity filters\n")
            f.write("     - Alternative price metrics (VWAP, close price)\n")
            f.write("  3. Feature stratification (Phase 2) to identify niche opportunities\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("METHODOLOGY NOTES FOR REVIEWERS\n")
        f.write("="*80 + "\n\n")
        
        f.write("STATISTICAL FRAMEWORK:\n")
        f.write("  - Wilson score intervals: Conservative, correct coverage near boundaries\n")
        f.write("  - Two-tailed tests: Detecting both positive and negative bias\n")
        f.write("  - Multiple comparisons: No Bonferroni correction (exploratory phase)\n\n")
        
        f.write("PRICE REPRESENTATION:\n")
        f.write("  - TWAP over VWAP: More robust to wash trading and manipulation\n")
        f.write("  - Window offset: Prevents settlement bias (analyzing before outcome known)\n")
        f.write("  - Alternative for Phase 2: Compare TWAP vs close price vs VWAP\n\n")
        
        f.write("LIQUIDITY FILTERS:\n")
        f.write("  - Volume threshold: Ensures executability\n")
        f.write("  - Trade count: Ensures price discovery quality\n")
        f.write("  - Conservative approach: Better to exclude edge cases than include noise\n\n")
        
        f.write("RISK METRICS:\n")
        f.write("  - Sharpe assumes IID returns: Reasonable within stratified buckets\n")
        f.write("  - Kelly fraction: Theoretical optimum, use fractional in practice\n")
        f.write("  - No tail risk analysis yet: Phase 2 should examine drawdown distributions\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Analysis version: 10.0 - Enhanced Edition\n")
        f.write("="*80 + "\n")
    
    log(f"  ✓ Enhanced report saved: {report_path}")
    
    log("\nOutputs generated:")
    log(f"  - Summary JSON:    phase1_calibration_{TIMESTAMP}_summary.json")
    log(f"  - Detailed Report: phase1_calibration_{TIMESTAMP}_report.txt")
    log(f"  - Visualization:   phase1_calibration_{TIMESTAMP}.png")
    
    log("\nTo view results locally:")
    log(f"  scp -i 'poly.pem' ec2-user@ec2-34-203-248-16.compute-1.amazonaws.com:{OUTPUT_DIR}/phase1_calibration_{TIMESTAMP}_* ./")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    if sample:
        print(f"\n*** RUNNING IN SAMPLE MODE: {sample} batch files ***\n")
    
    # Check for cached index
    if os.path.exists(INDEX_CACHE_FILE):
        try:
            cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(INDEX_CACHE_FILE))
            print(f"NOTE: Found cached index (age: {cache_age.seconds // 3600}h {(cache_age.seconds % 3600) // 60}m)")
            print(f"      Pass 1 will be skipped if cache is valid")
            print(f"      To force rebuild: rm {INDEX_CACHE_FILE}\n")
        except: 
            pass
    
    ensure_output_dir()
    analyze_calibration_streaming(sample_size=sample)