#!/usr/bin/env python3
"""
OOS Token Outcome Enrichment - Parameterized Version for Out-of-Sample Collection
==================================================================================
Based on enrich_token_outcomes.py v1.0

CHANGES FROM ORIGINAL:
- Added --input-dir command-line argument
- Checkpoint and mapping files scoped to input directory
- Safety checks to prevent accidentally enriching training data
- All CLOB API logic UNCHANGED (battle-tested)

Usage:
    python oos_enrich_token_outcomes.py --input-dir oos_collection/order_history_batches/

PURPOSE:
Adds 'token_outcome' and 'token_winner' fields to parquet files by:
1. Building condition_id → {token_id: "Yes"/"No", winner: bool} mapping from CLOB API
2. Enriching each parquet file with the token_outcome and token_winner columns

INPUTS:
- {input_dir}/*.parquet (order history from oos_fetch_order_history.py)
- CLOB API: https://clob.polymarket.com/markets

OUTPUTS:
- Enriched parquet files (in-place update)
- {input_dir}/oos_token_mapping.json (cached API results)
- {input_dir}/oos_enrich_checkpoint.json (progress tracking)
"""

import argparse
import requests
import pandas as pd
import time
import json
import os
import sys
import glob
import psutil
import shutil
import tempfile
from datetime import datetime
from typing import Dict, Set, List, Optional
from collections import OrderedDict

# ==========================================
# ARGUMENT PARSING
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Enrich OOS parquet files with token outcomes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python oos_enrich_token_outcomes.py --input-dir oos_collection/order_history_batches/
        """
    )
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing parquet batch files to enrich')
    return parser.parse_args()

# ==========================================
# CONFIGURATION
# ==========================================
CLOB_API_URL = "https://clob.polymarket.com/markets"

# Rate Limiting
MIN_DELAY = 1.5
MAX_DELAY = 60.0
SUCCESS_DECAY = 0.995
ERROR_MULTIPLIER = 2.5
COOLDOWN_AFTER_ERROR = 10.0

# Memory Protection
MAX_MEMORY_PCT = 80.0
MEMORY_CHECK_INTERVAL = 10

MAX_RETRIES = 5

# ==========================================
# LOGGING SYSTEM
# ==========================================
class Logger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.start_time = time.time()
        
    def log(self, msg: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start_time
        elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"
        
        log_line = f"[{timestamp}] [{elapsed_str}] [{level}] {msg}"
        print(log_line)
        sys.stdout.flush()
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_line + '\n')
        except:
            pass

# Global logger - initialized in main()
logger = None

# ==========================================
# MEMORY MONITOR
# ==========================================
class MemoryMonitor:
    def __init__(self, max_pct: float = MAX_MEMORY_PCT):
        self.max_pct = max_pct
        self.process = psutil.Process()
        
    def check(self) -> bool:
        """Returns True if memory is safe, False if dangerous"""
        try:
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            mem_pct = self.process.memory_percent()
            
            if mem_pct > self.max_pct:
                logger.log(f"⚠️ MEMORY CRITICAL: {mem_pct:.1f}% ({mem_mb:.0f}MB)", "ERROR")
                return False
                
            return True
        except:
            return True
    
    def get_usage(self) -> str:
        """Returns memory usage string"""
        try:
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            mem_pct = self.process.memory_percent()
            return f"{mem_mb:.0f}MB ({mem_pct:.1f}%)"
        except:
            return "unknown"

memory_monitor = MemoryMonitor()

# ==========================================
# ADAPTIVE RATE LIMITER
# ==========================================
class SafeRateLimiter:
    def __init__(self):
        self.delay = MIN_DELAY
        self.consecutive_successes = 0
        self.consecutive_errors = 0
        self.total_requests = 0
        self.total_errors = 0
        
    def on_error(self, error_type: str = "unknown"):
        """Called when any error occurs"""
        old_delay = self.delay
        self.consecutive_errors += 1
        self.consecutive_successes = 0
        self.total_errors += 1
        
        if error_type == "RATE_LIMIT":
            self.delay = min(self.delay * ERROR_MULTIPLIER, MAX_DELAY)
            logger.log(
                f"⚠️ RATE LIMIT: Delay {old_delay:.2f}s → {self.delay:.2f}s",
                "WARN"
            )
        elif error_type in ["SERVER_ERROR", "TIMEOUT", "CONNECTION"]:
            self.delay = min(self.delay * 1.5, MAX_DELAY)
            logger.log(
                f"⚠️ {error_type}: Delay {old_delay:.2f}s → {self.delay:.2f}s",
                "WARN"
            )
        else:
            self.delay = min(self.delay * 2.0, MAX_DELAY)
            logger.log(
                f"⚠️ ERROR ({error_type}): Delay {old_delay:.2f}s → {self.delay:.2f}s",
                "WARN"
            )
        
        if self.consecutive_errors >= 3:
            logger.log(f"🧊 COOLING DOWN {COOLDOWN_AFTER_ERROR}s", "WARN")
            time.sleep(COOLDOWN_AFTER_ERROR)
        
    def on_success(self):
        """Called when request succeeds"""
        self.consecutive_successes += 1
        self.consecutive_errors = 0
        self.total_requests += 1
        
        old_delay = self.delay
        
        # Fast recovery if way above baseline with many successes
        if self.delay > MIN_DELAY * 5 and self.consecutive_successes >= 20:
            self.delay = (self.delay + MIN_DELAY) / 2
            if abs(self.delay - old_delay) > 0.5:
                logger.log(f"✅ FAST RECOVERY: {old_delay:.2f}s → {self.delay:.2f}s", "INFO")
        # Normal decay
        elif self.delay > MIN_DELAY:
            self.delay = max(MIN_DELAY, self.delay * SUCCESS_DECAY)
            
    def wait(self):
        """Sleep for current delay"""
        time.sleep(self.delay)
        
    def get_stats(self) -> str:
        """Returns statistics string"""
        if self.total_requests == 0:
            return "No requests yet"
        error_rate = (self.total_errors / self.total_requests) * 100
        return (f"{self.total_requests} reqs, {self.total_errors} errors ({error_rate:.1f}%), "
                f"delay={self.delay:.2f}s")

limiter = SafeRateLimiter()

# ==========================================
# CLOB API CLIENT
# ==========================================
def fetch_condition_tokens(condition_id: str, token_id: str = None) -> Optional[Dict[str, Dict[str, any]]]:
    """
    Fetch token_id → {outcome, winner} mapping from CLOB API
    
    Args:
        condition_id: The market condition to query
        token_id: Optional token to validate is in the mapping
    
    Returns:
        {
            "token_id_1": {"outcome": "Yes", "winner": False},
            "token_id_2": {"outcome": "No", "winner": True}
        }
        None for failures or non-binary markets
    """
    
    url = f"{CLOB_API_URL}/{condition_id}"
    
    for attempt in range(MAX_RETRIES):
        try:
            limiter.wait()
            
            response = requests.get(url, timeout=30)
            
            # Check status code
            if response.status_code == 429:
                limiter.on_error("RATE_LIMIT")
                continue
            elif response.status_code == 404:
                logger.log(f"   404: Condition {condition_id[:16]} not found in CLOB", "WARN")
                limiter.on_success()
                return None
            elif response.status_code >= 500:
                limiter.on_error("SERVER_ERROR")
                continue
            elif response.status_code != 200:
                logger.log(f"   Unexpected status {response.status_code} for {condition_id[:16]}", "WARN")
                limiter.on_error("HTTP_ERROR")
                continue
            
            data = response.json()
            limiter.on_success()
            
            # Parse CLOB response
            if not isinstance(data, dict):
                logger.log(f"   Unexpected response type {type(data)} for {condition_id[:16]}", "WARN")
                return None
            
            tokens = data.get('tokens', [])
            
            if len(tokens) != 2:
                logger.log(f"   Non-binary market ({len(tokens)} tokens) - skipping {condition_id[:16]}", "DEBUG")
                return None
            
            # Build mapping from CLOB tokens array - includes outcome AND winner
            mapping = {}
            for token in tokens:
                tid = token.get('token_id')
                outcome = token.get('outcome')
                winner = token.get('winner', False)  # Default to False if missing
                
                if tid is None or outcome is None:
                    logger.log(f"   Missing token_id or outcome for {condition_id[:16]}", "WARN")
                    return None
                
                mapping[tid] = {
                    'outcome': outcome,
                    'winner': winner
                }
            
            # Validate we have exactly 2 tokens
            if len(mapping) != 2:
                logger.log(f"   Invalid mapping size {len(mapping)} for {condition_id[:16]}", "WARN")
                return None
            
            # Optional: Verify target token is in mapping
            if token_id and token_id not in mapping:
                logger.log(f"   Token {token_id[:16]} not in condition {condition_id[:16]}", "WARN")
                return None
            
            return mapping
            
        except requests.exceptions.Timeout:
            limiter.on_error("TIMEOUT")
        except requests.exceptions.ConnectionError:
            limiter.on_error("CONNECTION")
        except json.JSONDecodeError as e:
            logger.log(f"   JSON decode error for {condition_id[:16]}: {e}", "ERROR")
            limiter.on_error("JSON_ERROR")
        except Exception as e:
            logger.log(f"   Unexpected error fetching {condition_id[:16]}: {e}", "ERROR")
            limiter.on_error("UNKNOWN")
    
    # All retries exhausted
    logger.log(f"   ❌ Failed to fetch {condition_id[:16]} after {MAX_RETRIES} attempts", "ERROR")
    return None

# ==========================================
# CHECKPOINT MANAGER
# ==========================================
class Checkpoint:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = self.load()
        
    def load(self) -> dict:
        """Load checkpoint or create new one"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                logger.log(f"📄 Loaded checkpoint: {self.filepath}", "INFO")
                return data
            except:
                logger.log(f"⚠️ Checkpoint corrupted, starting fresh", "WARN")
        
        return {
            'phase': 'mapping',  # 'mapping' or 'enrichment'
            'mapping_complete': False,
            'processed_conditions': [],
            'enriched_files': [],
            'failed_conditions': [],
            'failed_files': [],
            'stats': {
                'binary_markets': 0,
                'multi_outcome_markets': 0,
                'api_failures': 0
            }
        }
    
    def save(self):
        """Save checkpoint"""
        self.data['timestamp'] = time.time()
        self.data['memory'] = memory_monitor.get_usage()
        self.data['rate_limit_delay'] = limiter.delay
        
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.log(f"❌ Checkpoint save failed: {e}", "ERROR")

# Global checkpoint - initialized in main()
checkpoint = None

# ==========================================
# MAPPING MANAGER
# ==========================================
class MappingManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.mapping = self.load()
        
    def load(self) -> Dict[str, Dict[str, str]]:
        """Load mapping from disk or create empty"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                logger.log(f"📄 Loaded {len(data)} conditions from {self.filepath}", "INFO")
                return data
            except:
                logger.log(f"⚠️ Mapping file corrupted", "WARN")
        
        return {}
    
    def save(self):
        """Save mapping to disk"""
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.mapping, f, indent=2)
        except Exception as e:
            logger.log(f"❌ Mapping save failed: {e}", "ERROR")
    
    def get(self, condition_id: str) -> Optional[Dict[str, str]]:
        """Get token mapping for condition"""
        return self.mapping.get(condition_id)
    
    def add(self, condition_id: str, token_map: Dict[str, str]):
        """Add token mapping for condition"""
        self.mapping[condition_id] = token_map

# Global mapping manager - initialized in main()
mapping_manager = None

# ==========================================
# PHASE 1: BUILD MAPPING
# ==========================================
def build_token_mapping(input_dir: str):
    """
    Phase 1: Build condition_id → {token_id: outcome} mapping
    """
    logger.log("="*70)
    logger.log("PHASE 1: Building Token Outcome Mapping")
    logger.log("="*70)
    
    # Get all unique condition_ids from parquet files
    logger.log("🔍 Scanning parquet files for condition_ids...")
    
    # Track (condition_id, sample_token_id) pairs
    condition_token_pairs = {}  # condition_id -> sample_token_id
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "batch_*.parquet")))
    
    if not parquet_files:
        logger.log(f"❌ No parquet files found in {input_dir}", "ERROR")
        return False
    
    logger.log(f"   Found {len(parquet_files)} parquet files")
    
    # Scan ALL files to extract unique condition_ids (memory efficient - only reads two columns)
    logger.log(f"   Scanning ALL files for condition_ids...")
    
    for i, filepath in enumerate(parquet_files):
        try:
            df = pd.read_parquet(filepath, columns=['condition_id', 'token_id'])
            # For each unique condition, store one sample token
            for _, row in df.drop_duplicates(subset=['condition_id']).iterrows():
                cond_id = row['condition_id']
                if cond_id not in condition_token_pairs:
                    condition_token_pairs[cond_id] = row['token_id']
            del df  # Explicit cleanup
            
            if (i + 1) % 100 == 0:
                logger.log(f"   Scanned {i+1}/{len(parquet_files)} files: {len(condition_token_pairs)} unique conditions", "DEBUG")
        except Exception as e:
            logger.log(f"   ⚠️ Error reading {filepath}: {e}", "WARN")
    
    condition_ids = set(condition_token_pairs.keys())
    
    # If we have checkpoint progress, filter out already processed
    processed = set(checkpoint.data.get('processed_conditions', []))
    remaining = condition_ids - processed
    
    logger.log(f"📊 Total unique conditions: {len(condition_ids)}")
    logger.log(f"   Already processed: {len(processed)}")
    logger.log(f"   Remaining: {len(remaining)}")
    
    if len(remaining) == 0:
        logger.log("✅ Mapping already complete")
        checkpoint.data['mapping_complete'] = True
        checkpoint.data['phase'] = 'enrichment'
        checkpoint.save()
        return True
    
    # Fetch token mappings for each condition
    logger.log(f"🌐 Fetching token mappings from CLOB API...")
    
    success_count = 0
    fail_count = 0
    
    for idx, condition_id in enumerate(sorted(remaining)):
        # Get sample token for this condition
        sample_token_id = condition_token_pairs.get(condition_id)
        
        # Progress
        if (idx + 1) % 50 == 0:
            logger.log(
                f"   [{idx+1}/{len(remaining)}] Fetching... "
                f"Success: {success_count}, Failed: {fail_count}",
                "INFO"
            )
        
        # Memory check
        if idx % MEMORY_CHECK_INTERVAL == 0:
            if not memory_monitor.check():
                logger.log("⚠️ Memory critical - saving and stopping", "ERROR")
                checkpoint.save()
                mapping_manager.save()
                return False
        
        # Fetch from API
        token_map = fetch_condition_tokens(condition_id, sample_token_id)
        
        if token_map:
            mapping_manager.add(condition_id, token_map)
            success_count += 1
            checkpoint.data['stats']['binary_markets'] = \
                checkpoint.data['stats'].get('binary_markets', 0) + 1
        else:
            fail_count += 1
            checkpoint.data['failed_conditions'].append(condition_id)
            checkpoint.data['stats']['api_failures'] = \
                checkpoint.data['stats'].get('api_failures', 0) + 1
        
        checkpoint.data['processed_conditions'].append(condition_id)
        
        # Save progress periodically
        if (idx + 1) % 100 == 0:
            checkpoint.save()
            mapping_manager.save()
    
    # Final save
    checkpoint.data['mapping_complete'] = True
    checkpoint.data['phase'] = 'enrichment'
    checkpoint.save()
    mapping_manager.save()
    
    logger.log("="*70)
    logger.log("✅ PHASE 1 COMPLETE")
    logger.log(f"   Successful mappings: {success_count}")
    logger.log(f"   Failed mappings: {fail_count}")
    logger.log(f"   Total in mapping: {len(mapping_manager.mapping)}")
    logger.log("="*70)
    
    return True

# ==========================================
# PHASE 2: ENRICH PARQUET FILES
# ==========================================
def enrich_parquet_files(input_dir: str):
    """
    Phase 2: Enrich parquet files with token_outcome column
    """
    logger.log("="*70)
    logger.log("PHASE 2: Enriching Parquet Files")
    logger.log("="*70)
    
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "batch_*.parquet")))
    
    if not parquet_files:
        logger.log(f"❌ No parquet files found in {input_dir}", "ERROR")
        return False
    
    # Filter out already enriched files
    enriched = set(checkpoint.data.get('enriched_files', []))
    remaining_files = [f for f in parquet_files if os.path.basename(f) not in enriched]
    
    logger.log(f"📊 Total files: {len(parquet_files)}")
    logger.log(f"   Already enriched: {len(enriched)}")
    logger.log(f"   Remaining: {len(remaining_files)}")
    
    if len(remaining_files) == 0:
        logger.log("✅ All files already enriched")
        return True
    
    success_count = 0
    fail_count = 0
    
    for idx, filepath in enumerate(remaining_files):
        filename = os.path.basename(filepath)
        
        # Progress
        if (idx + 1) % 100 == 0:
            logger.log(
                f"   [{idx+1}/{len(remaining_files)}] Enriching files... "
                f"Success: {success_count}, Failed: {fail_count}",
                "INFO"
            )
        
        # Memory check
        if idx % MEMORY_CHECK_INTERVAL == 0:
            if not memory_monitor.check():
                logger.log("⚠️ Memory critical - saving and stopping", "ERROR")
                checkpoint.save()
                return False
        
        try:
            # Read file
            df = pd.read_parquet(filepath)
            original_size = len(df)
            
            # Skip if already has token_winner column with data
            if 'token_winner' in df.columns and df['token_winner'].notna().any():
                logger.log(f"   ⏭️ {filename}: Already enriched, skipping", "DEBUG")
                checkpoint.data['enriched_files'].append(filename)
                continue
            
            # ---------------------------------------------------------
            # ENRICHMENT LOGIC
            # ---------------------------------------------------------
            def get_token_data(row):
                """Get token outcome and winner status from mapping"""
                cond_id = row['condition_id']
                token_id = row['token_id']
                
                cond_mapping = mapping_manager.get(cond_id)
                if cond_mapping is None:
                    return pd.Series(['UNKNOWN', False])
                
                token_data = cond_mapping.get(token_id)
                if token_data is None:
                    return pd.Series(['UNKNOWN', False])
                
                return pd.Series([token_data['outcome'], token_data['winner']])
            
            # Apply function to create the new columns
            df[['token_outcome', 'token_winner']] = df.apply(
                lambda row: pd.Series(get_token_data(row)),
                axis=1
            )
            
            # ---------------------------------------------------------
            # Streaming Diagnostics
            # ---------------------------------------------------------
            n_yes = (df['token_outcome'] == 'Yes').sum()
            n_no = (df['token_outcome'] == 'No').sum()
            n_winners = df['token_winner'].sum()
            n_unknown = (df['token_outcome'] == 'UNKNOWN').sum()
            
            # Sample Row
            sample_msg = ""
            if len(df) > 0:
                s_row = df.iloc[0]
                s_out = s_row['token_outcome']
                s_win = s_row['token_winner']
                sample_msg = f"| Sample: {s_out} (Winner={s_win})"
            
            # Log the data stream stats
            if n_unknown == len(df):
                logger.log(f"   ⚠️ {filename}: 100% UNKNOWN outcomes (Check Mapping!)", "WARN")
            else:
                logger.log(
                    f"   ✓ {filename} | Yes:{n_yes} No:{n_no} Win:{n_winners} {sample_msg}", 
                    "INFO"
                )
            
            # ---------------------------------------------------------
            # Safe File Write
            # ---------------------------------------------------------
            temp_fd, temp_path = tempfile.mkstemp(suffix='.parquet', dir=input_dir)
            os.close(temp_fd)
            
            try:
                df.to_parquet(temp_path, index=False)
                
                # Verify temp file
                df_verify = pd.read_parquet(temp_path)
                if len(df_verify) != original_size:
                    raise ValueError(f"Size mismatch: {original_size} → {len(df_verify)}")
                
                # Verify columns persist
                if 'token_outcome' not in df_verify.columns:
                    raise ValueError("token_outcome column missing after write")
                if 'token_winner' not in df_verify.columns:
                    raise ValueError("token_winner column missing after write")
                
                # Atomic rename (replaces original)
                shutil.move(temp_path, filepath)
                
                checkpoint.data['enriched_files'].append(filename)
                success_count += 1
                
                del df_verify
                
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
            
            # Clear memory
            del df
            
        except Exception as e:
            logger.log(f"   ❌ {filename}: {e}", "ERROR")
            checkpoint.data['failed_files'].append(filename)
            fail_count += 1
        
        # Save checkpoint every 10 files
        if idx % 10 == 0:
            checkpoint.save()
    
    # Final save
    checkpoint.save()
    
    logger.log("="*70)
    logger.log("✅ PHASE 2 COMPLETE")
    logger.log(f"   Successful: {success_count}")
    logger.log(f"   Failed: {fail_count}")
    logger.log(f"   Total enriched: {len(checkpoint.data['enriched_files'])}")
    logger.log("="*70)
    
    return True

# ==========================================
# PRE-FLIGHT VALIDATION
# ==========================================
def run_preflight_checks(input_dir: str) -> bool:
    """
    Run validation tests before main execution
    """
    logger.log("="*70)
    logger.log("🔍 PRE-FLIGHT VALIDATION")
    logger.log("="*70)
    
    # Test 1: Check input directory
    logger.log("1. Checking input directory...")
    if not os.path.exists(input_dir):
        logger.log(f"   ❌ Directory not found: {input_dir}", "ERROR")
        return False
    
    parquet_files = glob.glob(os.path.join(input_dir, "batch_*.parquet"))
    if len(parquet_files) == 0:
        logger.log(f"   ❌ No parquet files found in {input_dir}", "ERROR")
        return False
    
    logger.log(f"   ✓ Found {len(parquet_files)} parquet files", "INFO")
    
    # Test 2: Read sample file
    logger.log("2. Testing parquet file read...")
    try:
        sample_file = parquet_files[0]
        df = pd.read_parquet(sample_file, columns=['token_id', 'condition_id'])
        logger.log(f"   ✓ Successfully read {len(df)} rows from {os.path.basename(sample_file)}", "INFO")
        
        # Extract sample condition_id
        sample_condition = df['condition_id'].iloc[0]
        sample_token = df['token_id'].iloc[0]
        logger.log(f"   Sample: condition_id={sample_condition[:32]}..., token_id={sample_token[:32]}...", "DEBUG")
        del df
    except Exception as e:
        logger.log(f"   ❌ Failed to read parquet: {e}", "ERROR")
        return False
    
    # Test 3: Test CLOB API connectivity
    logger.log("3. Testing CLOB API connectivity...")
    try:
        test_mapping = fetch_condition_tokens(sample_condition, sample_token)
        if test_mapping is None:
            logger.log(f"   ⚠️ Could not fetch mapping for sample condition", "WARN")
            logger.log(f"   This may be OK if condition doesn't exist in CLOB", "INFO")
        else:
            logger.log(f"   ✓ API working: got mapping with {len(test_mapping)} tokens", "INFO")
            
            # Verify sample token is in mapping
            if sample_token in test_mapping:
                logger.log(f"   ✓ Sample token found: {test_mapping[sample_token]}", "INFO")
            else:
                logger.log(f"   ⚠️ Sample token not in mapping (unexpected)", "WARN")
    except Exception as e:
        logger.log(f"   ❌ API test failed: {e}", "ERROR")
        return False
    
    # Test 4: Check write permissions
    logger.log("4. Testing write permissions...")
    try:
        test_file = os.path.join(input_dir, ".write_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        logger.log(f"   ✓ Write permissions OK", "INFO")
    except Exception as e:
        logger.log(f"   ❌ No write permission: {e}", "ERROR")
        return False
    
    logger.log("="*70)
    logger.log("✅ PRE-FLIGHT CHECKS PASSED")
    logger.log("="*70)
    return True

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    global logger, checkpoint, mapping_manager
    
    args = parse_args()
    
    # Safety check
    if args.input_dir in ['order_history_batches', '.', './', 'order_history_batches/']:
        print("ERROR: Cannot use 'order_history_batches' as input directory.")
        print("       This is a safety measure to prevent modifying training data.")
        print("       Use the OOS output directory like 'oos_collection/order_history_batches/'")
        sys.exit(1)
    
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Initialize paths scoped to input directory
    log_file = os.path.join(args.input_dir, 'oos_enrich_log.txt')
    checkpoint_file = os.path.join(args.input_dir, 'oos_enrich_checkpoint.json')
    mapping_file = os.path.join(args.input_dir, 'oos_token_mapping.json')
    
    # Initialize globals
    logger = Logger(log_file)
    checkpoint = Checkpoint(checkpoint_file)
    mapping_manager = MappingManager(mapping_file)
    
    logger.log("="*70)
    logger.log("🚀 OOS TOKEN OUTCOME ENRICHMENT")
    logger.log("="*70)
    logger.log(f"Input directory: {args.input_dir}")
    logger.log(f"Log file:        {log_file}")
    logger.log(f"Checkpoint:      {checkpoint_file}")
    logger.log(f"Mapping file:    {mapping_file}")
    
    # Pre-flight validation
    if not run_preflight_checks(args.input_dir):
        logger.log("❌ Pre-flight checks failed - aborting", "ERROR")
        return
    
    # Determine starting phase
    current_phase = checkpoint.data.get('phase', 'mapping')
    mapping_complete = checkpoint.data.get('mapping_complete', False)
    
    logger.log(f"📍 Current phase: {current_phase}")
    logger.log(f"   Mapping complete: {mapping_complete}")
    
    # Phase 1: Build mapping
    if not mapping_complete:
        success = build_token_mapping(args.input_dir)
        if not success:
            logger.log("❌ Phase 1 incomplete - restart to continue", "ERROR")
            return
    
    # Phase 2: Enrich files
    success = enrich_parquet_files(args.input_dir)
    if not success:
        logger.log("❌ Phase 2 incomplete - restart to continue", "ERROR")
        return
    
    # Summary
    logger.log("="*70)
    logger.log("🎉 ENRICHMENT COMPLETE")
    logger.log(f"   Conditions mapped: {len(mapping_manager.mapping)}")
    logger.log(f"   Files enriched: {len(checkpoint.data['enriched_files'])}")
    logger.log(f"   Failed conditions: {len(checkpoint.data.get('failed_conditions', []))}")
    logger.log(f"   Failed files: {len(checkpoint.data.get('failed_files', []))}")
    logger.log(f"   {limiter.get_stats()}")
    logger.log("="*70)
    logger.log("\nOOS data collection complete! Ready for validation.")
    logger.log("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if logger:
            logger.log("\n⚠️ INTERRUPTED BY USER", "WARN")
            logger.log(f"   Progress saved. Restart to continue.", "INFO")
        else:
            print("\n⚠️ INTERRUPTED BY USER")
    except Exception as e:
        if logger:
            logger.log(f"\n❌ FATAL ERROR: {e}", "ERROR")
            import traceback
            logger.log(traceback.format_exc(), "ERROR")
        else:
            print(f"\n❌ FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()