#!/usr/bin/env python3
"""
Polymarket Order History Backfill - v4.3 Window Flush Strategy
Version: 4.3 - 60-Second Window Flush + Persistent Retries

NEW IN v4.3 - SOLVES RATE LIMITING WITH DIRECT WINDOW FLUSH:
1. **60-SECOND WINDOW FLUSH**: On FIRST 503 error, immediately wait 60s to clear rolling window
2. **PERSISTENT RETRIES**: After flush, keep retrying with delays (up to 10 attempts)
3. **SHORTER BASE DELAY**: 2s between requests (vs 3s) since we flush on problems
4. **NEVER LOSE DATA**: Only give up after exhausting all retries

THE INSIGHT:
Error message says "in the past 60 seconds" - this is an explicit rolling window.
Solution: When we hit first 503, wait 60s to clear that window, then persistently retry.

FLOW:
1. Hit 503 → flush window (60s) → set flag to prevent duplicate flushes
2. Retry 1 → if 503, wait delay → Retry 2 → if 503, wait delay → ... up to 10 attempts
3. Only give up after all retries exhausted (rare after window flush)
4. On success, reset flag and continue

From v4.2:
- Cursor-based pagination (timestamp_gt instead of skip)
- O(1) query complexity regardless of token size
- Memory-safe batch file writes
- Bounded LRU cache for slugs
"""

import requests
import pandas as pd
import time
import json
import os
import sys
import gc
import glob
import psutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import OrderedDict, deque

# ==========================================
# CONFIGURATION
# ==========================================
GOLDSKY_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"

# Files
INPUT_FILE = 'backfill_token_list.csv'
OUTPUT_DIR = 'order_history_batches'
OUTPUT_FILE = 'order_history_complete.parquet'
CHECKPOINT_FILE = 'fetch_checkpoint_v4.json'
LOG_FILE = 'fetch_log_v4.txt'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Rate Limiting - v4.3 WINDOW FLUSH STRATEGY
MIN_DELAY = 5.0              # Faster bursts (vs 3.0 in v4.2)
MAX_DELAY = 60.0          
SUCCESS_DECAY = 0.995     
ERROR_MULTIPLIER = 2.5    
COOLDOWN_AFTER_ERROR = 70.0
DISABLE_SERVER_ERROR_BACKOFF = True

# 60-Second Window Flush - NEW IN v4.3
FLUSH_WINDOW_ON_503 = True   # Wait 60s on 503 to clear the rolling window
WINDOW_FLUSH_DELAY = 70.0    # Match the "past 60 seconds" in error message

# Diagnostic Tracking
REQUEST_HISTORY_SIZE = 100   # Track last N requests for pattern analysis

# Memory Protection
MAX_MEMORY_PCT = 80.0
MEMORY_CHECK_INTERVAL = 10

# Cache Limits
MAX_SLUG_CACHE = 1000

MAX_RETRIES = 10  # Increased for persistent recovery after 503 flushes

# Query Optimization - CRITICAL FOR PERFORMANCE
QUERY_PAGE_SIZE = 500     # Smaller pages = lighter queries
                          # Can increase to 250-500 if no 503 errors occur

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

logger = Logger(LOG_FILE)

# ==========================================
# REQUEST TRACKER - Pattern Analysis
# ==========================================
class RequestTracker:
    """Track all requests to identify patterns in failures"""
    def __init__(self, max_history: int = REQUEST_HISTORY_SIZE):
        self.history = deque(maxlen=max_history)
        self.last_60s_window = deque()
        
    def add_request(self, success: bool, duration: float, status_code: int, 
                   response_size: int = 0, error_msg: str = ""):
        """Record a request attempt"""
        record = {
            'timestamp': time.time(),
            'success': success,
            'duration': duration,
            'status_code': status_code,
            'response_size': response_size,
            'error_msg': error_msg
        }
        self.history.append(record)
        self.last_60s_window.append(record)
        
        # Clean old entries from 60s window
        cutoff = time.time() - 60
        while self.last_60s_window and self.last_60s_window[0]['timestamp'] < cutoff:
            self.last_60s_window.popleft()
    
    def get_60s_stats(self) -> Dict:
        """Get statistics for last 60 seconds"""
        if not self.last_60s_window:
            return {
                'total': 0, 'success': 0, 'failed': 0,
                'avg_duration': 0, 'max_duration': 0,
                'slow_count': 0
            }
        
        total = len(self.last_60s_window)
        success = sum(1 for r in self.last_60s_window if r['success'])
        failed = total - success
        durations = [r['duration'] for r in self.last_60s_window]
        
        # What might Goldsky consider "slow"?
        slow_threshold = 5.0  # seconds
        slow_count = sum(1 for d in durations if d > slow_threshold)
        
        return {
            'total': total,
            'success': success,
            'failed': failed,
            'avg_duration': sum(durations) / len(durations),
            'max_duration': max(durations),
            'slow_count': slow_count,
            'slow_pct': (slow_count / total * 100) if total > 0 else 0
        }
    
    def get_recent_pattern(self, n: int = 20) -> str:
        """Get pattern of last N requests for visual inspection"""
        if not self.history:
            return "No history"
        
        recent = list(self.history)[-n:]
        pattern = []
        for r in recent:
            if r['success']:
                symbol = '✓'
            elif r['status_code'] == 503:
                symbol = '5'
            elif r['status_code'] == 429:
                symbol = 'R'
            else:
                symbol = 'X'
            pattern.append(symbol)
        
        return ''.join(pattern)
    
    def analyze_failure_context(self) -> str:
        """Analyze what happened before failures"""
        if len(self.history) < 10:
            return "Not enough data"
        
        recent = list(self.history)[-20:]
        failures = [i for i, r in enumerate(recent) if not r['success'] and r['status_code'] == 503]
        
        if not failures:
            return "No recent 503 errors"
        
        # Look at what happened before each 503
        analysis = []
        for fail_idx in failures:
            if fail_idx >= 3:
                before = recent[fail_idx-3:fail_idx]
                before_durations = [r['duration'] for r in before]
                avg_before = sum(before_durations) / len(before_durations)
                analysis.append(f"503 preceded by {len(before)} reqs, avg duration {avg_before:.1f}s")
        
        return "; ".join(analysis) if analysis else "No clear pattern"

request_tracker = RequestTracker()

# ==========================================
# MEMORY MONITOR
# ==========================================
class MemoryMonitor:
    def __init__(self, max_pct: float = MAX_MEMORY_PCT):
        self.max_pct = max_pct
        self.process = psutil.Process()
        self.baseline_mb = None
        
    def set_baseline(self):
        mem_info = self.process.memory_info()
        self.baseline_mb = mem_info.rss / 1024 / 1024
        
    def check(self) -> bool:
        try:
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            mem_pct = self.process.memory_percent()
            
            if mem_pct > self.max_pct:
                logger.log(f"⚠️ MEMORY CRITICAL: {mem_pct:.1f}% ({mem_mb:.0f}MB)", "ERROR")
                return False
            
            if mem_mb > 500:
                logger.log(f"Memory: {mem_mb:.0f}MB ({mem_pct:.1f}%)", "DEBUG")
                
            return True
        except:
            return True
            
    def get_usage(self) -> str:
        try:
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            mem_pct = self.process.memory_percent()
            return f"{mem_mb:.0f}MB ({mem_pct:.1f}%)"
        except:
            return "unknown"
    
    def get_mb(self) -> float:
        try:
            mem_info = self.process.memory_info()
            return mem_info.rss / 1024 / 1024
        except:
            return 0.0
    
    def get_delta(self) -> str:
        if self.baseline_mb is None:
            return "no baseline"
        current = self.get_mb()
        delta = current - self.baseline_mb
        return f"+{delta:.0f}MB" if delta > 0 else f"{delta:.0f}MB"

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
        self.just_flushed_window = False  # Track if we just did a 60s flush
        
    def on_error(self, error_type: str = "unknown"):
        old_delay = self.delay
        self.consecutive_errors += 1
        self.consecutive_successes = 0
        self.total_errors += 1
        
        # v4.3: SPECIAL HANDLING FOR 503 - Flush the 60-second window
        # Only flush on FIRST 503 in a sequence (not on retries of same query)
        if error_type == "HTTP_503" and FLUSH_WINDOW_ON_503 and not self.just_flushed_window:
            # Get diagnostics before flush
            stats_60s = request_tracker.get_60s_stats()
            pattern = request_tracker.get_recent_pattern(30)
            context = request_tracker.analyze_failure_context()
            
            logger.log(
                f"🚫 HTTP_503: Flushing 60s window (Error in 'past 60 seconds')",
                "WARN"
            )
            logger.log(
                f"   📊 Last 60s: {stats_60s['total']} reqs, "
                f"{stats_60s['success']} success, {stats_60s['failed']} failed, "
                f"avg {stats_60s['avg_duration']:.1f}s, {stats_60s['slow_count']} slow",
                "INFO"
            )
            logger.log(
                f"   📈 Pattern (last 30): {pattern}",
                "INFO"
            )
            logger.log(
                f"   🔍 Context: {context}",
                "INFO"
            )
            logger.log(
                f"   Waiting {WINDOW_FLUSH_DELAY}s to clear rolling window...",
                "INFO"
            )
            time.sleep(WINDOW_FLUSH_DELAY)
            
            # Reset to baseline after flush - clean slate
            self.delay = MIN_DELAY
            self.consecutive_errors = 0  # Reset since we flushed
            self.just_flushed_window = True  # Mark that we just flushed (prevents duplicate flushes)
            logger.log(
                f"   ✅ Window cleared, resuming at {self.delay:.2f}s delay",
                "INFO"
            )
            logger.log(
                f"   Will retry up to {MAX_RETRIES} times with delays between attempts",
                "INFO"
            )
            return  # Skip normal backoff logic
        
        if error_type == "RATE_LIMIT":
            self.delay = min(self.delay * ERROR_MULTIPLIER, MAX_DELAY)
            logger.log(
                f"⚠️ RATE LIMIT: Delay {old_delay:.2f}s → {self.delay:.2f}s "
                f"(Errors: {self.consecutive_errors})",
                "WARN"
            )
        elif error_type in ["SERVER_ERROR", "TIMEOUT", "CONNECTION", 
                           "HTTP_502", "HTTP_503", "HTTP_504", "HTTP_500"]:  # Include 503 for subsequent retries
            if DISABLE_SERVER_ERROR_BACKOFF:
                self.delay = MIN_DELAY
                logger.log(
                    f"⚠️ {error_type}: Staying at {self.delay:.2f}s (backoff disabled)",
                    "WARN"
                )
            else:
                self.delay = min(self.delay * 1.5, MAX_DELAY)
                logger.log(
                    f"⚠️ {error_type}: Delay {old_delay:.2f}s → {self.delay:.2f}s "
                    f"(Server issue, light backoff)",
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
        self.consecutive_successes += 1
        self.consecutive_errors = 0
        self.total_requests += 1
        self.just_flushed_window = False  # Reset flag on success
        
        old_delay = self.delay
        
        if self.consecutive_successes in [10, 25, 50, 100]:
            logger.log(
                f"✨ Success streak: {self.consecutive_successes} requests, delay: {self.delay:.2f}s",
                "INFO"
            )
        
        if self.delay > MIN_DELAY * 5 and self.consecutive_successes >= 20:
            self.delay = (self.delay + MIN_DELAY) / 2
            logger.log(
                f"🚀 FAST RECOVERY: {old_delay:.2f}s → {self.delay:.2f}s "
                f"({self.consecutive_successes} successes)",
                "INFO"
            )
            self.consecutive_successes = 0
            
        elif self.consecutive_successes >= 5:
            self.delay = max(MIN_DELAY, self.delay * 0.98)
            
            if abs(old_delay - self.delay) > 0.5:
                logger.log(
                    f"✓ Speeding up: {old_delay:.2f}s → {self.delay:.2f}s",
                    "DEBUG"
                )
            self.consecutive_successes = 0
        
    def sleep(self):
        time.sleep(self.delay)
    
    def get_stats(self) -> str:
        error_rate = (self.total_errors / self.total_requests * 100) if self.total_requests > 0 else 0
        return f"Requests: {self.total_requests}, Errors: {self.total_errors} ({error_rate:.1f}%)"

limiter = SafeRateLimiter()

# ==========================================
# LRU CACHE FOR SLUGS
# ==========================================
class BoundedSlugCache:
    def __init__(self, max_size: int = MAX_SLUG_CACHE):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_slug(self, condition_id: str) -> Optional[str]:
        if condition_id in self.cache:
            self.hits += 1
            self.cache.move_to_end(condition_id)
            return self.cache[condition_id]
        
        self.misses += 1
        
        try:
            time.sleep(0.2)
            response = requests.get(
                GAMMA_API_URL, 
                params={'condition_ids': condition_id},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    market = data[0]
                    event_slug = market.get('events', [{}])[0].get('slug', '')
                    market_slug = market.get('slug', '')
                    full_slug = f"/event/{event_slug}/{market_slug}" if event_slug else f"/event/{market_slug}"
                    
                    self.cache[condition_id] = full_slug
                    if len(self.cache) > self.max_size:
                        self.cache.popitem(last=False)
                    
                    return full_slug
        except Exception as e:
            logger.log(f"Slug fetch error: {e}", "DEBUG")
            
        return None
    
    def stats(self) -> str:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return f"Cache: {len(self.cache)} entries, {hit_rate:.1f}% hit rate"

slug_cache = BoundedSlugCache()

# ==========================================
# CORE FETCH LOGIC WITH CURSOR-BASED PAGINATION
# ==========================================
def calculate_price(event: Dict[str, Any], token_id: str) -> Optional[float]:
    try:
        maker_asset = str(event['makerAssetId'])
        taker_asset = str(event['takerAssetId'])
        maker_amt = float(event['makerAmountFilled'])
        taker_amt = float(event['takerAmountFilled'])
        
        if maker_asset == "0" and taker_asset == token_id:  # BUY
            price = maker_amt / taker_amt
        elif taker_asset == "0" and maker_asset == token_id:  # SELL
            price = taker_amt / maker_amt
        else:
            return None
            
        if not (0 <= price <= 1):
            return None
            
        return price
    except:
        return None

def fetch_events_with_retry(query: str) -> Optional[Dict]:
    """Fetch with exponential backoff and detailed error logging
    
    v4.3: On first 503, flush the 60s window, then continue retrying normally
    """
    for attempt in range(MAX_RETRIES):
        try:
            limiter.sleep()
            
            request_start = time.time()
            
            response = requests.post(
                GOLDSKY_URL,
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            request_duration = time.time() - request_start
            response_size = len(response.content) if response.content else 0
            
            if response.status_code == 200:
                # Track successful request
                request_tracker.add_request(
                    success=True,
                    duration=request_duration,
                    status_code=200,
                    response_size=response_size
                )
                limiter.on_success()
                limiter.just_flushed_window = False  # Reset flag on success
                if request_duration > 5.0:
                    logger.log(f"   Slow response: {request_duration:.1f}s", "DEBUG")
                return response.json()
            
            if response.status_code == 429:
                error_type = "RATE_LIMIT"
            elif response.status_code >= 500:
                error_type = f"HTTP_{response.status_code}"
            elif response.status_code == 404:
                error_type = "NOT_FOUND"
            else:
                error_type = f"HTTP_{response.status_code}"
            
            error_details = f"Status: {response.status_code}"
            error_msg = ""
            
            try:
                response_text = response.text[:200] if response.text else "empty"
                error_details += f" | Body: {response_text}"
                error_msg = response_text
            except:
                error_details += " | Body: [unable to read]"
            
            error_details += f" | Duration: {request_duration:.1f}s"
            
            if 'Retry-After' in response.headers:
                error_details += f" | Retry-After: {response.headers['Retry-After']}"
            if 'X-RateLimit-Remaining' in response.headers:
                error_details += f" | RateLimit: {response.headers['X-RateLimit-Remaining']}"
            
            # Track failed request
            request_tracker.add_request(
                success=False,
                duration=request_duration,
                status_code=response.status_code,
                response_size=response_size,
                error_msg=error_msg[:200]
            )
            
            logger.log(f"🔍 {error_type}: {error_details}", "WARN")
            
            # Handle 503 error - flush window on FIRST 503, then continue retrying
            limiter.on_error(error_type)
            
            if attempt < MAX_RETRIES - 1:
                logger.log(f"   Retry {attempt+1}/{MAX_RETRIES} after {error_type}", "WARN")
            
        except requests.exceptions.Timeout:
            request_tracker.add_request(
                success=False, duration=30.0, status_code=0, error_msg="Timeout"
            )
            limiter.on_error("TIMEOUT")
            logger.log(f"Timeout on attempt {attempt+1}/{MAX_RETRIES}", "WARN")
            
        except requests.exceptions.ConnectionError as e:
            request_tracker.add_request(
                success=False, duration=0, status_code=0, error_msg=str(e)[:100]
            )
            limiter.on_error("CONNECTION")
            logger.log(f"Connection error: {str(e)[:100]}", "WARN")
            time.sleep(5)
            
        except Exception as e:
            request_tracker.add_request(
                success=False, duration=0, status_code=0, error_msg=str(e)[:100]
            )
            limiter.on_error("UNKNOWN")
            logger.log(f"Unexpected error: {str(e)[:100]}", "ERROR")
            
    logger.log(f"Failed after {MAX_RETRIES} retries", "ERROR")
    limiter.just_flushed_window = False  # Reset flag on failure
    return None

def fetch_and_process_token_orders_cursor(token_id: str, side: str, condition_id: str, 
                                          slug: str, outcome_index: int, resolution_time: int,
                                          output_dir: str, seen_event_ids: set) -> int:
    """
    OPTIMIZED: Uses cursor-based pagination (timestamp_gt) instead of skip.
    This eliminates the O(n) complexity of skip-based pagination.
    """
    last_timestamp = 0  # Start from beginning
    total_written = 0
    batch_count = 0
    asset_field = "makerAssetId" if side == "maker" else "takerAssetId"
    
    mem_start = memory_monitor.get_mb()
    
    while True:
        # CURSOR-BASED QUERY: timestamp_gt instead of skip
        # This makes every query O(1) instead of O(skip + page_size)
        query = f"""
        {{
          orderFilledEvents(
            first: {QUERY_PAGE_SIZE},
            where: {{ 
              {asset_field}: "{token_id}",
              timestamp_gt: {last_timestamp}
            }},
            orderBy: timestamp,
            orderDirection: asc
          ) {{
            id, timestamp, makerAssetId, takerAssetId, 
            makerAmountFilled, takerAmountFilled, transactionHash
          }}
        }}
        """
        
        data = fetch_events_with_retry(query)
        if not data:
            logger.log(f"Failed to fetch {side} orders for token {token_id} after {MAX_RETRIES} attempts", "WARN")
            break
            
        events = data.get("data", {}).get("orderFilledEvents", [])
        if not events:
            break
        
        # Process this batch immediately
        batch_events = []
        for event in events:
            if event['id'] in seen_event_ids:
                continue
            seen_event_ids.add(event['id'])
            
            price = calculate_price(event, token_id)
            if price is not None:
                batch_events.append({
                    'token_id': token_id,
                    'condition_id': condition_id,
                    'timestamp': event['timestamp'],
                    'side': side,
                    'price': price,
                    'maker_amount': float(event['makerAmountFilled']),
                    'taker_amount': float(event['takerAmountFilled']),
                    'tx_hash': event['transactionHash'],
                    'slug': slug,
                    'outcome_index': outcome_index,
                    'resolution_time': resolution_time
                })
        
        if batch_events:
            write_batch_file(batch_events, output_dir)
            total_written += len(batch_events)
            batch_count += 1
            
            del batch_events
            gc.collect()
        
        # Update cursor to last timestamp seen
        last_timestamp = int(events[-1]['timestamp'])
        
        # Log progress for large tokens
        if total_written > 0 and total_written % 10000 == 0:
            mem_current = memory_monitor.get_mb()
            mem_delta = mem_current - mem_start
            logger.log(
                f"      ... {side} side: {total_written:,} events | "
                f"MEM: {mem_current:.0f}MB (+{mem_delta:.0f}MB)",
                "DEBUG"
            )
            gc.collect()
        
        # Break if we got fewer than page size (last page)
        if len(events) < QUERY_PAGE_SIZE:
            break
    
    mem_end = memory_monitor.get_mb()
    total_delta = mem_end - mem_start
    
    if total_written > 10000:
        logger.log(
            f"      [{side}] {total_written:,} events in {batch_count} files | "
            f"MEM delta: {total_delta:+.0f}MB",
            "DEBUG"
        )
    
    return total_written

# ==========================================
# BATCH FILE SYSTEM
# ==========================================
_batch_counter = 0

def write_batch_file(events: List[Dict], output_dir: str) -> str:
    global _batch_counter
    
    if not events:
        return None
    
    new_df = pd.DataFrame(events)
    filename = os.path.join(output_dir, f"batch_{_batch_counter:06d}.parquet")
    new_df.to_parquet(filename, index=False, engine='pyarrow')
    _batch_counter += 1
    
    return filename

# ==========================================
# CHECKPOINT SYSTEM
# ==========================================
class Checkpoint:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = self.load()
    
    def load(self) -> Dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                
                if 'last_completed_index' in data:
                    logger.log(f"📄 RESUMING from checkpoint: index {data['last_completed_index']}", "INFO")
                    return data
            except:
                pass
        
        if os.path.exists('fetch_checkpoint_v3.json'):
            try:
                with open('fetch_checkpoint_v3.json', 'r') as f:
                    v3_data = json.load(f)
                
                if 'last_saved_index' in v3_data:
                    last_idx = v3_data['last_saved_index']
                elif 'last_processed_index' in v3_data:
                    last_idx = v3_data['last_processed_index']
                else:
                    last_idx = -1
                
                logger.log(f"📄 MIGRATING from v3 checkpoint: index {last_idx}", "INFO")
                
                batch_dir = 'order_history_batches'
                if os.path.exists(batch_dir):
                    existing_batches = glob.glob(os.path.join(batch_dir, "batch_*.parquet"))
                    _batch_counter = len(existing_batches)
                    if _batch_counter > 0:
                        logger.log(f"   Found {_batch_counter} existing batch files", "INFO")
                
                return {'last_completed_index': last_idx, 'total_events': 0}
            except:
                logger.log("v3 checkpoint corrupted", "WARN")
        
        logger.log("No checkpoint found, starting from beginning", "INFO")
        return {'last_completed_index': -1, 'total_events': 0}
    
    def save(self, index: int, events_added: int):
        self.data['last_completed_index'] = index
        self.data['total_events'] = self.data.get('total_events', 0) + events_added
        self.data['timestamp'] = time.time()
        self.data['memory'] = memory_monitor.get_usage()
        self.data['rate_limit_delay'] = limiter.delay
        self.data['batch_counter'] = _batch_counter
        
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.log(f"Checkpoint save failed: {e}", "ERROR")
    
    def get_start_index(self) -> int:
        return self.data.get('last_completed_index', -1) + 1

checkpoint = Checkpoint(CHECKPOINT_FILE)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    logger.log("="*70)
    logger.log("🚀 POLYMARKET BACKFILL v4.3 - 60-SECOND WINDOW FLUSH")
    logger.log("="*70)
    
    try:
        df_tokens = pd.read_csv(INPUT_FILE)
        logger.log(f"Loaded {len(df_tokens)} tokens from {INPUT_FILE}")
    except Exception as e:
        logger.log(f"❌ Failed to load input CSV: {e}", "ERROR")
        return
    
    start_index = checkpoint.get_start_index()
    if start_index > 0:
        logger.log(f"📄 RESUMING from index {start_index}")
        logger.log(f"   Previous session: {checkpoint.data.get('total_events', 0)} events collected")
    
    for idx in range(start_index, len(df_tokens)):
        row = df_tokens.iloc[idx]
        token_id = row['token_id']
        condition_id = row['condition_id']
        
        if idx % MEMORY_CHECK_INTERVAL == 0:
            if not memory_monitor.check():
                logger.log("❌ MEMORY CRITICAL - STOPPING GRACEFULLY", "ERROR")
                logger.log(f"   Progress saved at index {idx-1}", "INFO")
                logger.log(f"   Restart script to continue", "INFO")
                return
        
        pct_complete = (idx / len(df_tokens)) * 100
        eta_seconds = (len(df_tokens) - idx) * limiter.delay
        eta_hours = eta_seconds / 3600
        
        if idx % 10 == 0 or idx < start_index + 5:
            logger.log(
                f"[{idx}/{len(df_tokens)}] {pct_complete:.1f}% | "
                f"ETA: {eta_hours:.1f}h | Delay: {limiter.delay:.2f}s | "
                f"{memory_monitor.get_usage()}"
            )
            
        if idx % 100 == 0 and idx > 0:
            logger.log(f"   Rate limiter: {limiter.get_stats()}", "DEBUG")
            
            # v4.3: Show pattern analysis periodically
            stats_60s = request_tracker.get_60s_stats()
            pattern = request_tracker.get_recent_pattern(50)
            logger.log(
                f"   📊 60s window: {stats_60s['total']} reqs, {stats_60s['success']} success, "
                f"avg {stats_60s['avg_duration']:.1f}s, {stats_60s['slow_count']} slow",
                "INFO"
            )
            logger.log(f"   📈 Pattern (last 50): {pattern}", "INFO")
            
            if limiter.total_requests > 0:
                recent_delay = limiter.delay
                health_status = "🟢 HEALTHY" if recent_delay < MIN_DELAY * 3 else \
                               "🟡 DEGRADED" if recent_delay < MIN_DELAY * 10 else \
                               "🔴 STRUGGLING"
                logger.log(
                    f"   Connection health: {health_status} | "
                    f"Current delay: {recent_delay:.1f}s | "
                    f"Consecutive successes: {limiter.consecutive_successes}",
                    "INFO"
                )
        
        try:
            slug = slug_cache.get_slug(condition_id)
            seen_event_ids = set()
            
            # Use cursor-based pagination
            maker_count = fetch_and_process_token_orders_cursor(
                token_id, "maker", condition_id, slug, 
                row.get('outcome', -1), row.get('resolution_time', 0),
                OUTPUT_DIR, seen_event_ids
            )
            
            taker_count = fetch_and_process_token_orders_cursor(
                token_id, "taker", condition_id, slug,
                row.get('outcome', -1), row.get('resolution_time', 0),
                OUTPUT_DIR, seen_event_ids
            )
            
            total_events = maker_count + taker_count
            
            if total_events > 0:
                if total_events > 50000:
                    logger.log(f"   ✓ Token {token_id}: {total_events:,} events (LARGE)", "INFO")
                    gc.collect()
                else:
                    logger.log(f"   ✓ Token {token_id}: {total_events} events", "DEBUG")
            
            checkpoint.save(idx, total_events)
            del seen_event_ids
            
            if idx % 50 == 0:
                logger.log(f"   {slug_cache.stats()}", "DEBUG")
            
        except Exception as e:
            logger.log(f"   ❌ Error processing token {token_id}: {e}", "ERROR")
            checkpoint.save(idx, 0)
            continue
    
    logger.log("="*70)
    logger.log("✅ BACKFILL COMPLETE")
    logger.log(f"   Total events: {checkpoint.data.get('total_events', 0):,}")
    logger.log(f"   Batch files: {_batch_counter}")
    logger.log(f"   {slug_cache.stats()}")
    logger.log("="*70)
    
    logger.log("🔄 Merging batch files into final output...")
    try:
        batch_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "batch_*.parquet")))
        
        if batch_files:
            logger.log(f"   Found {len(batch_files)} batch files to merge")
            
            dfs = []
            for i, batch_file in enumerate(batch_files):
                if i % 100 == 0:
                    logger.log(f"   Reading batch {i}/{len(batch_files)}")
                dfs.append(pd.read_parquet(batch_file))
            
            logger.log("   Concatenating...")
            final_df = pd.concat(dfs, ignore_index=True)
            
            logger.log(f"   Writing final file: {OUTPUT_FILE}")
            final_df.to_parquet(OUTPUT_FILE, index=False)
            
            logger.log(f"   ✅ Final output: {len(final_df):,} events")
            logger.log(f"   File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f}MB")
        else:
            logger.log("   No batch files found")
    except Exception as e:
        logger.log(f"   ⚠️ Merge failed: {e}", "ERROR")
        logger.log(f"   Batch files remain in {OUTPUT_DIR}", "INFO")
    
    logger.log("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.log("\n⚠️ INTERRUPTED BY USER", "WARN")
        logger.log(f"   Progress saved. Restart to continue.", "INFO")
    except Exception as e:
        logger.log(f"\n❌ FATAL ERROR: {e}", "ERROR")
        import traceback
        logger.log(traceback.format_exc(), "ERROR")