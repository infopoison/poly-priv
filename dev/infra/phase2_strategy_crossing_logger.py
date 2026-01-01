#!/usr/bin/env python3
"""
Strategy B Dip Monitor
======================

PURPOSE:
  Detect and log Strategy B dip signals: 10%+ dips from anchor price
  in longshot tokens approaching resolution.

STRATEGY B SIGNAL (exactly as backtested):
  - Token is 10-51% (longshot side, but can trigger 10pt dip)
  - In specified observation window (48h→24h OR 24h→12h)
  - Price DROPS 10 POINTS from ANCHOR (e.g., 35% → 25%)
  - We FADE the dip (buy after the drop)

ANCHOR DEFINITION:
  Anchor = Price at the START of the observation window.
  The exact price when the token entered the window.
  
  For tokens already mid-window at script start: anchor is approximated 
  using current price (imperfect but necessary).
  
  For tokens entering the window during monitoring: anchor would be 
  captured exactly at window entry (requires longer monitoring runs).

NOTE ON THRESHOLD:
  10 POINTS means absolute probability points, NOT relative percentage.
  - 40% anchor → triggers at 30% (dropped 10 points) ✓
  - 15% anchor → triggers at 5% (dropped 10 points) ✓
  - 8% anchor → CANNOT TRIGGER (excluded from monitoring)

WINDOW SELECTION:
  --window 48h_to_24h   Monitor tokens 24-48h from resolution
  --window 24h_to_12h   Monitor tokens 12-24h from resolution
  
  Windows are SEPARATE strategies. Run one at a time for clean analysis.

OUTPUT:
  - dip_signals_WINDOW_YYYYMMDD_HHMMSS.jsonl

BACKTEST VALIDATION FIELDS (in each signal):
  - trigger_source: "book" | "price_change" | "trade" - what event caused signal
  - spread_at_signal: bid-ask spread at trigger time
  - volume_during_dip: volume traded while price was below anchor
  - trades_during_dip: trade count while price was below anchor
  
  These allow post-hoc filtering to match backtest assumptions (trade-driven only).

Usage:
    # Probe mode for 48h→24h window
    python strategy_b_dip_monitor.py --window 48h_to_24h --probe
    
    # Full collection for 24h→12h window  
    python strategy_b_dip_monitor.py --window 24h_to_12h --duration 86400

Requirements:
    pip install websockets requests
"""

import json
import asyncio
import websockets
import requests
import sys
import argparse
import time
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

GAMMA_API = "https://gamma-api.polymarket.com/markets"
WS_URI = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Strategy B parameters
DIP_THRESHOLD = 0.10  # 10 POINTS (not percent) - e.g., 40% → 30% is a 10 point dip
SUB_51_THRESHOLD = 0.51  # Longshot definition
MIN_ANCHOR_FOR_SIGNAL = 0.10  # Anchors below 10% can't produce 10-point dips

# Window definitions: (min_hours_to_resolution, max_hours_to_resolution)
WINDOWS = {
    "48h_to_24h": (24, 48),
    "24h_to_12h": (12, 24),
}

# Defaults
DEFAULT_WINDOW = "48h_to_24h"
DEFAULT_MIN_LIQUIDITY = 1000
DEFAULT_DURATION = 86400  # 24 hours
PROBE_DURATION = 300      # 5 minutes
PROBE_MAX_MARKETS = 50

# Monitoring
STATS_INTERVAL = 30
PING_INTERVAL = 10

# Reconnection settings
RECONNECT_BASE_DELAY = 5      # Initial reconnect delay (seconds)
RECONNECT_MAX_DELAY = 60      # Maximum reconnect delay
RECONNECT_BACKOFF = 2         # Exponential backoff multiplier


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TokenState:
    """State for a single token with anchor-based dip tracking."""
    token_id: str
    condition_id: str
    side: str  # "YES" or "NO"
    question: str
    end_date: datetime
    liquidity: float
    window: str  # "48h_to_24h" or "24h_to_12h"
    
    # Anchor = price at window start (captured from first websocket update)
    anchor_price: Optional[float] = None  # Set on first websocket price, not REST
    anchor_time: Optional[datetime] = None
    anchor_is_approximate: bool = True  # True since we start mid-window
    
    # Initial price from REST (for reference only, not used as anchor)
    rest_api_price: Optional[float] = None
    
    # Current state
    current_price: Optional[float] = None
    last_price: Optional[float] = None
    last_update: Optional[datetime] = None
    current_spread: Optional[float] = None  # Best ask - best bid
    current_best_ask: Optional[float] = None
    
    # Dip tracking
    max_dip_pct: float = 0.0
    dip_triggered: bool = False
    dip_trigger_time: Optional[datetime] = None
    dip_trigger_price: Optional[float] = None
    
    # Volume tracking - cumulative since anchor
    cumulative_volume: float = 0.0
    trade_count: int = 0
    
    # Volume tracking - during dip specifically (when price < anchor)
    volume_during_dip: float = 0.0
    trades_during_dip: int = 0
    
    def get_dip_points(self) -> float:
        """Calculate current dip in POINTS from anchor (not percentage)."""
        if self.current_price is None or self.anchor_price is None:
            return 0.0
        return self.anchor_price - self.current_price
    
    def update_price(self, new_price: float, trigger_source: str = "unknown", 
                     spread: Optional[float] = None, best_ask: Optional[float] = None) -> Optional[dict]:
        """
        Update price and detect dip threshold crossing.
        
        Args:
            new_price: New best bid price
            trigger_source: "book", "price_change", or "trade"
            spread: Current spread (ask - bid) if available
            best_ask: Current best ask if available
        """
        self.last_price = self.current_price
        self.current_price = new_price
        self.last_update = datetime.now(timezone.utc)
        
        if spread is not None:
            self.current_spread = spread
        if best_ask is not None:
            self.current_best_ask = best_ask
        
        # Set anchor on FIRST websocket price (not REST API price)
        if self.anchor_price is None:
            self.anchor_price = new_price
            self.anchor_time = self.last_update
            return None  # No signal on anchor establishment
        
        dip_points = self.get_dip_points()
        self.max_dip_pct = max(self.max_dip_pct, dip_points)  # Now storing max points
        
        # Only trigger once per token
        if not self.dip_triggered and dip_points >= DIP_THRESHOLD:
            self.dip_triggered = True
            self.dip_trigger_time = self.last_update
            self.dip_trigger_price = new_price
            
            hours_to_res = (self.end_date - datetime.now(timezone.utc)).total_seconds() / 3600
            
            return {
                "ts": self.last_update.isoformat(),
                "event": "dip_signal",
                "window": self.window,
                "condition_id": self.condition_id,
                "token_id": self.token_id,
                "token_side": self.side,
                "anchor_price": round(self.anchor_price, 4),
                "anchor_is_approximate": self.anchor_is_approximate,
                "trigger_price": round(new_price, 4),
                "dip_points": round(dip_points * 100, 1),
                "hours_to_resolution": round(hours_to_res, 2),
                # Backtest validation fields
                "trigger_source": trigger_source,
                "spread_at_signal": round(self.current_spread, 4) if self.current_spread else None,
                "best_ask_at_signal": round(self.current_best_ask, 4) if self.current_best_ask else None,
                "volume_during_dip": round(self.volume_during_dip, 2),
                "trades_during_dip": self.trades_during_dip,
                # Cumulative stats
                "cumulative_volume": round(self.cumulative_volume, 2),
                "trade_count": self.trade_count,
                "liquidity": self.liquidity,
                "question": self.question[:100]
            }
        
        return None
    
    def add_trade_volume(self, size: float):
        """Add volume from a trade event. Track dip-specific volume if price below anchor."""
        self.cumulative_volume += size
        self.trade_count += 1
        
        # Track volume during dip (when current price is below anchor)
        if self.anchor_price is not None and self.current_price is not None:
            if self.current_price < self.anchor_price:
                self.volume_during_dip += size
                self.trades_during_dip += 1


@dataclass
class MonitorStats:
    """Aggregate monitoring statistics."""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    window: str = ""
    
    # Connection
    connection_attempts: int = 0
    successful_connections: int = 0
    disconnections: int = 0
    
    # Messages
    total_messages: int = 0
    total_events: int = 0
    book_events: int = 0
    price_change_events: int = 0
    trade_events: int = 0
    parse_errors: int = 0
    
    # Signals
    dip_signals: int = 0
    
    # Rate tracking
    message_timestamps: List[float] = field(default_factory=list)
    
    def record_message(self):
        now_ts = time.time()
        self.total_messages += 1
        self.message_timestamps.append(now_ts)
        cutoff = now_ts - 60
        self.message_timestamps = [t for t in self.message_timestamps if t > cutoff]
    
    def get_msg_rate(self) -> float:
        if len(self.message_timestamps) < 2:
            return 0.0
        window = self.message_timestamps[-1] - self.message_timestamps[0]
        return len(self.message_timestamps) / window if window > 0 else 0.0
    
    def get_runtime(self) -> float:
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()


# =============================================================================
# MARKET FETCHING
# =============================================================================

def fetch_strategy_b_targets(window: str, min_liquidity: float, max_markets: Optional[int]) -> List[TokenState]:
    """
    Fetch longshot tokens in the specified observation window.
    
    Args:
        window: "48h_to_24h" or "24h_to_12h"
        min_liquidity: Minimum liquidity filter
        max_markets: Optional cap on number of markets
    
    Returns:
        List of TokenState for sub-51% tokens in the window
    """
    min_hours, max_hours = WINDOWS[window]
    
    print(f"Fetching markets from Gamma API...")
    print(f"  Window: {window} ({min_hours}h to {max_hours}h from resolution)")
    
    # Fetch all active markets
    markets = []
    offset = 0
    while True:
        resp = requests.get(
            GAMMA_API,
            params={"active": "true", "closed": "false", "limit": 500, "offset": offset},
            timeout=30
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        markets.extend(batch)
        if len(batch) < 500:
            break
        offset += 500
    
    print(f"  Fetched {len(markets)} active markets")
    
    now = datetime.now(timezone.utc)
    tokens = []
    markets_in_window = 0
    
    for m in markets:
        # Parse end date
        end_date = None
        for field_name in ["endDateIso", "endDate"]:
            raw = m.get(field_name)
            if raw:
                try:
                    if raw.endswith("Z"):
                        raw = raw[:-1] + "+00:00"
                    end_date = datetime.fromisoformat(raw)
                    if end_date.tzinfo is None:
                        end_date = end_date.replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue
        
        if not end_date:
            continue
        
        # Check if in target window
        hours_to_res = (end_date - now).total_seconds() / 3600
        if not (min_hours <= hours_to_res <= max_hours):
            continue
        
        # Parse liquidity
        try:
            liquidity = float(m.get("liquidity", 0))
        except (ValueError, TypeError):
            liquidity = 0
        
        if liquidity < min_liquidity:
            continue
        
        # Parse token IDs
        clob_tokens = m.get("clobTokenIds")
        if not clob_tokens:
            continue
        
        try:
            if isinstance(clob_tokens, str):
                clob_tokens = json.loads(clob_tokens)
            if len(clob_tokens) < 2:
                continue
            yes_token = clob_tokens[0]
            no_token = clob_tokens[1]
        except (json.JSONDecodeError, IndexError):
            continue
        
        # Parse current prices
        raw_prices = m.get("outcomePrices")
        yes_price, no_price = None, None
        if raw_prices:
            try:
                if isinstance(raw_prices, str):
                    prices = json.loads(raw_prices)
                else:
                    prices = raw_prices
                yes_price = float(prices[0])
                no_price = float(prices[1])
            except:
                pass
        
        if yes_price is None or no_price is None:
            continue
        
        condition_id = m.get("conditionId", "")
        question = m.get("question", "")[:100]
        
        markets_in_window += 1
        
        # Only include tokens that:
        # 1. Are sub-51% (longshots)
        # 2. Are >= 10% (can actually produce a 10-point dip)
        # Note: anchor_price will be set on first websocket update, not from REST
        
        if MIN_ANCHOR_FOR_SIGNAL <= yes_price < SUB_51_THRESHOLD:
            tokens.append(TokenState(
                token_id=yes_token,
                condition_id=condition_id,
                side="YES",
                question=question,
                end_date=end_date,
                liquidity=liquidity,
                window=window,
                rest_api_price=yes_price,  # For reference only
                anchor_is_approximate=True,
            ))
        
        if MIN_ANCHOR_FOR_SIGNAL <= no_price < SUB_51_THRESHOLD:
            tokens.append(TokenState(
                token_id=no_token,
                condition_id=condition_id,
                side="NO",
                question=question,
                end_date=end_date,
                liquidity=liquidity,
                window=window,
                rest_api_price=no_price,  # For reference only
                anchor_is_approximate=True,
            ))
        
        if max_markets and markets_in_window >= max_markets:
            break
    
    print(f"  Markets in {window} window (liq >= ${min_liquidity:,.0f}): {markets_in_window}")
    print(f"  Tokens 10-51% (can trigger 10pt dip): {len(tokens)}")
    
    return tokens


# =============================================================================
# DIP MONITOR
# =============================================================================

class DipMonitor:
    """WebSocket monitor that detects Strategy B dip signals."""
    
    def __init__(self, tokens: List[TokenState], window: str, log_file: str, probe_mode: bool = False):
        self.tokens = {t.token_id: t for t in tokens}
        self.window = window
        self.log_file = log_file
        self.probe_mode = probe_mode
        self.stats = MonitorStats(window=window)
        self.running = False
        self.log_handle = None
    
    def _log_signal(self, signal: dict):
        """Write signal to log file and console."""
        if self.log_handle:
            self.log_handle.write(json.dumps(signal) + "\n")
            self.log_handle.flush()
        
        approx_note = " (approx anchor)" if signal.get("anchor_is_approximate") else ""
        side_emoji = "📈" if signal["token_side"] == "YES" else "📉"
        
        # Format spread
        spread_str = f"{signal['spread_at_signal']:.2%}" if signal.get('spread_at_signal') else "?"
        
        print(f"\n  🎯 DIP SIGNAL [{self.window}]")
        print(f"     {side_emoji} [{signal['token_side']}] {signal['question'][:50]}")
        print(f"     Anchor: {signal['anchor_price']:.1%} → Trigger: {signal['trigger_price']:.1%}{approx_note}")
        print(f"     Dip: {signal['dip_points']:.1f} pts  |  Hours to res: {signal['hours_to_resolution']:.1f}h")
        print(f"     Source: {signal['trigger_source']}  |  Spread: {spread_str}")
        print(f"     Vol during dip: ${signal['volume_during_dip']:,.0f} ({signal['trades_during_dip']} trades)")
        print()
    
    def process_message(self, raw_message: str):
        """Process websocket message."""
        self.stats.record_message()
        
        if raw_message == "PONG":
            return
        
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            self.stats.parse_errors += 1
            return
        
        events = data if isinstance(data, list) else [data]
        
        for event in events:
            if not isinstance(event, dict):
                continue
            
            self.stats.total_events += 1
            event_type = event.get("event_type", "")
            
            if event_type == "book":
                self.stats.book_events += 1
                self._process_book(event)
            elif event_type == "price_change":
                self.stats.price_change_events += 1
                self._process_price_change(event)
            elif event_type == "last_trade_price":
                self.stats.trade_events += 1
                self._process_trade(event)
    
    def _process_book(self, data: dict):
        """Process book snapshot."""
        asset_id = data.get("asset_id")
        if asset_id not in self.tokens:
            return
        
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        
        if bids:
            try:
                best_bid = max(float(b["price"]) for b in bids)
                best_ask = min(float(a["price"]) for a in asks) if asks else None
                spread = (best_ask - best_bid) if best_ask else None
                
                signal = self.tokens[asset_id].update_price(
                    best_bid, 
                    trigger_source="book",
                    spread=spread,
                    best_ask=best_ask
                )
                if signal:
                    self._record_signal(signal)
            except (ValueError, KeyError):
                pass
    
    def _process_price_change(self, data: dict):
        """Process price change."""
        for change in data.get("price_changes", []):
            asset_id = change.get("asset_id")
            if asset_id not in self.tokens:
                continue
            
            best_bid = change.get("best_bid")
            best_ask = change.get("best_ask")
            
            if best_bid:
                try:
                    best_bid_f = float(best_bid)
                    best_ask_f = float(best_ask) if best_ask else None
                    spread = (best_ask_f - best_bid_f) if best_ask_f else None
                    
                    signal = self.tokens[asset_id].update_price(
                        best_bid_f,
                        trigger_source="price_change",
                        spread=spread,
                        best_ask=best_ask_f
                    )
                    if signal:
                        self._record_signal(signal)
                except ValueError:
                    pass
    
    def _process_trade(self, data: dict):
        """Process trade event."""
        asset_id = data.get("asset_id")
        if asset_id not in self.tokens:
            return
        
        token = self.tokens[asset_id]
        
        # Track volume
        size = data.get("size")
        if size:
            try:
                token.add_trade_volume(float(size))
            except ValueError:
                pass
        
        # Update price (spread is kept from last book/price_change update)
        price = data.get("price")
        if price:
            try:
                signal = token.update_price(
                    float(price),
                    trigger_source="trade"
                )
                if signal:
                    self._record_signal(signal)
            except ValueError:
                pass
    
    def _record_signal(self, signal: dict):
        """Record a dip signal."""
        self.stats.dip_signals += 1
        self._log_signal(signal)
    
    def print_stats(self):
        """Print current statistics."""
        runtime = self.stats.get_runtime()
        rate = self.stats.get_msg_rate()
        
        # Dip distribution (now in points)
        anchored = sum(1 for t in self.tokens.values() if t.anchor_price is not None)
        dips = [t.max_dip_pct * 100 for t in self.tokens.values() if t.max_dip_pct > 0]  # max_dip_pct now stores points
        near_threshold = sum(1 for d in dips if 5 <= d < 10)
        at_threshold = sum(1 for d in dips if d >= 10)
        
        print(f"\n{'='*70}")
        print(f"STATS [{self.window}] @ {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC "
              f"(runtime: {runtime/60:.1f} min)")
        print(f"{'='*70}")
        print(f"Tokens monitored:    {len(self.tokens)} ({anchored} anchored)")
        print(f"Messages:            {self.stats.total_messages} ({rate:.1f}/sec)")
        print(f"")
        print(f"DIP TRACKING (points from anchor):")
        print(f"  Near threshold (5-10 pts): {near_threshold}")
        print(f"  At/past threshold (≥10 pts): {at_threshold}")
        print(f"  SIGNALS LOGGED:            {self.stats.dip_signals}")
        
        if runtime > 60:
            per_hour = self.stats.dip_signals * 3600 / runtime
            print(f"  Projected/hour:         {per_hour:.1f}")
        
        print(f"")
        print(f"Log file: {self.log_file}")
        print(f"{'='*70}\n")
    
    async def _ping_loop(self, ws):
        """Keepalive pings."""
        while self.running:
            try:
                await ws.send("PING")
                await asyncio.sleep(PING_INTERVAL)
            except:
                break
    
    async def _stats_loop(self):
        """Periodic stats."""
        while self.running:
            await asyncio.sleep(STATS_INTERVAL)
            if self.running:
                self.print_stats()
    
    async def connect_and_monitor(self):
        """Single connection attempt. Returns True if should retry, False if done."""
        token_ids = list(self.tokens.keys())
        
        print(f"\nConnecting to {WS_URI}")
        print(f"Subscribing to {len(token_ids)} tokens...")
        
        self.stats.connection_attempts += 1
        
        try:
            async with websockets.connect(
                WS_URI,
                ping_interval=None,
                ping_timeout=None,
                max_size=10_000_000,
                close_timeout=10,
            ) as ws:
                
                self.stats.successful_connections += 1
                print(f"[✓] Connected (attempt #{self.stats.connection_attempts})")
                
                await ws.send(json.dumps({
                    "assets_ids": token_ids,
                    "type": "market"
                }))
                print(f"[✓] Subscribed to {len(token_ids)} tokens")
                print(f"[✓] Logging to: {self.log_file}")
                print(f"\nMonitoring for 10pt+ dips from anchor...\n")
                
                self.running = True
                ping_task = asyncio.create_task(self._ping_loop(ws))
                stats_task = asyncio.create_task(self._stats_loop())
                
                try:
                    async for message in ws:
                        self.process_message(message)
                except websockets.ConnectionClosed as e:
                    print(f"\n[!] Connection closed: {e}")
                    self.stats.disconnections += 1
                    return True  # Should retry
                finally:
                    self.running = False
                    ping_task.cancel()
                    stats_task.cancel()
                    
        except asyncio.CancelledError:
            print(f"\n[!] Connection cancelled")
            return False  # Don't retry, we're shutting down
        except Exception as e:
            print(f"\n[!] Connection error: {e}")
            self.stats.disconnections += 1
            return True  # Should retry
        
        return False  # Normal exit
    
    async def run(self, duration_seconds: int):
        """Run the monitor with automatic reconnection."""
        mode_str = "PROBE" if self.probe_mode else "COLLECTION"
        
        print(f"\n{'='*70}")
        print(f"STRATEGY B DIP MONITOR - {mode_str} MODE")
        print(f"{'='*70}")
        print(f"Window:          {self.window}")
        print(f"Duration:        {duration_seconds}s ({duration_seconds/60:.1f} min)")
        print(f"Tokens:          {len(self.tokens)}")
        print(f"Dip threshold:   {int(DIP_THRESHOLD * 100)} points")
        print(f"Log file:        {self.log_file}")
        print(f"Reconnection:    ENABLED (auto-retry on disconnect)")
        print(f"{'='*70}\n")
        
        # REST price distribution (anchors set from first websocket price)
        rest_prices = [t.rest_api_price for t in self.tokens.values() if hasattr(t, 'rest_api_price') and t.rest_api_price]
        if rest_prices:
            print(f"REST API prices (anchors set on first websocket update):")
            print(f"  Min: {min(rest_prices):.1%}  Median: {sorted(rest_prices)[len(rest_prices)//2]:.1%}  Max: {max(rest_prices):.1%}")
            print()
        
        # Open log file in append mode so reconnections don't lose data
        self.log_handle = open(self.log_file, "a")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        reconnect_delay = RECONNECT_BASE_DELAY
        
        try:
            while time.time() < end_time:
                remaining = end_time - time.time()
                if remaining <= 0:
                    break
                
                try:
                    # Run connection with remaining time as timeout
                    should_retry = await asyncio.wait_for(
                        self.connect_and_monitor(),
                        timeout=remaining
                    )
                    
                    if not should_retry:
                        break
                    
                    # Connection dropped, check if we should retry
                    if time.time() >= end_time:
                        print(f"\n[✓] Duration complete (after disconnect)")
                        break
                    
                    # Exponential backoff
                    print(f"\n[*] Reconnecting in {reconnect_delay:.0f}s...")
                    await asyncio.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * RECONNECT_BACKOFF, RECONNECT_MAX_DELAY)
                    
                except asyncio.TimeoutError:
                    print("\n[✓] Duration complete")
                    break
                    
        except KeyboardInterrupt:
            print("\n[!] Interrupted by user")
        finally:
            self.running = False
            if self.log_handle:
                self.log_handle.close()
            self.print_final_report()
    
    def print_final_report(self):
        """Final report."""
        runtime = self.stats.get_runtime()
        
        print(f"\n{'='*70}")
        print(f"FINAL REPORT [{self.window}]")
        print(f"{'='*70}")
        print(f"")
        print(f"CONFIGURATION")
        print(f"  Window:                {self.window}")
        print(f"  Mode:                  {'Probe' if self.probe_mode else 'Collection'}")
        print(f"  Tokens monitored:      {len(self.tokens)}")
        print(f"  Runtime:               {runtime:.0f}s ({runtime/60:.1f} min)")
        print(f"  Dip threshold:         {int(DIP_THRESHOLD * 100)} points")
        print(f"")
        print(f"CONNECTION")
        print(f"  Attempts:              {self.stats.connection_attempts}")
        print(f"  Successful:            {self.stats.successful_connections}")
        print(f"  Disconnections:        {self.stats.disconnections}")
        print(f"")
        print(f"THROUGHPUT")
        print(f"  Total messages:        {self.stats.total_messages}")
        if runtime > 0:
            print(f"  Avg rate:              {self.stats.total_messages/runtime:.1f}/sec")
        print(f"")
        print(f"DIP SIGNALS")
        print(f"  Signals detected:      {self.stats.dip_signals}")
        
        if runtime > 60:
            per_hour = self.stats.dip_signals * 3600 / runtime
            print(f"  Projected/hour:        {per_hour:.1f}")
            print(f"  Projected/24h:         {per_hour * 24:.0f}")
        
        # Near misses (now in points)
        near_misses = [(t.question[:40], t.max_dip_pct*100) 
                       for t in self.tokens.values() 
                       if 5 <= t.max_dip_pct * 100 < 10]
        
        if near_misses:
            print(f"")
            print(f"NEAR MISSES (5-10 point dips):")
            for q, dip in sorted(near_misses, key=lambda x: -x[1])[:5]:
                print(f"  {dip:.1f} pts: {q}")
        
        print(f"")
        print(f"OUTPUT")
        print(f"  Log file:              {self.log_file}")
        
        if os.path.exists(self.log_file):
            size = os.path.getsize(self.log_file)
            print(f"  File size:             {size} bytes")
        
        print(f"")
        
        if self.probe_mode:
            if self.stats.dip_signals > 0:
                print(f"✅ PROBE SUCCESS - Dip signals detected!")
                print(f"   Ready for full collection: --window {self.window} --duration 86400")
            elif self.stats.total_messages > 0:
                near_count = sum(1 for t in self.tokens.values() if t.max_dip_pct > 0.05)
                if near_count > 0:
                    print(f"⚠️  PROBE PARTIAL - {near_count} tokens saw 5+ point dips (threshold is 10).")
                    print(f"   Infrastructure working. Run longer for signals.")
                else:
                    print(f"⚠️  PROBE PARTIAL - Connection works, markets stable.")
            else:
                print(f"❌ PROBE FAILED - No messages received.")
        
        print(f"{'='*70}\n")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Strategy B Dip Monitor - Detects 10%+ dips from anchor price",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Probe 48h_to_24h window
  python strategy_b_dip_monitor.py --window 48h_to_24h --probe
  
  # Full 24h collection for 24h_to_12h window
  python strategy_b_dip_monitor.py --window 24h_to_12h --duration 86400
"""
    )
    parser.add_argument("--window", type=str, default=DEFAULT_WINDOW,
                       choices=list(WINDOWS.keys()),
                       help=f"Observation window (default: {DEFAULT_WINDOW})")
    parser.add_argument("--probe", action="store_true",
                       help="Probe mode: short verification run")
    parser.add_argument("--duration", type=int, default=None,
                       help="Duration in seconds (default: 300 probe, 86400 collection)")
    parser.add_argument("--min-liquidity", type=float, default=DEFAULT_MIN_LIQUIDITY,
                       help=f"Minimum liquidity (default: ${DEFAULT_MIN_LIQUIDITY})")
    parser.add_argument("--max-markets", type=int, default=None,
                       help="Maximum markets to monitor")
    parser.add_argument("--output", type=str, default=None,
                       help="Output log file")
    
    args = parser.parse_args()
    
    probe_mode = args.probe
    window = args.window
    
    if args.duration:
        duration = args.duration
    else:
        duration = PROBE_DURATION if probe_mode else DEFAULT_DURATION
    
    max_markets = args.max_markets
    if max_markets is None and probe_mode:
        max_markets = PROBE_MAX_MARKETS
    
    if args.output:
        log_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"dip_signals_{window}_{timestamp}.jsonl"
    
    print(f"\n{'='*70}")
    print(f"STRATEGY B DIP MONITOR")
    print(f"{'='*70}")
    print(f"")
    print(f"STRATEGY B SIGNAL (exactly as backtested):")
    print(f"  • Token is 10-51% (longshot, can dip 10 points)")
    print(f"  • In {window} observation window")
    print(f"  • Price dips 10 POINTS from ANCHOR (e.g., 35%→25%)")
    print(f"  • We FADE the dip")
    print(f"")
    print(f"Configuration:")
    print(f"  Window:        {window}")
    print(f"  Mode:          {'PROBE' if probe_mode else 'COLLECTION'}")
    print(f"  Duration:      {duration}s ({duration/60:.1f} min)")
    print(f"  Min liquidity: ${args.min_liquidity:,.0f}")
    print(f"  Max markets:   {max_markets or 'unlimited'}")
    print(f"  Output:        {log_file}")
    print(f"{'='*70}\n")
    
    # Fetch tokens
    tokens = fetch_strategy_b_targets(window, args.min_liquidity, max_markets)
    
    if not tokens:
        print(f"❌ No tokens 10-51% found in {window} window.")
        sys.exit(1)
    
    # Show sample
    print(f"\nSample tokens (first 6):")
    for t in tokens[:6]:
        hours = (t.end_date - datetime.now(timezone.utc)).total_seconds() / 3600
        print(f"  [{t.side}] REST price: {t.rest_api_price:.1%} | {hours:.1f}h to res | {t.question[:40]}")
    
    if len(tokens) > 6:
        print(f"  ... and {len(tokens) - 6} more")
    
    print(f"\n  Note: Anchor will be set from first websocket price (more accurate)")
    
    # Run
    monitor = DipMonitor(tokens, window, log_file, probe_mode)
    await monitor.run(duration)


if __name__ == "__main__":
    asyncio.run(main())