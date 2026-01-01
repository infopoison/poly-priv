#!/usr/bin/env python3
"""
Strategy B WebSocket Scaling Probe
==================================

PURPOSE:
  Empirically answer: What are the websocket scaling limits for Strategy B?
  
MEASURES:
  1. Connection stability with N concurrent asset subscriptions
  2. Message throughput (events/second)
  3. Threshold crossing frequency (price enters/exits sub-51%)
  4. Memory footprint over time
  
ARCHITECTURE:
  - Fetch Strategy B candidates via REST (Gamma API)
  - Apply liquidity filter
  - Subscribe to ALL candidates on single websocket
  - Track lightweight state (current price only, not full orderbook)
  - Log threshold crossings for paper trading validation

DEPLOYMENT TARGET: t3.micro (1 vCPU, 1GB RAM)

Usage:
    # Full probe with all Strategy B targets
    python strategy_b_ws_probe.py
    
    # Limit subscription count (for testing)
    python strategy_b_ws_probe.py --max-markets 50
    
    # Custom duration
    python strategy_b_ws_probe.py --duration 3600
    
    # Higher liquidity filter
    python strategy_b_ws_probe.py --min-liquidity 5000

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
import traceback
import resource
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# =============================================================================
# CONFIGURATION
# =============================================================================

GAMMA_API = "https://gamma-api.polymarket.com/markets"
WS_URI = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Strategy B parameters
SUB_51_THRESHOLD = 0.51  # Target bucket: YES price < 51%
RESOLUTION_WINDOW_MIN_HOURS = 12
RESOLUTION_WINDOW_MAX_HOURS = 48

# Default filters
DEFAULT_MIN_LIQUIDITY = 1000  # USD
DEFAULT_MAX_MARKETS = None    # No limit by default
DEFAULT_DURATION = 1800       # 30 minutes

# Monitoring intervals
STATS_INTERVAL = 30           # Print stats every N seconds
PING_INTERVAL = 10            # WebSocket keepalive


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MarketState:
    """Lightweight state for a single market"""
    condition_id: str
    token_id: str
    question: str
    end_date: datetime
    liquidity: float
    
    # Price tracking
    current_price: Optional[float] = None
    last_price: Optional[float] = None
    last_update: Optional[datetime] = None
    
    # Threshold tracking
    in_sub_51: bool = False
    threshold_crossings: int = 0
    
    def update_price(self, new_price: float) -> Optional[str]:
        """
        Update price and detect threshold crossings.
        Returns crossing type: 'enter_sub51', 'exit_sub51', or None
        """
        self.last_price = self.current_price
        self.current_price = new_price
        self.last_update = datetime.now(timezone.utc)
        
        was_sub_51 = self.in_sub_51
        self.in_sub_51 = new_price < SUB_51_THRESHOLD
        
        if was_sub_51 and not self.in_sub_51:
            self.threshold_crossings += 1
            return 'exit_sub51'
        elif not was_sub_51 and self.in_sub_51:
            self.threshold_crossings += 1
            return 'enter_sub51'
        
        return None


@dataclass
class ProbeStats:
    """Aggregate statistics for the probe"""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Connection stats
    connection_attempts: int = 0
    successful_connections: int = 0
    disconnections: int = 0
    
    # Message stats
    total_messages: int = 0
    book_events: int = 0
    price_change_events: int = 0
    trade_events: int = 0
    other_events: int = 0
    parse_errors: int = 0
    
    # Threshold crossings
    enter_sub51_crossings: int = 0
    exit_sub51_crossings: int = 0
    
    # Timing
    last_message_time: Optional[datetime] = None
    max_message_gap_seconds: float = 0.0
    
    # Throughput tracking (rolling window)
    message_timestamps: List[float] = field(default_factory=list)
    
    def record_message(self):
        """Record a message receipt"""
        now = datetime.now(timezone.utc)
        now_ts = now.timestamp()
        
        if self.last_message_time:
            gap = (now - self.last_message_time).total_seconds()
            self.max_message_gap_seconds = max(self.max_message_gap_seconds, gap)
        
        self.last_message_time = now
        self.total_messages += 1
        
        # Keep last 60 seconds of timestamps for throughput calculation
        self.message_timestamps.append(now_ts)
        cutoff = now_ts - 60
        self.message_timestamps = [t for t in self.message_timestamps if t > cutoff]
    
    def get_messages_per_second(self) -> float:
        """Calculate recent message throughput"""
        if len(self.message_timestamps) < 2:
            return 0.0
        
        window = self.message_timestamps[-1] - self.message_timestamps[0]
        if window < 1:
            return 0.0
        
        return len(self.message_timestamps) / window
    
    def get_runtime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()


# =============================================================================
# MARKET FETCHING (from universe probe)
# =============================================================================

def fetch_active_markets(limit: int = 500) -> List[dict]:
    """Fetch active, non-closed markets from Gamma API"""
    markets = []
    offset = 0
    
    while True:
        params = {
            "active": "true",
            "closed": "false", 
            "limit": limit,
            "offset": offset,
        }
        
        resp = requests.get(GAMMA_API, params=params, timeout=30)
        resp.raise_for_status()
        
        batch = resp.json()
        if not batch:
            break
        
        markets.extend(batch)
        
        if len(batch) < limit:
            break
        
        offset += limit
    
    return markets


def parse_market(market: dict, now: datetime) -> Optional[MarketState]:
    """Parse a market into MarketState if it's a Strategy B target"""
    
    # Parse prices
    raw_prices = market.get("outcomePrices")
    if not raw_prices:
        return None
    
    try:
        if isinstance(raw_prices, str):
            prices = json.loads(raw_prices)
        else:
            prices = raw_prices
        yes_price = float(prices[0])
    except (json.JSONDecodeError, ValueError, IndexError):
        return None
    
    # Parse end date
    end_date = None
    for field in ["endDateIso", "endDate"]:
        raw = market.get(field)
        if not raw:
            continue
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            end_date = datetime.fromisoformat(raw)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
            break
        except ValueError:
            continue
    
    if end_date is None:
        return None
    
    # Check time window
    hours_to_resolution = (end_date - now).total_seconds() / 3600
    if not (RESOLUTION_WINDOW_MIN_HOURS <= hours_to_resolution <= RESOLUTION_WINDOW_MAX_HOURS):
        return None
    
    # Check price bucket (sub-51%)
    if yes_price >= SUB_51_THRESHOLD:
        return None
    
    # Parse liquidity
    try:
        liquidity = float(market.get("liquidity", 0))
    except (ValueError, TypeError):
        liquidity = 0
    
    # Get token ID (clobTokenIds contains YES and NO token IDs)
    clob_tokens = market.get("clobTokenIds")
    if not clob_tokens:
        return None
    
    try:
        if isinstance(clob_tokens, str):
            clob_tokens = json.loads(clob_tokens)
        token_id = clob_tokens[0]  # YES token
    except (json.JSONDecodeError, IndexError):
        return None
    
    return MarketState(
        condition_id=market.get("conditionId", ""),
        token_id=token_id,
        question=market.get("question", "")[:100],
        end_date=end_date,
        liquidity=liquidity,
        current_price=yes_price,
        in_sub_51=yes_price < SUB_51_THRESHOLD,
    )


def get_strategy_b_targets(min_liquidity: float, max_markets: Optional[int]) -> List[MarketState]:
    """Fetch and filter Strategy B target markets"""
    
    print(f"Fetching markets from Gamma API...")
    now = datetime.now(timezone.utc)
    raw_markets = fetch_active_markets()
    print(f"  Fetched {len(raw_markets)} active markets")
    
    # Parse and filter
    targets = []
    for m in raw_markets:
        state = parse_market(m, now)
        if state and state.liquidity >= min_liquidity:
            targets.append(state)
    
    print(f"  Strategy B targets (liquidity >= ${min_liquidity:,.0f}): {len(targets)}")
    
    # Sort by liquidity (highest first) and optionally limit
    targets.sort(key=lambda x: x.liquidity, reverse=True)
    
    if max_markets and len(targets) > max_markets:
        print(f"  Limiting to top {max_markets} by liquidity")
        targets = targets[:max_markets]
    
    return targets


# =============================================================================
# WEBSOCKET PROBE
# =============================================================================

class StrategyBProbe:
    """WebSocket probe for Strategy B market monitoring"""
    
    def __init__(self, markets: List[MarketState]):
        self.markets = {m.token_id: m for m in markets}
        self.stats = ProbeStats()
        self.running = False
        
    def process_message(self, raw_message: str):
        """Process a websocket message"""
        self.stats.record_message()
        
        # Handle PONG responses
        if raw_message == "PONG":
            return
        
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            self.stats.parse_errors += 1
            return
        
        # Handle both list and dict responses
        events = data if isinstance(data, list) else [data]
        
        for event in events:
            if not isinstance(event, dict):
                self.stats.other_events += 1
                continue
                
            event_type = event.get("event_type", "unknown")
            
            if event_type == "book":
                self.stats.book_events += 1
                self._process_book(event)
                
            elif event_type == "price_change":
                self.stats.price_change_events += 1
                self._process_price_change(event)
                
            elif event_type == "last_trade_price":
                self.stats.trade_events += 1
                self._process_trade(event)
                
            else:
                self.stats.other_events += 1
    
    def _process_book(self, data: dict):
        """Process orderbook snapshot - extract best bid as price proxy"""
        asset_id = data.get("asset_id")
        if asset_id not in self.markets:
            return
        
        bids = data.get("bids", [])
        if bids:
            # Best bid as price estimate
            best_bid = max(float(b["price"]) for b in bids)
            crossing = self.markets[asset_id].update_price(best_bid)
            self._record_crossing(asset_id, crossing)
    
    def _process_price_change(self, data: dict):
        """Process price change event"""
        for change in data.get("price_changes", []):
            asset_id = change.get("asset_id")
            if asset_id not in self.markets:
                continue
            
            # Use best_bid if available
            best_bid = change.get("best_bid")
            if best_bid:
                try:
                    price = float(best_bid)
                    crossing = self.markets[asset_id].update_price(price)
                    self._record_crossing(asset_id, crossing)
                except ValueError:
                    pass
    
    def _process_trade(self, data: dict):
        """Process trade/fill event"""
        asset_id = data.get("asset_id")
        if asset_id not in self.markets:
            return
        
        price = data.get("price")
        if price:
            try:
                crossing = self.markets[asset_id].update_price(float(price))
                self._record_crossing(asset_id, crossing)
            except ValueError:
                pass
    
    def _record_crossing(self, asset_id: str, crossing: Optional[str]):
        """Record threshold crossing"""
        if crossing == "enter_sub51":
            self.stats.enter_sub51_crossings += 1
            market = self.markets[asset_id]
            print(f"  [CROSSING] ENTER sub-51%: {market.current_price:.2%} <- {market.question[:60]}")
        elif crossing == "exit_sub51":
            self.stats.exit_sub51_crossings += 1
            market = self.markets[asset_id]
            print(f"  [CROSSING] EXIT sub-51%: {market.current_price:.2%} <- {market.question[:60]}")
    
    def print_stats(self):
        """Print current statistics"""
        runtime = self.stats.get_runtime_seconds()
        msg_rate = self.stats.get_messages_per_second()
        
        # Memory usage (Linux/Mac)
        try:
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if sys.platform == "darwin":
                mem_mb /= 1024  # macOS reports bytes, Linux reports KB
        except:
            mem_mb = 0
        
        print(f"\n{'='*70}")
        print(f"STATS @ {datetime.now(timezone.utc).strftime('%H:%M:%S')} (runtime: {runtime:.0f}s)")
        print(f"{'='*70}")
        print(f"Markets monitored:     {len(self.markets)}")
        print(f"Total messages:        {self.stats.total_messages}")
        print(f"Message rate:          {msg_rate:.2f}/sec")
        print(f"Max message gap:       {self.stats.max_message_gap_seconds:.1f}s")
        print(f"")
        print(f"Event breakdown:")
        print(f"  Book snapshots:      {self.stats.book_events}")
        print(f"  Price changes:       {self.stats.price_change_events}")
        print(f"  Trades/fills:        {self.stats.trade_events}")
        print(f"  Other:               {self.stats.other_events}")
        print(f"  Parse errors:        {self.stats.parse_errors}")
        print(f"")
        print(f"Threshold crossings:")
        print(f"  Enter sub-51%:       {self.stats.enter_sub51_crossings}")
        print(f"  Exit sub-51%:        {self.stats.exit_sub51_crossings}")
        print(f"")
        print(f"Memory usage:          {mem_mb:.1f} MB")
        print(f"{'='*70}\n")
    
    async def _ping_loop(self, websocket):
        """Send keepalive pings"""
        while self.running:
            try:
                await websocket.send("PING")
                await asyncio.sleep(PING_INTERVAL)
            except:
                break
    
    async def _stats_loop(self):
        """Periodic stats printing"""
        while self.running:
            await asyncio.sleep(STATS_INTERVAL)
            if self.running:
                self.print_stats()
    
    async def connect_and_monitor(self):
        """Main websocket connection and monitoring loop"""
        
        token_ids = list(self.markets.keys())
        
        print(f"\nConnecting to {WS_URI}")
        print(f"Subscribing to {len(token_ids)} assets...")
        
        self.stats.connection_attempts += 1
        
        try:
            async with websockets.connect(
                WS_URI,
                ping_interval=None,  # We handle our own pings
                ping_timeout=None,
                max_size=10_000_000,  # 10MB max message size
            ) as websocket:
                
                self.stats.successful_connections += 1
                print(f"[✓] Connected")
                
                # Subscribe to all assets
                subscribe_msg = {
                    "assets_ids": token_ids,
                    "type": "market"
                }
                await websocket.send(json.dumps(subscribe_msg))
                print(f"[✓] Subscribed to {len(token_ids)} markets")
                
                # Start background tasks
                self.running = True
                ping_task = asyncio.create_task(self._ping_loop(websocket))
                stats_task = asyncio.create_task(self._stats_loop())
                
                try:
                    async for message in websocket:
                        self.process_message(message)
                except websockets.ConnectionClosed as e:
                    print(f"[!] Connection closed: {e}")
                    self.stats.disconnections += 1
                finally:
                    self.running = False
                    ping_task.cancel()
                    stats_task.cancel()
                    
        except Exception as e:
            print(f"[!] Connection error: {e}")
            traceback.print_exc()
    
    async def run(self, duration_seconds: int):
        """Run the probe for specified duration"""
        
        print(f"\n{'='*70}")
        print(f"STRATEGY B WEBSOCKET SCALING PROBE")
        print(f"{'='*70}")
        print(f"Duration:       {duration_seconds}s ({duration_seconds/60:.1f} min)")
        print(f"Markets:        {len(self.markets)}")
        print(f"Stats interval: {STATS_INTERVAL}s")
        print(f"{'='*70}\n")
        
        try:
            await asyncio.wait_for(
                self.connect_and_monitor(),
                timeout=duration_seconds
            )
        except asyncio.TimeoutError:
            print("\n[✓] Probe duration complete")
        except KeyboardInterrupt:
            print("\n[!] Interrupted")
        finally:
            self.running = False
            self.print_final_report()
    
    def print_final_report(self):
        """Print final probe report"""
        runtime = self.stats.get_runtime_seconds()
        
        print(f"\n{'='*70}")
        print(f"FINAL REPORT")
        print(f"{'='*70}")
        print(f"")
        print(f"CONFIGURATION")
        print(f"  Markets monitored:        {len(self.markets)}")
        print(f"  Runtime:                  {runtime:.0f}s ({runtime/60:.1f} min)")
        print(f"")
        print(f"CONNECTION")
        print(f"  Attempts:                 {self.stats.connection_attempts}")
        print(f"  Successful:               {self.stats.successful_connections}")
        print(f"  Disconnections:           {self.stats.disconnections}")
        print(f"")
        print(f"THROUGHPUT")
        print(f"  Total messages:           {self.stats.total_messages}")
        print(f"  Avg rate:                 {self.stats.total_messages/runtime:.2f}/sec" if runtime > 0 else "  Avg rate: N/A")
        print(f"  Max message gap:          {self.stats.max_message_gap_seconds:.1f}s")
        print(f"")
        print(f"EVENTS")
        print(f"  Book snapshots:           {self.stats.book_events}")
        print(f"  Price changes:            {self.stats.price_change_events}")
        print(f"  Trades/fills:             {self.stats.trade_events}")
        print(f"  Parse errors:             {self.stats.parse_errors}")
        print(f"")
        print(f"THRESHOLD CROSSINGS (PAPER TRADE SIGNALS)")
        print(f"  Enter sub-51%:            {self.stats.enter_sub51_crossings}")
        print(f"  Exit sub-51%:             {self.stats.exit_sub51_crossings}")
        print(f"  Total crossings:          {self.stats.enter_sub51_crossings + self.stats.exit_sub51_crossings}")
        
        if runtime > 0:
            crossings_per_hour = (self.stats.enter_sub51_crossings + self.stats.exit_sub51_crossings) * 3600 / runtime
            print(f"  Estimated crossings/hour: {crossings_per_hour:.1f}")
        
        print(f"")
        
        # Architecture recommendation
        print(f"ARCHITECTURE ASSESSMENT")
        if self.stats.total_messages == 0:
            print(f"  ⚠️  No messages received - check connection/subscription")
        elif self.stats.disconnections > 0:
            print(f"  ⚠️  Experienced disconnections - may need reconnection logic")
        else:
            msg_rate = self.stats.total_messages / runtime if runtime > 0 else 0
            if msg_rate > 100:
                print(f"  ⚠️  High message rate ({msg_rate:.0f}/sec) - may need throttling on t3.micro")
            elif msg_rate > 10:
                print(f"  ✓  Moderate message rate ({msg_rate:.1f}/sec) - should be manageable")
            else:
                print(f"  ✓  Low message rate ({msg_rate:.2f}/sec) - well within capacity")
        
        print(f"{'='*70}\n")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Strategy B WebSocket Scaling Probe")
    parser.add_argument("--min-liquidity", type=float, default=DEFAULT_MIN_LIQUIDITY,
                       help=f"Minimum liquidity filter (default: ${DEFAULT_MIN_LIQUIDITY})")
    parser.add_argument("--max-markets", type=int, default=DEFAULT_MAX_MARKETS,
                       help="Maximum markets to monitor (default: no limit)")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                       help=f"Probe duration in seconds (default: {DEFAULT_DURATION})")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"STRATEGY B WEBSOCKET SCALING PROBE")
    print(f"{'='*70}")
    print(f"")
    print(f"This probe empirically measures:")
    print(f"  1. Websocket connection stability with N subscriptions")
    print(f"  2. Message throughput (events/second)")
    print(f"  3. Threshold crossing frequency")
    print(f"  4. Resource usage (memory)")
    print(f"")
    print(f"Parameters:")
    print(f"  Min liquidity: ${args.min_liquidity:,.0f}")
    print(f"  Max markets:   {args.max_markets or 'unlimited'}")
    print(f"  Duration:      {args.duration}s ({args.duration/60:.1f} min)")
    print(f"{'='*70}\n")
    
    # Fetch Strategy B targets
    targets = get_strategy_b_targets(args.min_liquidity, args.max_markets)
    
    if not targets:
        print("❌ No Strategy B targets found. Check filters.")
        sys.exit(1)
    
    # Show sample of targets
    print(f"\nSample targets (top 5 by liquidity):")
    for m in targets[:5]:
        hours_left = (m.end_date - datetime.now(timezone.utc)).total_seconds() / 3600
        print(f"  {m.current_price:.1%} | ${m.liquidity:,.0f} liq | {hours_left:.1f}h | {m.question[:50]}")
    
    if len(targets) > 5:
        print(f"  ... and {len(targets) - 5} more")
    
    # Run probe
    probe = StrategyBProbe(targets)
    await probe.run(args.duration)


if __name__ == "__main__":
    asyncio.run(main())