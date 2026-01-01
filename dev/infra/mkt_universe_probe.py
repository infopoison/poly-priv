#!/usr/bin/env python3
"""
Market Universe Sizing Probe
============================
Answers: How many markets are typically in Strategy B target windows?

Strategy B targets:
- Markets in sub_51% bucket (YES price < 0.51)
- Resolution windows: 48h-24h and 24h-12h from now

This determines websocket scaling requirements:
- <100 markets: Simple "monitor all, filter locally" approach
- >200 markets: Need pre-filtering architecture

Usage:
    python market_universe_probe.py
"""

import requests
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict


GAMMA_API = "https://gamma-api.polymarket.com/markets"


def fetch_active_markets(limit=500):
    """
    Fetch active, non-closed markets from Gamma API.
    Returns raw market data.
    """
    markets = []
    offset = 0
    
    while True:
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
        }
        
        print(f"  Fetching markets (offset={offset})...")
        resp = requests.get(GAMMA_API, params=params)
        resp.raise_for_status()
        
        batch = resp.json()
        if not batch:
            break
            
        markets.extend(batch)
        
        if len(batch) < limit:
            break
            
        offset += limit
    
    return markets


def parse_outcome_prices(market):
    """
    Parse outcomePrices field (stringified JSON array).
    Returns (yes_price, no_price) or (None, None) if unparseable.
    """
    raw = market.get("outcomePrices")
    if not raw:
        return None, None
    
    try:
        if isinstance(raw, str):
            prices = json.loads(raw)
        else:
            prices = raw
        
        if len(prices) >= 2:
            return float(prices[0]), float(prices[1])
    except (json.JSONDecodeError, ValueError, IndexError):
        pass
    
    return None, None


def parse_end_date(market):
    """
    Parse endDate field to datetime.
    Returns None if unparseable or missing.
    """
    # Try endDateIso first, then endDate
    for field in ["endDateIso", "endDate"]:
        raw = market.get(field)
        if not raw:
            continue
        
        try:
            # Handle various ISO formats
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return datetime.fromisoformat(raw)
        except ValueError:
            continue
    
    return None


def categorize_market(market, now):
    """
    Categorize a market by:
    1. Time to resolution (window bucket)
    2. Current YES price (probability bucket)
    
    Returns dict with categorization or None if market is unparseable.
    """
    yes_price, _ = parse_outcome_prices(market)
    end_date = parse_end_date(market)
    
    if yes_price is None or end_date is None:
        return None
    
    # Make end_date timezone-aware if it isn't
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    
    hours_to_resolution = (end_date - now).total_seconds() / 3600
    
    # Define time windows
    if hours_to_resolution < 0:
        time_bucket = "past_resolution"
    elif hours_to_resolution < 12:
        time_bucket = "0h-12h"
    elif hours_to_resolution < 24:
        time_bucket = "12h-24h"  # Strategy B window
    elif hours_to_resolution < 48:
        time_bucket = "24h-48h"  # Strategy B window
    elif hours_to_resolution < 72:
        time_bucket = "48h-72h"
    elif hours_to_resolution < 168:
        time_bucket = "72h-1w"
    else:
        time_bucket = ">1w"
    
    # Define price buckets (Strategy B focuses on sub_51%)
    if yes_price < 0.10:
        price_bucket = "0-10%"
    elif yes_price < 0.25:
        price_bucket = "10-25%"
    elif yes_price < 0.40:
        price_bucket = "25-40%"
    elif yes_price < 0.51:
        price_bucket = "40-51%"  # Primary Strategy B target
    elif yes_price < 0.60:
        price_bucket = "51-60%"
    elif yes_price < 0.75:
        price_bucket = "60-75%"
    elif yes_price < 0.90:
        price_bucket = "75-90%"
    else:
        price_bucket = "90-100%"
    
    return {
        "condition_id": market.get("conditionId"),
        "question": market.get("question", "")[:80],
        "yes_price": yes_price,
        "end_date": end_date,
        "hours_to_resolution": hours_to_resolution,
        "time_bucket": time_bucket,
        "price_bucket": price_bucket,
        "is_strategy_b_window": time_bucket in ["12h-24h", "24h-48h"],
        "is_sub_51": yes_price < 0.51,
    }


def main():
    print("=" * 70)
    print("MARKET UNIVERSE SIZING PROBE")
    print("=" * 70)
    print()
    
    now = datetime.now(timezone.utc)
    print(f"Current time (UTC): {now.isoformat()}")
    print()
    
    # Fetch all active markets
    print("Fetching active markets from Gamma API...")
    markets = fetch_active_markets()
    print(f"  Total active markets fetched: {len(markets)}")
    print()
    
    # Categorize markets
    print("Categorizing markets...")
    categorized = []
    parse_failures = 0
    
    for m in markets:
        cat = categorize_market(m, now)
        if cat:
            categorized.append(cat)
        else:
            parse_failures += 1
    
    print(f"  Successfully categorized: {len(categorized)}")
    print(f"  Parse failures: {parse_failures}")
    print()
    
    # Build cross-tabulation: time_bucket x price_bucket
    cross_tab = defaultdict(lambda: defaultdict(int))
    for c in categorized:
        cross_tab[c["time_bucket"]][c["price_bucket"]] += 1
    
    # Print cross-tabulation
    print("=" * 70)
    print("CROSS-TABULATION: Time Window × Price Bucket")
    print("=" * 70)
    
    time_order = ["0h-12h", "12h-24h", "24h-48h", "48h-72h", "72h-1w", ">1w", "past_resolution"]
    price_order = ["0-10%", "10-25%", "25-40%", "40-51%", "51-60%", "60-75%", "75-90%", "90-100%"]
    
    # Header
    header = f"{'Time Window':<15}" + "".join(f"{p:>10}" for p in price_order) + f"{'TOTAL':>10}"
    print(header)
    print("-" * len(header))
    
    for time_bucket in time_order:
        row = f"{time_bucket:<15}"
        row_total = 0
        for price_bucket in price_order:
            count = cross_tab[time_bucket][price_bucket]
            row_total += count
            row += f"{count:>10}"
        row += f"{row_total:>10}"
        
        # Highlight Strategy B windows
        if time_bucket in ["12h-24h", "24h-48h"]:
            row = ">>> " + row + " <<<"
        
        print(row)
    
    print()
    
    # Strategy B specific summary
    print("=" * 70)
    print("STRATEGY B TARGET UNIVERSE")
    print("=" * 70)
    
    strategy_b_markets = [c for c in categorized if c["is_strategy_b_window"] and c["is_sub_51"]]
    
    print(f"Markets in target windows (12h-48h to resolution): ", end="")
    in_window = [c for c in categorized if c["is_strategy_b_window"]]
    print(f"{len(in_window)}")
    
    print(f"Markets in sub_51% bucket (all time windows): ", end="")
    sub_51 = [c for c in categorized if c["is_sub_51"]]
    print(f"{len(sub_51)}")
    
    print(f"Markets in BOTH (Strategy B targets): {len(strategy_b_markets)}")
    print()
    
    # Architecture recommendation
    print("=" * 70)
    print("ARCHITECTURE RECOMMENDATION")
    print("=" * 70)
    
    target_count = len(strategy_b_markets)
    all_active_count = len(categorized)
    
    if target_count < 50:
        rec = "SIMPLE: Monitor all active markets, filter locally"
        detail = f"Only {target_count} targets - single websocket connection is fine"
    elif target_count < 100:
        rec = "MODERATE: Monitor all active markets, filter locally"
        detail = f"{target_count} targets - manageable with single connection"
    elif target_count < 200:
        rec = "HYBRID: Consider pre-filtering by time window via REST"
        detail = f"{target_count} targets - may benefit from selective subscription"
    else:
        rec = "COMPLEX: Pre-filter candidates via REST, selective websocket subscription"
        detail = f"{target_count} targets - need bounded resource usage"
    
    print(f"Target count: {target_count}")
    print(f"Recommendation: {rec}")
    print(f"Detail: {detail}")
    print()
    
    # If there are Strategy B targets, show some examples
    if strategy_b_markets:
        print("=" * 70)
        print("SAMPLE STRATEGY B TARGETS (first 10)")
        print("=" * 70)
        
        # Sort by hours to resolution
        strategy_b_markets.sort(key=lambda x: x["hours_to_resolution"])
        
        for m in strategy_b_markets[:10]:
            print(f"  [{m['time_bucket']}] [{m['price_bucket']}] {m['yes_price']:.2%} - {m['question']}")
    
    print()
    print("=" * 70)
    print("PROBE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()