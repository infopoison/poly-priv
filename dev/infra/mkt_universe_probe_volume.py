#!/usr/bin/env python3
"""
Market Universe Sizing Probe v2
===============================
Extends v1 with volume/liquidity analysis to determine filtered target counts.

Answers:
1. How many Strategy B targets exist (raw)?
2. How does volume/liquidity distribute across these targets?
3. How many targets remain after applying various filters?

Usage:
    python market_universe_probe_v2.py
"""

import requests
import json
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import statistics


GAMMA_API = "https://gamma-api.polymarket.com/markets"

# Volume thresholds to test (total market volume in USD)
VOLUME_THRESHOLDS = [0, 1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

# Liquidity thresholds to test (current liquidity in USD)
LIQUIDITY_THRESHOLDS = [0, 1_000, 5_000, 10_000, 50_000, 100_000]


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
    for field in ["endDateIso", "endDate"]:
        raw = market.get(field)
        if not raw:
            continue
        
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return datetime.fromisoformat(raw)
        except ValueError:
            continue
    
    return None


def parse_volume(market):
    """
    Parse volume field (total USD volume).
    Returns float or 0 if unparseable.
    """
    raw = market.get("volume")
    if raw is None:
        return 0
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0


def parse_liquidity(market):
    """
    Parse liquidity field (current USD liquidity).
    Returns float or 0 if unparseable.
    """
    raw = market.get("liquidity")
    if raw is None:
        return 0
    try:
        return float(raw)
    except (ValueError, TypeError):
        return 0


def categorize_market(market, now):
    """
    Categorize a market with full metadata.
    Returns dict with categorization or None if market is unparseable.
    """
    yes_price, _ = parse_outcome_prices(market)
    end_date = parse_end_date(market)
    volume = parse_volume(market)
    liquidity = parse_liquidity(market)
    
    if yes_price is None or end_date is None:
        return None
    
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)
    
    hours_to_resolution = (end_date - now).total_seconds() / 3600
    
    # Time windows
    if hours_to_resolution < 0:
        time_bucket = "past_resolution"
    elif hours_to_resolution < 12:
        time_bucket = "0h-12h"
    elif hours_to_resolution < 24:
        time_bucket = "12h-24h"
    elif hours_to_resolution < 48:
        time_bucket = "24h-48h"
    elif hours_to_resolution < 72:
        time_bucket = "48h-72h"
    elif hours_to_resolution < 168:
        time_bucket = "72h-1w"
    else:
        time_bucket = ">1w"
    
    # Price buckets
    if yes_price < 0.10:
        price_bucket = "0-10%"
    elif yes_price < 0.25:
        price_bucket = "10-25%"
    elif yes_price < 0.40:
        price_bucket = "25-40%"
    elif yes_price < 0.51:
        price_bucket = "40-51%"
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
        "slug": market.get("slug", ""),
        "yes_price": yes_price,
        "end_date": end_date,
        "hours_to_resolution": hours_to_resolution,
        "time_bucket": time_bucket,
        "price_bucket": price_bucket,
        "volume": volume,
        "liquidity": liquidity,
        "is_strategy_b_window": time_bucket in ["12h-24h", "24h-48h"],
        "is_sub_51": yes_price < 0.51,
    }


def compute_percentiles(values, percentiles=[10, 25, 50, 75, 90, 95, 99]):
    """Compute percentiles for a list of values."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result = {}
    for p in percentiles:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        result[f"p{p}"] = sorted_vals[idx]
    return result


def format_usd(value):
    """Format USD value with appropriate suffix."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.0f}"


def main():
    print("=" * 80)
    print("MARKET UNIVERSE SIZING PROBE v2 - WITH VOLUME/LIQUIDITY ANALYSIS")
    print("=" * 80)
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
    
    # Strategy B targets (raw)
    strategy_b_raw = [c for c in categorized if c["is_strategy_b_window"] and c["is_sub_51"]]
    
    print("=" * 80)
    print("STRATEGY B RAW TARGETS")
    print("=" * 80)
    print(f"Markets in target windows (12h-48h): {len([c for c in categorized if c['is_strategy_b_window']])}")
    print(f"Markets in sub_51% bucket: {len([c for c in categorized if c['is_sub_51']])}")
    print(f"Strategy B targets (intersection): {len(strategy_b_raw)}")
    print()
    
    # Volume/Liquidity distribution for Strategy B targets
    if strategy_b_raw:
        volumes = [c["volume"] for c in strategy_b_raw]
        liquidities = [c["liquidity"] for c in strategy_b_raw]
        
        print("=" * 80)
        print("VOLUME DISTRIBUTION (Strategy B Targets)")
        print("=" * 80)
        
        vol_pct = compute_percentiles(volumes)
        print(f"  Count: {len(volumes)}")
        print(f"  Min:   {format_usd(min(volumes))}")
        print(f"  p10:   {format_usd(vol_pct.get('p10', 0))}")
        print(f"  p25:   {format_usd(vol_pct.get('p25', 0))}")
        print(f"  p50:   {format_usd(vol_pct.get('p50', 0))} (median)")
        print(f"  p75:   {format_usd(vol_pct.get('p75', 0))}")
        print(f"  p90:   {format_usd(vol_pct.get('p90', 0))}")
        print(f"  p95:   {format_usd(vol_pct.get('p95', 0))}")
        print(f"  p99:   {format_usd(vol_pct.get('p99', 0))}")
        print(f"  Max:   {format_usd(max(volumes))}")
        print()
        
        print("=" * 80)
        print("LIQUIDITY DISTRIBUTION (Strategy B Targets)")
        print("=" * 80)
        
        liq_pct = compute_percentiles(liquidities)
        print(f"  Count: {len(liquidities)}")
        print(f"  Min:   {format_usd(min(liquidities))}")
        print(f"  p10:   {format_usd(liq_pct.get('p10', 0))}")
        print(f"  p25:   {format_usd(liq_pct.get('p25', 0))}")
        print(f"  p50:   {format_usd(liq_pct.get('p50', 0))} (median)")
        print(f"  p75:   {format_usd(liq_pct.get('p75', 0))}")
        print(f"  p90:   {format_usd(liq_pct.get('p90', 0))}")
        print(f"  p95:   {format_usd(liq_pct.get('p95', 0))}")
        print(f"  p99:   {format_usd(liq_pct.get('p99', 0))}")
        print(f"  Max:   {format_usd(max(liquidities))}")
        print()
        
        # Filtered counts by volume threshold
        print("=" * 80)
        print("FILTERED COUNTS BY VOLUME THRESHOLD")
        print("=" * 80)
        print(f"{'Volume Threshold':<20} {'Count':<10} {'% of Raw':<10} {'Median Vol':<15} {'Median Liq':<15}")
        print("-" * 80)
        
        for thresh in VOLUME_THRESHOLDS:
            filtered = [c for c in strategy_b_raw if c["volume"] >= thresh]
            if filtered:
                med_vol = statistics.median([c["volume"] for c in filtered])
                med_liq = statistics.median([c["liquidity"] for c in filtered])
            else:
                med_vol = 0
                med_liq = 0
            pct = len(filtered) / len(strategy_b_raw) * 100 if strategy_b_raw else 0
            print(f">= {format_usd(thresh):<17} {len(filtered):<10} {pct:>6.1f}%    {format_usd(med_vol):<15} {format_usd(med_liq):<15}")
        
        print()
        
        # Filtered counts by liquidity threshold
        print("=" * 80)
        print("FILTERED COUNTS BY LIQUIDITY THRESHOLD")
        print("=" * 80)
        print(f"{'Liquidity Threshold':<20} {'Count':<10} {'% of Raw':<10} {'Median Vol':<15} {'Median Liq':<15}")
        print("-" * 80)
        
        for thresh in LIQUIDITY_THRESHOLDS:
            filtered = [c for c in strategy_b_raw if c["liquidity"] >= thresh]
            if filtered:
                med_vol = statistics.median([c["volume"] for c in filtered])
                med_liq = statistics.median([c["liquidity"] for c in filtered])
            else:
                med_vol = 0
                med_liq = 0
            pct = len(filtered) / len(strategy_b_raw) * 100 if strategy_b_raw else 0
            print(f">= {format_usd(thresh):<17} {len(filtered):<10} {pct:>6.1f}%    {format_usd(med_vol):<15} {format_usd(med_liq):<15}")
        
        print()
        
        # Architecture recommendation based on filtered count
        print("=" * 80)
        print("ARCHITECTURE RECOMMENDATIONS BY FILTER LEVEL")
        print("=" * 80)
        
        recommendations = [
            (0, "No filter (raw)"),
            (10_000, "Min $10K volume"),
            (50_000, "Min $50K volume"),
            (100_000, "Min $100K volume"),
        ]
        
        for vol_thresh, label in recommendations:
            filtered = [c for c in strategy_b_raw if c["volume"] >= vol_thresh]
            count = len(filtered)
            
            if count < 50:
                arch = "SIMPLE: Single websocket, local filtering"
            elif count < 100:
                arch = "MODERATE: Single websocket, manageable"
            elif count < 200:
                arch = "HYBRID: Consider REST pre-filtering"
            else:
                arch = "COMPLEX: REST pre-filter required"
            
            print(f"  {label}: {count} markets → {arch}")
        
        print()
        
        # Sample high-liquidity targets
        print("=" * 80)
        print("SAMPLE HIGH-LIQUIDITY STRATEGY B TARGETS (top 15 by volume)")
        print("=" * 80)
        
        sorted_by_vol = sorted(strategy_b_raw, key=lambda x: x["volume"], reverse=True)
        
        for m in sorted_by_vol[:15]:
            print(f"  [{m['time_bucket']}] {m['yes_price']:.1%} | Vol: {format_usd(m['volume']):>10} | Liq: {format_usd(m['liquidity']):>10}")
            print(f"    └─ {m['question']}")
        
        print()
        
        # Sample low-liquidity targets (to see what we're filtering out)
        print("=" * 80)
        print("SAMPLE LOW-LIQUIDITY STRATEGY B TARGETS (bottom 10 by volume)")
        print("=" * 80)
        
        sorted_by_vol_asc = sorted(strategy_b_raw, key=lambda x: x["volume"])
        
        for m in sorted_by_vol_asc[:10]:
            print(f"  [{m['time_bucket']}] {m['yes_price']:.1%} | Vol: {format_usd(m['volume']):>10} | Liq: {format_usd(m['liquidity']):>10}")
            print(f"    └─ {m['question']}")
    
    print()
    print("=" * 80)
    print("12h-24h WINDOW INVESTIGATION")
    print("=" * 80)
    
    # Check what's in the 12h-24h window
    in_12_24 = [c for c in categorized if c["time_bucket"] == "12h-24h"]
    print(f"Total markets in 12h-24h window: {len(in_12_24)}")
    
    if in_12_24:
        print("\nSample markets in 12h-24h window:")
        for m in in_12_24[:5]:
            print(f"  {m['yes_price']:.1%} | {m['hours_to_resolution']:.1f}h | {m['question']}")
    else:
        # Check nearby markets to understand the gap
        near_12h = [c for c in categorized if 10 <= c["hours_to_resolution"] <= 26]
        print(f"\nMarkets with 10-26h to resolution: {len(near_12h)}")
        if near_12h:
            print("Sample markets near the 12-24h boundary:")
            for m in sorted(near_12h, key=lambda x: x["hours_to_resolution"])[:10]:
                print(f"  {m['hours_to_resolution']:.1f}h | {m['yes_price']:.1%} | {m['question'][:60]}")
    
    print()
    print("=" * 80)
    print("PROBE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()