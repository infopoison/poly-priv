#!/usr/bin/env python3
"""
Phase 3C DIAGNOSTIC: Self-Contained Validation
Version: 1.0

PURPOSE:
  Validate the 2D surface analysis methodology using synthetic data.
  Run this BEFORE the full analysis to ensure the logic is correct.

TESTS:
  1. Price extraction at horizons
  2. Move categorization (drops vs rises)
  3. Threshold bucketing
  4. Fill simulation mechanics
  5. Edge calculations
  6. Streaming aggregation

USAGE:
  python phase3c_diagnostic.py
"""

import numpy as np
from datetime import datetime
import sys

# ==============================================================================
# CONFIGURATION (same as main script)
# ==============================================================================

INTERVAL_PAIRS = [
    (48, 24, '48h_to_24h'),
    (24, 12, '24h_to_12h'),
    (12, 6, '12h_to_6h'),
    (6, 3, '6h_to_3h'),
    (3, 1, '3h_to_1h'),
]

MOVE_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20]
DIRECTIONS = ['drop', 'rise']

MIN_TRADES = 10
MIN_SAMPLES_PER_CELL = 5  # Lower for diagnostic

# ==============================================================================
# STREAMING STATISTICS (copied from main)
# ==============================================================================

class StreamingStats:
    """Compute mean, variance incrementally using Welford's algorithm."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        
    def update(self, value):
        if value is None or not np.isfinite(value):
            return
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
    
    def get_stats(self):
        if self.n == 0:
            return None
        variance = self.M2 / self.n if self.n > 0 else 0
        std = np.sqrt(variance)
        return {'n': self.n, 'mean': self.mean, 'std': std}


# ==============================================================================
# CORE FUNCTIONS (copied from main)
# ==============================================================================

def extract_price_at_horizon(trades, resolution_time, hours_before):
    """Extract price closest to specific time horizon before resolution."""
    target_time = resolution_time - (hours_before * 3600)
    
    tolerance_hours = max(0.5, hours_before * 0.25)
    tolerance_seconds = tolerance_hours * 3600
    
    best_trade = None
    best_distance = float('inf')
    
    for ts, price, size in trades:
        distance = abs(ts - target_time)
        if distance < best_distance and distance < tolerance_seconds:
            best_distance = distance
            best_trade = (ts, price)
    
    return best_trade if best_trade else (None, None)


def simulate_limit_order_fill(trades, placement_time, limit_price, is_buy=True):
    """Simulate whether a limit order would be filled."""
    future_trades = [(ts, p, s) for ts, p, s in trades if ts > placement_time]
    
    if not future_trades:
        return {'filled': False, 'fill_time': None, 'fill_price': None}
    
    for ts, price, size in future_trades:
        if is_buy and price <= limit_price:
            return {'filled': True, 'fill_time': ts, 'fill_price': price}
        elif not is_buy and price >= limit_price:
            return {'filled': True, 'fill_time': ts, 'fill_price': price}
    
    return {'filled': False, 'fill_time': None, 'fill_price': None}


def compute_interval_data(trades, resolution_time):
    """Compute price changes across all interval pairs."""
    results = {}
    
    for start_h, end_h, label in INTERVAL_PAIRS:
        start_time, start_price = extract_price_at_horizon(trades, resolution_time, start_h)
        end_time, end_price = extract_price_at_horizon(trades, resolution_time, end_h)
        
        if start_price is None or end_price is None:
            continue
        
        fill_result = None
        if end_time is not None:
            fill_result = simulate_limit_order_fill(trades, end_time, end_price, is_buy=True)
        
        results[label] = {
            'start_hours': start_h,
            'end_hours': end_h,
            'start_price': start_price,
            'end_price': end_price,
            'move': end_price - start_price,
            'fill_simulation': fill_result,
        }
    
    return results


# ==============================================================================
# SYNTHETIC DATA GENERATION
# ==============================================================================

def generate_synthetic_market(resolution_time, price_path, winner):
    """Generate synthetic trades following a price path.
    
    Args:
        resolution_time: Unix timestamp of resolution
        price_path: List of (hours_before, price) tuples
        winner: Boolean, whether this token won
    
    Returns:
        List of (timestamp, price, size) tuples
    """
    trades = []
    
    for hours_before, price in price_path:
        ts = resolution_time - (hours_before * 3600)
        # Add some jitter
        ts += np.random.uniform(-1800, 1800)  # ±30 min
        price += np.random.uniform(-0.005, 0.005)  # ±0.5%
        price = max(0.01, min(0.99, price))
        size = np.random.uniform(10, 1000)
        trades.append((ts, price, size))
    
    # Sort by timestamp
    trades.sort(key=lambda x: x[0])
    return trades


def generate_test_scenarios():
    """Generate test scenarios with known outcomes."""
    base_resolution = 1700000000  # Arbitrary Unix timestamp
    
    scenarios = []
    
    # Scenario 1: Clear DROP (48h->24h), price drops 15%, winner = True
    # This should show UNDERREACTION (dropped too much, but won)
    path = [
        (72, 0.60), (60, 0.58), (48, 0.55),  # Start at 0.55
        (36, 0.48), (24, 0.40),  # End at 0.40 = 15% drop
        (18, 0.42), (12, 0.45),
        (6, 0.50), (3, 0.55),
        (1, 0.60), (0.5, 0.65),
    ]
    scenarios.append({
        'name': 'DROP_15pct_winner',
        'trades': generate_synthetic_market(base_resolution, path, True),
        'winner': True,
        'resolution_time': base_resolution,
        'expected': {
            '48h_to_24h': {'direction': 'drop', 'move_approx': -0.15},
        }
    })
    
    # Scenario 2: Clear RISE (48h->24h), price rises 12%, winner = True
    # This is consistent
    path = [
        (72, 0.50), (60, 0.52), (48, 0.55),
        (36, 0.60), (24, 0.67),  # Rise from 0.55 to 0.67 = 12% rise
        (18, 0.70), (12, 0.72),
        (6, 0.75), (3, 0.78),
        (1, 0.80), (0.5, 0.82),
    ]
    scenarios.append({
        'name': 'RISE_12pct_winner',
        'trades': generate_synthetic_market(base_resolution + 100000, path, True),
        'winner': True,
        'resolution_time': base_resolution + 100000,
        'expected': {
            '48h_to_24h': {'direction': 'rise', 'move_approx': 0.12},
        }
    })
    
    # Scenario 3: Clear DROP (24h->12h), price drops 8%, winner = False
    # This is consistent (dropped and lost)
    path = [
        (72, 0.45), (48, 0.42), (24, 0.40),
        (18, 0.38), (12, 0.32),  # Drop from 0.40 to 0.32 = 8% drop
        (6, 0.28), (3, 0.22),
        (1, 0.15), (0.5, 0.10),
    ]
    scenarios.append({
        'name': 'DROP_8pct_loser',
        'trades': generate_synthetic_market(base_resolution + 200000, path, False),
        'winner': False,
        'resolution_time': base_resolution + 200000,
        'expected': {
            '24h_to_12h': {'direction': 'drop', 'move_approx': -0.08},
        }
    })
    
    # Scenario 4: RISE (6h->3h), price rises 20%, winner = False
    # This should show OVERREACTION (rose too much, but lost)
    path = [
        (48, 0.50), (24, 0.52), (12, 0.55),
        (6, 0.50), (3, 0.70),  # Big rise from 0.50 to 0.70 = 20%
        (1, 0.65), (0.5, 0.55),
    ]
    scenarios.append({
        'name': 'RISE_20pct_loser',
        'trades': generate_synthetic_market(base_resolution + 300000, path, False),
        'winner': False,
        'resolution_time': base_resolution + 300000,
        'expected': {
            '6h_to_3h': {'direction': 'rise', 'move_approx': 0.20},
        }
    })
    
    # Scenario 5: Small move (stable market)
    path = [
        (72, 0.75), (48, 0.74), (24, 0.75),
        (12, 0.74), (6, 0.75),
        (3, 0.76), (1, 0.77),
    ]
    scenarios.append({
        'name': 'STABLE_winner',
        'trades': generate_synthetic_market(base_resolution + 400000, path, True),
        'winner': True,
        'resolution_time': base_resolution + 400000,
        'expected': {
            '48h_to_24h': {'direction': 'rise', 'move_approx': 0.01},  # Small
        }
    })
    
    return scenarios


# ==============================================================================
# TESTS
# ==============================================================================

def test_price_extraction():
    """Test that price extraction at horizons works correctly."""
    print("\n" + "="*60)
    print("TEST 1: Price Extraction at Horizons")
    print("="*60)
    
    resolution_time = 1700000000
    
    # Create trades at known times
    trades = [
        (resolution_time - 50*3600, 0.50, 100),  # T-50h
        (resolution_time - 48*3600, 0.52, 100),  # T-48h
        (resolution_time - 46*3600, 0.54, 100),  # T-46h
        (resolution_time - 24*3600, 0.60, 100),  # T-24h
        (resolution_time - 12*3600, 0.65, 100),  # T-12h
        (resolution_time - 6*3600, 0.70, 100),   # T-6h
        (resolution_time - 1*3600, 0.80, 100),   # T-1h
    ]
    
    tests = [
        (48, 0.52),  # Should find T-48h trade
        (24, 0.60),  # Should find T-24h trade
        (12, 0.65),  # Should find T-12h trade
        (6, 0.70),   # Should find T-6h trade
        (1, 0.80),   # Should find T-1h trade
    ]
    
    all_passed = True
    for horizon, expected_price in tests:
        _, price = extract_price_at_horizon(trades, resolution_time, horizon)
        passed = price is not None and abs(price - expected_price) < 0.001
        status = "✓ PASS" if passed else "✗ FAIL"
        price_str = f"{price:.2f}" if price is not None else "None"
        print(f"  H-{horizon}h: expected {expected_price:.2f}, got {price_str} {status}")
        all_passed = all_passed and passed
    
    return all_passed


def test_move_categorization():
    """Test that moves are categorized correctly."""
    print("\n" + "="*60)
    print("TEST 2: Move Categorization")
    print("="*60)
    
    scenarios = generate_test_scenarios()
    
    all_passed = True
    for scenario in scenarios:
        trades = scenario['trades']
        resolution_time = scenario['resolution_time']
        
        interval_data = compute_interval_data(trades, resolution_time)
        
        print(f"\n  Scenario: {scenario['name']}")
        
        for interval_label, expected in scenario['expected'].items():
            if interval_label not in interval_data:
                print(f"    {interval_label}: MISSING - ✗ FAIL")
                all_passed = False
                continue
            
            data = interval_data[interval_label]
            actual_move = data['move']
            expected_move = expected['move_approx']
            expected_dir = expected['direction']
            
            actual_dir = 'drop' if actual_move < 0 else 'rise'
            
            # Check direction
            dir_passed = actual_dir == expected_dir
            
            # Check magnitude (within 5% tolerance due to jitter)
            mag_passed = abs(abs(actual_move) - abs(expected_move)) < 0.05
            
            passed = dir_passed and mag_passed
            status = "✓ PASS" if passed else "✗ FAIL"
            
            print(f"    {interval_label}: move={actual_move:+.3f} ({actual_dir}), "
                  f"expected ~{expected_move:+.2f} ({expected_dir}) {status}")
            
            all_passed = all_passed and passed
    
    return all_passed


def test_threshold_bucketing():
    """Test that thresholds are correctly applied."""
    print("\n" + "="*60)
    print("TEST 3: Threshold Bucketing")
    print("="*60)
    
    test_cases = [
        (0.015, []),                        # Below all thresholds
        (0.025, [0.02]),                    # Only >= 2%
        (0.06, [0.02, 0.05]),               # >= 2% and >= 5%
        (0.12, [0.02, 0.05, 0.10]),         # >= 2%, 5%, 10%
        (0.18, [0.02, 0.05, 0.10, 0.15]),   # >= 2%, 5%, 10%, 15%
        (0.25, MOVE_THRESHOLDS),            # All thresholds
    ]
    
    all_passed = True
    for move_size, expected_thresholds in test_cases:
        matching = [t for t in MOVE_THRESHOLDS if move_size >= t]
        passed = matching == expected_thresholds
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  move={move_size:.2%}: thresholds={matching} "
              f"(expected {expected_thresholds}) {status}")
        all_passed = all_passed and passed
    
    return all_passed


def test_fill_simulation():
    """Test that fill simulation works correctly."""
    print("\n" + "="*60)
    print("TEST 4: Fill Simulation")
    print("="*60)
    
    resolution_time = 1700000000
    placement_time = resolution_time - 24*3600  # T-24h
    
    # Trades after placement time
    future_trades = [
        (placement_time + 1*3600, 0.55, 100),   # T-23h
        (placement_time + 6*3600, 0.50, 100),   # T-18h - hits 0.50 limit
        (placement_time + 12*3600, 0.48, 100),  # T-12h
    ]
    
    all_trades = [
        (placement_time - 1*3600, 0.60, 100),  # Before placement
    ] + future_trades
    
    tests = [
        (0.55, True, 0.55),   # Limit at 0.55, should fill at first trade
        (0.50, True, 0.50),   # Limit at 0.50, should fill at second trade
        (0.45, True, 0.48),   # Limit at 0.45, should fill at third trade (0.48 <= 0.45 is False!)
        (0.40, False, None),  # Limit at 0.40, never fills (no trade <= 0.40)
    ]
    
    all_passed = True
    for limit_price, should_fill, expected_fill_price in tests:
        result = simulate_limit_order_fill(all_trades, placement_time, limit_price, is_buy=True)
        
        filled = result['filled']
        fill_price = result['fill_price']
        
        # For limit at 0.45, we won't fill since min future price is 0.48 > 0.45
        # Update expected
        if limit_price == 0.45:
            should_fill = False
            expected_fill_price = None
        
        passed = (filled == should_fill)
        if should_fill and filled:
            passed = passed and (fill_price == expected_fill_price)
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Limit={limit_price:.2f}: filled={filled}, "
              f"fill_price={fill_price} (expected {should_fill}, {expected_fill_price}) {status}")
        all_passed = all_passed and passed
    
    return all_passed


def test_edge_calculation():
    """Test edge calculation logic."""
    print("\n" + "="*60)
    print("TEST 5: Edge Calculation Logic")
    print("="*60)
    
    # Edge = (win_rate - avg_price) * 10000 bps
    test_cases = [
        # (win_rate, avg_price, expected_edge_bps)
        (0.80, 0.75, 500),    # 80% win rate, 75% price = +500 bps edge
        (0.60, 0.60, 0),      # No edge
        (0.50, 0.65, -1500),  # Negative edge
        (0.95, 0.90, 500),    # High-prob case
    ]
    
    all_passed = True
    for win_rate, avg_price, expected_edge in test_cases:
        calculated_edge = (win_rate - avg_price) * 10000
        passed = abs(calculated_edge - expected_edge) < 1
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  WR={win_rate:.2%}, P={avg_price:.2%}: "
              f"edge={calculated_edge:+.0f}bps (expected {expected_edge:+.0f}bps) {status}")
        all_passed = all_passed and passed
    
    return all_passed


def test_streaming_aggregation():
    """Test that streaming aggregation produces correct results."""
    print("\n" + "="*60)
    print("TEST 6: Streaming Aggregation")
    print("="*60)
    
    stats = StreamingStats()
    
    values = [10, 20, 30, 40, 50]
    for v in values:
        stats.update(v)
    
    result = stats.get_stats()
    
    expected_mean = np.mean(values)
    expected_std = np.std(values)
    
    mean_passed = abs(result['mean'] - expected_mean) < 0.001
    std_passed = abs(result['std'] - expected_std) < 0.001
    n_passed = result['n'] == len(values)
    
    all_passed = mean_passed and std_passed and n_passed
    
    print(f"  Values: {values}")
    print(f"  Mean: {result['mean']:.2f} (expected {expected_mean:.2f}) {'✓' if mean_passed else '✗'}")
    print(f"  Std:  {result['std']:.2f} (expected {expected_std:.2f}) {'✓' if std_passed else '✗'}")
    print(f"  N:    {result['n']} (expected {len(values)}) {'✓' if n_passed else '✗'}")
    
    status = "✓ PASS" if all_passed else "✗ FAIL"
    print(f"  Overall: {status}")
    
    return all_passed


def test_full_pipeline():
    """Test the full pipeline with synthetic data."""
    print("\n" + "="*60)
    print("TEST 7: Full Pipeline Integration")
    print("="*60)
    
    scenarios = generate_test_scenarios()
    
    # Simulate what the aggregator would collect
    surface_data = {}
    for interval_label in [x[2] for x in INTERVAL_PAIRS]:
        surface_data[interval_label] = {}
        for direction in DIRECTIONS:
            surface_data[interval_label][direction] = {}
            for threshold in MOVE_THRESHOLDS:
                surface_data[interval_label][direction][threshold] = {
                    'n': 0, 'wins': 0, 'prices': []
                }
    
    for scenario in scenarios:
        trades = scenario['trades']
        resolution_time = scenario['resolution_time']
        winner = scenario['winner']
        
        interval_data = compute_interval_data(trades, resolution_time)
        
        for interval_label, data in interval_data.items():
            move = data['move']
            end_price = data['end_price']
            
            direction = 'drop' if move < 0 else 'rise'
            abs_move = abs(move)
            
            for threshold in MOVE_THRESHOLDS:
                if abs_move >= threshold:
                    cell = surface_data[interval_label][direction][threshold]
                    cell['n'] += 1
                    if winner:
                        cell['wins'] += 1
                    cell['prices'].append(end_price)
    
    # Print summary
    print("\n  Aggregated Surface Data:")
    
    has_data = False
    for interval_label in [x[2] for x in INTERVAL_PAIRS]:
        for direction in DIRECTIONS:
            for threshold in MOVE_THRESHOLDS:
                cell = surface_data[interval_label][direction][threshold]
                if cell['n'] > 0:
                    has_data = True
                    win_rate = cell['wins'] / cell['n']
                    avg_price = np.mean(cell['prices'])
                    edge_bps = (win_rate - avg_price) * 10000
                    
                    print(f"    {interval_label} | {direction} | >={threshold*100:.0f}%: "
                          f"n={cell['n']}, WR={win_rate:.2%}, P={avg_price:.2%}, "
                          f"edge={edge_bps:+.0f}bps")
    
    passed = has_data
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n  Pipeline produced data: {status}")
    
    return passed


# ==============================================================================
# MAIN
# ==============================================================================

def run_all_tests():
    """Run all diagnostic tests."""
    print("="*70)
    print("PHASE 3C DIAGNOSTIC: Self-Contained Validation")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Price Extraction", test_price_extraction),
        ("Move Categorization", test_move_categorization),
        ("Threshold Bucketing", test_threshold_bucketing),
        ("Fill Simulation", test_fill_simulation),
        ("Edge Calculation", test_edge_calculation),
        ("Streaming Aggregation", test_streaming_aggregation),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "-"*70)
    if all_passed:
        print("ALL TESTS PASSED - Ready to run on production data")
        return 0
    else:
        print("SOME TESTS FAILED - Review before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())