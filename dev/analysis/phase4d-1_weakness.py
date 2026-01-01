#!/usr/bin/env python3
"""
Analysis Script 3: Strategy Weakness Analysis

PURPOSE:
  Your concern: "We haven't really asked 'where are the weakest performances, 
  and why might that be?' I had hypothesized that smaller dips on longer time 
  intervals, far from resolution (e.g. 0.05% dips in 48 to 24h interval with 
  third tercile) should be noise and should not have much edge or should 
  represent some type of baseline."

This script:
  1. Identifies the WORST performing cells
  2. Tests your baseline hypothesis (small dips, far from resolution, late tercile)
  3. Looks for patterns in weakness
  4. Checks if seeing edge in "should be noise" conditions is suspicious
"""

import json
import sys
import argparse
import numpy as np
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def collect_all_cells(data):
    """
    Collect all cells with enough data for analysis.
    """
    surface = data.get('surface', {})
    all_cells = []
    
    for interval_label, interval_data in surface.items():
        # Parse interval to get end hour
        parts = interval_label.replace('h_to_', '_').replace('h', '').split('_')
        try:
            start_hour = int(parts[0])
            end_hour = int(parts[1])
            observation_window = start_hour - end_hour
        except:
            start_hour, end_hour, observation_window = 0, 0, 0
        
        for threshold, threshold_data in interval_data.items():
            for prob_bucket, bucket_data in threshold_data.items():
                for tercile in ['early', 'mid', 'late']:
                    tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
                    
                    if tercile_data.get('status') == 'ok':
                        cell = {
                            'interval': interval_label,
                            'start_hour': start_hour,
                            'end_hour': end_hour,
                            'observation_window': observation_window,
                            'threshold': threshold,
                            'prob_bucket': prob_bucket,
                            'tercile': tercile,
                            'edge_after_fill': tercile_data.get('edge_after_fill_bps'),
                            'uncond_edge': tercile_data.get('unconditional_edge_bps'),
                            'se': tercile_data.get('se_edge_after_fill'),
                            'n': tercile_data.get('n_samples'),
                            'fill_rate': tercile_data.get('fill_rate'),
                            'avg_fill_price': tercile_data.get('avg_fill_price'),
                            'avg_crossing_price': tercile_data.get('avg_crossing_price'),
                        }
                        all_cells.append(cell)
    
    return all_cells


def identify_worst_cells(all_cells, n=20):
    """
    Identify the worst performing cells by edge.
    """
    print("="*100)
    print("WORST PERFORMING CELLS")
    print("="*100)
    
    # Filter to cells with edge data
    cells_with_edge = [c for c in all_cells if c['edge_after_fill'] is not None]
    
    # Sort by edge (lowest first)
    cells_with_edge.sort(key=lambda x: x['edge_after_fill'])
    
    print(f"\nBOTTOM {n} CELLS BY EDGE AFTER FILL:")
    print("-" * 120)
    print(f"{'Interval':<15} {'Thresh':>7} {'Bucket':>10} {'Terc':>6} {'Edge':>8} {'SE':>7} {'n':>7} {'Fill%':>7} {'Sig?':>12}")
    
    for c in cells_with_edge[:n]:
        se = c['se'] if c['se'] else 0
        ci_low = c['edge_after_fill'] - 1.96 * se
        ci_high = c['edge_after_fill'] + 1.96 * se
        
        sig = "SIG NEG" if ci_high < 0 else ("SIG POS" if ci_low > 0 else "not sig")
        
        thresh_pct = float(c['threshold']) * 100
        print(f"{c['interval']:<15} {thresh_pct:>6.0f}% {c['prob_bucket']:>10} {c['tercile']:>6} "
              f"{c['edge_after_fill']:>+8.0f} {se:>7.1f} {c['n']:>7,} {c['fill_rate']*100:>6.1f}% {sig:>12}")
    
    # Count significantly negative
    sig_negative = [c for c in cells_with_edge if c['se'] and c['edge_after_fill'] + 1.96 * c['se'] < 0]
    print(f"\n\nTotal cells with SIGNIFICANTLY NEGATIVE edge: {len(sig_negative)}")
    
    return cells_with_edge[:n]


def test_baseline_hypothesis(all_cells):
    """
    Test hypothesis: small dips + far from resolution + late tercile = baseline/noise
    """
    print("\n")
    print("="*100)
    print("BASELINE HYPOTHESIS TEST")
    print("="*100)
    print("\nHypothesis: Small dips (5%) on long intervals (48h->24h), late tercile")
    print("            should represent baseline with minimal edge (just noise).")
    print()
    
    # Define baseline conditions - use string key as stored in JSON
    baseline_conditions = {
        'threshold': '0.05',  # smallest threshold (string key)
        'end_hour_min': 24,  # far from resolution
        'tercile': 'late',  # late reaction
    }
    
    # Find matching cells
    baseline_cells = []
    for c in all_cells:
        if (c['threshold'] == baseline_conditions['threshold'] and
            c['end_hour'] >= baseline_conditions['end_hour_min'] and
            c['tercile'] == baseline_conditions['tercile']):
            baseline_cells.append(c)
    
    if not baseline_cells:
        print("No cells matching baseline conditions found!")
        print("Checking what thresholds exist...")
        thresholds = set(c['threshold'] for c in all_cells)
        print(f"Available thresholds: {sorted(thresholds)}")
        
        # Try with minimum available threshold
        min_thresh = min(thresholds)
        print(f"\nRetrying with minimum threshold {min_thresh}...")
        for c in all_cells:
            if (c['threshold'] == min_thresh and
                c['end_hour'] >= baseline_conditions['end_hour_min'] and
                c['tercile'] == baseline_conditions['tercile']):
                baseline_cells.append(c)
    
    if baseline_cells:
        print(f"Found {len(baseline_cells)} baseline cells:")
        print("-" * 100)
        print(f"{'Interval':<15} {'Bucket':>10} {'Edge':>10} {'SE':>8} {'n':>8} {'Assessment'}")
        
        total_edge = 0
        total_n = 0
        
        for c in baseline_cells:
            se = c['se'] if c['se'] else 0
            ci_low = c['edge_after_fill'] - 1.96 * se
            ci_high = c['edge_after_fill'] + 1.96 * se
            
            if ci_low > 50:
                assessment = "UNEXPECTED: Positive edge in baseline!"
            elif ci_high < -50:
                assessment = "Negative edge (adverse selection?)"
            else:
                assessment = "As expected: ~zero edge (noise)"
            
            print(f"{c['interval']:<15} {c['prob_bucket']:>10} {c['edge_after_fill']:>+10.0f} {se:>8.1f} {c['n']:>8,} {assessment}")
            
            total_edge += c['edge_after_fill'] * c['n']
            total_n += c['n']
        
        avg_baseline_edge = total_edge / total_n if total_n > 0 else 0
        print(f"\nWeighted average baseline edge: {avg_baseline_edge:>+.1f} bps")
        
        if avg_baseline_edge > 50:
            print("\n*** WARNING: Baseline shows positive edge!")
            print("    This could indicate:")
            print("    1. Limit order mechanics provide edge even in 'noise' conditions")
            print("    2. Some systematic bias in the analysis")
            print("    3. Market microstructure effects (adverse selection working in our favor)")
        elif avg_baseline_edge < -50:
            print("\n*** Finding: Baseline shows negative edge (as might be expected from adverse selection)")
        else:
            print("\n*** Finding: Baseline is approximately zero-edge (validates methodology)")
    
    # Compare to optimal conditions
    print("\n\nCOMPARISON TO 'OPTIMAL' CONDITIONS:")
    print("-" * 60)
    
    optimal_cells = []
    for c in all_cells:
        # Convert string threshold to float for comparison
        thresh_val = float(c['threshold'])
        if (thresh_val >= 0.10 and
            4 <= c['end_hour'] <= 12 and
            c['tercile'] == 'early'):
            optimal_cells.append(c)
    
    if optimal_cells:
        total_edge = sum(c['edge_after_fill'] * c['n'] for c in optimal_cells)
        total_n = sum(c['n'] for c in optimal_cells)
        avg_optimal_edge = total_edge / total_n if total_n > 0 else 0
        
        print(f"Optimal conditions (≥10% dip, 4-12h to resolution, early tercile):")
        print(f"  Average edge: {avg_optimal_edge:>+.1f} bps")
        print(f"  Total n:      {total_n:,}")
        
        if baseline_cells:
            baseline_avg = sum(c['edge_after_fill'] * c['n'] for c in baseline_cells) / sum(c['n'] for c in baseline_cells)
            improvement = avg_optimal_edge - baseline_avg
            print(f"\n  Edge improvement vs baseline: {improvement:>+.1f} bps")


def analyze_weakness_patterns(all_cells):
    """
    Look for patterns in what makes cells perform poorly.
    """
    print("\n")
    print("="*100)
    print("PATTERNS IN WEAK PERFORMANCE")
    print("="*100)
    
    # Only use 'all' bucket to avoid double-counting
    all_bucket_cells = [c for c in all_cells if c['prob_bucket'] == 'all' and c['edge_after_fill'] is not None]
    
    # 1. By tercile
    print("\n1. EDGE BY TERCILE (all prob bucket):")
    print("-" * 60)
    for tercile in ['early', 'mid', 'late']:
        cells = [c for c in all_bucket_cells if c['tercile'] == tercile]
        if cells:
            avg_edge = np.mean([c['edge_after_fill'] for c in cells])
            std_edge = np.std([c['edge_after_fill'] for c in cells])
            n_neg = sum(1 for c in cells if c['edge_after_fill'] < 0)
            print(f"  {tercile:<8}: avg edge = {avg_edge:>+8.1f} bps, std = {std_edge:>7.1f}, {n_neg}/{len(cells)} negative")
    
    # 2. By end hour
    print("\n2. EDGE BY END HOUR (time to resolution):")
    print("-" * 60)
    by_end_hour = defaultdict(list)
    for c in all_bucket_cells:
        by_end_hour[c['end_hour']].append(c)
    
    for end_hour in sorted(by_end_hour.keys(), reverse=True):
        cells = by_end_hour[end_hour]
        avg_edge = np.mean([c['edge_after_fill'] for c in cells])
        std_edge = np.std([c['edge_after_fill'] for c in cells])
        n_neg = sum(1 for c in cells if c['edge_after_fill'] < 0)
        print(f"  {end_hour:>2}h to resolution: avg edge = {avg_edge:>+8.1f} bps, std = {std_edge:>7.1f}, {n_neg}/{len(cells)} negative")
    
    # 3. By threshold
    print("\n3. EDGE BY THRESHOLD:")
    print("-" * 60)
    by_threshold = defaultdict(list)
    for c in all_bucket_cells:
        by_threshold[c['threshold']].append(c)
    
    for threshold in sorted(by_threshold.keys()):
        cells = by_threshold[threshold]
        avg_edge = np.mean([c['edge_after_fill'] for c in cells])
        std_edge = np.std([c['edge_after_fill'] for c in cells])
        n_neg = sum(1 for c in cells if c['edge_after_fill'] < 0)
        thresh_pct = float(threshold) * 100
        print(f"  ≥{thresh_pct:>4.0f}% dip: avg edge = {avg_edge:>+8.1f} bps, std = {std_edge:>7.1f}, {n_neg}/{len(cells)} negative")
    
    # 4. By observation window (longer windows = more time for dip to develop)
    print("\n4. EDGE BY OBSERVATION WINDOW:")
    print("-" * 60)
    by_window = defaultdict(list)
    for c in all_bucket_cells:
        by_window[c['observation_window']].append(c)
    
    for window in sorted(by_window.keys()):
        cells = by_window[window]
        avg_edge = np.mean([c['edge_after_fill'] for c in cells])
        std_edge = np.std([c['edge_after_fill'] for c in cells])
        print(f"  {window:>2}h window: avg edge = {avg_edge:>+8.1f} bps (n_cells={len(cells)})")


def check_adverse_selection_signature(all_cells):
    """
    Check if there's evidence of adverse selection in the data.
    Adverse selection: getting filled tends to be bad (price moved against you for a reason).
    """
    print("\n")
    print("="*100)
    print("ADVERSE SELECTION SIGNATURE CHECK")
    print("="*100)
    print("\nComparing unconditional edge to edge after fill.")
    print("If fill hurts you (edge after fill < uncond edge), that's adverse selection.")
    print()
    
    cells_with_both = [c for c in all_cells if c['uncond_edge'] is not None and c['edge_after_fill'] is not None and c['prob_bucket'] == 'all']
    
    fill_benefit = []
    for c in cells_with_both:
        benefit = c['edge_after_fill'] - c['uncond_edge']
        fill_benefit.append({
            **c,
            'fill_benefit': benefit
        })
    
    # Summary
    avg_benefit = np.mean([c['fill_benefit'] for c in fill_benefit])
    n_positive = sum(1 for c in fill_benefit if c['fill_benefit'] > 0)
    n_negative = sum(1 for c in fill_benefit if c['fill_benefit'] < 0)
    
    print(f"Average fill benefit: {avg_benefit:>+.1f} bps")
    print(f"Cells where fill helps:  {n_positive} ({n_positive/len(fill_benefit)*100:.1f}%)")
    print(f"Cells where fill hurts:  {n_negative} ({n_negative/len(fill_benefit)*100:.1f}%)")
    
    if avg_benefit > 20:
        print("\n--> FINDING: Fill mechanism HELPS on average (favorable selection)")
        print("    This suggests limit orders capture favorable dips.")
    elif avg_benefit < -20:
        print("\n--> FINDING: Fill mechanism HURTS on average (adverse selection)")
        print("    This suggests getting filled signals negative information.")
    else:
        print("\n--> FINDING: Fill mechanism is approximately neutral")
    
    # Break down by tercile
    print("\n\nFILL BENEFIT BY TERCILE:")
    print("-" * 40)
    for tercile in ['early', 'mid', 'late']:
        tercile_cells = [c for c in fill_benefit if c['tercile'] == tercile]
        if tercile_cells:
            avg = np.mean([c['fill_benefit'] for c in tercile_cells])
            print(f"  {tercile:<8}: fill benefit = {avg:>+8.1f} bps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze strategy weakness')
    parser.add_argument('json_file', help='Path to phase4d JSON file')
    
    args = parser.parse_args()
    
    data = load_json(args.json_file)
    
    # Collect all cells
    all_cells = collect_all_cells(data)
    print(f"\nLoaded {len(all_cells)} cells from JSON\n")
    
    # Identify worst cells
    worst = identify_worst_cells(all_cells)
    
    # Test baseline hypothesis
    test_baseline_hypothesis(all_cells)
    
    # Analyze patterns
    analyze_weakness_patterns(all_cells)
    
    # Check adverse selection
    check_adverse_selection_signature(all_cells)
