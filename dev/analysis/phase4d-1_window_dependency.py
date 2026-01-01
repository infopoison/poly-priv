#!/usr/bin/env python3
"""
Analysis Script 1: Window-Dependent Performance by Probability Bucket

PURPOSE:
  Show how edge varies by interval AND probability bucket.
  Your hypothesis: "Fading favorites far from resolution is catastrophically bad,
  but becomes rapidly favorable close to resolution."

This script will print a matrix showing edge by (interval × prob_bucket)
to reveal whether this pattern holds.
"""

import json
import sys
import argparse
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_window_dependency(data, threshold=0.10, tercile='early'):
    """
    For a given threshold and tercile, show edge across all intervals × prob buckets.
    
    Args:
        data: loaded JSON
        threshold: move threshold (default 0.10 = 10%)
        tercile: which tercile to analyze (early/mid/late)
    """
    
    surface = data.get('surface', {})
    prob_buckets = [b[0] for b in data.get('prob_buckets', [])]
    
    # Convert threshold to string key as stored in JSON
    # JSON has '0.05', '0.1', '0.15', '0.2' (not '0.10')
    threshold_key = str(threshold)
    if threshold_key == '0.1':
        threshold_key = '0.1'  # already correct
    elif threshold_key == '0.05':
        threshold_key = '0.05'
    # Handle case where user passes 0.10 which becomes '0.1'
    
    # Order intervals by end hour (most meaningful ordering)
    # Parse interval labels to get end hour
    interval_order = []
    for interval_label in surface.keys():
        # Parse like '48h_to_24h' -> end=24
        parts = interval_label.replace('h_to_', '_').replace('h', '').split('_')
        if len(parts) >= 2:
            try:
                end_hour = int(parts[1])
                interval_order.append((interval_label, end_hour))
            except:
                interval_order.append((interval_label, 999))
        else:
            interval_order.append((interval_label, 999))
    
    interval_order.sort(key=lambda x: -x[1])  # Sort by end hour descending (farther first)
    intervals = [x[0] for x in interval_order]
    
    print("="*100)
    print(f"WINDOW-DEPENDENT PERFORMANCE BY PROBABILITY BUCKET")
    print(f"Threshold: ≥{threshold*100:.0f}% move | Tercile: {tercile}")
    print("="*100)
    print()
    
    # Build matrix
    results = {}
    for interval_label in intervals:
        results[interval_label] = {}
        threshold_data = surface.get(interval_label, {}).get(threshold_key, {})
        
        for prob_bucket in prob_buckets + ['all']:
            bucket_data = threshold_data.get(prob_bucket, {})
            tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
            
            if tercile_data.get('status') == 'ok':
                results[interval_label][prob_bucket] = {
                    'edge': tercile_data.get('edge_after_fill_bps'),
                    'uncond_edge': tercile_data.get('unconditional_edge_bps'),
                    'se': tercile_data.get('se_edge_after_fill'),
                    'n': tercile_data.get('n_samples'),
                    'fill_rate': tercile_data.get('fill_rate'),
                }
            else:
                results[interval_label][prob_bucket] = None
    
    # Print header
    header_buckets = prob_buckets + ['all']
    header = f"{'Interval':<15} | " + " | ".join([f"{b:>10}" for b in header_buckets])
    print(header)
    print("-" * len(header))
    
    # Print rows - Edge After Fill
    print("\nEDGE AFTER FILL (bps):")
    print("-" * 100)
    for interval_label in intervals:
        row = f"{interval_label:<15} | "
        for bucket in header_buckets:
            r = results[interval_label].get(bucket)
            if r and r['edge'] is not None:
                edge = r['edge']
                # Highlight very negative vs very positive
                if edge < -100:
                    row += f"{edge:>10.0f}** | "
                elif edge > 100:
                    row += f"{edge:>10.0f}++ | "
                else:
                    row += f"{edge:>10.0f}   | "
            else:
                row += f"{'---':>10}   | "
        print(row)
    
    # Print sample sizes
    print("\n\nSAMPLE SIZES (n):")
    print("-" * 100)
    for interval_label in intervals:
        row = f"{interval_label:<15} | "
        for bucket in header_buckets:
            r = results[interval_label].get(bucket)
            if r and r['n'] is not None:
                row += f"{r['n']:>10,}   | "
            else:
                row += f"{'---':>10}   | "
        print(row)
    
    # Print fill rates
    print("\n\nFILL RATES (%):")
    print("-" * 100)
    for interval_label in intervals:
        row = f"{interval_label:<15} | "
        for bucket in header_buckets:
            r = results[interval_label].get(bucket)
            if r and r['fill_rate'] is not None:
                row += f"{r['fill_rate']*100:>10.1f}%  | "
            else:
                row += f"{'---':>10}   | "
        print(row)
    
    # Summary analysis
    print("\n")
    print("="*100)
    print("KEY PATTERNS")
    print("="*100)
    
    # Check hypothesis: does edge improve closer to resolution for favorites?
    print("\n1. FAVORITES (75_90 and 90_99) BY INTERVAL:")
    print("-" * 60)
    for prob_bucket in ['75_90', '90_99']:
        print(f"\n   {prob_bucket}:")
        for interval_label in intervals:
            r = results[interval_label].get(prob_bucket)
            if r and r['edge'] is not None:
                edge = r['edge']
                n = r['n']
                se = r['se'] if r['se'] else 0
                ci_low = edge - 1.96 * se
                ci_high = edge + 1.96 * se
                print(f"      {interval_label:<15}: {edge:>+7.0f} bps (95% CI: [{ci_low:>+7.0f}, {ci_high:>+7.0f}]) n={n:,}")
    
    # Check overall edge significance
    print("\n2. OVERALL EDGE ('all' bucket) WITH CONFIDENCE INTERVALS:")
    print("-" * 60)
    for interval_label in intervals:
        r = results[interval_label].get('all')
        if r and r['edge'] is not None:
            edge = r['edge']
            se = r['se'] if r['se'] else 0
            n = r['n']
            ci_low = edge - 1.96 * se
            ci_high = edge + 1.96 * se
            
            # Statistical significance indicator
            sig = ""
            if ci_low > 0:
                sig = "*** SIGNIFICANT POSITIVE"
            elif ci_high < 0:
                sig = "*** SIGNIFICANT NEGATIVE"
            else:
                sig = "(CI includes zero)"
            
            print(f"   {interval_label:<15}: {edge:>+7.0f} bps (95% CI: [{ci_low:>+7.0f}, {ci_high:>+7.0f}]) {sig}")
    
    return results


def compare_terciles(data, threshold=0.10, prob_bucket='all'):
    """
    Compare early vs late tercile performance to see timing effects.
    """
    print("\n")
    print("="*100)
    print(f"TERCILE COMPARISON (EARLY vs LATE) - {prob_bucket} bucket, ≥{threshold*100:.0f}% moves")
    print("="*100)
    
    surface = data.get('surface', {})
    threshold_key = str(threshold)  # Convert to string key
    
    print(f"\n{'Interval':<15} | {'Early Edge':>12} | {'Late Edge':>12} | {'Diff (E-L)':>12} | {'Early SE':>10} | {'Late SE':>10}")
    print("-" * 90)
    
    for interval_label in sorted(surface.keys()):
        threshold_data = surface.get(interval_label, {}).get(threshold_key, {})
        bucket_data = threshold_data.get(prob_bucket, {})
        
        early = bucket_data.get('terciles', {}).get('early', {})
        late = bucket_data.get('terciles', {}).get('late', {})
        
        if early.get('status') == 'ok' and late.get('status') == 'ok':
            early_edge = early.get('edge_after_fill_bps', 0)
            late_edge = late.get('edge_after_fill_bps', 0)
            early_se = early.get('se_edge_after_fill', 0)
            late_se = late.get('se_edge_after_fill', 0)
            diff = early_edge - late_edge
            
            print(f"{interval_label:<15} | {early_edge:>+12.0f} | {late_edge:>+12.0f} | {diff:>+12.0f} | {early_se:>10.1f} | {late_se:>10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze window-dependent performance')
    parser.add_argument('json_file', help='Path to phase4d JSON file')
    parser.add_argument('--threshold', '-t', type=float, default=0.10, 
                        help='Move threshold (default: 0.10 = 10%%)')
    parser.add_argument('--tercile', choices=['early', 'mid', 'late'], default='early',
                        help='Which tercile to analyze (default: early)')
    
    args = parser.parse_args()
    
    data = load_json(args.json_file)
    
    # Main analysis
    results = analyze_window_dependency(data, args.threshold, args.tercile)
    
    # Tercile comparison
    compare_terciles(data, args.threshold, 'all')
    
    # Also show for favorites
    compare_terciles(data, args.threshold, '75_90')
