#!/usr/bin/env python3
"""
Analysis Script 1: Window-Dependent Performance by Probability Bucket

PURPOSE:
  Show how edge varies by interval AND probability bucket.
  Your hypothesis: "Fading favorites far from resolution is catastrophically bad,
  but becomes rapidly favorable close to resolution."

This script prints edge, sample size, and fill rate for ALL combinations of:
  - Intervals (windows)
  - Probability buckets
  - Move thresholds (5%, 10%, 15%, 20%)
  - Terciles (early, mid, late)
"""

import json
import sys
import argparse
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def get_ordered_intervals(surface):
    """Order intervals by end hour (farther from resolution first)."""
    interval_order = []
    for interval_label in surface.keys():
        parts = interval_label.replace('h_to_', '_').replace('h', '').split('_')
        if len(parts) >= 2:
            try:
                end_hour = int(parts[1])
                interval_order.append((interval_label, end_hour))
            except:
                interval_order.append((interval_label, 999))
        else:
            interval_order.append((interval_label, 999))
    
    interval_order.sort(key=lambda x: -x[1])  # Sort by end hour descending
    return [x[0] for x in interval_order]


def analyze_all_combinations(data):
    """
    Print edge, n, and fill rate for ALL (threshold, tercile) combinations.
    Each combination gets its own table showing interval × prob_bucket.
    """
    
    surface = data.get('surface', {})
    prob_buckets = [b[0] for b in data.get('prob_buckets', [])]
    intervals = get_ordered_intervals(surface)
    
    # All thresholds and terciles
    threshold_keys = ['0.05', '0.1', '0.15', '0.2']
    threshold_labels = ['5%', '10%', '15%', '20%']
    terciles = ['early', 'mid', 'late']
    
    print("="*120)
    print("WINDOW-DEPENDENT PERFORMANCE BY PROBABILITY BUCKET")
    print("="*120)
    print("\nShowing Edge After Fill (bps) | Sample Size (n) | Fill Rate (%)")
    print("for each (Threshold, Tercile) combination\n")
    
    # Iterate through all threshold/tercile combinations
    for thresh_key, thresh_label in zip(threshold_keys, threshold_labels):
        for tercile in terciles:
            print("\n" + "="*120)
            print(f"THRESHOLD: ≥{thresh_label} move | TERCILE: {tercile.upper()}")
            print("="*120)
            
            # Build results matrix
            results = {}
            for interval_label in intervals:
                results[interval_label] = {}
                threshold_data = surface.get(interval_label, {}).get(thresh_key, {})
                
                for prob_bucket in prob_buckets + ['all']:
                    bucket_data = threshold_data.get(prob_bucket, {})
                    tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
                    
                    if tercile_data.get('status') == 'ok':
                        results[interval_label][prob_bucket] = {
                            'edge': tercile_data.get('edge_after_fill_bps'),
                            'se': tercile_data.get('se_edge_after_fill'),
                            'n': tercile_data.get('n_samples'),
                            'fill_rate': tercile_data.get('fill_rate'),
                        }
                    else:
                        results[interval_label][prob_bucket] = None
            
            # Print table for this combination
            header_buckets = prob_buckets + ['all']
            
            # --- EDGE TABLE ---
            print(f"\n  EDGE AFTER FILL (bps):")
            header = f"  {'Interval':<12} |" + "".join([f"{b:>12} |" for b in header_buckets])
            print(header)
            print("  " + "-" * (len(header) - 2))
            
            for interval_label in intervals:
                row = f"  {interval_label:<12} |"
                for bucket in header_buckets:
                    r = results[interval_label].get(bucket)
                    if r and r['edge'] is not None:
                        edge = r['edge']
                        row += f"{edge:>+12.0f} |"
                    else:
                        row += f"{'---':>12} |"
                print(row)
            
            # --- SAMPLE SIZE TABLE ---
            print(f"\n  SAMPLE SIZE (n):")
            print(header)
            print("  " + "-" * (len(header) - 2))
            
            for interval_label in intervals:
                row = f"  {interval_label:<12} |"
                for bucket in header_buckets:
                    r = results[interval_label].get(bucket)
                    if r and r['n'] is not None:
                        row += f"{r['n']:>12,} |"
                    else:
                        row += f"{'---':>12} |"
                print(row)
            
            # --- FILL RATE TABLE ---
            print(f"\n  FILL RATE (%):")
            print(header)
            print("  " + "-" * (len(header) - 2))
            
            for interval_label in intervals:
                row = f"  {interval_label:<12} |"
                for bucket in header_buckets:
                    r = results[interval_label].get(bucket)
                    if r and r['fill_rate'] is not None:
                        row += f"{r['fill_rate']*100:>11.1f}% |"
                    else:
                        row += f"{'---':>12} |"
                print(row)


def print_summary_patterns(data):
    """
    Print a summary highlighting key patterns across all combinations.
    """
    surface = data.get('surface', {})
    prob_buckets = [b[0] for b in data.get('prob_buckets', [])]
    intervals = get_ordered_intervals(surface)
    
    threshold_keys = ['0.05', '0.1', '0.15', '0.2']
    terciles = ['early', 'mid', 'late']
    
    print("\n\n" + "="*120)
    print("SUMMARY: KEY PATTERNS")
    print("="*120)
    
    # Collect all cells for pattern analysis
    all_cells = []
    
    for interval_label in intervals:
        # Parse end hour
        parts = interval_label.replace('h_to_', '_').replace('h', '').split('_')
        try:
            end_hour = int(parts[1])
        except:
            end_hour = None
        
        for thresh_key in threshold_keys:
            threshold_data = surface.get(interval_label, {}).get(thresh_key, {})
            
            for prob_bucket in prob_buckets:
                bucket_data = threshold_data.get(prob_bucket, {})
                
                for tercile in terciles:
                    tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
                    
                    if tercile_data.get('status') == 'ok':
                        all_cells.append({
                            'interval': interval_label,
                            'end_hour': end_hour,
                            'threshold': thresh_key,
                            'prob_bucket': prob_bucket,
                            'tercile': tercile,
                            'edge': tercile_data.get('edge_after_fill_bps'),
                            'se': tercile_data.get('se_edge_after_fill'),
                            'n': tercile_data.get('n_samples'),
                            'fill_rate': tercile_data.get('fill_rate'),
                        })
    
    # 1. Best and worst cells overall
    cells_with_edge = [c for c in all_cells if c['edge'] is not None]
    cells_with_edge.sort(key=lambda x: x['edge'], reverse=True)
    
    print("\n1. TOP 10 BEST PERFORMING CELLS:")
    print("-" * 100)
    print(f"   {'Interval':<12} {'Thresh':>7} {'Bucket':>10} {'Tercile':>8} {'Edge':>10} {'SE':>8} {'n':>8}")
    for c in cells_with_edge[:10]:
        thresh_pct = float(c['threshold']) * 100
        se = c['se'] if c['se'] else 0
        print(f"   {c['interval']:<12} {thresh_pct:>6.0f}% {c['prob_bucket']:>10} {c['tercile']:>8} {c['edge']:>+10.0f} {se:>8.1f} {c['n']:>8,}")
    
    print("\n2. TOP 10 WORST PERFORMING CELLS:")
    print("-" * 100)
    print(f"   {'Interval':<12} {'Thresh':>7} {'Bucket':>10} {'Tercile':>8} {'Edge':>10} {'SE':>8} {'n':>8}")
    for c in cells_with_edge[-10:]:
        thresh_pct = float(c['threshold']) * 100
        se = c['se'] if c['se'] else 0
        print(f"   {c['interval']:<12} {thresh_pct:>6.0f}% {c['prob_bucket']:>10} {c['tercile']:>8} {c['edge']:>+10.0f} {se:>8.1f} {c['n']:>8,}")
    
    # 2. Average edge by probability bucket (across all conditions)
    print("\n3. AVERAGE EDGE BY PROBABILITY BUCKET (across all intervals/thresholds/terciles):")
    print("-" * 60)
    
    for bucket in prob_buckets:
        bucket_cells = [c for c in cells_with_edge if c['prob_bucket'] == bucket]
        if bucket_cells:
            avg_edge = sum(c['edge'] for c in bucket_cells) / len(bucket_cells)
            total_n = sum(c['n'] for c in bucket_cells)
            n_positive = sum(1 for c in bucket_cells if c['edge'] > 0)
            print(f"   {bucket:<12}: avg edge = {avg_edge:>+8.1f} bps, {n_positive}/{len(bucket_cells)} positive, total n = {total_n:,}")
    
    # 3. Average edge by tercile
    print("\n4. AVERAGE EDGE BY TERCILE (across all intervals/thresholds/buckets):")
    print("-" * 60)
    
    for tercile in terciles:
        tercile_cells = [c for c in cells_with_edge if c['tercile'] == tercile]
        if tercile_cells:
            avg_edge = sum(c['edge'] for c in tercile_cells) / len(tercile_cells)
            total_n = sum(c['n'] for c in tercile_cells)
            n_positive = sum(1 for c in tercile_cells if c['edge'] > 0)
            print(f"   {tercile:<12}: avg edge = {avg_edge:>+8.1f} bps, {n_positive}/{len(tercile_cells)} positive, total n = {total_n:,}")
    
    # 4. Average edge by threshold
    print("\n5. AVERAGE EDGE BY THRESHOLD (across all intervals/terciles/buckets):")
    print("-" * 60)
    
    for thresh_key in threshold_keys:
        thresh_cells = [c for c in cells_with_edge if c['threshold'] == thresh_key]
        if thresh_cells:
            avg_edge = sum(c['edge'] for c in thresh_cells) / len(thresh_cells)
            total_n = sum(c['n'] for c in thresh_cells)
            thresh_pct = float(thresh_key) * 100
            n_positive = sum(1 for c in thresh_cells if c['edge'] > 0)
            print(f"   ≥{thresh_pct:>4.0f}%: avg edge = {avg_edge:>+8.1f} bps, {n_positive}/{len(thresh_cells)} positive, total n = {total_n:,}")
    
    # 5. Favorites analysis - does edge differ by proximity to resolution?
    print("\n6. FAVORITES (75_90, 90_99) BY INTERVAL:")
    print("-" * 80)
    print("   Does fading favorites become better closer to resolution?")
    print()
    
    fav_buckets = ['75_90', '90_99']
    
    for interval_label in intervals:
        fav_cells = [c for c in cells_with_edge 
                     if c['interval'] == interval_label and c['prob_bucket'] in fav_buckets]
        if fav_cells:
            avg_edge = sum(c['edge'] for c in fav_cells) / len(fav_cells)
            n_positive = sum(1 for c in fav_cells if c['edge'] > 0)
            print(f"   {interval_label:<12}: avg edge = {avg_edge:>+8.1f} bps, {n_positive}/{len(fav_cells)} positive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze window-dependent performance')
    parser.add_argument('json_file', help='Path to phase4d JSON file')
    parser.add_argument('--summary-only', '-s', action='store_true',
                        help='Only print summary patterns, skip individual tables')
    
    args = parser.parse_args()
    
    data = load_json(args.json_file)
    
    if not args.summary_only:
        # Print all individual tables
        analyze_all_combinations(data)
    
    # Always print summary
    print_summary_patterns(data)