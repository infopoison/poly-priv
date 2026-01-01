#!/usr/bin/env python3
"""
Analysis Script 4: Resolution Proximity and Overreaction Analysis

PURPOSE:
  Your observation: "markets seem prone to overreaction closer to resolution, 
  and it seems like the overreaction is slightly outsized into favorites"

This script:
  1. Tests whether edge increases as resolution approaches
  2. Checks if favorites show different patterns than non-favorites
  3. Identifies the transition point where fading stops working
  4. Computes the "overreaction profile" by probability bucket
"""

import json
import sys
import argparse
import numpy as np
from collections import defaultdict

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def parse_interval_label(interval_label):
    """Parse interval label like '48h_to_24h' to (start, end, window)."""
    parts = interval_label.replace('h_to_', '_').replace('h', '').split('_')
    try:
        start = int(parts[0])
        end = int(parts[1])
        window = start - end
        return start, end, window
    except:
        return None, None, None


def analyze_resolution_proximity(data, tercile='early', threshold=0.10):
    """
    Analyze how edge changes as we get closer to resolution.
    """
    print("="*100)
    print(f"RESOLUTION PROXIMITY ANALYSIS")
    print(f"Tercile: {tercile}, Threshold: ≥{threshold*100:.0f}%")
    print("="*100)
    
    surface = data.get('surface', {})
    threshold_key = str(threshold)  # Convert to string key
    
    # Collect by end hour
    by_end_hour = defaultdict(dict)
    
    for interval_label, interval_data in surface.items():
        start, end, window = parse_interval_label(interval_label)
        if end is None:
            continue
        
        threshold_data = interval_data.get(threshold_key, {})
        
        for prob_bucket in ['all', '51_60', '60_75', '75_90', '90_99']:
            bucket_data = threshold_data.get(prob_bucket, {})
            tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
            
            if tercile_data.get('status') == 'ok':
                if prob_bucket not in by_end_hour[end]:
                    by_end_hour[end][prob_bucket] = []
                
                by_end_hour[end][prob_bucket].append({
                    'interval': interval_label,
                    'window': window,
                    'edge': tercile_data.get('edge_after_fill_bps'),
                    'uncond_edge': tercile_data.get('unconditional_edge_bps'),
                    'se': tercile_data.get('se_edge_after_fill'),
                    'n': tercile_data.get('n_samples'),
                    'fill_rate': tercile_data.get('fill_rate'),
                })
    
    # Print summary table
    print("\n1. EDGE BY HOURS TO RESOLUTION (using 'all' bucket):")
    print("-" * 80)
    print(f"{'End Hour':>10} {'Interval':>15} {'Edge After Fill':>16} {'SE':>10} {'n':>10} {'Fill Rate':>10}")
    
    for end_hour in sorted(by_end_hour.keys(), reverse=True):
        all_data = by_end_hour[end_hour].get('all', [])
        for d in all_data:
            print(f"{end_hour:>10}h {d['interval']:>15} {d['edge']:>+16.0f} {d['se']:>10.1f} {d['n']:>10,} {d['fill_rate']*100:>9.1f}%")
    
    # Analyze transition point
    print("\n\n2. IDENTIFYING THE TRANSITION POINT:")
    print("-" * 80)
    print("Where does edge flip from positive to negative?")
    
    positive_hours = []
    negative_hours = []
    
    for end_hour in sorted(by_end_hour.keys(), reverse=True):
        all_data = by_end_hour[end_hour].get('all', [])
        if all_data:
            # Average across windows ending at this hour
            avg_edge = np.mean([d['edge'] for d in all_data])
            if avg_edge > 0:
                positive_hours.append(end_hour)
                print(f"  {end_hour:>2}h to resolution: avg edge = {avg_edge:>+8.1f} bps  [POSITIVE]")
            else:
                negative_hours.append(end_hour)
                print(f"  {end_hour:>2}h to resolution: avg edge = {avg_edge:>+8.1f} bps  [NEGATIVE]")
    
    if positive_hours and negative_hours:
        transition = max(negative_hours)
        print(f"\n  --> TRANSITION POINT: Edge becomes negative at ≤{transition}h to resolution")
        print(f"      Safe zone appears to be ≥{min(positive_hours)}h to resolution")
    
    return by_end_hour


def compare_favorites_vs_others(data, tercile='early'):
    """
    Compare overreaction patterns in favorites vs non-favorites.
    """
    print("\n")
    print("="*100)
    print("FAVORITES vs NON-FAVORITES COMPARISON")
    print("="*100)
    print("\nDoes overreaction behavior differ by probability bucket?")
    
    surface = data.get('surface', {})
    
    # Group buckets
    bucket_groups = {
        'longshots': ['sub_51'],
        'toss_up': ['51_60', '60_75'],
        'favorites': ['75_90', '90_99'],
    }
    
    # Threshold configs: (key_in_json, display_value)
    threshold_configs = [('0.05', 0.05), ('0.1', 0.10), ('0.15', 0.15), ('0.2', 0.20)]
    
    results = defaultdict(lambda: defaultdict(list))
    
    for interval_label, interval_data in surface.items():
        start, end, window = parse_interval_label(interval_label)
        if end is None:
            continue
        
        for threshold_key, threshold_val in threshold_configs:
            threshold_data = interval_data.get(threshold_key, {})
            
            for group_name, bucket_list in bucket_groups.items():
                for bucket in bucket_list:
                    bucket_data = threshold_data.get(bucket, {})
                    tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
                    
                    if tercile_data.get('status') == 'ok':
                        results[group_name][threshold_key].append({
                            'interval': interval_label,
                            'end_hour': end,
                            'edge': tercile_data.get('edge_after_fill_bps'),
                            'uncond_edge': tercile_data.get('unconditional_edge_bps'),
                            'n': tercile_data.get('n_samples'),
                        })
    
    # Print comparison by threshold
    for threshold_key, threshold_val in [('0.1', 0.10), ('0.15', 0.15), ('0.2', 0.20)]:
        print(f"\n\nThreshold: ≥{threshold_val*100:.0f}% dip")
        print("-" * 70)
        print(f"{'Group':<15} {'Avg Edge':>12} {'Avg Uncond':>12} {'Total n':>12} {'# Cells':>10}")
        
        for group_name in ['longshots', 'toss_up', 'favorites']:
            cells = results[group_name][threshold_key]
            if cells:
                avg_edge = np.mean([c['edge'] for c in cells if c['edge'] is not None])
                avg_uncond = np.mean([c['uncond_edge'] for c in cells if c['uncond_edge'] is not None])
                total_n = sum(c['n'] for c in cells)
                
                print(f"{group_name:<15} {avg_edge:>+12.1f} {avg_uncond:>+12.1f} {total_n:>12,} {len(cells):>10}")
    
    # Detailed by prob bucket and interval
    print("\n\n3. DETAILED VIEW: FAVORITES (75_90 and 90_99) BY INTERVAL")
    print("-" * 80)
    
    fav_buckets = ['75_90', '90_99']
    
    for threshold_key in ['0.1']:
        print(f"\nThreshold: ≥10%")
        print(f"{'Interval':<15} {'Bucket':>10} {'Edge':>10} {'Uncond':>10} {'Fill Ben':>10} {'n':>8}")
        
        for interval_label, interval_data in sorted(surface.items()):
            start, end, window = parse_interval_label(interval_label)
            threshold_data = interval_data.get(threshold_key, {})
            
            for bucket in fav_buckets:
                bucket_data = threshold_data.get(bucket, {})
                tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
                
                if tercile_data.get('status') == 'ok':
                    edge = tercile_data.get('edge_after_fill_bps', 0)
                    uncond = tercile_data.get('unconditional_edge_bps', 0)
                    fill_benefit = edge - uncond
                    n = tercile_data.get('n_samples', 0)
                    
                    print(f"{interval_label:<15} {bucket:>10} {edge:>+10.0f} {uncond:>+10.0f} {fill_benefit:>+10.0f} {n:>8,}")


def analyze_overreaction_asymmetry(data, threshold=0.10, tercile='early'):
    """
    Test whether overreaction is symmetric across probability buckets.
    """
    print("\n")
    print("="*100)
    print(f"OVERREACTION ASYMMETRY ANALYSIS (≥{threshold*100:.0f}% dips)")
    print("="*100)
    print("\nIs overreaction stronger for favorites or longshots?")
    print("'Unconditional edge' measures pure overreaction before fill mechanics.")
    
    surface = data.get('surface', {})
    threshold_key = str(threshold)  # Convert to string key
    
    # Collect unconditional edges by probability bucket
    bucket_edges = defaultdict(list)
    
    for interval_label, interval_data in surface.items():
        start, end, window = parse_interval_label(interval_label)
        
        # Only use intervals ending at 4h+ to avoid contamination
        if end is not None and end < 4:
            continue
        
        threshold_data = interval_data.get(threshold_key, {})
        
        for prob_bucket in ['sub_51', '51_60', '60_75', '75_90', '90_99', '99_plus']:
            bucket_data = threshold_data.get(prob_bucket, {})
            tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
            
            if tercile_data.get('status') == 'ok':
                bucket_edges[prob_bucket].append({
                    'interval': interval_label,
                    'uncond_edge': tercile_data.get('unconditional_edge_bps'),
                    'edge_after_fill': tercile_data.get('edge_after_fill_bps'),
                    'n': tercile_data.get('n_samples'),
                })
    
    print("\n\nUnconditional Edge by Probability Bucket (higher = more overreaction):")
    print("-" * 70)
    print(f"{'Bucket':<12} {'Avg Uncond Edge':>16} {'Avg After Fill':>16} {'Total n':>12}")
    
    bucket_order = ['sub_51', '51_60', '60_75', '75_90', '90_99', '99_plus']
    
    for bucket in bucket_order:
        cells = bucket_edges.get(bucket, [])
        if cells:
            # Weighted average
            total_n = sum(c['n'] for c in cells)
            if total_n > 0:
                avg_uncond = sum(c['uncond_edge'] * c['n'] for c in cells if c['uncond_edge']) / total_n
                avg_fill = sum(c['edge_after_fill'] * c['n'] for c in cells if c['edge_after_fill']) / total_n
                print(f"{bucket:<12} {avg_uncond:>+16.1f} {avg_fill:>+16.1f} {total_n:>12,}")
    
    # Test symmetry hypothesis
    print("\n\nSYMMETRY TEST:")
    print("-" * 40)
    
    # Compare favorites vs longshots
    fav_cells = bucket_edges.get('75_90', []) + bucket_edges.get('90_99', [])
    long_cells = bucket_edges.get('sub_51', [])
    toss_cells = bucket_edges.get('51_60', []) + bucket_edges.get('60_75', [])
    
    if fav_cells:
        fav_uncond = np.mean([c['uncond_edge'] for c in fav_cells if c['uncond_edge']])
        fav_n = sum(c['n'] for c in fav_cells)
        print(f"Favorites (75-99%):  avg uncond edge = {fav_uncond:>+8.1f} bps (n={fav_n:,})")
    
    if long_cells:
        long_uncond = np.mean([c['uncond_edge'] for c in long_cells if c['uncond_edge']])
        long_n = sum(c['n'] for c in long_cells)
        print(f"Longshots (<51%):    avg uncond edge = {long_uncond:>+8.1f} bps (n={long_n:,})")
    
    if toss_cells:
        toss_uncond = np.mean([c['uncond_edge'] for c in toss_cells if c['uncond_edge']])
        toss_n = sum(c['n'] for c in toss_cells)
        print(f"Toss-up (51-75%):    avg uncond edge = {toss_uncond:>+8.1f} bps (n={toss_n:,})")
    
    print("\nNOTE: Positive uncond edge = market dropped too much (overreaction)")
    print("      If favorites show LOWER uncond edge, they may be more informationally efficient.")


def sampling_noise_assessment(data, threshold=0.10):
    """
    Assess how much of the patterns could be sampling noise.
    """
    print("\n")
    print("="*100)
    print("SAMPLING NOISE ASSESSMENT")
    print("="*100)
    print("\nHow confident can we be in the patterns?")
    
    surface = data.get('surface', {})
    threshold_key = str(threshold)  # Convert to string key
    
    # Collect all SEs for 'all' bucket
    all_se_data = []
    
    for interval_label, interval_data in surface.items():
        threshold_data = interval_data.get(threshold_key, {})
        bucket_data = threshold_data.get('all', {})
        
        for tercile in ['early', 'mid', 'late']:
            tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
            
            if tercile_data.get('status') == 'ok':
                se = tercile_data.get('se_edge_after_fill')
                n = tercile_data.get('n_samples')
                edge = tercile_data.get('edge_after_fill_bps')
                
                if se and n and edge:
                    all_se_data.append({
                        'interval': interval_label,
                        'tercile': tercile,
                        'edge': edge,
                        'se': se,
                        'n': n,
                        'z': edge / se if se > 0 else 0,
                    })
    
    if not all_se_data:
        print("No SE data available")
        return
    
    # Summary statistics
    avg_se = np.mean([d['se'] for d in all_se_data])
    min_se = min([d['se'] for d in all_se_data])
    max_se = max([d['se'] for d in all_se_data])
    
    print(f"\nStandard Error Statistics (≥{threshold*100:.0f}% threshold, 'all' bucket):")
    print(f"  Average SE:  {avg_se:.1f} bps")
    print(f"  Min SE:      {min_se:.1f} bps")
    print(f"  Max SE:      {max_se:.1f} bps")
    
    # How many would remain significant if edge was halved?
    print("\nROBUSTNESS CHECK:")
    print("  If true edge is only HALF of what we measure...")
    
    robust_positive = 0
    total = len(all_se_data)
    
    for d in all_se_data:
        halved_edge = d['edge'] / 2
        ci_low = halved_edge - 1.96 * d['se']
        if ci_low > 0:
            robust_positive += 1
    
    print(f"  Cells still significantly positive: {robust_positive}/{total} ({robust_positive/total*100:.1f}%)")
    
    # Z-score distribution
    z_scores = [d['z'] for d in all_se_data]
    z_positive = sum(1 for z in z_scores if z > 1.96)
    z_negative = sum(1 for z in z_scores if z < -1.96)
    
    print(f"\nZ-SCORE DISTRIBUTION:")
    print(f"  z > 1.96 (sig positive):  {z_positive} ({z_positive/total*100:.1f}%)")
    print(f"  z < -1.96 (sig negative): {z_negative} ({z_negative/total*100:.1f}%)")
    print(f"  |z| < 1.96 (not sig):     {total - z_positive - z_negative} ({(total-z_positive-z_negative)/total*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze resolution proximity effects')
    parser.add_argument('json_file', help='Path to phase4d JSON file')
    parser.add_argument('--tercile', choices=['early', 'mid', 'late'], default='early',
                        help='Which tercile to analyze (default: early)')
    parser.add_argument('--threshold', '-t', type=float, default=0.10,
                        help='Move threshold (default: 0.10 = 10%%)')
    
    args = parser.parse_args()
    
    data = load_json(args.json_file)
    
    # Resolution proximity
    by_end_hour = analyze_resolution_proximity(data, args.tercile, args.threshold)
    
    # Favorites vs others
    compare_favorites_vs_others(data, args.tercile)
    
    # Asymmetry
    analyze_overreaction_asymmetry(data, args.threshold, args.tercile)
    
    # Sampling noise
    sampling_noise_assessment(data, args.threshold)
