#!/usr/bin/env python3
"""
Analysis Script 2: Edge Existence and Confidence Interval Analysis

PURPOSE:
  Your concern: "The confidence intervals are showing that there could be 
  little to no edge, but it seems like there definitely is some edge overall."

This script:
  1. Aggregates across all cells to estimate overall edge
  2. Shows where edge is statistically significant vs noise
  3. Computes weighted average edge across conditions
  4. Tests hypothesis: is there genuine edge or just noise?
"""

import json
import sys
import argparse
import numpy as np
from scipy import stats

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def collect_all_edges(data, tercile='early'):
    """
    Collect edge estimates from all (interval, threshold, prob_bucket) cells.
    Returns list of (edge, se, n, cell_info) tuples.
    """
    surface = data.get('surface', {})
    all_edges = []
    
    for interval_label, interval_data in surface.items():
        for threshold, threshold_data in interval_data.items():
            for prob_bucket, bucket_data in threshold_data.items():
                tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
                
                if tercile_data.get('status') == 'ok':
                    edge = tercile_data.get('edge_after_fill_bps')
                    se = tercile_data.get('se_edge_after_fill')
                    n = tercile_data.get('n_samples')
                    uncond_edge = tercile_data.get('unconditional_edge_bps')
                    uncond_se = tercile_data.get('se_unconditional_edge')
                    
                    if edge is not None and se is not None and n is not None:
                        all_edges.append({
                            'interval': interval_label,
                            'threshold': threshold,
                            'prob_bucket': prob_bucket,
                            'edge_after_fill': edge,
                            'se_after_fill': se,
                            'uncond_edge': uncond_edge,
                            'uncond_se': uncond_se,
                            'n': n,
                            'z_score': edge / se if se > 0 else 0,
                        })
    
    return all_edges


def analyze_edge_significance(all_edges):
    """
    Analyze how many cells show significant positive/negative edge.
    """
    print("="*100)
    print("EDGE SIGNIFICANCE ANALYSIS")
    print("="*100)
    
    sig_positive = []
    sig_negative = []
    insignificant = []
    
    for e in all_edges:
        ci_low = e['edge_after_fill'] - 1.96 * e['se_after_fill']
        ci_high = e['edge_after_fill'] + 1.96 * e['se_after_fill']
        
        if ci_low > 0:
            sig_positive.append(e)
        elif ci_high < 0:
            sig_negative.append(e)
        else:
            insignificant.append(e)
    
    total = len(all_edges)
    
    print(f"\nTotal cells analyzed: {total}")
    print(f"  - Significantly POSITIVE (CI > 0): {len(sig_positive)} ({len(sig_positive)/total*100:.1f}%)")
    print(f"  - Significantly NEGATIVE (CI < 0): {len(sig_negative)} ({len(sig_negative)/total*100:.1f}%)")
    print(f"  - Insignificant (CI includes 0):   {len(insignificant)} ({len(insignificant)/total*100:.1f}%)")
    
    # Top positive cells
    if sig_positive:
        print("\n\nTOP 10 SIGNIFICANTLY POSITIVE CELLS (by z-score):")
        print("-" * 100)
        sig_positive.sort(key=lambda x: -x['z_score'])
        print(f"{'Interval':<15} {'Thresh':>8} {'ProbBucket':>12} {'Edge':>10} {'SE':>8} {'n':>8} {'z':>8}")
        for e in sig_positive[:10]:
            thresh_pct = float(e['threshold']) * 100
            print(f"{e['interval']:<15} {thresh_pct:>7.0f}% {e['prob_bucket']:>12} {e['edge_after_fill']:>+10.0f} {e['se_after_fill']:>8.1f} {e['n']:>8,} {e['z_score']:>8.2f}")
    
    # Top negative cells
    if sig_negative:
        print("\n\nTOP 10 SIGNIFICANTLY NEGATIVE CELLS (by z-score):")
        print("-" * 100)
        sig_negative.sort(key=lambda x: x['z_score'])
        print(f"{'Interval':<15} {'Thresh':>8} {'ProbBucket':>12} {'Edge':>10} {'SE':>8} {'n':>8} {'z':>8}")
        for e in sig_negative[:10]:
            thresh_pct = float(e['threshold']) * 100
            print(f"{e['interval']:<15} {thresh_pct:>7.0f}% {e['prob_bucket']:>12} {e['edge_after_fill']:>+10.0f} {e['se_after_fill']:>8.1f} {e['n']:>8,} {e['z_score']:>8.2f}")
    
    return sig_positive, sig_negative, insignificant


def compute_weighted_average_edge(all_edges, exclude_prob_buckets=None):
    """
    Compute sample-weighted average edge across all cells.
    Uses inverse-variance weighting for optimal combination.
    """
    print("\n")
    print("="*100)
    print("WEIGHTED AVERAGE EDGE ANALYSIS")
    print("="*100)
    
    # Filter
    if exclude_prob_buckets:
        edges = [e for e in all_edges if e['prob_bucket'] not in exclude_prob_buckets]
    else:
        edges = all_edges
    
    if not edges:
        print("No edges to analyze after filtering")
        return
    
    # Simple n-weighted average
    total_n = sum(e['n'] for e in edges)
    weighted_edge = sum(e['edge_after_fill'] * e['n'] for e in edges) / total_n
    
    # Inverse-variance weighted (optimal under independence)
    weights = [1 / (e['se_after_fill']**2) if e['se_after_fill'] > 0 else 0 for e in edges]
    total_weight = sum(weights)
    
    if total_weight > 0:
        iv_weighted_edge = sum(e['edge_after_fill'] * w for e, w in zip(edges, weights)) / total_weight
        iv_se = np.sqrt(1 / total_weight)  # Combined SE
    else:
        iv_weighted_edge = weighted_edge
        iv_se = 0
    
    print(f"\nNumber of cells: {len(edges)}")
    print(f"Total observations: {total_n:,}")
    print()
    print(f"Sample-weighted average edge:          {weighted_edge:>+8.1f} bps")
    print(f"Inverse-variance weighted edge:        {iv_weighted_edge:>+8.1f} bps")
    print(f"Combined SE (inv-var):                 {iv_se:>8.1f} bps")
    print(f"95% CI:                                [{iv_weighted_edge - 1.96*iv_se:>+8.1f}, {iv_weighted_edge + 1.96*iv_se:>+8.1f}]")
    
    # Statistical test
    z = iv_weighted_edge / iv_se if iv_se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    print(f"\nHypothesis test (H0: edge = 0):")
    print(f"  z-statistic: {z:.3f}")
    print(f"  p-value:     {p_value:.6f}")
    
    if p_value < 0.01:
        print(f"  --> REJECT H0 at 1% level: There IS significant edge")
    elif p_value < 0.05:
        print(f"  --> REJECT H0 at 5% level: There IS significant edge")
    else:
        print(f"  --> FAIL TO REJECT H0: Cannot confirm edge exists")
    
    return iv_weighted_edge, iv_se


def analyze_by_condition(all_edges):
    """
    Break down weighted edge by interval and by probability bucket.
    """
    print("\n")
    print("="*100)
    print("EDGE BY CONDITION")
    print("="*100)
    
    # Group by interval
    print("\n1. BY INTERVAL (using 'all' prob bucket only):")
    print("-" * 80)
    
    by_interval = {}
    for e in all_edges:
        if e['prob_bucket'] != 'all':
            continue
        interval = e['interval']
        if interval not in by_interval:
            by_interval[interval] = []
        by_interval[interval].append(e)
    
    print(f"{'Interval':<15} {'Avg Edge':>12} {'Tot SE':>10} {'Total n':>12} {'Sig?':>15}")
    for interval, edges in sorted(by_interval.items()):
        if not edges:
            continue
        
        total_n = sum(e['n'] for e in edges)
        weights = [1/(e['se_after_fill']**2) if e['se_after_fill'] > 0 else 0 for e in edges]
        total_weight = sum(weights)
        
        if total_weight > 0:
            avg_edge = sum(e['edge_after_fill'] * w for e, w in zip(edges, weights)) / total_weight
            combined_se = np.sqrt(1 / total_weight)
        else:
            avg_edge = np.mean([e['edge_after_fill'] for e in edges])
            combined_se = 0
        
        ci_low = avg_edge - 1.96 * combined_se
        ci_high = avg_edge + 1.96 * combined_se
        
        sig = ""
        if ci_low > 0:
            sig = "SIG POSITIVE"
        elif ci_high < 0:
            sig = "SIG NEGATIVE"
        else:
            sig = "not sig"
        
        print(f"{interval:<15} {avg_edge:>+12.1f} {combined_se:>10.1f} {total_n:>12,} {sig:>15}")
    
    # Group by prob bucket
    print("\n\n2. BY PROBABILITY BUCKET (using all intervals, 0.10 threshold only):")
    print("-" * 80)
    
    by_bucket = {}
    for e in all_edges:
        if e['threshold'] != '0.1':  # Focus on 10% threshold (string key)
            continue
        bucket = e['prob_bucket']
        if bucket not in by_bucket:
            by_bucket[bucket] = []
        by_bucket[bucket].append(e)
    
    bucket_order = ['sub_51', '51_60', '60_75', '75_90', '90_99', '99_plus', 'all']
    print(f"{'Prob Bucket':<12} {'Avg Edge':>12} {'Tot SE':>10} {'Total n':>12} {'Sig?':>15}")
    
    for bucket in bucket_order:
        edges = by_bucket.get(bucket, [])
        if not edges:
            continue
        
        total_n = sum(e['n'] for e in edges)
        weights = [1/(e['se_after_fill']**2) if e['se_after_fill'] > 0 else 0 for e in edges]
        total_weight = sum(weights)
        
        if total_weight > 0:
            avg_edge = sum(e['edge_after_fill'] * w for e, w in zip(edges, weights)) / total_weight
            combined_se = np.sqrt(1 / total_weight)
        else:
            avg_edge = np.mean([e['edge_after_fill'] for e in edges])
            combined_se = 0
        
        ci_low = avg_edge - 1.96 * combined_se
        ci_high = avg_edge + 1.96 * combined_se
        
        sig = ""
        if ci_low > 0:
            sig = "SIG POSITIVE"
        elif ci_high < 0:
            sig = "SIG NEGATIVE"
        else:
            sig = "not sig"
        
        print(f"{bucket:<12} {avg_edge:>+12.1f} {combined_se:>10.1f} {total_n:>12,} {sig:>15}")


def analyze_unconditional_vs_conditional(all_edges):
    """
    Compare unconditional edge to edge after fill.
    Shows how much the fill mechanism helps/hurts.
    """
    print("\n")
    print("="*100)
    print("UNCONDITIONAL vs CONDITIONAL (AFTER FILL) EDGE")
    print("="*100)
    print("\nThis shows whether the fill mechanism rescues or hurts your edge.")
    print("Fill benefit = (edge after fill) - (unconditional edge)")
    print()
    
    edges_with_both = [e for e in all_edges if e['uncond_edge'] is not None]
    
    if not edges_with_both:
        print("No data with both unconditional and conditional edges")
        return
    
    # Group by interval (using 'all' bucket, 0.10 threshold)
    by_interval = {}
    for e in edges_with_both:
        if e['prob_bucket'] != 'all' or e['threshold'] != '0.1':
            continue
        interval = e['interval']
        by_interval[interval] = e
    
    print(f"{'Interval':<15} {'Uncond':>10} {'After Fill':>12} {'Fill Benefit':>14} {'Interpretation':>20}")
    print("-" * 80)
    
    for interval, e in sorted(by_interval.items()):
        uncond = e['uncond_edge']
        cond = e['edge_after_fill']
        benefit = cond - uncond
        
        if benefit > 50:
            interp = "Fill HELPS"
        elif benefit < -50:
            interp = "Fill HURTS"
        else:
            interp = "~neutral"
        
        print(f"{interval:<15} {uncond:>+10.0f} {cond:>+12.0f} {benefit:>+14.0f} {interp:>20}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze edge existence and confidence')
    parser.add_argument('json_file', help='Path to phase4d JSON file')
    parser.add_argument('--tercile', choices=['early', 'mid', 'late'], default='early',
                        help='Which tercile to analyze (default: early)')
    
    args = parser.parse_args()
    
    data = load_json(args.json_file)
    
    # Collect all edges
    all_edges = collect_all_edges(data, args.tercile)
    print(f"\nLoaded {len(all_edges)} edge estimates from JSON\n")
    
    # Significance analysis
    sig_pos, sig_neg, insig = analyze_edge_significance(all_edges)
    
    # Weighted average
    compute_weighted_average_edge(all_edges)
    
    # By condition
    analyze_by_condition(all_edges)
    
    # Unconditional vs conditional
    analyze_unconditional_vs_conditional(all_edges)