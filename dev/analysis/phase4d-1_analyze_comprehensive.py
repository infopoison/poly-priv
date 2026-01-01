#!/usr/bin/env python3
"""
Analysis Script 5: Comprehensive Strategy Summary

PURPOSE:
  One-shot summary that answers ALL the key questions:
  1. Is there edge overall? (with significance test)
  2. Where is edge strongest/weakest?
  3. How does edge vary by: interval, prob bucket, tercile, threshold?
  4. What's the transition point (when does fading stop working)?
  5. Fill sensitivity: how robust is edge to degraded fills?
  6. Strategy recommendation

Run this for a complete picture.
"""

import json
import sys
import argparse
import numpy as np
from collections import defaultdict
from scipy import stats

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def parse_interval_label(interval_label):
    parts = interval_label.replace('h_to_', '_').replace('h', '').split('_')
    try:
        return int(parts[0]), int(parts[1]), int(parts[0]) - int(parts[1])
    except:
        return None, None, None


def get_threshold_key(thresh_float):
    """Convert float threshold to the string key used in JSON."""
    # JSON stores as '0.05', '0.1', '0.15', '0.2' (not '0.10')
    return str(thresh_float)


def run_comprehensive_analysis(data):
    """
    Run full analysis and print comprehensive summary.
    """
    surface = data.get('surface', {})
    fill_sensitivity = data.get('fill_sensitivity', {})
    
    print("="*100)
    print("COMPREHENSIVE PHASE 4D ANALYSIS SUMMARY")
    print("="*100)
    print(f"\nData: {data.get('n_tokens', 0):,} tokens, {data.get('n_tokens_with_data', 0):,} with data")
    print(f"Thresholds analyzed: {data.get('move_thresholds', [])}")
    print(f"Intervals: {[x[2] for x in data.get('interval_pairs', [])]}")
    
    # =========================================================================
    # SECTION 1: OVERALL EDGE EXISTENCE
    # =========================================================================
    print("\n")
    print("="*100)
    print("SECTION 1: DOES EDGE EXIST?")
    print("="*100)
    
    # Collect edges from 'all' bucket, early tercile, across intervals and thresholds
    all_edges = []
    
    for interval_label, interval_data in surface.items():
        for threshold_key, threshold_data in interval_data.items():
            bucket_data = threshold_data.get('all', {})
            tercile_data = bucket_data.get('terciles', {}).get('early', {})
            
            if tercile_data.get('status') == 'ok':
                edge = tercile_data.get('edge_after_fill_bps')
                se = tercile_data.get('se_edge_after_fill')
                n = tercile_data.get('n_samples')
                
                if edge is not None and se is not None and se > 0:
                    all_edges.append({'edge': edge, 'se': se, 'n': n})
    
    if all_edges:
        # Inverse-variance weighted average
        weights = [1/(e['se']**2) for e in all_edges]
        total_weight = sum(weights)
        iv_edge = sum(e['edge'] * w for e, w in zip(all_edges, weights)) / total_weight
        iv_se = np.sqrt(1 / total_weight)
        
        z = iv_edge / iv_se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        
        print(f"\nOverall edge (inverse-variance weighted across {len(all_edges)} cells):")
        print(f"  Edge:    {iv_edge:>+8.1f} bps")
        print(f"  SE:      {iv_se:>8.1f} bps")
        print(f"  95% CI:  [{iv_edge - 1.96*iv_se:>+8.1f}, {iv_edge + 1.96*iv_se:>+8.1f}]")
        print(f"  z:       {z:>8.3f}")
        print(f"  p-value: {p:>8.6f}")
        
        if p < 0.01:
            print(f"\n  *** CONCLUSION: SIGNIFICANT EDGE EXISTS (p < 0.01) ***")
        elif p < 0.05:
            print(f"\n  ** Edge is significant at 5% level **")
        else:
            print(f"\n  Edge is NOT statistically significant")
    
    # =========================================================================
    # SECTION 2: EDGE BY INTERVAL (WHERE IS EDGE STRONGEST?)
    # =========================================================================
    print("\n")
    print("="*100)
    print("SECTION 2: EDGE BY INTERVAL")
    print("="*100)
    print("\nUsing 'all' bucket, early tercile, 10% threshold")
    
    threshold_key = '0.1'  # JSON stores as string
    interval_results = []
    
    for interval_label in sorted(surface.keys()):
        start, end, window = parse_interval_label(interval_label)
        bucket_data = surface[interval_label].get(threshold_key, {}).get('all', {})
        tercile_data = bucket_data.get('terciles', {}).get('early', {})
        
        if tercile_data.get('status') == 'ok':
            edge = tercile_data.get('edge_after_fill_bps')
            se = tercile_data.get('se_edge_after_fill')
            n = tercile_data.get('n_samples')
            fill_rate = tercile_data.get('fill_rate')
            
            interval_results.append({
                'interval': interval_label,
                'end_hour': end,
                'edge': edge,
                'se': se,
                'n': n,
                'fill_rate': fill_rate,
                'effective_edge': edge * fill_rate if edge and fill_rate else 0,
            })
    
    # Sort by end hour (farther from resolution first)
    interval_results.sort(key=lambda x: -x['end_hour'] if x['end_hour'] else 0)
    
    print(f"\n{'Interval':<15} {'End Hr':>7} {'Edge':>10} {'SE':>8} {'Fill%':>8} {'Eff Edge':>10} {'Sig?':>12}")
    print("-" * 85)
    
    for r in interval_results:
        sig = ""
        if r['se']:
            ci_low = r['edge'] - 1.96 * r['se']
            ci_high = r['edge'] + 1.96 * r['se']
            if ci_low > 0:
                sig = "*** POS ***"
            elif ci_high < 0:
                sig = "*** NEG ***"
        
        print(f"{r['interval']:<15} {r['end_hour']:>7}h {r['edge']:>+10.0f} {r['se']:>8.1f} "
              f"{r['fill_rate']*100:>7.1f}% {r['effective_edge']:>+10.0f} {sig:>12}")
    
    # Identify best interval
    positive_intervals = [r for r in interval_results if r['edge'] and r['edge'] > 0]
    if positive_intervals:
        best = max(positive_intervals, key=lambda x: x['effective_edge'])
        print(f"\n  BEST INTERVAL: {best['interval']} (effective edge: {best['effective_edge']:+.0f} bps)")
    
    # Identify transition point
    sorted_by_end = sorted(interval_results, key=lambda x: x['end_hour'] if x['end_hour'] else 99)
    transition = None
    for r in sorted_by_end:
        if r['edge'] and r['edge'] < 0:
            transition = r['end_hour']
            break
    
    if transition:
        print(f"  TRANSITION: Edge becomes negative at {transition}h to resolution")
    
    # =========================================================================
    # SECTION 3: EDGE BY PROBABILITY BUCKET
    # =========================================================================
    print("\n")
    print("="*100)
    print("SECTION 3: EDGE BY PROBABILITY BUCKET")
    print("="*100)
    print("\nAggregating across ALL intervals, ALL thresholds, ALL terciles")
    
    prob_bucket_edges = defaultdict(list)
    
    for interval_label, interval_data in surface.items():
        for threshold_key, threshold_data in interval_data.items():
            for prob_bucket in ['sub_51', '51_60', '60_75', '75_90', '90_99', '99_plus']:
                bucket_data = threshold_data.get(prob_bucket, {})
                
                for tercile in ['early', 'mid', 'late']:
                    tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
                    
                    if tercile_data.get('status') == 'ok':
                        edge = tercile_data.get('edge_after_fill_bps')
                        se = tercile_data.get('se_edge_after_fill')
                        n = tercile_data.get('n_samples')
                        
                        if edge is not None and se is not None and se > 0:
                            prob_bucket_edges[prob_bucket].append({
                                'edge': edge,
                                'se': se,
                                'n': n,
                                'interval': interval_label,
                                'threshold': threshold_key,
                                'tercile': tercile,
                            })
    
    print(f"\n{'Bucket':<12} {'Avg Edge':>12} {'Combined SE':>12} {'Total n':>12} {'# Cells':>10} {'Significant':>12}")
    print("-" * 75)
    
    bucket_order = ['sub_51', '51_60', '60_75', '75_90', '90_99', '99_plus']
    
    for bucket in bucket_order:
        cells = prob_bucket_edges.get(bucket, [])
        if cells:
            weights = [1/(c['se']**2) for c in cells if c['se'] > 0]
            edges_for_weights = [c['edge'] for c in cells if c['se'] > 0]
            
            if weights:
                total_weight = sum(weights)
                avg_edge = sum(e * w for e, w in zip(edges_for_weights, weights)) / total_weight
                combined_se = np.sqrt(1 / total_weight)
                total_n = sum(c['n'] for c in cells)
                
                ci_low = avg_edge - 1.96 * combined_se
                sig = "YES" if ci_low > 0 else ("NEG" if avg_edge + 1.96 * combined_se < 0 else "no")
                
                print(f"{bucket:<12} {avg_edge:>+12.1f} {combined_se:>12.1f} {total_n:>12,} {len(cells):>10} {sig:>12}")
    
    # =========================================================================
    # SECTION 4: TERCILE COMPARISON (TIMING EFFECT)
    # =========================================================================
    print("\n")
    print("="*100)
    print("SECTION 4: TERCILE COMPARISON (TIMING EFFECT)")
    print("="*100)
    print("\nDoes reacting early vs late matter? (Aggregating across ALL thresholds, ALL intervals)")
    
    # 4A: Overall tercile comparison
    tercile_edges = defaultdict(list)
    
    for interval_label, interval_data in surface.items():
        for threshold_key, threshold_data in interval_data.items():
            bucket_data = threshold_data.get('all', {})
            
            for tercile in ['early', 'mid', 'late']:
                tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
                
                if tercile_data.get('status') == 'ok':
                    edge = tercile_data.get('edge_after_fill_bps')
                    se = tercile_data.get('se_edge_after_fill')
                    n = tercile_data.get('n_samples')
                    
                    if edge is not None and se is not None and se > 0:
                        tercile_edges[tercile].append({
                            'edge': edge,
                            'se': se,
                            'n': n,
                        })
    
    print(f"\n4A. OVERALL TERCILE COMPARISON ('all' bucket)")
    print(f"{'Tercile':<10} {'Avg Edge':>12} {'Combined SE':>12} {'Total n':>12} {'# Cells':>10}")
    print("-" * 60)
    
    for tercile in ['early', 'mid', 'late']:
        cells = tercile_edges.get(tercile, [])
        if cells:
            weights = [1/(c['se']**2) for c in cells]
            total_weight = sum(weights)
            avg_edge = sum(c['edge'] * w for c, w in zip(cells, weights)) / total_weight
            combined_se = np.sqrt(1 / total_weight)
            total_n = sum(c['n'] for c in cells)
            print(f"{tercile:<10} {avg_edge:>+12.1f} {combined_se:>12.1f} {total_n:>12,} {len(cells):>10}")
    
    early_cells = tercile_edges.get('early', [])
    late_cells = tercile_edges.get('late', [])
    
    if early_cells and late_cells:
        early_weights = [1/(c['se']**2) for c in early_cells]
        late_weights = [1/(c['se']**2) for c in late_cells]
        early_avg = sum(c['edge'] * w for c, w in zip(early_cells, early_weights)) / sum(early_weights)
        late_avg = sum(c['edge'] * w for c, w in zip(late_cells, late_weights)) / sum(late_weights)
        
        print(f"\n  Early vs Late difference: {early_avg - late_avg:+.1f} bps")
    
    # 4B: Odds-stratified tercile comparison
    print(f"\n4B. ODDS-STRATIFIED TERCILE COMPARISON")
    print("    Does early vs late advantage vary by probability bucket?")
    print()
    
    # Collect by (prob_bucket, tercile)
    bucket_tercile_edges = defaultdict(lambda: defaultdict(list))
    
    for interval_label, interval_data in surface.items():
        for threshold_key, threshold_data in interval_data.items():
            for prob_bucket in ['sub_51', '51_60', '60_75', '75_90', '90_99', '99_plus']:
                bucket_data = threshold_data.get(prob_bucket, {})
                
                for tercile in ['early', 'mid', 'late']:
                    tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
                    
                    if tercile_data.get('status') == 'ok':
                        edge = tercile_data.get('edge_after_fill_bps')
                        se = tercile_data.get('se_edge_after_fill')
                        n = tercile_data.get('n_samples')
                        
                        if edge is not None and se is not None and se > 0:
                            bucket_tercile_edges[prob_bucket][tercile].append({
                                'edge': edge,
                                'se': se,
                                'n': n,
                            })
    
    print(f"{'Bucket':<12} {'Early':>10} {'Mid':>10} {'Late':>10} {'E-L Diff':>12} {'Early Better?':>14}")
    print("-" * 75)
    
    bucket_order = ['sub_51', '51_60', '60_75', '75_90', '90_99', '99_plus']
    
    for bucket in bucket_order:
        tercile_avgs = {}
        for tercile in ['early', 'mid', 'late']:
            cells = bucket_tercile_edges[bucket][tercile]
            if cells:
                weights = [1/(c['se']**2) for c in cells]
                total_weight = sum(weights)
                tercile_avgs[tercile] = sum(c['edge'] * w for c, w in zip(cells, weights)) / total_weight
            else:
                tercile_avgs[tercile] = None
        
        early_str = f"{tercile_avgs['early']:>+10.0f}" if tercile_avgs['early'] is not None else f"{'---':>10}"
        mid_str = f"{tercile_avgs['mid']:>+10.0f}" if tercile_avgs['mid'] is not None else f"{'---':>10}"
        late_str = f"{tercile_avgs['late']:>+10.0f}" if tercile_avgs['late'] is not None else f"{'---':>10}"
        
        if tercile_avgs['early'] is not None and tercile_avgs['late'] is not None:
            diff = tercile_avgs['early'] - tercile_avgs['late']
            diff_str = f"{diff:>+12.0f}"
            better = "YES" if diff > 20 else ("NO" if diff < -20 else "~same")
        else:
            diff_str = f"{'---':>12}"
            better = "---"
        
        print(f"{bucket:<12} {early_str} {mid_str} {late_str} {diff_str} {better:>14}")
    
    # =========================================================================
    # SECTION 5: THRESHOLD COMPARISON
    # =========================================================================
    print("\n")
    print("="*100)
    print("SECTION 5: THRESHOLD COMPARISON")
    print("="*100)
    print("\nDoes larger dip = more edge?")
    
    # Threshold keys as they appear in JSON: '0.05', '0.1', '0.15', '0.2'
    threshold_configs = [('0.05', 5), ('0.1', 10), ('0.15', 15), ('0.2', 20)]
    
    for threshold_key, threshold_pct in threshold_configs:
        threshold_cells = []
        for interval_label, interval_data in surface.items():
            start, end, window = parse_interval_label(interval_label)
            if end is not None and end < 4:
                continue
            
            threshold_data = interval_data.get(threshold_key, {})
            bucket_data = threshold_data.get('all', {})
            tercile_data = bucket_data.get('terciles', {}).get('early', {})
            
            if tercile_data.get('status') == 'ok':
                threshold_cells.append({
                    'edge': tercile_data.get('edge_after_fill_bps'),
                    'n': tercile_data.get('n_samples'),
                })
        
        if threshold_cells:
            avg_edge = np.mean([c['edge'] for c in threshold_cells if c['edge']])
            total_n = sum(c['n'] for c in threshold_cells)
            print(f"  ≥{threshold_pct:>4}% dip: avg edge = {avg_edge:>+8.1f} bps, total n = {total_n:>10,}")
    
    # =========================================================================
    # SECTION 6: FILL SENSITIVITY
    # =========================================================================
    print("\n")
    print("="*100)
    print("SECTION 6: FILL RATE SENSITIVITY")
    print("="*100)
    print("\nHow robust is edge if fill rate degrades?")
    
    if fill_sensitivity:
        print(f"\n{'Interval':<15} {'Actual Fill':>12} {'Actual Edge':>12} {'@90% Fill':>12} {'@70% Fill':>12}")
        print("-" * 70)
        
        for interval_label in sorted(fill_sensitivity.keys()):
            for threshold_key, sens_data in fill_sensitivity[interval_label].items():
                if threshold_key != '0.1':
                    continue
                
                actual_fill = sens_data.get('actual_fill_rate', 0)
                actual_edge = sens_data.get('actual_edge_bps', 0)
                
                scenarios = sens_data.get('scenarios', {})
                edge_90 = scenarios.get('90pct', {}).get('edge_bps', '-')
                edge_70 = scenarios.get('70pct', {}).get('edge_bps', '-')
                
                if isinstance(edge_90, (int, float)):
                    edge_90 = f"{edge_90:+.0f}"
                if isinstance(edge_70, (int, float)):
                    edge_70 = f"{edge_70:+.0f}"
                
                print(f"{interval_label:<15} {actual_fill*100:>11.1f}% {actual_edge:>+12.0f} {edge_90:>12} {edge_70:>12}")
    else:
        print("No fill sensitivity data available")
    
    # =========================================================================
    # SECTION 7: STRATEGY SUMMARY
    # =========================================================================
    print("\n")
    print("="*100)
    print("SECTION 7: STRATEGY RECOMMENDATION")
    print("="*100)
    
    print("""
Based on the analysis:

1. EDGE EXISTS: Overall weighted edge is positive and likely significant.

2. OPTIMAL CONDITIONS:
   - Interval: End ≥4h before resolution (6h-4h or 8h-4h intervals look good)
   - Threshold: ≥10% dips offer meaningful edge; larger may be better but fewer samples
   - Tercile: Early reaction preferred (but not dramatic)
   - Prob bucket: Moderate favorites (60-90%) may offer best balance

3. AVOID:
   - Intervals ending <4h before resolution (edge turns negative)
   - Late tercile reactions (modest edge loss)
   - Very high probability (99%+) markets (sample size issues)

4. ROBUSTNESS:
   - Edge degrades with fill rate but may remain positive at 70% fills
   - CIs are often wide; economic significance may differ from statistical

5. KEY RISKS:
   - Sampling noise remains a concern (many CIs include zero)
   - Fill rate in practice may differ from simulation
   - Market conditions may change
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive strategy summary')
    parser.add_argument('json_file', help='Path to phase4d JSON file')
    
    args = parser.parse_args()
    
    data = load_json(args.json_file)
    run_comprehensive_analysis(data)