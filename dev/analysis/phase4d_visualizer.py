#!/usr/bin/env python3
"""
Phase 4D Data Visualizer
Generates presentation-ready charts from the probability-stratified analysis.

Usage:
    python phase4d_visualizer.py --json phase4d_data.json --output ./charts/
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('dark_background')

# Color palette
COLORS = {
    'bg': '#0a0a0f',
    'card': '#12121a',
    'grid': '#2a2a3a',
    'text': '#e4e4e7',
    'muted': '#71717a',
    'accent': '#22d3ee',
    'success': '#4ade80',
    'warning': '#facc15',
    'danger': '#f87171',
    'purple': '#a78bfa'
}

BUCKET_COLORS = {
    'sub_51': '#64748b',
    '51_60': '#f59e0b',
    '60_75': '#10b981',
    '75_90': '#3b82f6',
    '90_99': '#22d3ee',
    '99_plus': '#a78bfa',
    'all': '#e4e4e7'
}

BUCKET_LABELS = {
    'sub_51': 'Longshots (<51%)',
    '51_60': 'Toss-up (51-60%)',
    '60_75': 'Moderate Fav (60-75%)',
    '75_90': 'Strong Fav (75-90%)',
    '90_99': 'Heavy Fav (90-99%)',
    '99_plus': 'Near-Certain (99%+)',
    'all': 'All Markets'
}


def load_json(filepath):
    """Load the Phase 4D JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_edge_by_bucket(data, interval='8h_to_4h', threshold=0.15, tercile='early'):
    """Extract edge data by probability bucket for a given configuration."""
    surface = data.get('surface', {})
    interval_data = surface.get(interval, {})
    
    # Handle both string and float threshold keys
    threshold_key = str(threshold) if str(threshold) in interval_data else threshold
    threshold_data = interval_data.get(threshold_key, {})
    
    results = []
    for bucket in ['sub_51', '51_60', '60_75', '75_90', '90_99']:
        bucket_data = threshold_data.get(bucket, {})
        tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
        
        if tercile_data.get('status') == 'ok':
            results.append({
                'bucket': bucket,
                'label': BUCKET_LABELS.get(bucket, bucket),
                'edge': tercile_data.get('edge_after_fill_bps', 0),
                'n': tercile_data.get('n_samples', 0),
                'se': tercile_data.get('se_edge_after_fill', 0),
                'fill_rate': tercile_data.get('fill_rate', 0),
                'win_rate': tercile_data.get('conditional_win_rate', 0)
            })
    
    return results


def extract_timing_effect(data, threshold=0.1):
    """Extract early vs late tercile edge comparison."""
    surface = data.get('surface', {})
    results = []
    
    for interval in ['48h_to_24h', '24h_to_12h', '12h_to_6h', '8h_to_4h', '6h_to_4h']:
        interval_data = surface.get(interval, {})
        
        # Handle both string and float threshold keys
        threshold_key = str(threshold) if str(threshold) in interval_data else threshold
        threshold_data = interval_data.get(threshold_key, {})
        
        # Try 'all' bucket first, then aggregate from individual buckets
        all_data = threshold_data.get('all', {})
        
        if not all_data:
            # Aggregate from individual probability buckets
            early_edges = []
            late_edges = []
            early_n = 0
            late_n = 0
            
            for bucket in ['sub_51', '51_60', '60_75', '75_90', '90_99']:
                bucket_data = threshold_data.get(bucket, {})
                early = bucket_data.get('terciles', {}).get('early', {})
                late = bucket_data.get('terciles', {}).get('late', {})
                
                if early.get('status') == 'ok':
                    n = early.get('n_samples', 0)
                    early_edges.append(early.get('edge_after_fill_bps', 0) * n)
                    early_n += n
                    
                if late.get('status') == 'ok':
                    n = late.get('n_samples', 0)
                    late_edges.append(late.get('edge_after_fill_bps', 0) * n)
                    late_n += n
            
            if early_n > 0 and late_n > 0:
                results.append({
                    'interval': interval.replace('_to_', '→').replace('h', 'h'),
                    'early_edge': sum(early_edges) / early_n,
                    'late_edge': sum(late_edges) / late_n,
                    'early_n': early_n,
                    'late_n': late_n
                })
        else:
            early = all_data.get('terciles', {}).get('early', {})
            late = all_data.get('terciles', {}).get('late', {})
            
            if early.get('status') == 'ok' and late.get('status') == 'ok':
                results.append({
                    'interval': interval.replace('_to_', '→').replace('h', 'h'),
                    'early_edge': early.get('edge_after_fill_bps', 0),
                    'late_edge': late.get('edge_after_fill_bps', 0),
                    'early_n': early.get('n_samples', 0),
                    'late_n': late.get('n_samples', 0)
                })
    
    return results


def extract_fill_sensitivity(data, interval='8h_to_4h', threshold=0.15, bucket='90_99'):
    """Extract fill sensitivity data."""
    surface = data.get('surface', {})
    fill_sens = data.get('fill_sensitivity', {})
    
    sens_data = fill_sens.get(interval, {}).get(threshold, {})
    if not sens_data:
        return []
    
    results = [{'fill_rate': '100%', 'edge': sens_data.get('actual_edge_bps', 0)}]
    
    scenarios = sens_data.get('scenarios', {})
    for rate_key in ['90pct', '80pct', '70pct', '60pct']:
        if rate_key in scenarios:
            pct = rate_key.replace('pct', '%')
            results.append({'fill_rate': pct, 'edge': scenarios[rate_key].get('edge_bps', 0)})
    
    return results


def extract_transition_boundary(data, threshold=0.1):
    """Extract edge by hours to resolution."""
    surface = data.get('surface', {})
    
    # Map intervals to end hours
    interval_map = {
        '48h_to_24h': 24,
        '24h_to_12h': 12,
        '12h_to_6h': 6,
        '9h_to_6h': 6,
        '8h_to_4h': 4,
        '6h_to_4h': 4,
        '6h_to_3h': 3,
        '5h_to_2h': 2,
        '4h_to_2h': 2,
        '3h_to_1h': 1
    }
    
    results = {}
    for interval, end_hour in interval_map.items():
        interval_data = surface.get(interval, {})
        if not interval_data:
            continue
            
        # Handle both string and float threshold keys
        threshold_key = str(threshold) if str(threshold) in interval_data else threshold
        threshold_data = interval_data.get(threshold_key, {})
        
        # Aggregate across buckets
        total_edge_weighted = 0
        total_n = 0
        total_fill_rate_weighted = 0
        
        for bucket in ['sub_51', '51_60', '60_75', '75_90', '90_99']:
            bucket_data = threshold_data.get(bucket, {})
            early = bucket_data.get('terciles', {}).get('early', {})
            
            if early.get('status') == 'ok':
                n = early.get('n_samples', 0)
                edge = early.get('edge_after_fill_bps', 0)
                fill_rate = early.get('fill_rate', 0)
                
                total_edge_weighted += edge * n
                total_fill_rate_weighted += fill_rate * n
                total_n += n
        
        if total_n > 0:
            avg_edge = total_edge_weighted / total_n
            avg_fill = total_fill_rate_weighted / total_n
            
            # Keep best data point per end hour
            if end_hour not in results or total_n > results[end_hour]['n']:
                results[end_hour] = {
                    'end_hour': end_hour,
                    'edge': avg_edge,
                    'fill_rate': avg_fill,
                    'interval': interval,
                    'n': total_n
                }
    
    return sorted(results.values(), key=lambda x: -x['end_hour'])


def extract_edge_evolution_by_window(data, threshold=0.15, tercile='early'):
    """
    Extract edge evolution across windows for each probability bucket.
    
    For a fixed tercile and threshold, returns how edge evolves from distant
    windows (48h_to_24h) to close-to-resolution windows (6h_to_4h), stratified
    by probability bucket.
    
    This captures the dominant temporal effect: fading favorites far from 
    resolution is catastrophic, but becomes favorable close to resolution.
    """
    surface = data.get('surface', {})
    
    # Order windows from distant to close-to-resolution
    # The key insight: edge structure changes dramatically with window
    ordered_windows = [
        '48h_to_24h',
        '24h_to_12h', 
        '12h_to_6h',
        '9h_to_6h',
        '8h_to_4h',
        '6h_to_4h',
        '6h_to_3h',
        '5h_to_2h',
        '4h_to_2h',
        '3h_to_1h'
    ]
    
    buckets = ['sub_51', '51_60', '60_75', '75_90', '90_99']
    
    results = {bucket: [] for bucket in buckets}
    results['all'] = []  # Aggregate across all buckets
    
    for window in ordered_windows:
        interval_data = surface.get(window, {})
        if not interval_data:
            continue
            
        # Handle both string and float threshold keys
        threshold_key = str(threshold) if str(threshold) in interval_data else threshold
        threshold_data = interval_data.get(threshold_key, {})
        
        if not threshold_data:
            continue
        
        # Extract per-bucket data
        for bucket in buckets:
            bucket_data = threshold_data.get(bucket, {})
            tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
            
            if tercile_data.get('status') == 'ok':
                results[bucket].append({
                    'window': window,
                    'window_display': window.replace('_to_', '→').replace('h', 'h'),
                    'edge': tercile_data.get('edge_after_fill_bps', 0),
                    'se': tercile_data.get('se_edge_after_fill', 0),
                    'n': tercile_data.get('n_samples', 0),
                    'fill_rate': tercile_data.get('fill_rate', 0),
                    'win_rate': tercile_data.get('conditional_win_rate', 0)
                })
        
        # Compute weighted aggregate across buckets for this window
        total_edge_weighted = 0
        total_se_sq_weighted = 0  # For pooled SE
        total_n = 0
        
        for bucket in buckets:
            bucket_data = threshold_data.get(bucket, {})
            tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
            
            if tercile_data.get('status') == 'ok':
                n = tercile_data.get('n_samples', 0)
                edge = tercile_data.get('edge_after_fill_bps', 0)
                se = tercile_data.get('se_edge_after_fill', 0)
                
                total_edge_weighted += edge * n
                total_se_sq_weighted += (se ** 2) * (n ** 2)  # Variance weighting
                total_n += n
        
        if total_n > 0:
            avg_edge = total_edge_weighted / total_n
            # Pooled SE approximation
            pooled_se = np.sqrt(total_se_sq_weighted) / total_n if total_n > 0 else 0
            
            results['all'].append({
                'window': window,
                'window_display': window.replace('_to_', '→').replace('h', 'h'),
                'edge': avg_edge,
                'se': pooled_se,
                'n': total_n,
                'fill_rate': None,
                'win_rate': None
            })
    
    return results


def plot_edge_evolution(data, output_path, threshold=0.15, tercile='early', show_all_aggregate=False):
    """
    Plot edge evolution across windows for each probability bucket.
    
    This is the key plot showing the dominant temporal effect: how edge changes
    from distant windows (48h_to_24h) to close-to-resolution windows.
    
    The hypothesis: small dips on long windows far from resolution (e.g., 5% dips 
    in 48h→24h, late tercile) should show no edge - just noise. But the same 
    dips close to resolution may show strong edge.
    """
    evolution = extract_edge_evolution_by_window(data, threshold, tercile)
    
    if not any(evolution[b] for b in ['sub_51', '51_60', '60_75', '75_90', '90_99']):
        print(f"No evolution data found for threshold={threshold}, tercile={tercile}")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    # Get all unique windows that appear in any bucket (in order)
    all_windows = []
    window_set = set()
    for bucket in ['sub_51', '51_60', '60_75', '75_90', '90_99']:
        for point in evolution[bucket]:
            if point['window'] not in window_set:
                all_windows.append(point['window'])
                window_set.add(point['window'])
    
    # Create x positions for windows
    x_positions = {w: i for i, w in enumerate(all_windows)}
    
    # Plot each bucket as a line with error bars
    buckets_to_plot = ['sub_51', '51_60', '60_75', '75_90', '90_99']
    if show_all_aggregate:
        buckets_to_plot.append('all')
    
    # Offset for visual clarity when points overlap
    offsets = np.linspace(-0.15, 0.15, len(buckets_to_plot))
    
    for idx, bucket in enumerate(buckets_to_plot):
        if not evolution[bucket]:
            continue
            
        x_vals = [x_positions[p['window']] + offsets[idx] for p in evolution[bucket]]
        y_vals = [p['edge'] for p in evolution[bucket]]
        errors = [p['se'] * 1.96 for p in evolution[bucket]]  # 95% CI
        
        color = BUCKET_COLORS.get(bucket, COLORS['muted'])
        label = BUCKET_LABELS.get(bucket, bucket)
        
        # Plot line with markers
        line = ax.plot(x_vals, y_vals, 'o-', color=color, linewidth=2, 
                       markersize=8, label=label, alpha=0.9)
        
        # Add error bars
        ax.errorbar(x_vals, y_vals, yerr=errors, fmt='none', color=color, 
                    capsize=4, capthick=1.5, linewidth=1.5, alpha=0.7)
    
    # Zero line
    ax.axhline(y=0, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7)
    
    # X-axis formatting
    ax.set_xticks(range(len(all_windows)))
    window_labels = [w.replace('_to_', '→\n').replace('h', 'h') for w in all_windows]
    ax.set_xticklabels(window_labels, fontsize=10, color=COLORS['text'])
    
    ax.set_xlabel('Observation Window', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('Edge After Fill (bps)', fontsize=12, color=COLORS['text'])
    ax.set_title(f'Edge Evolution Across Windows by Probability Bucket\n'
                 f'{int(threshold*100)}% dip threshold | {tercile.capitalize()} tercile reaction',
                 fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)
    
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['muted'])
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    ax.xaxis.grid(True, linestyle='--', alpha=0.15, color=COLORS['grid'])
    
    # Add temporal direction annotation (after limits are set)
    ymin, ymax = ax.get_ylim()
    annotation_y = ymin - (ymax - ymin) * 0.12
    ax.annotate('', xy=(len(all_windows) - 0.5, annotation_y),
                xytext=(-0.5, annotation_y),
                arrowprops=dict(arrowstyle='->', color=COLORS['muted'], lw=1.5),
                annotation_clip=False)
    ax.text(len(all_windows) / 2, annotation_y - (ymax - ymin) * 0.03, 
            '← Distant from resolution                    Close to resolution →',
            ha='center', va='top', fontsize=9, color=COLORS['muted'], style='italic',
            clip_on=False)
    
    plt.tight_layout()
    plt.savefig(output_path / f'edge_evolution_t{int(threshold*100)}_{tercile}.png', 
                dpi=150, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / f'edge_evolution_t{int(threshold*100)}_{tercile}.png'}")


def plot_edge_evolution_grid(data, output_path):
    """
    Generate a grid of edge evolution plots across all tercile/threshold combinations.
    
    This reveals the full temporal structure: where edge is real vs noise.
    Hypothesis to test: small dips (5%) on distant windows (48h→24h) in late tercile
    should show no edge - just noise.
    """
    thresholds = [0.05, 0.1, 0.15, 0.2]
    terciles = ['early', 'mid', 'late']
    
    fig, axes = plt.subplots(len(terciles), len(thresholds), 
                             figsize=(20, 12), facecolor=COLORS['bg'])
    
    # Get consistent window ordering from data
    surface = data.get('surface', {})
    ordered_windows = ['48h_to_24h', '24h_to_12h', '12h_to_6h', '9h_to_6h', 
                       '8h_to_4h', '6h_to_4h', '6h_to_3h', '5h_to_2h', 
                       '4h_to_2h', '3h_to_1h']
    
    for row, tercile in enumerate(terciles):
        for col, threshold in enumerate(thresholds):
            ax = axes[row, col]
            ax.set_facecolor(COLORS['bg'])
            
            evolution = extract_edge_evolution_by_window(data, threshold, tercile)
            
            # Find windows that have data
            available_windows = []
            for w in ordered_windows:
                for bucket in ['75_90', '90_99']:  # Focus on favorites
                    if any(p['window'] == w for p in evolution.get(bucket, [])):
                        if w not in available_windows:
                            available_windows.append(w)
            
            if not available_windows:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       color=COLORS['muted'], fontsize=10)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                x_positions = {w: i for i, w in enumerate(available_windows)}
                
                # Plot key buckets only for clarity
                for bucket in ['75_90', '90_99', '60_75']:
                    if not evolution.get(bucket):
                        continue
                    
                    points = [p for p in evolution[bucket] if p['window'] in available_windows]
                    if not points:
                        continue
                    
                    x_vals = [x_positions[p['window']] for p in points]
                    y_vals = [p['edge'] for p in points]
                    errors = [p['se'] * 1.96 for p in points]
                    
                    color = BUCKET_COLORS.get(bucket, COLORS['muted'])
                    ax.plot(x_vals, y_vals, 'o-', color=color, linewidth=1.5, 
                           markersize=5, alpha=0.9)
                    ax.errorbar(x_vals, y_vals, yerr=errors, fmt='none', color=color, 
                               capsize=2, capthick=1, linewidth=1, alpha=0.6)
                
                ax.axhline(y=0, color=COLORS['danger'], linestyle='--', linewidth=1, alpha=0.5)
                
                # Minimal x-axis labels
                ax.set_xticks(range(len(available_windows)))
                labels = [w.split('_to_')[1].replace('h', '') for w in available_windows]
                ax.set_xticklabels(labels, fontsize=7, color=COLORS['muted'])
            
            # Title for each subplot
            ax.set_title(f'{int(threshold*100)}% | {tercile}', fontsize=9, 
                        color=COLORS['text'], fontweight='bold')
            
            # Only label outer axes
            if row == len(terciles) - 1:
                ax.set_xlabel('End hour', fontsize=8, color=COLORS['muted'])
            if col == 0:
                ax.set_ylabel('Edge (bps)', fontsize=8, color=COLORS['muted'])
            
            # Styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(COLORS['grid'])
            ax.spines['bottom'].set_color(COLORS['grid'])
            ax.tick_params(colors=COLORS['muted'], labelsize=7)
            ax.yaxis.grid(True, linestyle='--', alpha=0.2, color=COLORS['grid'])
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color=BUCKET_COLORS['60_75'], marker='o', label='60-75%'),
        plt.Line2D([0], [0], color=BUCKET_COLORS['75_90'], marker='o', label='75-90%'),
        plt.Line2D([0], [0], color=BUCKET_COLORS['90_99'], marker='o', label='90-99%'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=9, 
               framealpha=0.9, bbox_to_anchor=(0.99, 0.99))
    
    fig.suptitle('Edge Evolution: Temporal Structure Across Parameter Space\n'
                 'X-axis: Hours to resolution (descending) | Key insight: edge structure is window-dependent',
                 fontsize=14, fontweight='bold', color=COLORS['text'], y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path / 'edge_evolution_grid.png', dpi=150, facecolor=COLORS['bg'], 
                edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'edge_evolution_grid.png'}")


def plot_edge_by_bucket(data, output_path, interval='8h_to_4h', threshold=0.15, tercile='early'):
    """Plot edge by probability bucket - the main finding chart."""
    edges = extract_edge_by_bucket(data, interval, threshold, tercile)
    
    if not edges:
        print(f"No data found for {interval} at {threshold} threshold")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    buckets = [e['bucket'] for e in edges]
    values = [e['edge'] for e in edges]
    errors = [e['se'] * 1.96 for e in edges]  # 95% CI
    colors = [BUCKET_COLORS.get(b, COLORS['accent']) for b in buckets]
    labels = [e['label'].replace(' ', '\n') for e in edges]
    
    x = np.arange(len(buckets))
    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor='none', alpha=0.9)
    ax.errorbar(x, values, yerr=errors, fmt='none', color=COLORS['text'], capsize=5, capthick=2, linewidth=2)
    
    # Highlight the peak
    max_idx = np.argmax(values)
    bars[max_idx].set_edgecolor(COLORS['accent'])
    bars[max_idx].set_linewidth(3)
    
    ax.axhline(y=0, color=COLORS['muted'], linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color=COLORS['text'])
    ax.set_ylabel('Edge After Fill (bps)', fontsize=12, color=COLORS['text'])
    interval_display = interval.replace('_to_', '→').replace('h', 'h')
    ax.set_title(f'Edge by Probability Bucket\n{interval_display} | {int(threshold*100)}% threshold | {tercile.capitalize()} tercile', 
                 fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)
    
    # Add sample sizes
    for i, (bar, edge_data) in enumerate(zip(bars, edges)):
        height = bar.get_height()
        ax.annotate(f'n={edge_data["n"]}', 
                    xy=(bar.get_x() + bar.get_width()/2, height + errors[i] + 30),
                    ha='center', va='bottom', fontsize=9, color=COLORS['muted'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['muted'])
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig(output_path / 'edge_by_bucket.png', dpi=150, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'edge_by_bucket.png'}")


def plot_timing_effect(data, output_path, threshold=0.1, interval=None):
    """Plot early vs late tercile comparison."""
    timing = extract_timing_effect(data, threshold)
    
    if not timing:
        print("No timing effect data found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    intervals = [t['interval'] for t in timing]
    early = [t['early_edge'] for t in timing]
    late = [t['late_edge'] for t in timing]
    
    x = np.arange(len(intervals))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, early, width, label='Early Tercile', color=COLORS['success'], alpha=0.9)
    bars2 = ax.bar(x + width/2, late, width, label='Late Tercile', color=COLORS['muted'], alpha=0.7)
    
    # Highlight the selected interval if specified
    if interval:
        interval_display = interval.replace('_to_', '→').replace('h', 'h')
        for i, intv in enumerate(intervals):
            if intv == interval_display:
                bars1[i].set_edgecolor(COLORS['accent'])
                bars1[i].set_linewidth(2)
                bars2[i].set_edgecolor(COLORS['accent'])
                bars2[i].set_linewidth(2)
    
    ax.axhline(y=0, color=COLORS['muted'], linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(intervals, fontsize=11, color=COLORS['text'])
    ax.set_ylabel('Edge After Fill (bps)', fontsize=12, color=COLORS['text'])
    ax.set_title(f'Timing Effect: Early vs Late Reaction\n{int(threshold*100)}% threshold', 
                 fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)
    ax.legend(loc='upper right', framealpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['muted'])
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig(output_path / 'timing_effect.png', dpi=150, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'timing_effect.png'}")


def plot_transition_boundary(data, output_path, threshold=0.1):
    """Plot edge vs hours to resolution."""
    boundary = extract_transition_boundary(data, threshold)
    
    if not boundary:
        print("No transition boundary data found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    hours = [b['end_hour'] for b in boundary]
    edges = [b['edge'] for b in boundary]
    
    # Color by status
    colors = []
    for edge, hour in zip(edges, hours):
        if hour >= 4 and edge > 0:
            colors.append(COLORS['success'])
        elif hour <= 1 or edge < -50:
            colors.append(COLORS['danger'])
        else:
            colors.append(COLORS['warning'])
    
    bars = ax.bar(range(len(hours)), edges, color=colors, width=0.6, alpha=0.9)
    
    ax.axhline(y=0, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7)
    
    # Add vertical line at 4h boundary
    if 4 in hours:
        idx = hours.index(4)
        ax.axvline(x=idx + 0.5, color=COLORS['accent'], linestyle=':', linewidth=2, alpha=0.7)
        ax.annotate('Safe Boundary', xy=(idx + 0.6, max(edges) * 0.8), 
                    fontsize=10, color=COLORS['accent'], rotation=90, va='top')
    
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f'{h}h' for h in hours], fontsize=11, color=COLORS['text'])
    ax.set_xlabel('Hours to Resolution', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('Edge After Fill (bps)', fontsize=12, color=COLORS['text'])
    ax.set_title(f'Resolution Proximity Boundary\n{int(threshold*100)}% threshold | Early tercile', 
                 fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['success'], label='Safe (≥4h)'),
        mpatches.Patch(facecolor=COLORS['warning'], label='Marginal'),
        mpatches.Patch(facecolor=COLORS['danger'], label='Danger (≤1h)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['muted'])
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    plt.tight_layout()
    plt.savefig(output_path / 'transition_boundary.png', dpi=150, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'transition_boundary.png'}")


def plot_edge_heatmap(data, output_path, tercile='early', interval='8h_to_4h'):
    """Plot edge surface heatmap (threshold × probability bucket)."""
    surface = data.get('surface', {})
    
    # Use specified interval
    interval_data = surface.get(interval, {})
    
    thresholds = ['0.05', '0.1', '0.15', '0.2']
    buckets = ['sub_51', '51_60', '60_75', '75_90', '90_99']
    
    # Build heatmap matrix
    matrix = np.zeros((len(thresholds), len(buckets)))
    
    for i, thresh in enumerate(thresholds):
        thresh_data = interval_data.get(thresh, {})
        for j, bucket in enumerate(buckets):
            bucket_data = thresh_data.get(bucket, {})
            tercile_data = bucket_data.get('terciles', {}).get(tercile, {})
            if tercile_data.get('status') == 'ok':
                matrix[i, j] = tercile_data.get('edge_after_fill_bps', 0)
            else:
                matrix[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    
    # Create heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-200, vmax=1500)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Edge (bps)', rotation=-90, va='bottom', color=COLORS['text'])
    cbar.ax.tick_params(colors=COLORS['muted'])
    
    # Set labels
    ax.set_xticks(np.arange(len(buckets)))
    ax.set_yticks(np.arange(len(thresholds)))
    ax.set_xticklabels([BUCKET_LABELS[b].split('(')[0].strip() for b in buckets], fontsize=10, color=COLORS['text'])
    ax.set_yticklabels([f'{float(t)*100:.0f}%' for t in thresholds], fontsize=10, color=COLORS['text'])
    ax.set_xlabel('Probability Bucket', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('Dip Threshold', fontsize=12, color=COLORS['text'])
    
    # Add text annotations
    for i in range(len(thresholds)):
        for j in range(len(buckets)):
            if not np.isnan(matrix[i, j]):
                text_color = 'black' if matrix[i, j] > 500 else 'white'
                ax.text(j, i, f'{matrix[i, j]:.0f}', ha='center', va='center', 
                       color=text_color, fontsize=10, fontweight='bold')
    
    interval_display = interval.replace('_to_', '→').replace('h', 'h')
    ax.set_title(f'Edge Surface: Threshold × Probability Bucket\n{interval_display} | {tercile.capitalize()} tercile', 
                 fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path / 'edge_heatmap.png', dpi=150, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'edge_heatmap.png'}")


def plot_summary_dashboard(data, output_path, interval='8h_to_4h', threshold=0.15, tercile='early'):
    """Generate a combined summary dashboard."""
    fig = plt.figure(figsize=(16, 12), facecolor=COLORS['bg'])
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    interval_display = interval.replace('_to_', '→').replace('h', 'h')
    
    # Top left: Edge by bucket
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['bg'])
    
    edges = extract_edge_by_bucket(data, interval, threshold, tercile)
    if edges:
        buckets = [e['bucket'] for e in edges]
        values = [e['edge'] for e in edges]
        colors = [BUCKET_COLORS.get(b, COLORS['accent']) for b in buckets]
        
        bars = ax1.bar(range(len(buckets)), values, color=colors, width=0.6, alpha=0.9)
        ax1.set_xticks(range(len(buckets)))
        ax1.set_xticklabels(buckets, fontsize=9, color=COLORS['text'])
        ax1.set_ylabel('Edge (bps)', fontsize=10, color=COLORS['text'])
        ax1.set_title(f'Edge by Probability Bucket ({tercile.capitalize()})', fontsize=12, fontweight='bold', color=COLORS['text'])
        ax1.axhline(y=0, color=COLORS['muted'], linestyle='--', linewidth=1, alpha=0.5)
    
    # Top right: Timing effect
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLORS['bg'])
    
    timing = extract_timing_effect(data, threshold)
    if timing:
        intervals = [t['interval'].replace('→', '→\n') for t in timing]
        early = [t['early_edge'] for t in timing]
        late = [t['late_edge'] for t in timing]
        
        x = np.arange(len(intervals))
        width = 0.35
        bars1 = ax2.bar(x - width/2, early, width, label='Early', color=COLORS['success'], alpha=0.9)
        bars2 = ax2.bar(x + width/2, late, width, label='Late', color=COLORS['muted'], alpha=0.7)
        
        # Highlight selected interval
        for i, t in enumerate(timing):
            if t['interval'] == interval_display:
                bars1[i].set_edgecolor(COLORS['accent'])
                bars1[i].set_linewidth(2)
                bars2[i].set_edgecolor(COLORS['accent'])
                bars2[i].set_linewidth(2)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(intervals, fontsize=8, color=COLORS['text'])
        ax2.set_ylabel('Edge (bps)', fontsize=10, color=COLORS['text'])
        ax2.set_title(f'Timing Effect ({int(threshold*100)}% threshold)', fontsize=12, fontweight='bold', color=COLORS['text'])
        ax2.legend(loc='upper right', fontsize=8)
        ax2.axhline(y=0, color=COLORS['muted'], linestyle='--', linewidth=1, alpha=0.5)
    
    # Bottom left: Transition boundary
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(COLORS['bg'])
    
    boundary = extract_transition_boundary(data, threshold)
    if boundary:
        hours = [b['end_hour'] for b in boundary]
        edge_vals = [b['edge'] for b in boundary]
        colors = [COLORS['success'] if h >= 4 and e > 0 else COLORS['danger'] if e < 0 else COLORS['warning'] 
                  for h, e in zip(hours, edge_vals)]
        
        ax3.bar(range(len(hours)), edge_vals, color=colors, width=0.6, alpha=0.9)
        ax3.set_xticks(range(len(hours)))
        ax3.set_xticklabels([f'{h}h' for h in hours], fontsize=9, color=COLORS['text'])
        ax3.set_xlabel('Hours to Resolution', fontsize=10, color=COLORS['text'])
        ax3.set_ylabel('Edge (bps)', fontsize=10, color=COLORS['text'])
        ax3.set_title('Resolution Boundary', fontsize=12, fontweight='bold', color=COLORS['text'])
        ax3.axhline(y=0, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Bottom right: Key stats
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(COLORS['bg'])
    ax4.axis('off')
    
    # Calculate peak edge from current settings
    peak_edge = max([e['edge'] for e in edges]) if edges else 0
    peak_bucket = max(edges, key=lambda x: x['edge'])['bucket'] if edges else 'N/A'
    
    stats_text = f"""
    KEY FINDINGS ({interval_display})
    ════════════════════════════════════
    
    Peak Edge:           +{peak_edge:,.0f} bps
    Best Bucket:         {peak_bucket}
    Threshold:           {int(threshold*100)}%
    Tercile:             {tercile.capitalize()}
    Safe Boundary:       ≥4 hours to resolution
    
    STRATEGY PARAMETERS
    ════════════════════════════════════
    
    Target:              90-99% probability markets
    Entry Signal:        ≥{int(threshold*100)}% dip from pre-dip price
    Timing:              React in {tercile} tercile
    Resolution Rule:     Maintain ≥4h to resolution
    Exit:                Hold to resolution
    
    CAPACITY (estimated)
    ════════════════════════════════════
    
    See capacity table for details
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace', color=COLORS['text'])
    
    # Style all axes
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLORS['grid'])
        ax.spines['bottom'].set_color(COLORS['grid'])
        ax.tick_params(colors=COLORS['muted'])
        ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
    
    fig.suptitle(f'Phase 4D: Probability-Stratified Edge Analysis\n{interval_display} | {int(threshold*100)}% threshold | {tercile.capitalize()} tercile', 
                 fontsize=16, fontweight='bold', color=COLORS['text'], y=0.98)
    
    plt.savefig(output_path / 'summary_dashboard.png', dpi=150, facecolor=COLORS['bg'], 
                edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'summary_dashboard.png'}")


def main():
    parser = argparse.ArgumentParser(description='Phase 4D Data Visualizer')
    parser.add_argument('--json', '-j', type=str, default='phase4d_data.json',
                        help='Path to Phase 4D JSON file')
    parser.add_argument('--output', '-o', type=str, default='./charts',
                        help='Output directory for charts')
    parser.add_argument('--interval', '-i', type=str, default='8h_to_4h',
                        choices=['48h_to_24h', '24h_to_12h', '12h_to_6h', '9h_to_6h', '8h_to_4h', '6h_to_4h'],
                        help='Interval to analyze (default: 8h_to_4h)')
    parser.add_argument('--threshold', '-t', type=float, default=0.15,
                        choices=[0.05, 0.1, 0.15, 0.2],
                        help='Dip threshold (default: 0.15)')
    parser.add_argument('--tercile', type=str, default='early',
                        choices=['early', 'mid', 'late'],
                        help='Tercile to focus on (default: early)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.json}...")
    data = load_json(args.json)
    print(f"Loaded Phase 4D data: {data.get('n_tokens', 0):,} tokens analyzed")
    
    # Generate all charts
    print("\nGenerating charts...")
    print(f"  Interval: {args.interval}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Tercile: {args.tercile}")
    
    # Original charts
    plot_edge_by_bucket(data, output_path, args.interval, args.threshold, args.tercile)
    plot_timing_effect(data, output_path, args.threshold, args.interval)
    plot_transition_boundary(data, output_path, args.threshold)
    plot_edge_heatmap(data, output_path, args.tercile, args.interval)
    plot_summary_dashboard(data, output_path, args.interval, args.threshold, args.tercile)
    
    # NEW: Edge evolution plots - the dominant temporal effect
    print("\nGenerating edge evolution plots (temporal structure)...")
    plot_edge_evolution(data, output_path, args.threshold, args.tercile)
    
    # Generate evolution plots for all terciles at current threshold
    for t in ['early', 'mid', 'late']:
        if t != args.tercile:
            plot_edge_evolution(data, output_path, args.threshold, t)
    
    # Full grid showing all parameter combinations
    plot_edge_evolution_grid(data, output_path)
    
    # Print temporal structure summary
    print("\n" + "="*70)
    print("TEMPORAL STRUCTURE SUMMARY")
    print("="*70)
    evolution = extract_edge_evolution_by_window(data, args.threshold, args.tercile)
    
    for bucket in ['75_90', '90_99']:
        if evolution.get(bucket):
            edges = evolution[bucket]
            if len(edges) >= 2:
                distant = [e for e in edges if '48h' in e['window'] or '24h' in e['window'].split('_to_')[0]]
                close = [e for e in edges if '4h' in e['window'] or '3h' in e['window']]
                
                if distant and close:
                    distant_avg = np.mean([e['edge'] for e in distant])
                    close_avg = np.mean([e['edge'] for e in close])
                    print(f"\n{BUCKET_LABELS.get(bucket, bucket)}:")
                    print(f"  Distant windows (48h-24h): {distant_avg:+.0f} bps avg")
                    print(f"  Close windows (≤4h):       {close_avg:+.0f} bps avg")
                    print(f"  Δ (close - distant):       {close_avg - distant_avg:+.0f} bps")
    
    print("\n" + "-"*70)
    print("Key insight: Edge structure is heavily window-dependent.")
    print("Fading favorites far from resolution may be catastrophic noise.")
    print("-"*70)
    
    print(f"\nAll charts saved to {output_path}/")
    print("Files generated:")
    for f in output_path.glob('*.png'):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()