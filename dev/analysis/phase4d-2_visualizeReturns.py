#!/usr/bin/env python3
"""
Phase 4D Returns Distribution Visualizer
Version: 1.0

PURPOSE:
    Generate comprehensive visualizations of returns distributions from 
    precomputed percentile CSV data. Designed for systematic exploration
    across all combinations of windows, terciles, and thresholds.

USAGE:
    # Generate all plots (one 4-panel figure per window)
    python phase4d_visualize_returns.py --csv percentiles.csv --output ./plots
    
    # Focus on specific window
    python phase4d_visualize_returns.py --csv percentiles.csv --window 24h_to_12h
    
    # Combine all terciles into single view
    python phase4d_visualize_returns.py --csv percentiles.csv --combine-terciles
    
    # Combine all thresholds into single view  
    python phase4d_visualize_returns.py --csv percentiles.csv --combine-thresholds
    
    # Generate heatmaps across all dimensions
    python phase4d_visualize_returns.py --csv percentiles.csv --heatmaps
    
    # Full analysis mode (all visualizations)
    python phase4d_visualize_returns.py --csv percentiles.csv --full

OUTPUT:
    - 4-panel diagnostic plots per window (boxplot by bucket, boxplot by interval, 
      histogram, cumulative distribution)
    - Heatmaps of median returns by cell
    - Summary tables as CSV
    - Text report with key findings

NOTES:
    - Returns are in basis points (bps), where 10000 bps = 100%
    - Positive returns = profitable trade, negative = loss
    - 'all' bucket aggregates across probability buckets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import argparse
import os
import sys
from datetime import datetime
import warnings
from io import StringIO

warnings.filterwarnings('ignore')


# ==============================================================================
# TEXT REPORTER - Accumulates and outputs text data
# ==============================================================================

class TextReporter:
    """
    Accumulates text output from all visualization functions.
    Prints to console and saves to file.
    """
    
    def __init__(self):
        self.sections = []
        self.current_section = None
        
    def start_section(self, title):
        """Start a new section."""
        self.current_section = {
            'title': title,
            'content': []
        }
        
    def add_line(self, line=""):
        """Add a line to current section."""
        if self.current_section is None:
            self.start_section("UNNAMED SECTION")
        self.current_section['content'].append(line)
        
    def add_lines(self, lines):
        """Add multiple lines."""
        for line in lines:
            self.add_line(line)
            
    def add_dataframe(self, df, title=None, float_format='.0f'):
        """Add a dataframe as formatted text."""
        if title:
            self.add_line(f"\n{title}")
            self.add_line("-" * len(title))
        
        # Format the dataframe
        formatted = df.to_string(index=True, float_format=lambda x: f'{x:{float_format}}' if pd.notna(x) else 'NaN')
        self.add_line(formatted)
        self.add_line("")
        
    def add_pivot_table(self, pivot, title=None):
        """Add a pivot table with nice formatting."""
        if title:
            self.add_line(f"\n{title}")
            self.add_line("-" * len(title))
        
        # Convert to string with formatting
        formatted = pivot.to_string(float_format=lambda x: f'{x:.0f}' if pd.notna(x) else '---')
        self.add_line(formatted)
        self.add_line("")
        
    def end_section(self):
        """End current section and add to list."""
        if self.current_section:
            self.sections.append(self.current_section)
            self.current_section = None
            
    def print_to_console(self):
        """Print all sections to console."""
        for section in self.sections:
            print("\n" + "=" * 80)
            print(f" {section['title']}")
            print("=" * 80)
            for line in section['content']:
                print(line)
                
    def save_to_file(self, filepath):
        """Save all sections to text file."""
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PHASE 4D RETURNS DISTRIBUTION - COMPLETE DATA DUMP\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for section in self.sections:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f" {section['title']}\n")
                f.write("=" * 80 + "\n")
                for line in section['content']:
                    f.write(line + "\n")
                    
        print(f"\nText data dump saved: {filepath}")
        return filepath


# Global reporter instance
REPORTER = TextReporter()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INTERVALS_ORDER = ['48h_to_24h', '24h_to_12h', '12h_to_6h', '9h_to_6h', '8h_to_4h', '6h_to_4h']
PROB_BUCKETS_ORDER = ['sub_51', '51_60', '60_75', '75_90', '90_99', '99_plus']
TERCILES_ORDER = ['early', 'mid', 'late']
THRESHOLDS_ORDER = [0.05, 0.10, 0.15, 0.20]

# Color schemes
BUCKET_COLORS = {
    'sub_51': '#1f77b4',
    '51_60': '#ff7f0e', 
    '60_75': '#2ca02c',
    '75_90': '#d62728',
    '90_99': '#9467bd',
    '99_plus': '#8c564b',
    'all': '#7f7f7f'
}

TERCILE_COLORS = {
    'early': '#2ecc71',
    'mid': '#f39c12',
    'late': '#e74c3c'
}

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


# ==============================================================================
# DATA LOADING AND VALIDATION
# ==============================================================================

def load_percentiles(csv_path):
    """Load and validate percentiles CSV."""
    print(f"Loading: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    required_cols = ['interval', 'threshold', 'tercile', 'prob_bucket', 
                     'n', 'p10', 'p25', 'p50', 'p75', 'p90', 'mean', 'std']
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"  Loaded {len(df)} rows")
    print(f"  Intervals: {df['interval'].unique().tolist()}")
    print(f"  Thresholds: {sorted(df['threshold'].unique().tolist())}")
    print(f"  Terciles: {df['tercile'].unique().tolist()}")
    print(f"  Prob buckets: {df['prob_bucket'].unique().tolist()}")
    
    return df


# ==============================================================================
# SYNTHETIC DATA GENERATION FOR BOX PLOTS
# ==============================================================================

def synthesize_distribution(row, n_samples=1000):
    """
    Synthesize approximate distribution from percentiles for visualization.
    
    Uses a mixture approach:
    - Generate points at each percentile
    - Interpolate between them
    - Add noise consistent with the reported std
    
    This is an approximation for visualization only - actual data would be better.
    """
    if row['n'] < 5:
        return np.array([])
    
    # Key percentiles
    percentiles = {
        0: row.get('min', row['p10'] - 2 * row['std']),
        10: row['p10'],
        25: row['p25'],
        50: row['p50'],
        75: row['p75'],
        90: row['p90'],
        100: row.get('max', row['p90'] + 2 * row['std'])
    }
    
    # Generate samples weighted toward each percentile region
    samples = []
    
    # 0-10th percentile region
    samples.extend(np.linspace(percentiles[0], percentiles[10], int(n_samples * 0.10)))
    # 10-25th
    samples.extend(np.linspace(percentiles[10], percentiles[25], int(n_samples * 0.15)))
    # 25-50th
    samples.extend(np.linspace(percentiles[25], percentiles[50], int(n_samples * 0.25)))
    # 50-75th
    samples.extend(np.linspace(percentiles[50], percentiles[75], int(n_samples * 0.25)))
    # 75-90th
    samples.extend(np.linspace(percentiles[75], percentiles[90], int(n_samples * 0.15)))
    # 90-100th
    samples.extend(np.linspace(percentiles[90], percentiles[100], int(n_samples * 0.10)))
    
    return np.array(samples)


# ==============================================================================
# FOUR-PANEL DIAGNOSTIC PLOT
# ==============================================================================

def create_four_panel_diagnostic(df, interval, threshold, tercile, output_dir):
    """
    Create the 4-panel diagnostic plot for a specific cell configuration.
    
    Panel 1: Box plot of returns by probability bucket (for this interval/threshold/tercile)
    Panel 2: Box plot of returns by interval (for the most interesting prob bucket)
    Panel 3: Histogram of returns for the focal cell
    Panel 4: Cumulative distribution with percentile markers
    """
    global REPORTER
    REPORTER.start_section(f"DIAGNOSTIC: {interval}, {int(threshold*100)}%, {tercile}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Filter data
    subset = df[(df['interval'] == interval) & 
                (df['threshold'] == threshold) & 
                (df['tercile'] == tercile)]
    
    if len(subset) == 0:
        plt.close()
        REPORTER.add_line("No data available for this combination.")
        REPORTER.end_section()
        return None
    
    # Add percentile table for this slice
    REPORTER.add_line("PERCENTILE DATA FOR THIS CONFIGURATION")
    REPORTER.add_line("-" * 100)
    REPORTER.add_line(f"{'Bucket':<12} {'n':>6} {'P10':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8} {'Mean':>8} {'Std':>8}")
    REPORTER.add_line("-" * 100)
    
    for _, row in subset.sort_values('prob_bucket').iterrows():
        REPORTER.add_line(f"{row['prob_bucket']:<12} {row['n']:>6.0f} {row['p10']:>8.0f} {row['p25']:>8.0f} {row['p50']:>8.0f} {row['p75']:>8.0f} {row['p90']:>8.0f} {row['mean']:>8.0f} {row['std']:>8.0f}")
    REPORTER.add_line("")
    
    # -------------------------------------------------------------------------
    # Panel 1: Box plot by probability bucket
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    
    bucket_data = []
    bucket_labels = []
    
    for bucket in PROB_BUCKETS_ORDER:
        row = subset[subset['prob_bucket'] == bucket]
        if len(row) == 1:
            row = row.iloc[0]
            synth = synthesize_distribution(row)
            if len(synth) > 0:
                bucket_data.append(synth)
                bucket_labels.append(bucket)
    
    if bucket_data:
        bp = ax1.boxplot(bucket_data, labels=bucket_labels, patch_artist=True)
        for patch, label in zip(bp['boxes'], bucket_labels):
            patch.set_facecolor(BUCKET_COLORS.get(label, '#cccccc'))
            patch.set_alpha(0.7)
    
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_title(f'{interval}: Returns by Prob Bucket\n({int(threshold*100)}% threshold, {tercile} tercile)')
    ax1.set_xlabel('Probability Bucket')
    ax1.set_ylabel('Return (bps)')
    ax1.tick_params(axis='x', rotation=45)
    
    # -------------------------------------------------------------------------
    # Panel 2: Box plot by interval for a specific bucket (90_99 is usually interesting)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    
    # Find the most interesting bucket (prefer 90_99, then 75_90)
    focus_bucket = '90_99' if '90_99' in df['prob_bucket'].values else '75_90'
    
    interval_subset = df[(df['prob_bucket'] == focus_bucket) & 
                         (df['threshold'] == threshold) & 
                         (df['tercile'] == tercile)]
    
    # Add interval comparison to text
    REPORTER.add_line(f"\n{focus_bucket} BUCKET COMPARISON ACROSS INTERVALS")
    REPORTER.add_line("-" * 80)
    REPORTER.add_line(f"{'Interval':<15} {'n':>6} {'P50':>8} {'Mean':>8} {'Std':>8}")
    REPORTER.add_line("-" * 80)
    
    interval_data = []
    interval_labels = []
    
    for intv in INTERVALS_ORDER:
        row = interval_subset[interval_subset['interval'] == intv]
        if len(row) == 1:
            row = row.iloc[0]
            REPORTER.add_line(f"{intv:<15} {row['n']:>6.0f} {row['p50']:>8.0f} {row['mean']:>8.0f} {row['std']:>8.0f}")
            synth = synthesize_distribution(row)
            if len(synth) > 0:
                interval_data.append(synth)
                interval_labels.append(intv)
    REPORTER.add_line("")
    
    if interval_data:
        bp2 = ax2.boxplot(interval_data, labels=interval_labels, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('#87CEEB')
            patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_title(f'{focus_bucket} Bucket: Returns by Interval\n({int(threshold*100)}% threshold, {tercile} tercile)')
    ax2.set_xlabel('Interval')
    ax2.set_ylabel('Return (bps)')
    ax2.tick_params(axis='x', rotation=45)
    
    # -------------------------------------------------------------------------
    # Panel 3: Histogram for focal cell
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    
    # Use the 90_99 or most interesting bucket for this interval
    focal_row = subset[subset['prob_bucket'] == focus_bucket]
    if len(focal_row) == 1:
        focal_row = focal_row.iloc[0]
        synth = synthesize_distribution(focal_row, n_samples=500)
        
        if len(synth) > 0:
            ax3.hist(synth, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
            ax3.axvline(x=focal_row['mean'], color='green', linewidth=2, label=f"Mean: {focal_row['mean']:.0f}")
            ax3.axvline(x=focal_row['p50'], color='orange', linewidth=2, linestyle='--', label=f"Median: {focal_row['p50']:.0f}")
            ax3.axvline(x=0, color='red', linewidth=1, linestyle='--', alpha=0.7)
            ax3.legend(loc='upper right')
    
    ax3.set_title(f'{interval}, {focus_bucket}, {tercile} Tercile\nReturns Distribution')
    ax3.set_xlabel('Return (bps)')
    ax3.set_ylabel('Frequency')
    
    # -------------------------------------------------------------------------
    # Panel 4: Cumulative distribution with percentile markers
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    
    if len(focal_row) == 1 if isinstance(focal_row, pd.DataFrame) else focal_row is not None:
        if isinstance(focal_row, pd.DataFrame):
            focal_row = focal_row.iloc[0]
        
        synth = synthesize_distribution(focal_row, n_samples=1000)
        
        if len(synth) > 0:
            sorted_data = np.sort(synth)
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            ax4.plot(sorted_data, cumulative, color='steelblue', linewidth=2)
            
            # Mark percentiles
            percentile_markers = [
                (focal_row['p10'], 0.10, 'P10', 'blue'),
                (focal_row['p25'], 0.25, 'P25', 'green'),
                (focal_row['p50'], 0.50, 'P50', 'orange'),
                (focal_row['p75'], 0.75, 'P75', 'purple'),
                (focal_row['p90'], 0.90, 'P90', 'red'),
            ]
            
            for val, prob, label, color in percentile_markers:
                ax4.scatter([val], [prob], color=color, s=80, zorder=5)
                ax4.annotate(f'{label}: {val:.0f}', (val, prob), 
                            textcoords='offset points', xytext=(10, 5),
                            fontsize=9, color=color)
            
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    ax4.set_title(f'{interval}, {focus_bucket}, {tercile} Tercile\nCumulative Distribution')
    ax4.set_xlabel('Return (bps)')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Save
    filename = f'diagnostic_{interval}_{int(threshold*100)}pct_{tercile}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    REPORTER.end_section()
    
    return filepath


# ==============================================================================
# WINDOW-LEVEL SUMMARY PLOT
# ==============================================================================

def create_window_summary(df, interval, output_dir):
    """
    Create a comprehensive summary for a single window across all thresholds and terciles.
    
    Shows:
    - Median returns heatmap (threshold × bucket) for each tercile
    - Sample sizes heatmap
    - Edge significance indicators
    """
    global REPORTER
    REPORTER.start_section(f"WINDOW SUMMARY: {interval}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    interval_df = df[df['interval'] == interval]
    
    if len(interval_df) == 0:
        plt.close()
        REPORTER.add_line("No data available for this interval.")
        REPORTER.end_section()
        return None
    
    # Row 1: Median returns by tercile
    for idx, tercile in enumerate(TERCILES_ORDER):
        ax = axes[0, idx]
        
        tercile_df = interval_df[interval_df['tercile'] == tercile]
        
        # Create pivot table
        pivot = tercile_df.pivot_table(
            values='p50', 
            index='threshold', 
            columns='prob_bucket',
            aggfunc='first'
        )
        
        # Reorder columns
        cols_present = [c for c in PROB_BUCKETS_ORDER if c in pivot.columns]
        pivot = pivot[cols_present]
        
        if pivot.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{tercile.upper()} Tercile')
            continue
        
        # Add to text report
        REPORTER.add_line(f"\n{tercile.upper()} TERCILE - MEDIAN RETURNS (P50)")
        REPORTER.add_line("-" * 70)
        REPORTER.add_line(pivot.to_string(float_format=lambda x: f'{x:.0f}' if pd.notna(x) else '---'))
        REPORTER.add_line("")
        
        # Create heatmap with diverging colormap centered at 0
        vmax = max(abs(pivot.max().max()), abs(pivot.min().min()))
        vmax = max(vmax, 100)  # Ensure some minimum range
        
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        sns.heatmap(pivot, ax=ax, cmap='RdYlGn', center=0, 
                    annot=True, fmt='.0f', norm=norm,
                    cbar_kws={'label': 'Median Return (bps)'})
        
        ax.set_title(f'{tercile.upper()} Tercile: Median Returns')
        ax.set_xlabel('Probability Bucket')
        ax.set_ylabel('Threshold')
    
    # Row 2: Sample sizes and significance
    REPORTER.add_line("\n" + "=" * 70)
    REPORTER.add_line("SAMPLE SIZES BY TERCILE")
    REPORTER.add_line("=" * 70)
    
    for idx, tercile in enumerate(TERCILES_ORDER):
        ax = axes[1, idx]
        
        tercile_df = interval_df[interval_df['tercile'] == tercile]
        
        # Create pivot for sample sizes
        pivot_n = tercile_df.pivot_table(
            values='n', 
            index='threshold', 
            columns='prob_bucket',
            aggfunc='first'
        )
        
        cols_present = [c for c in PROB_BUCKETS_ORDER if c in pivot_n.columns]
        pivot_n = pivot_n[cols_present]
        
        if pivot_n.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{tercile.upper()} Tercile: Sample Sizes')
            continue
        
        # Add to text report
        REPORTER.add_line(f"\n{tercile.upper()} TERCILE - SAMPLE SIZES")
        REPORTER.add_line("-" * 70)
        REPORTER.add_line(pivot_n.to_string(float_format=lambda x: f'{x:.0f}' if pd.notna(x) else '---'))
        REPORTER.add_line("")
        
        sns.heatmap(pivot_n, ax=ax, cmap='Blues', annot=True, fmt='.0f',
                    cbar_kws={'label': 'n'})
        
        ax.set_title(f'{tercile.upper()} Tercile: Sample Sizes')
        ax.set_xlabel('Probability Bucket')
        ax.set_ylabel('Threshold')
    
    plt.suptitle(f'Window: {interval} - Summary Across Terciles and Thresholds', fontsize=14, y=1.02)
    plt.tight_layout()
    
    filename = f'window_summary_{interval}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    REPORTER.end_section()
    
    return filepath


# ==============================================================================
# COMBINED TERCILE VIEW
# ==============================================================================

def create_combined_tercile_view(df, interval, threshold, output_dir):
    """
    Create view combining all terciles for a given interval/threshold.
    Shows how edge varies by reaction timing.
    """
    global REPORTER
    REPORTER.start_section(f"TERCILE COMPARISON: {interval}, {int(threshold*100)}% Threshold")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    subset = df[(df['interval'] == interval) & (df['threshold'] == threshold)]
    
    if len(subset) == 0:
        plt.close()
        REPORTER.add_line("No data available for this combination.")
        REPORTER.end_section()
        return None
    
    # Left: Grouped bar chart of median returns by bucket and tercile
    ax1 = axes[0]
    
    buckets = [b for b in PROB_BUCKETS_ORDER if b in subset['prob_bucket'].values]
    x = np.arange(len(buckets))
    width = 0.25
    
    REPORTER.add_line("MEDIAN RETURNS BY BUCKET AND TERCILE")
    REPORTER.add_line("-" * 70)
    header = f"{'Bucket':<12}" + "".join([f"{t:>12}" for t in TERCILES_ORDER])
    REPORTER.add_line(header)
    REPORTER.add_line("-" * 70)
    
    for bucket in buckets:
        row_text = f"{bucket:<12}"
        for tercile in TERCILES_ORDER:
            tercile_data = subset[subset['tercile'] == tercile]
            row = tercile_data[tercile_data['prob_bucket'] == bucket]
            if len(row) == 1:
                row_text += f"{row['p50'].iloc[0]:>12.0f}"
            else:
                row_text += f"{'---':>12}"
        REPORTER.add_line(row_text)
    REPORTER.add_line("")
    
    for i, tercile in enumerate(TERCILES_ORDER):
        tercile_data = subset[subset['tercile'] == tercile]
        values = []
        for bucket in buckets:
            row = tercile_data[tercile_data['prob_bucket'] == bucket]
            if len(row) == 1:
                values.append(row['p50'].iloc[0])
            else:
                values.append(np.nan)
        
        ax1.bar(x + i * width, values, width, label=tercile, color=TERCILE_COLORS[tercile], alpha=0.8)
    
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Probability Bucket')
    ax1.set_ylabel('Median Return (bps)')
    ax1.set_title(f'{interval}, {int(threshold*100)}% Threshold\nMedian Returns by Tercile')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(buckets, rotation=45)
    ax1.legend(title='Tercile')
    
    # Right: Tercile effect (early - late) by bucket
    ax2 = axes[1]
    
    early_late_diff = []
    
    REPORTER.add_line("\nTERCILE EFFECT: EARLY - LATE (Positive = Early Better)")
    REPORTER.add_line("-" * 50)
    REPORTER.add_line(f"{'Bucket':<12} {'Early':>10} {'Late':>10} {'Diff':>10}")
    REPORTER.add_line("-" * 50)
    
    for bucket in buckets:
        early = subset[(subset['prob_bucket'] == bucket) & (subset['tercile'] == 'early')]
        late = subset[(subset['prob_bucket'] == bucket) & (subset['tercile'] == 'late')]
        
        early_val = early['p50'].iloc[0] if len(early) == 1 else np.nan
        late_val = late['p50'].iloc[0] if len(late) == 1 else np.nan
        
        if len(early) == 1 and len(late) == 1:
            diff = early_val - late_val
            early_late_diff.append(diff)
            REPORTER.add_line(f"{bucket:<12} {early_val:>10.0f} {late_val:>10.0f} {diff:>10.0f}")
        else:
            early_late_diff.append(np.nan)
            REPORTER.add_line(f"{bucket:<12} {'---':>10} {'---':>10} {'---':>10}")
    
    REPORTER.add_line("")
    
    colors = ['green' if d > 0 else 'red' for d in early_late_diff]
    ax2.bar(buckets, early_late_diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Probability Bucket')
    ax2.set_ylabel('Early - Late (bps)')
    ax2.set_title(f'Tercile Effect: Early vs Late\n(Positive = Early reaction better)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    filename = f'tercile_comparison_{interval}_{int(threshold*100)}pct.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    REPORTER.end_section()
    
    return filepath


# ==============================================================================
# GLOBAL HEATMAPS
# ==============================================================================

def create_global_heatmaps(df, output_dir):
    """
    Create heatmaps showing median returns across all dimensions.
    """
    global REPORTER
    REPORTER.start_section("GLOBAL HEATMAPS DATA")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Filter to early tercile, 10% threshold for the main view
    main_df = df[(df['tercile'] == 'early') & (df['threshold'] == 0.10)]
    
    # Top row: By interval × bucket for different thresholds
    for idx, thresh in enumerate([0.05, 0.10, 0.15]):
        ax = axes[0, idx]
        
        thresh_df = df[(df['tercile'] == 'early') & (df['threshold'] == thresh)]
        
        pivot = thresh_df.pivot_table(
            values='p50',
            index='interval',
            columns='prob_bucket',
            aggfunc='first'
        )
        
        # Reorder
        rows_present = [r for r in INTERVALS_ORDER if r in pivot.index]
        cols_present = [c for c in PROB_BUCKETS_ORDER if c in pivot.columns]
        pivot = pivot.loc[rows_present, cols_present] if rows_present and cols_present else pivot
        
        if pivot.empty:
            continue
        
        # Add to text report
        REPORTER.add_pivot_table(pivot, f"MEDIAN RETURNS (P50) - {int(thresh*100)}% Threshold, Early Tercile")
        
        vmax = max(abs(pivot.max().max()), abs(pivot.min().min()), 500)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        sns.heatmap(pivot, ax=ax, cmap='RdYlGn', center=0,
                    annot=True, fmt='.0f', norm=norm,
                    cbar_kws={'label': 'Median (bps)'})
        
        ax.set_title(f'{int(thresh*100)}% Threshold, Early Tercile')
        ax.set_xlabel('Probability Bucket')
        ax.set_ylabel('Interval')
    
    # Bottom row: Mean returns, standard deviation, win rate
    metrics = [('mean', 'Mean Return (bps)', 'RdYlGn'),
               ('std', 'Std Dev (bps)', 'YlOrRd'),
               ('win_rate', 'Win Rate', 'RdYlGn')]
    
    for idx, (metric, title, cmap) in enumerate(metrics):
        ax = axes[1, idx]
        
        if metric not in main_df.columns:
            ax.text(0.5, 0.5, f'{metric} not in data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        pivot = main_df.pivot_table(
            values=metric,
            index='interval',
            columns='prob_bucket',
            aggfunc='first'
        )
        
        rows_present = [r for r in INTERVALS_ORDER if r in pivot.index]
        cols_present = [c for c in PROB_BUCKETS_ORDER if c in pivot.columns]
        pivot = pivot.loc[rows_present, cols_present] if rows_present and cols_present else pivot
        
        if pivot.empty:
            continue
        
        # Add to text report
        fmt = '.2f' if metric == 'win_rate' else '.0f'
        REPORTER.add_line(f"\n{title.upper()} - 10% Threshold, Early Tercile")
        REPORTER.add_line("-" * 60)
        REPORTER.add_line(pivot.to_string(float_format=lambda x: f'{x:{fmt}}' if pd.notna(x) else '---'))
        REPORTER.add_line("")
        
        if metric == 'win_rate':
            sns.heatmap(pivot, ax=ax, cmap=cmap, vmin=0, vmax=1,
                        annot=True, fmt='.2f', cbar_kws={'label': title})
        elif metric == 'mean':
            vmax = max(abs(pivot.max().max()), abs(pivot.min().min()), 500)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            sns.heatmap(pivot, ax=ax, cmap=cmap, center=0,
                        annot=True, fmt='.0f', norm=norm, cbar_kws={'label': title})
        else:
            sns.heatmap(pivot, ax=ax, cmap=cmap,
                        annot=True, fmt='.0f', cbar_kws={'label': title})
        
        ax.set_title(f'{title}\n(10% threshold, early tercile)')
        ax.set_xlabel('Probability Bucket')
        ax.set_ylabel('Interval')
    
    plt.suptitle('Global Heatmaps: Returns Distribution Summary', fontsize=14, y=1.02)
    plt.tight_layout()
    
    filename = 'global_heatmaps.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    REPORTER.end_section()
    
    return filepath


# ==============================================================================
# ANOMALY FOCUS: 24h_to_12h DEEP DIVE
# ==============================================================================

def create_anomaly_deep_dive(df, output_dir):
    """
    Create detailed analysis of the 24h_to_12h anomaly.
    """
    global REPORTER
    REPORTER.start_section("24h_to_12h ANOMALY DEEP DIVE")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    interval_24h = df[df['interval'] == '24h_to_12h']
    
    # Panel 1: 24h_to_12h returns by bucket (early tercile, all thresholds)
    ax1 = axes[0, 0]
    
    buckets = [b for b in PROB_BUCKETS_ORDER if b in interval_24h['prob_bucket'].values]
    x = np.arange(len(buckets))
    width = 0.2
    
    REPORTER.add_line("24h_to_12h MEDIAN RETURNS BY BUCKET AND THRESHOLD (Early Tercile)")
    REPORTER.add_line("-" * 70)
    
    # Build text table
    header = f"{'Bucket':<12}" + "".join([f"{int(t*100)}%{' ':>8}" for t in THRESHOLDS_ORDER])
    REPORTER.add_line(header)
    REPORTER.add_line("-" * 70)
    
    for bucket in buckets:
        row_text = f"{bucket:<12}"
        for i, thresh in enumerate(THRESHOLDS_ORDER):
            thresh_data = interval_24h[(interval_24h['threshold'] == thresh) & 
                                        (interval_24h['tercile'] == 'early')]
            row = thresh_data[thresh_data['prob_bucket'] == bucket]
            if len(row) == 1:
                val = row['p50'].iloc[0]
                row_text += f"{val:>10.0f}"
            else:
                row_text += f"{'---':>10}"
        REPORTER.add_line(row_text)
    REPORTER.add_line("")
    
    for i, thresh in enumerate(THRESHOLDS_ORDER):
        thresh_data = interval_24h[(interval_24h['threshold'] == thresh) & 
                                    (interval_24h['tercile'] == 'early')]
        values = []
        for bucket in buckets:
            row = thresh_data[thresh_data['prob_bucket'] == bucket]
            if len(row) == 1:
                values.append(row['p50'].iloc[0])
            else:
                values.append(np.nan)
        
        ax1.bar(x + i * width, values, width, label=f'{int(thresh*100)}%', alpha=0.8)
    
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Probability Bucket')
    ax1.set_ylabel('Median Return (bps)')
    ax1.set_title('24h_to_12h: Returns by Bucket & Threshold\n(Early Tercile)')
    ax1.set_xticks(x + 1.5 * width)
    ax1.set_xticklabels(buckets, rotation=45)
    ax1.legend(title='Threshold')
    
    # Panel 2: Compare 24h_to_12h vs adjacent intervals for 90_99 bucket
    ax2 = axes[0, 1]
    
    bucket_90_99 = df[(df['prob_bucket'] == '90_99') & 
                      (df['threshold'] == 0.10) & 
                      (df['tercile'] == 'early')]
    
    intervals = [i for i in INTERVALS_ORDER if i in bucket_90_99['interval'].values]
    values = []
    errors = []
    
    REPORTER.add_line("\n90_99 BUCKET: COMPARISON ACROSS INTERVALS (10% threshold, early tercile)")
    REPORTER.add_line("-" * 70)
    REPORTER.add_line(f"{'Interval':<15} {'Median':>10} {'Mean':>10} {'StdDev':>10} {'n':>8} {'SE':>10}")
    REPORTER.add_line("-" * 70)
    
    for intv in intervals:
        row = bucket_90_99[bucket_90_99['interval'] == intv]
        if len(row) == 1:
            r = row.iloc[0]
            values.append(r['p50'])
            se = r['std'] / np.sqrt(r['n']) if r['n'] > 0 else 0
            errors.append(se)
            REPORTER.add_line(f"{intv:<15} {r['p50']:>10.0f} {r['mean']:>10.0f} {r['std']:>10.0f} {r['n']:>8.0f} {se:>10.1f}")
        else:
            values.append(np.nan)
            errors.append(0)
    REPORTER.add_line("")
    
    colors = ['red' if v < 0 else 'green' for v in values]
    ax2.bar(intervals, values, color=colors, alpha=0.7, yerr=errors, capsize=3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Interval')
    ax2.set_ylabel('Median Return (bps)')
    ax2.set_title('90_99 Bucket: Returns by Interval\n(10% threshold, early tercile)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Highlight 24h_to_12h
    if '24h_to_12h' in intervals:
        idx = intervals.index('24h_to_12h')
        ax2.get_children()[idx].set_edgecolor('black')
        ax2.get_children()[idx].set_linewidth(3)
    
    # Panel 3: P10 (worst case) comparison
    ax3 = axes[1, 0]
    
    p10_comparison = df[(df['threshold'] == 0.10) & (df['tercile'] == 'early')]
    
    pivot = p10_comparison.pivot_table(
        values='p10',
        index='interval',
        columns='prob_bucket',
        aggfunc='first'
    )
    
    rows_present = [r for r in INTERVALS_ORDER if r in pivot.index]
    cols_present = [c for c in PROB_BUCKETS_ORDER if c in pivot.columns]
    pivot = pivot.loc[rows_present, cols_present] if rows_present and cols_present else pivot
    
    REPORTER.add_line("\nP10 (WORST CASE) BY INTERVAL AND BUCKET (10% threshold, early tercile)")
    REPORTER.add_line("-" * 70)
    REPORTER.add_line(pivot.to_string(float_format=lambda x: f'{x:.0f}' if pd.notna(x) else '---'))
    REPORTER.add_line("")
    
    if not pivot.empty:
        sns.heatmap(pivot, ax=ax3, cmap='RdYlGn', center=0,
                    annot=True, fmt='.0f', cbar_kws={'label': 'P10 (bps)'})
    
    ax3.set_title('Worst Case (P10): Loss Exposure\n(10% threshold, early tercile)')
    ax3.set_xlabel('Probability Bucket')
    ax3.set_ylabel('Interval')
    
    # Panel 4: Risk/Reward ratio (P50 / abs(P10))
    ax4 = axes[1, 1]
    
    risk_reward = df[(df['threshold'] == 0.10) & (df['tercile'] == 'early')].copy()
    risk_reward['risk_reward'] = risk_reward['p50'] / (risk_reward['p10'].abs() + 1)
    
    pivot_rr = risk_reward.pivot_table(
        values='risk_reward',
        index='interval',
        columns='prob_bucket',
        aggfunc='first'
    )
    
    rows_present = [r for r in INTERVALS_ORDER if r in pivot_rr.index]
    cols_present = [c for c in PROB_BUCKETS_ORDER if c in pivot_rr.columns]
    pivot_rr = pivot_rr.loc[rows_present, cols_present] if rows_present and cols_present else pivot_rr
    
    REPORTER.add_line("\nRISK/REWARD RATIO (Median / |P10|) - Higher = Better")
    REPORTER.add_line("-" * 70)
    REPORTER.add_line(pivot_rr.to_string(float_format=lambda x: f'{x:.2f}' if pd.notna(x) else '---'))
    REPORTER.add_line("")
    
    if not pivot_rr.empty:
        sns.heatmap(pivot_rr, ax=ax4, cmap='RdYlGn', center=0,
                    annot=True, fmt='.2f', cbar_kws={'label': 'Median / |P10|'})
    
    ax4.set_title('Risk/Reward Ratio: Median / |P10|\n(Higher = better risk-adjusted)')
    ax4.set_xlabel('Probability Bucket')
    ax4.set_ylabel('Interval')
    
    plt.suptitle('24h_to_12h Anomaly Deep Dive', fontsize=14, y=1.02)
    plt.tight_layout()
    
    filename = 'anomaly_24h_to_12h_deep_dive.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    REPORTER.end_section()
    
    return filepath


# ==============================================================================
# FAT TAIL ANALYSIS
# ==============================================================================

def create_fat_tail_analysis(df, output_dir):
    """
    Create analysis focusing on fat-tailed loss potential.
    """
    global REPORTER
    REPORTER.start_section("FAT TAIL ANALYSIS")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: P90 - P10 spread by cell (measure of distribution width)
    ax1 = axes[0, 0]
    
    df_copy = df.copy()
    df_copy['spread'] = df_copy['p90'] - df_copy['p10']
    
    main_df = df_copy[(df_copy['tercile'] == 'early') & (df_copy['threshold'] == 0.10)]
    
    pivot = main_df.pivot_table(
        values='spread',
        index='interval',
        columns='prob_bucket',
        aggfunc='first'
    )
    
    rows_present = [r for r in INTERVALS_ORDER if r in pivot.index]
    cols_present = [c for c in PROB_BUCKETS_ORDER if c in pivot.columns]
    pivot = pivot.loc[rows_present, cols_present] if rows_present and cols_present else pivot
    
    REPORTER.add_line("DISTRIBUTION WIDTH (P90 - P10) - 10% Threshold, Early Tercile")
    REPORTER.add_line("-" * 70)
    REPORTER.add_line(pivot.to_string(float_format=lambda x: f'{x:.0f}' if pd.notna(x) else '---'))
    REPORTER.add_line("")
    
    if not pivot.empty:
        sns.heatmap(pivot, ax=ax1, cmap='YlOrRd', annot=True, fmt='.0f',
                    cbar_kws={'label': 'P90 - P10 (bps)'})
    
    ax1.set_title('Distribution Width (P90 - P10)\n(10% threshold, early tercile)')
    ax1.set_xlabel('Probability Bucket')
    ax1.set_ylabel('Interval')
    
    # Panel 2: Mean vs Median comparison (skewness indicator)
    ax2 = axes[0, 1]
    
    main_df['mean_median_diff'] = main_df['mean'] - main_df['p50']
    
    pivot_skew = main_df.pivot_table(
        values='mean_median_diff',
        index='interval',
        columns='prob_bucket',
        aggfunc='first'
    )
    
    rows_present = [r for r in INTERVALS_ORDER if r in pivot_skew.index]
    cols_present = [c for c in PROB_BUCKETS_ORDER if c in pivot_skew.columns]
    pivot_skew = pivot_skew.loc[rows_present, cols_present] if rows_present and cols_present else pivot_skew
    
    REPORTER.add_line("\nSKEWNESS INDICATOR (Mean - Median) - Negative = Left-Skewed / Fat Left Tail")
    REPORTER.add_line("-" * 70)
    REPORTER.add_line(pivot_skew.to_string(float_format=lambda x: f'{x:.0f}' if pd.notna(x) else '---'))
    REPORTER.add_line("")
    
    if not pivot_skew.empty:
        vmax = max(abs(pivot_skew.max().max()), abs(pivot_skew.min().min()), 100)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        sns.heatmap(pivot_skew, ax=ax2, cmap='RdBu_r', center=0,
                    annot=True, fmt='.0f', norm=norm,
                    cbar_kws={'label': 'Mean - Median (bps)'})
    
    ax2.set_title('Skewness Indicator (Mean - Median)\n(Negative = left-skewed / fat left tail)')
    ax2.set_xlabel('Probability Bucket')
    ax2.set_ylabel('Interval')
    
    # Panel 3: Cells with worst P10 (catastrophic loss potential)
    ax3 = axes[1, 0]
    
    worst_cells = df.nsmallest(15, 'p10')[['interval', 'threshold', 'tercile', 
                                            'prob_bucket', 'n', 'p10', 'p50', 'mean']]
    
    REPORTER.add_line("\nCELLS WITH WORST P10 (Highest Loss Potential) - Top 15")
    REPORTER.add_line("-" * 90)
    REPORTER.add_line(f"{'Interval':<12} {'Thresh':>8} {'Tercile':<8} {'Bucket':<10} {'n':>6} {'P10':>10} {'P50':>10} {'Mean':>10}")
    REPORTER.add_line("-" * 90)
    for _, row in worst_cells.iterrows():
        REPORTER.add_line(f"{row['interval']:<12} {row['threshold']:>8.0%} {row['tercile']:<8} {row['prob_bucket']:<10} {row['n']:>6.0f} {row['p10']:>10.0f} {row['p50']:>10.0f} {row['mean']:>10.0f}")
    REPORTER.add_line("")
    
    ax3.axis('off')
    table = ax3.table(
        cellText=worst_cells.values,
        colLabels=['Interval', 'Thresh', 'Tercile', 'Bucket', 'n', 'P10', 'P50', 'Mean'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color cells based on P10 value
    for i in range(len(worst_cells) + 1):
        for j in range(len(worst_cells.columns)):
            if i == 0:
                table[(i, j)].set_facecolor('#4472C4')
                table[(i, j)].set_text_props(color='white', weight='bold')
            elif j == 5:  # P10 column
                val = worst_cells.iloc[i-1]['p10']
                if val < -7000:
                    table[(i, j)].set_facecolor('#FF6B6B')
                elif val < -5000:
                    table[(i, j)].set_facecolor('#FFB4B4')
    
    ax3.set_title('Cells with Worst P10 (Highest Loss Potential)', fontweight='bold', pad=20)
    
    # Panel 4: Cells with highest positive edge but also high risk
    ax4 = axes[1, 1]
    
    # High edge cells: mean > 500, spread > 8000
    high_edge_high_risk = df_copy[(df_copy['mean'] > 500) & (df_copy['spread'] > 8000)]
    high_edge_high_risk = high_edge_high_risk.nlargest(15, 'mean')[
        ['interval', 'threshold', 'tercile', 'prob_bucket', 'n', 'mean', 'spread']
    ]
    
    REPORTER.add_line("\nHIGH EDGE + HIGH RISK CELLS (Mean > 500 bps, Spread > 8000 bps)")
    REPORTER.add_line("-" * 90)
    
    ax4.axis('off')
    
    if len(high_edge_high_risk) > 0:
        REPORTER.add_line(f"{'Interval':<12} {'Thresh':>8} {'Tercile':<8} {'Bucket':<10} {'n':>6} {'Mean':>10} {'Spread':>10}")
        REPORTER.add_line("-" * 90)
        for _, row in high_edge_high_risk.iterrows():
            REPORTER.add_line(f"{row['interval']:<12} {row['threshold']:>8.0%} {row['tercile']:<8} {row['prob_bucket']:<10} {row['n']:>6.0f} {row['mean']:>10.0f} {row['spread']:>10.0f}")
        
        table2 = ax4.table(
            cellText=high_edge_high_risk.values,
            colLabels=['Interval', 'Thresh', 'Tercile', 'Bucket', 'n', 'Mean', 'Spread'],
            loc='center',
            cellLoc='center'
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1.2, 1.5)
        
        for i in range(len(high_edge_high_risk) + 1):
            for j in range(len(high_edge_high_risk.columns)):
                if i == 0:
                    table2[(i, j)].set_facecolor('#4472C4')
                    table2[(i, j)].set_text_props(color='white', weight='bold')
        
        ax4.set_title('High Edge + High Risk Cells\n(Mean > 500 bps, Spread > 8000 bps)', fontweight='bold', pad=20)
    else:
        REPORTER.add_line("No cells meet criteria (Mean > 500, Spread > 8000)")
        ax4.text(0.5, 0.5, 'No cells meet criteria\n(Mean > 500, Spread > 8000)', 
                ha='center', va='center', fontsize=12)
    
    REPORTER.add_line("")
    
    plt.suptitle('Fat Tail Analysis: Loss Distribution Characteristics', fontsize=14, y=1.02)
    plt.tight_layout()
    
    filename = 'fat_tail_analysis.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    REPORTER.end_section()
    
    return filepath


# ==============================================================================
# GENERATE TEXT REPORT
# ==============================================================================

def generate_text_report(df, output_dir):
    """Generate summary text report."""
    
    report_path = os.path.join(output_dir, f'returns_analysis_report_{TIMESTAMP}.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 4D RETURNS DISTRIBUTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        f.write("-" * 80 + "\n")
        f.write("DATASET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total cells: {len(df)}\n")
        f.write(f"Intervals: {', '.join(df['interval'].unique())}\n")
        f.write(f"Thresholds: {', '.join([f'{t:.0%}' for t in sorted(df['threshold'].unique())])}\n")
        f.write(f"Terciles: {', '.join(df['tercile'].unique())}\n")
        f.write(f"Prob buckets: {', '.join([b for b in PROB_BUCKETS_ORDER if b in df['prob_bucket'].values])}\n\n")
        
        # Key findings
        f.write("-" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-" * 80 + "\n\n")
        
        # Best performing cells
        f.write("TOP 10 CELLS BY MEDIAN RETURN:\n")
        top_cells = df.nlargest(10, 'p50')[['interval', 'threshold', 'tercile', 'prob_bucket', 'n', 'p50', 'mean']]
        f.write(top_cells.to_string(index=False))
        f.write("\n\n")
        
        # Worst performing cells
        f.write("BOTTOM 10 CELLS BY MEDIAN RETURN:\n")
        bottom_cells = df.nsmallest(10, 'p50')[['interval', 'threshold', 'tercile', 'prob_bucket', 'n', 'p50', 'mean']]
        f.write(bottom_cells.to_string(index=False))
        f.write("\n\n")
        
        # Fat tail identification
        f.write("-" * 80 + "\n")
        f.write("FAT TAIL IDENTIFICATION\n")
        f.write("-" * 80 + "\n\n")
        
        fat_tail_cells = df[df['p10'] < -6000].sort_values('p10')
        f.write(f"Cells with P10 < -6000 bps (extreme loss potential): {len(fat_tail_cells)}\n\n")
        
        if len(fat_tail_cells) > 0:
            f.write(fat_tail_cells[['interval', 'threshold', 'tercile', 'prob_bucket', 'n', 'p10', 'p50', 'mean']].to_string(index=False))
        f.write("\n\n")
        
        # 24h_to_12h anomaly
        f.write("-" * 80 + "\n")
        f.write("24h_to_12h ANOMALY ANALYSIS\n")
        f.write("-" * 80 + "\n\n")
        
        anomaly_df = df[(df['interval'] == '24h_to_12h') & (df['prob_bucket'] == '90_99')]
        f.write("90_99 bucket at 24h_to_12h interval:\n")
        f.write(anomaly_df[['threshold', 'tercile', 'n', 'p10', 'p50', 'mean', 'std']].to_string(index=False))
        f.write("\n\n")
        
        f.write("INTERPRETATION:\n")
        f.write("This interval shows consistently poor performance for high-probability markets.\n")
        f.write("The likely explanation is information regime transition - dips in this window\n")
        f.write("may represent genuine resolving information rather than noise.\n\n")
        
        # Recommendations
        f.write("-" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("1. AVOID: 24h_to_12h interval for 90_99 and 99_plus buckets\n")
        f.write("   - Consistently negative edge across all thresholds and terciles\n\n")
        
        f.write("2. PREFER: Close-to-resolution windows (6h_to_4h, 8h_to_4h, 9h_to_6h)\n")
        f.write("   - Most robust positive edge\n")
        f.write("   - Higher sample sizes\n\n")
        
        f.write("3. STOP-LOSS CONSIDERATION:\n")
        
        high_spread = df[(df['p90'] - df['p10']) > 10000]
        f.write(f"   - {len(high_spread)} cells have P90-P10 spread > 10000 bps\n")
        f.write("   - These cells have high variance and may benefit from stop-loss rules\n\n")
        
        negative_mean = df[df['mean'] < 0]
        f.write(f"4. NEGATIVE EDGE CELLS: {len(negative_mean)} cells have negative mean return\n")
        if len(negative_mean) > 0:
            f.write("   Top negative edge cells:\n")
            for _, row in negative_mean.nsmallest(5, 'mean').iterrows():
                f.write(f"   - {row['interval']}, {row['threshold']:.0%}, {row['tercile']}, {row['prob_bucket']}: {row['mean']:.0f} bps\n")
    
    print(f"Report saved: {report_path}")
    return report_path


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    global REPORTER
    
    parser = argparse.ArgumentParser(
        description='Phase 4D Returns Distribution Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots
  python phase4d_visualize_returns.py --csv percentiles.csv --output ./plots --full
  
  # Focus on specific window
  python phase4d_visualize_returns.py --csv percentiles.csv --window 24h_to_12h
  
  # Generate only heatmaps
  python phase4d_visualize_returns.py --csv percentiles.csv --heatmaps
        """
    )
    
    parser.add_argument('--csv', '-c', type=str, required=True,
                        help='Path to percentiles CSV file')
    parser.add_argument('--output', '-o', type=str, default='./plots',
                        help='Output directory for plots')
    parser.add_argument('--window', '-w', type=str, default=None,
                        help='Focus on specific window (e.g., 24h_to_12h)')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='Focus on specific threshold (e.g., 0.10)')
    parser.add_argument('--tercile', type=str, default=None,
                        help='Focus on specific tercile (early, mid, late)')
    parser.add_argument('--combine-terciles', action='store_true',
                        help='Generate combined tercile comparison views')
    parser.add_argument('--heatmaps', action='store_true',
                        help='Generate global heatmaps')
    parser.add_argument('--anomaly', action='store_true',
                        help='Generate 24h_to_12h anomaly deep dive')
    parser.add_argument('--fat-tails', action='store_true',
                        help='Generate fat tail analysis')
    parser.add_argument('--full', action='store_true',
                        help='Generate all visualizations')
    parser.add_argument('--diagnostic', action='store_true',
                        help='Generate 4-panel diagnostic plots for each cell')
    parser.add_argument('--no-console', action='store_true',
                        help='Suppress console output of text data')
    
    args = parser.parse_args()
    
    # Load data
    df = load_percentiles(args.csv)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"\nOutput directory: {args.output}\n")
    
    generated_files = []
    
    # Full mode generates everything
    if args.full:
        args.heatmaps = True
        args.anomaly = True
        args.fat_tails = True
        args.combine_terciles = True
        args.diagnostic = True
    
    # Add comprehensive data dump section first
    REPORTER.start_section("DATASET OVERVIEW")
    REPORTER.add_line(f"Source file: {args.csv}")
    REPORTER.add_line(f"Total rows: {len(df)}")
    REPORTER.add_line(f"Intervals: {', '.join(sorted(df['interval'].unique()))}")
    REPORTER.add_line(f"Thresholds: {', '.join([f'{t:.0%}' for t in sorted(df['threshold'].unique())])}")
    REPORTER.add_line(f"Terciles: {', '.join(sorted(df['tercile'].unique()))}")
    REPORTER.add_line(f"Prob buckets: {', '.join([b for b in PROB_BUCKETS_ORDER if b in df['prob_bucket'].values])}")
    REPORTER.add_line("")
    
    # Add summary statistics
    REPORTER.add_line("SUMMARY STATISTICS")
    REPORTER.add_line("-" * 50)
    for col in ['p10', 'p25', 'p50', 'p75', 'p90', 'mean', 'std', 'n']:
        if col in df.columns:
            REPORTER.add_line(f"{col:>8}: min={df[col].min():>10.1f}, max={df[col].max():>10.1f}, mean={df[col].mean():>10.1f}")
    REPORTER.end_section()
    
    # Global heatmaps
    if args.heatmaps:
        print("Generating global heatmaps...")
        fp = create_global_heatmaps(df, args.output)
        if fp:
            generated_files.append(fp)
            print(f"  Created: {fp}")
    
    # Anomaly deep dive
    if args.anomaly:
        print("Generating 24h_to_12h anomaly analysis...")
        fp = create_anomaly_deep_dive(df, args.output)
        if fp:
            generated_files.append(fp)
            print(f"  Created: {fp}")
    
    # Fat tail analysis
    if args.fat_tails:
        print("Generating fat tail analysis...")
        fp = create_fat_tail_analysis(df, args.output)
        if fp:
            generated_files.append(fp)
            print(f"  Created: {fp}")
    
    # Window summaries
    windows_to_process = [args.window] if args.window else INTERVALS_ORDER
    
    for window in windows_to_process:
        if window not in df['interval'].values:
            print(f"  Skipping {window} (not in data)")
            continue
        
        print(f"Generating summary for {window}...")
        fp = create_window_summary(df, window, args.output)
        if fp:
            generated_files.append(fp)
            print(f"  Created: {fp}")
    
    # Combined tercile views
    if args.combine_terciles:
        print("Generating combined tercile views...")
        
        thresholds = [args.threshold] if args.threshold else THRESHOLDS_ORDER
        
        for window in windows_to_process:
            if window not in df['interval'].values:
                continue
            for thresh in thresholds:
                fp = create_combined_tercile_view(df, window, thresh, args.output)
                if fp:
                    generated_files.append(fp)
                    print(f"  Created: {fp}")
    
    # Diagnostic 4-panel plots
    if args.diagnostic:
        print("Generating 4-panel diagnostic plots...")
        
        for window in windows_to_process:
            if window not in df['interval'].values:
                continue
            
            thresholds = [args.threshold] if args.threshold else THRESHOLDS_ORDER
            terciles = [args.tercile] if args.tercile else TERCILES_ORDER
            
            for thresh in thresholds:
                for terc in terciles:
                    fp = create_four_panel_diagnostic(df, window, thresh, terc, args.output)
                    if fp:
                        generated_files.append(fp)
                        print(f"  Created: {fp}")
    
    # Add complete raw data dump
    REPORTER.start_section("COMPLETE RAW DATA DUMP")
    REPORTER.add_line("All cells sorted by interval, threshold, tercile, prob_bucket")
    REPORTER.add_line("-" * 120)
    
    # Create formatted table header
    header = f"{'Interval':<12} {'Thresh':>6} {'Terc':<6} {'Bucket':<10} {'n':>6} {'P10':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8} {'Mean':>8} {'Std':>8}"
    REPORTER.add_line(header)
    REPORTER.add_line("-" * 120)
    
    # Sort and dump all data
    sorted_df = df.sort_values(['interval', 'threshold', 'tercile', 'prob_bucket'])
    for _, row in sorted_df.iterrows():
        line = f"{row['interval']:<12} {row['threshold']:>6.0%} {row['tercile']:<6} {row['prob_bucket']:<10} "
        line += f"{row['n']:>6.0f} {row['p10']:>8.0f} {row['p25']:>8.0f} {row['p50']:>8.0f} "
        line += f"{row['p75']:>8.0f} {row['p90']:>8.0f} {row['mean']:>8.0f} {row['std']:>8.0f}"
        REPORTER.add_line(line)
    
    REPORTER.end_section()
    
    # Add key findings summary
    REPORTER.start_section("KEY FINDINGS SUMMARY")
    
    # Best cells
    REPORTER.add_line("TOP 10 CELLS BY MEDIAN RETURN (P50)")
    REPORTER.add_line("-" * 90)
    top_cells = df.nlargest(10, 'p50')
    REPORTER.add_line(f"{'Interval':<12} {'Thresh':>6} {'Terc':<6} {'Bucket':<10} {'n':>6} {'P50':>8} {'Mean':>8}")
    REPORTER.add_line("-" * 90)
    for _, row in top_cells.iterrows():
        REPORTER.add_line(f"{row['interval']:<12} {row['threshold']:>6.0%} {row['tercile']:<6} {row['prob_bucket']:<10} {row['n']:>6.0f} {row['p50']:>8.0f} {row['mean']:>8.0f}")
    REPORTER.add_line("")
    
    # Worst cells
    REPORTER.add_line("BOTTOM 10 CELLS BY MEDIAN RETURN (P50)")
    REPORTER.add_line("-" * 90)
    bottom_cells = df.nsmallest(10, 'p50')
    REPORTER.add_line(f"{'Interval':<12} {'Thresh':>6} {'Terc':<6} {'Bucket':<10} {'n':>6} {'P50':>8} {'Mean':>8}")
    REPORTER.add_line("-" * 90)
    for _, row in bottom_cells.iterrows():
        REPORTER.add_line(f"{row['interval']:<12} {row['threshold']:>6.0%} {row['tercile']:<6} {row['prob_bucket']:<10} {row['n']:>6.0f} {row['p50']:>8.0f} {row['mean']:>8.0f}")
    REPORTER.add_line("")
    
    # Negative mean cells
    neg_mean = df[df['mean'] < 0]
    REPORTER.add_line(f"CELLS WITH NEGATIVE MEAN RETURN: {len(neg_mean)}")
    REPORTER.add_line("-" * 90)
    if len(neg_mean) > 0:
        REPORTER.add_line(f"{'Interval':<12} {'Thresh':>6} {'Terc':<6} {'Bucket':<10} {'n':>6} {'Mean':>8} {'P50':>8}")
        REPORTER.add_line("-" * 90)
        for _, row in neg_mean.sort_values('mean').iterrows():
            REPORTER.add_line(f"{row['interval']:<12} {row['threshold']:>6.0%} {row['tercile']:<6} {row['prob_bucket']:<10} {row['n']:>6.0f} {row['mean']:>8.0f} {row['p50']:>8.0f}")
    
    REPORTER.end_section()
    
    # Text report (original)
    print("\nGenerating text report...")
    report_path = generate_text_report(df, args.output)
    generated_files.append(report_path)
    
    # Save comprehensive text dump
    text_dump_path = os.path.join(args.output, f'complete_data_dump_{TIMESTAMP}.txt')
    REPORTER.save_to_file(text_dump_path)
    generated_files.append(text_dump_path)
    
    # Print to console if not suppressed
    if not args.no_console:
        REPORTER.print_to_console()
    
    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nTotal files generated: {len(generated_files)}")
    print(f"Output directory: {args.output}")
    print(f"\nText data dump: {text_dump_path}")
    
    if len(generated_files) <= 20:
        print("\nFiles:")
        for fp in generated_files:
            print(f"  - {os.path.basename(fp)}")


if __name__ == "__main__":
    main()