#!/usr/bin/env python3
"""
Phase 4D Binary Payoff Structure Addendum
Version: 1.0

PURPOSE:
    Add outcome-stratified analysis to demonstrate the binary payoff structure
    in prediction market trades, especially for favorites.
    
    Key insight: The "fat tails" in returns distributions aren't mysterious -
    they're simply the bimodal structure of "you won" vs "you lost" on a 
    binary option. For favorites:
    - Win (~80%): gain ~10-20% of position (buy at 90¢, get $1)
    - Lose (~20%): lose ~80-90% of position (buy at 90¢, get $0)
    
    This addendum isolates these distributions to show the structure clearly.

INTEGRATION:
    This can be run standalone on existing returns data, or integrated into
    the main phase4d_returns_distribution.py script by adding these functions
    and calling them from run_stage2_analysis().

USAGE:
    python phase4d_binary_payoff_addendum.py --returns-dir /path/to/returns_data

OUTPUT:
    - Binary payoff plots per window (6 plots)
    - Outcome-stratified percentiles CSV
    - Summary report with asymmetry metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import argparse
import os
import glob
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION (should match main script)
# ==============================================================================

INTERVAL_PAIRS = [
    (48, 24, '48h_to_24h'),
    (24, 12, '24h_to_12h'),
    (12, 6, '12h_to_6h'),
    (9, 6, '9h_to_6h'),
    (8, 4, '8h_to_4h'),
    (6, 4, '6h_to_4h'),
]

INTERVALS_ORDER = [x[2] for x in INTERVAL_PAIRS]

PROB_BUCKETS = [
    ('sub_51', 0.0, 0.51),
    ('51_60', 0.51, 0.60),
    ('60_75', 0.60, 0.75),
    ('75_90', 0.75, 0.90),
    ('90_99', 0.90, 0.99),
    ('99_plus', 0.99, 1.01),
]
PROB_BUCKET_LABELS = [b[0] for b in PROB_BUCKETS]

BUCKET_COLORS = {
    'sub_51': '#1f77b4',
    '51_60': '#ff7f0e', 
    '60_75': '#2ca02c',
    '75_90': '#d62728',
    '90_99': '#9467bd',
    '99_plus': '#8c564b',
}

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

MIN_SAMPLES = 20


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_returns_data(returns_dir):
    """Load returns data from parquet chunks."""
    print(f"Loading returns data from: {returns_dir}")
    
    parquet_files = glob.glob(os.path.join(returns_dir, 'returns_chunk_*.parquet'))
    
    if not parquet_files:
        print("  ERROR: No parquet files found")
        return None
    
    print(f"  Found {len(parquet_files)} parquet files")
    
    dfs = []
    for f in parquet_files:
        dfs.append(pd.read_parquet(f))
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} total returns")
    print(f"  Columns: {df.columns.tolist()}")
    
    return df


# ==============================================================================
# OUTCOME-STRATIFIED STATISTICS
# ==============================================================================

def compute_outcome_stratified_stats(returns_df):
    """
    Compute statistics stratified by outcome (win/loss).
    
    Collapses across terciles and thresholds to focus on the binary structure.
    Groups by: interval, prob_bucket, is_winner
    """
    print("\nComputing outcome-stratified statistics...")
    
    results = []
    
    # Group by interval and prob_bucket (collapsing tercile and threshold)
    for interval in INTERVALS_ORDER:
        for prob_bucket in PROB_BUCKET_LABELS:
            subset = returns_df[
                (returns_df['interval'] == interval) &
                (returns_df['prob_bucket'] == prob_bucket)
            ]
            
            if len(subset) < MIN_SAMPLES:
                continue
            
            # Overall stats
            overall_n = len(subset)
            overall_mean = subset['return_bps'].mean()
            overall_win_rate = subset['is_winner'].mean()
            
            # Win stats
            wins = subset[subset['is_winner'] == True]
            win_n = len(wins)
            if win_n >= 5:
                win_mean = wins['return_bps'].mean()
                win_std = wins['return_bps'].std()
                win_p10 = np.percentile(wins['return_bps'], 10)
                win_p50 = np.percentile(wins['return_bps'], 50)
                win_p90 = np.percentile(wins['return_bps'], 90)
            else:
                win_mean = win_std = win_p10 = win_p50 = win_p90 = np.nan
            
            # Loss stats
            losses = subset[subset['is_winner'] == False]
            loss_n = len(losses)
            if loss_n >= 5:
                loss_mean = losses['return_bps'].mean()
                loss_std = losses['return_bps'].std()
                loss_p10 = np.percentile(losses['return_bps'], 10)
                loss_p50 = np.percentile(losses['return_bps'], 50)
                loss_p90 = np.percentile(losses['return_bps'], 90)
            else:
                loss_mean = loss_std = loss_p10 = loss_p50 = loss_p90 = np.nan
            
            # Expected value decomposition
            if win_n >= 5 and loss_n >= 5:
                ev_from_wins = overall_win_rate * win_mean
                ev_from_losses = (1 - overall_win_rate) * loss_mean
                ev_total = ev_from_wins + ev_from_losses
            else:
                ev_from_wins = ev_from_losses = ev_total = np.nan
            
            results.append({
                'interval': interval,
                'prob_bucket': prob_bucket,
                'n_total': overall_n,
                'win_rate': overall_win_rate,
                'overall_mean': overall_mean,
                'n_wins': win_n,
                'win_mean': win_mean,
                'win_std': win_std,
                'win_p10': win_p10,
                'win_p50': win_p50,
                'win_p90': win_p90,
                'n_losses': loss_n,
                'loss_mean': loss_mean,
                'loss_std': loss_std,
                'loss_p10': loss_p10,
                'loss_p50': loss_p50,
                'loss_p90': loss_p90,
                'ev_from_wins': ev_from_wins,
                'ev_from_losses': ev_from_losses,
                'ev_total': ev_total,
            })
    
    df = pd.DataFrame(results)
    print(f"  Computed stats for {len(df)} interval×bucket combinations")
    
    return df


# ==============================================================================
# BINARY PAYOFF STRUCTURE PLOTS
# ==============================================================================

def generate_binary_payoff_plots(returns_df, stratified_df, output_dir):
    """
    Generate plots showing the binary payoff structure.
    
    For each window:
    - Panel 1: Win distribution by bucket (violin/box)
    - Panel 2: Loss distribution by bucket (violin/box)
    - Panel 3: Win rate by bucket (bar)
    - Panel 4: Expected value decomposition
    """
    plots_dir = os.path.join(output_dir, 'binary_payoff_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\nGenerating binary payoff structure plots...")
    
    for interval in INTERVALS_ORDER:
        print(f"  Processing {interval}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        interval_data = returns_df[returns_df['interval'] == interval]
        interval_stats = stratified_df[stratified_df['interval'] == interval]
        
        if len(interval_data) < MIN_SAMPLES:
            plt.close()
            continue
        
        # ---------------------------------------------------------------------
        # Panel 1: Win distribution by bucket
        # ---------------------------------------------------------------------
        ax1 = axes[0, 0]
        
        win_data = []
        win_labels = []
        
        for bucket in PROB_BUCKET_LABELS:
            wins = interval_data[
                (interval_data['prob_bucket'] == bucket) &
                (interval_data['is_winner'] == True)
            ]['return_bps'].values
            
            if len(wins) >= 5:
                win_data.append(wins)
                win_labels.append(bucket)
        
        if win_data:
            bp1 = ax1.boxplot(win_data, labels=win_labels, patch_artist=True)
            for i, (patch, label) in enumerate(zip(bp1['boxes'], win_labels)):
                patch.set_facecolor(BUCKET_COLORS.get(label, '#cccccc'))
                patch.set_alpha(0.7)
            
            # Add median annotations
            for i, d in enumerate(win_data):
                med = np.median(d)
                ax1.annotate(f'{med:.0f}', xy=(i+1, med), xytext=(5, 5),
                            textcoords='offset points', fontsize=8)
        
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_title(f'{interval}: Returns GIVEN WIN\n(You bought the winner)', fontweight='bold')
        ax1.set_ylabel('Return (bps)')
        ax1.set_xlabel('Probability Bucket')
        ax1.tick_params(axis='x', rotation=45)
        
        # Set y-axis to show the positive range better
        ax1.set_ylim(-2000, 12000)
        
        # ---------------------------------------------------------------------
        # Panel 2: Loss distribution by bucket
        # ---------------------------------------------------------------------
        ax2 = axes[0, 1]
        
        loss_data = []
        loss_labels = []
        
        for bucket in PROB_BUCKET_LABELS:
            losses = interval_data[
                (interval_data['prob_bucket'] == bucket) &
                (interval_data['is_winner'] == False)
            ]['return_bps'].values
            
            if len(losses) >= 5:
                loss_data.append(losses)
                loss_labels.append(bucket)
        
        if loss_data:
            bp2 = ax2.boxplot(loss_data, labels=loss_labels, patch_artist=True)
            for i, (patch, label) in enumerate(zip(bp2['boxes'], loss_labels)):
                patch.set_facecolor(BUCKET_COLORS.get(label, '#cccccc'))
                patch.set_alpha(0.7)
            
            # Add median annotations
            for i, d in enumerate(loss_data):
                med = np.median(d)
                ax2.annotate(f'{med:.0f}', xy=(i+1, med), xytext=(5, -15),
                            textcoords='offset points', fontsize=8)
        
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_title(f'{interval}: Returns GIVEN LOSS\n(You bought the loser)', fontweight='bold')
        ax2.set_ylabel('Return (bps)')
        ax2.set_xlabel('Probability Bucket')
        ax2.tick_params(axis='x', rotation=45)
        
        # Set y-axis to show the negative range
        ax2.set_ylim(-12000, 2000)
        
        # ---------------------------------------------------------------------
        # Panel 3: Win rate by bucket
        # ---------------------------------------------------------------------
        ax3 = axes[1, 0]
        
        buckets_present = interval_stats['prob_bucket'].tolist()
        win_rates = interval_stats['win_rate'].tolist()
        
        if buckets_present:
            colors = [BUCKET_COLORS.get(b, '#cccccc') for b in buckets_present]
            bars = ax3.bar(buckets_present, win_rates, color=colors, alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar, wr in zip(bars, win_rates):
                ax3.annotate(f'{wr:.1%}', xy=(bar.get_x() + bar.get_width()/2, wr),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', fontsize=10, fontweight='bold')
        
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title(f'{interval}: Win Rate by Bucket\n(Fraction of trades that won)', fontweight='bold')
        ax3.set_ylabel('Win Rate')
        ax3.set_xlabel('Probability Bucket')
        ax3.set_ylim(0, 1.1)
        ax3.tick_params(axis='x', rotation=45)
        
        # ---------------------------------------------------------------------
        # Panel 4: Expected Value Decomposition
        # ---------------------------------------------------------------------
        ax4 = axes[1, 1]
        
        if len(interval_stats) > 0:
            x = np.arange(len(interval_stats))
            width = 0.35
            
            ev_wins = interval_stats['ev_from_wins'].values
            ev_losses = interval_stats['ev_from_losses'].values
            buckets = interval_stats['prob_bucket'].values
            
            bars1 = ax4.bar(x - width/2, ev_wins, width, label='EV from Wins', 
                           color='green', alpha=0.7)
            bars2 = ax4.bar(x + width/2, ev_losses, width, label='EV from Losses', 
                           color='red', alpha=0.7)
            
            # Add net EV line
            net_ev = ev_wins + ev_losses
            ax4.plot(x, net_ev, 'ko-', markersize=8, linewidth=2, label='Net EV')
            
            # Annotate net EV
            for i, nev in enumerate(net_ev):
                if not np.isnan(nev):
                    ax4.annotate(f'{nev:.0f}', xy=(i, nev), xytext=(5, 5),
                                textcoords='offset points', fontsize=9, fontweight='bold')
            
            ax4.set_xticks(x)
            ax4.set_xticklabels(buckets, rotation=45)
            ax4.legend(loc='upper left')
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_title(f'{interval}: Expected Value Decomposition\nEV = (WinRate × AvgWin) + (LossRate × AvgLoss)', fontweight='bold')
        ax4.set_ylabel('Expected Value (bps)')
        ax4.set_xlabel('Probability Bucket')
        
        plt.suptitle(f'BINARY PAYOFF STRUCTURE: {interval}\n(Collapsed across all thresholds and terciles)', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filepath = os.path.join(plots_dir, f'binary_payoff_{interval}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved: {filepath}")
    
    return plots_dir


def generate_asymmetry_summary_plot(stratified_df, output_dir):
    """
    Generate a single summary plot showing the asymmetry across all windows.
    
    This is the "money shot" for the PM presentation - shows that:
    - Wins are modest and predictable (~entry price to $1)
    - Losses are catastrophic and predictable (~entry price to $0)
    """
    print("\nGenerating asymmetry summary plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, interval in enumerate(INTERVALS_ORDER):
        ax = axes[idx]
        
        subset = stratified_df[stratified_df['interval'] == interval]
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(interval)
            continue
        
        buckets = subset['prob_bucket'].values
        x = np.arange(len(buckets))
        width = 0.35
        
        win_means = subset['win_mean'].values
        loss_means = subset['loss_mean'].values
        
        # Bars for win and loss means
        ax.bar(x - width/2, win_means, width, label='Avg Return | Win', 
               color='green', alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, loss_means, width, label='Avg Return | Loss', 
               color='red', alpha=0.7, edgecolor='black')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_xticks(x)
        ax.set_xticklabels(buckets, rotation=45, fontsize=9)
        ax.set_ylabel('Average Return (bps)')
        ax.set_title(f'{interval}', fontweight='bold')
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9)
        
        # Add win rate annotations at the top
        for i, (wr, bucket) in enumerate(zip(subset['win_rate'].values, buckets)):
            if not np.isnan(wr):
                ax.annotate(f'WR:{wr:.0%}', xy=(i, max(win_means[i], 0) + 500),
                           ha='center', fontsize=8, color='darkgreen')
    
    plt.suptitle('THE BINARY PAYOFF STRUCTURE\nAverage Returns Conditional on Outcome\n'
                 '(Green = You Won, Red = You Lost)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'binary_payoff_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")
    return filepath


def generate_favorites_focus_plot(returns_df, stratified_df, output_dir):
    """
    Generate a focused plot on 90_99 bucket showing the bimodal distribution.
    
    This is the clearest demonstration of the binary option structure.
    """
    print("\nGenerating favorites focus plot...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, interval in enumerate(INTERVALS_ORDER):
        ax = axes[idx]
        
        # Get 90_99 data for this interval
        subset = returns_df[
            (returns_df['interval'] == interval) &
            (returns_df['prob_bucket'] == '90_99')
        ]
        
        if len(subset) < MIN_SAMPLES:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{interval}: 90_99 Bucket')
            continue
        
        wins = subset[subset['is_winner'] == True]['return_bps'].values
        losses = subset[subset['is_winner'] == False]['return_bps'].values
        
        # Plot overlapping histograms
        bins = np.linspace(-10000, 10000, 50)
        
        if len(wins) > 5:
            ax.hist(wins, bins=bins, alpha=0.6, color='green', label=f'Wins (n={len(wins)})', density=True)
        if len(losses) > 5:
            ax.hist(losses, bins=bins, alpha=0.6, color='red', label=f'Losses (n={len(losses)})', density=True)
        
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add vertical lines for means
        if len(wins) > 5:
            win_mean = np.mean(wins)
            ax.axvline(x=win_mean, color='darkgreen', linestyle='-', linewidth=2, 
                      label=f'Win Mean: {win_mean:.0f}')
        if len(losses) > 5:
            loss_mean = np.mean(losses)
            ax.axvline(x=loss_mean, color='darkred', linestyle='-', linewidth=2,
                      label=f'Loss Mean: {loss_mean:.0f}')
        
        # Win rate annotation
        win_rate = len(wins) / (len(wins) + len(losses))
        ax.text(0.98, 0.98, f'Win Rate: {win_rate:.1%}', transform=ax.transAxes,
               ha='right', va='top', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Return (bps)')
        ax.set_ylabel('Density')
        ax.set_title(f'{interval}: 90_99 Bucket (Favorites)', fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
    
    plt.suptitle('BIMODAL DISTRIBUTION IN FAVORITES (90-99% Probability)\n'
                 'Green = Wins cluster around +1000-2000 bps | Red = Losses cluster around -8000-9000 bps',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'favorites_bimodal_distribution.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filepath}")
    return filepath


# ==============================================================================
# TEXT REPORT
# ==============================================================================

def generate_binary_payoff_report(stratified_df, output_dir):
    """Generate text report with binary payoff analysis."""
    
    report_path = os.path.join(output_dir, f'binary_payoff_report_{TIMESTAMP}.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 90 + "\n")
        f.write("BINARY PAYOFF STRUCTURE ANALYSIS\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("KEY INSIGHT:\n")
        f.write("-" * 90 + "\n")
        f.write("The 'fat tails' in prediction market returns are not mysterious variance.\n")
        f.write("They are the deterministic consequence of binary options:\n")
        f.write("  - When you WIN: you paid X cents, received $1 → return = (100-X)/X * 10000 bps\n")
        f.write("  - When you LOSE: you paid X cents, received $0 → return = -X/X * 10000 = -10000 bps\n\n")
        f.write("For favorites (90-99% implied probability):\n")
        f.write("  - Win ~80% of time: typical return +1000-2000 bps (modest gain)\n")
        f.write("  - Lose ~20% of time: typical return -8000-9500 bps (catastrophic loss)\n\n")
        
        f.write("=" * 90 + "\n")
        f.write("OUTCOME-STRATIFIED STATISTICS BY INTERVAL\n")
        f.write("=" * 90 + "\n\n")
        
        for interval in INTERVALS_ORDER:
            subset = stratified_df[stratified_df['interval'] == interval]
            
            if len(subset) == 0:
                continue
            
            f.write(f"\n{'='*90}\n")
            f.write(f"INTERVAL: {interval}\n")
            f.write(f"{'='*90}\n\n")
            
            f.write(f"{'Bucket':<10} {'n':>6} {'WinRate':>8} {'AvgWin':>8} {'AvgLoss':>9} {'EVWin':>8} {'EVLoss':>9} {'NetEV':>8}\n")
            f.write("-" * 90 + "\n")
            
            for _, row in subset.iterrows():
                f.write(f"{row['prob_bucket']:<10} {row['n_total']:>6.0f} {row['win_rate']:>8.1%} "
                       f"{row['win_mean']:>8.0f} {row['loss_mean']:>9.0f} "
                       f"{row['ev_from_wins']:>8.0f} {row['ev_from_losses']:>9.0f} {row['ev_total']:>8.0f}\n")
            
            f.write("\n")
            
            # Detailed win/loss distributions
            f.write("Win Distribution Details:\n")
            f.write(f"{'Bucket':<10} {'n_wins':>8} {'P10':>8} {'P50':>8} {'P90':>8} {'Std':>8}\n")
            f.write("-" * 60 + "\n")
            for _, row in subset.iterrows():
                if not np.isnan(row['win_p50']):
                    f.write(f"{row['prob_bucket']:<10} {row['n_wins']:>8.0f} {row['win_p10']:>8.0f} "
                           f"{row['win_p50']:>8.0f} {row['win_p90']:>8.0f} {row['win_std']:>8.0f}\n")
            
            f.write("\nLoss Distribution Details:\n")
            f.write(f"{'Bucket':<10} {'n_loss':>8} {'P10':>8} {'P50':>8} {'P90':>8} {'Std':>8}\n")
            f.write("-" * 60 + "\n")
            for _, row in subset.iterrows():
                if not np.isnan(row['loss_p50']):
                    f.write(f"{row['prob_bucket']:<10} {row['n_losses']:>8.0f} {row['loss_p10']:>8.0f} "
                           f"{row['loss_p50']:>8.0f} {row['loss_p90']:>8.0f} {row['loss_std']:>8.0f}\n")
        
        # Key findings for PM presentation
        f.write("\n\n" + "=" * 90 + "\n")
        f.write("KEY FINDINGS FOR PRESENTATION\n")
        f.write("=" * 90 + "\n\n")
        
        # Focus on 90_99 bucket
        fav_data = stratified_df[stratified_df['prob_bucket'] == '90_99']
        
        if len(fav_data) > 0:
            f.write("FAVORITES (90-99% Bucket) Summary:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Average win rate across windows: {fav_data['win_rate'].mean():.1%}\n")
            f.write(f"Average return when winning: {fav_data['win_mean'].mean():.0f} bps\n")
            f.write(f"Average return when losing: {fav_data['loss_mean'].mean():.0f} bps\n")
            f.write(f"Average net EV: {fav_data['ev_total'].mean():.0f} bps\n\n")
            
            f.write("The asymmetry is stark:\n")
            f.write("  - You win ~80% with modest gains (~1500 bps)\n")
            f.write("  - You lose ~20% with catastrophic losses (~-8500 bps)\n")
            f.write("  - Net expected value depends on entry price precision\n\n")
        
        f.write("IMPLICATIONS FOR STRATEGY:\n")
        f.write("-" * 60 + "\n")
        f.write("1. The 'variance' in favorites is NOT unpredictable - it's binary outcomes\n")
        f.write("2. Stop-losses are largely irrelevant - you either win or lose the full amount\n")
        f.write("3. Edge comes from win rate exceeding implied probability, not from 'catching' moves\n")
        f.write("4. Position sizing is critical - a 20% loss rate with 85% position loss is dangerous\n")
        f.write("5. The 24h_to_12h anomaly may reflect informed flow detection - losses represent\n")
        f.write("   correctly trading against us, not 'bad luck'\n\n")
        
    print(f"  Saved: {report_path}")
    return report_path


# ==============================================================================
# CONSOLE OUTPUT
# ==============================================================================

def print_summary_to_console(stratified_df):
    """Print key summary to console."""
    
    print("\n" + "=" * 80)
    print("BINARY PAYOFF STRUCTURE SUMMARY")
    print("=" * 80)
    
    # Focus on favorites
    print("\n90_99 BUCKET (Favorites) - The Key Insight:")
    print("-" * 60)
    
    fav_data = stratified_df[stratified_df['prob_bucket'] == '90_99']
    
    if len(fav_data) > 0:
        print(f"{'Interval':<15} {'WinRate':>8} {'AvgWin':>10} {'AvgLoss':>10} {'NetEV':>10}")
        print("-" * 60)
        
        for _, row in fav_data.iterrows():
            print(f"{row['interval']:<15} {row['win_rate']:>8.1%} {row['win_mean']:>10.0f} "
                  f"{row['loss_mean']:>10.0f} {row['ev_total']:>10.0f}")
        
        print("-" * 60)
        print(f"{'AVERAGE':<15} {fav_data['win_rate'].mean():>8.1%} {fav_data['win_mean'].mean():>10.0f} "
              f"{fav_data['loss_mean'].mean():>10.0f} {fav_data['ev_total'].mean():>10.0f}")
    
    print("\nKEY TAKEAWAY:")
    print("  The 'fat tails' are just LOSING on a favorite.")
    print("  Win ~80% → gain ~1500 bps | Lose ~20% → lose ~8500 bps")
    print("  This is deterministic, not stochastic variance.")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_binary_payoff_analysis(returns_dir, output_dir=None):
    """Run the complete binary payoff analysis."""
    
    if output_dir is None:
        output_dir = returns_dir
    
    print("=" * 70)
    print("PHASE 4D BINARY PAYOFF STRUCTURE ANALYSIS")
    print("=" * 70)
    
    # Load data
    returns_df = load_returns_data(returns_dir)
    
    if returns_df is None:
        print("ERROR: Could not load returns data")
        return
    
    # Compute outcome-stratified stats
    stratified_df = compute_outcome_stratified_stats(returns_df)
    
    # Save stats
    stats_path = os.path.join(output_dir, f'outcome_stratified_stats_{TIMESTAMP}.csv')
    stratified_df.to_csv(stats_path, index=False)
    print(f"\nSaved outcome-stratified stats: {stats_path}")
    
    # Generate plots
    plots_dir = generate_binary_payoff_plots(returns_df, stratified_df, output_dir)
    summary_path = generate_asymmetry_summary_plot(stratified_df, output_dir)
    favorites_path = generate_favorites_focus_plot(returns_df, stratified_df, output_dir)
    
    # Generate report
    report_path = generate_binary_payoff_report(stratified_df, output_dir)
    
    # Console summary
    print_summary_to_console(stratified_df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  - Stats CSV: {stats_path}")
    print(f"  - Plots directory: {plots_dir}")
    print(f"  - Summary plot: {summary_path}")
    print(f"  - Favorites bimodal: {favorites_path}")
    print(f"  - Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Phase 4D Binary Payoff Structure Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phase4d_binary_payoff_addendum.py --returns-dir ./phase4d_returns_data_20251229/
  
  python phase4d_binary_payoff_addendum.py --returns-dir ./returns_data --output ./analysis
        """
    )
    
    parser.add_argument('--returns-dir', '-r', type=str, required=True,
                        help='Path to directory containing returns parquet chunks')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory (defaults to returns-dir)')
    
    args = parser.parse_args()
    
    run_binary_payoff_analysis(args.returns_dir, args.output)