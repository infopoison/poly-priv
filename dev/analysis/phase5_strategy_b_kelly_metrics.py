"""
Kelly Sizing Integration for Strategy B Script

This module provides functions to add directly into phase5_strategy_b_confirmatory.py.
The integration point is after filtered_df is created (line ~1404).

USAGE:
  Copy these functions into your script, then call:
  
    kelly_metrics = compute_kelly_and_risk_metrics(filtered_df)
    kelly_report = format_kelly_report(kelly_metrics)
    
  Right after you have filtered_df but before generating the main report.
"""

import numpy as np


# ==============================================================================
# KELLY COMPUTATION (add to your script)
# ==============================================================================

def compute_kelly_and_risk_metrics(filtered_df, trades_per_month=512, 
                                    initial_capital=10000, risk_free_rate=0.05):
    """
    Compute Kelly sizing and portfolio risk metrics from filtered trade data.
    
    Args:
        filtered_df: DataFrame with 'return_bps' and 'is_winner' columns
        trades_per_month: Expected trades per month (for annualization)
        initial_capital: Starting capital for simulation
        risk_free_rate: Annual risk-free rate
    
    Returns:
        dict with Kelly fractions, simulated portfolio results, and risk metrics
    """
    if len(filtered_df) == 0:
        return {'error': 'No trades in filtered data'}
    
    # Extract returns
    returns_bps = filtered_df['return_bps'].values
    returns_decimal = returns_bps / 10000
    
    n_trades = len(returns_bps)
    win_rate = filtered_df['is_winner'].mean()
    
    # Basic stats
    mean_ret = np.mean(returns_decimal)
    std_ret = np.std(returns_decimal)
    variance = std_ret ** 2
    
    # Kelly fraction (mean-variance approximation): f* = μ / σ²
    if variance > 0:
        kelly_full = mean_ret / variance
        kelly_full = max(0, min(kelly_full, 2.0))  # Bound to [0, 200%]
    else:
        kelly_full = 0
    
    kelly_half = kelly_full * 0.5
    kelly_quarter = kelly_full * 0.25
    
    # Expected growth rates: g(f) = μf - (σ²f²)/2
    def growth_rate(f):
        return mean_ret * f - (variance * f**2) / 2
    
    # Simulate portfolio at each Kelly level
    simulations = {}
    for label, frac in [('full', kelly_full), ('half', kelly_half), ('quarter', kelly_quarter)]:
        if frac > 0:
            sim = _simulate_portfolio(returns_decimal, frac, initial_capital)
            simulations[label] = sim
    
    # Compute annualized risk metrics
    trades_per_year = trades_per_month * 12
    dataset_years = n_trades / trades_per_year
    
    risk_metrics = {}
    for label, sim in simulations.items():
        metrics = _compute_risk_ratios(
            sim['period_returns'],
            sim['total_return'],
            sim['max_drawdown'],
            dataset_years,
            trades_per_year,
            risk_free_rate
        )
        risk_metrics[label] = metrics
    
    # Win/loss stats
    wins = returns_bps[returns_bps > 0]
    losses = returns_bps[returns_bps <= 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    
    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'mean_bps': np.mean(returns_bps),
        'std_bps': np.std(returns_bps),
        'avg_win_bps': avg_win,
        'avg_loss_bps': avg_loss,
        'kelly_full': kelly_full,
        'kelly_half': kelly_half,
        'kelly_quarter': kelly_quarter,
        'growth_rate_full': growth_rate(kelly_full),
        'growth_rate_half': growth_rate(kelly_half),
        'growth_rate_quarter': growth_rate(kelly_quarter),
        'simulations': simulations,
        'risk_metrics': risk_metrics,
        'trades_per_month': trades_per_month,
        'dataset_years': dataset_years,
        'initial_capital': initial_capital
    }


def _simulate_portfolio(returns_decimal, kelly_fraction, initial_capital):
    """Simulate sequential betting with Kelly sizing."""
    equity = [initial_capital]
    
    for ret in returns_decimal:
        current = equity[-1]
        bet_size = current * kelly_fraction
        pnl = bet_size * ret
        new_capital = max(current + pnl, 0)
        equity.append(new_capital)
    
    equity = np.array(equity)
    
    # Drawdowns
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_drawdown = np.min(drawdowns)
    
    # Per-trade returns
    period_returns = np.diff(equity) / np.maximum(equity[:-1], 1e-10)
    
    return {
        'equity_curve': equity,
        'final_equity': equity[-1],
        'total_return': (equity[-1] - initial_capital) / initial_capital,
        'max_drawdown': max_drawdown,
        'period_returns': period_returns
    }


def _compute_risk_ratios(period_returns, total_return, max_drawdown, 
                         dataset_years, periods_per_year, risk_free_rate):
    """Compute Sharpe, Sortino, Calmar."""
    rf_period = (1 + risk_free_rate) ** (1/periods_per_year) - 1
    excess = period_returns - rf_period
    
    mean_excess = np.mean(excess)
    std_returns = np.std(period_returns)
    
    # Sharpe
    sharpe = (mean_excess / std_returns) * np.sqrt(periods_per_year) if std_returns > 0 else 0
    
    # Sortino
    downside = period_returns[period_returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 0
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year) if downside_std > 0 else np.inf
    
    # Calmar
    if dataset_years > 0 and max_drawdown < 0:
        ann_return = (1 + total_return) ** (1/dataset_years) - 1
        calmar = ann_return / abs(max_drawdown)
    else:
        calmar = np.inf
    
    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'total_return': total_return,
        'max_drawdown': max_drawdown
    }


def format_kelly_report(metrics):
    """Format Kelly metrics as report string."""
    lines = []
    
    lines.append("\n" + "=" * 80)
    lines.append("KELLY SIZING AND PORTFOLIO RISK METRICS")
    lines.append("=" * 80)
    
    lines.append("\nKelly Fraction Analysis:")
    lines.append(f"  Full Kelly (f*):        {metrics['kelly_full']*100:.2f}% per trade")
    lines.append(f"  Half Kelly:             {metrics['kelly_half']*100:.2f}% per trade")
    lines.append(f"  Quarter Kelly:          {metrics['kelly_quarter']*100:.2f}% per trade")
    
    lines.append(f"\n  Derivation: f* = μ/σ² = {metrics['mean_bps']/10000*100:.4f}% / {(metrics['std_bps']/10000)**2*100:.6f}%")
    
    lines.append("\nExpected Log Growth (per trade):")
    lines.append(f"  Full Kelly:             {metrics['growth_rate_full']*10000:.2f} bps")
    lines.append(f"  Half Kelly:             {metrics['growth_rate_half']*10000:.2f} bps")
    lines.append(f"  Quarter Kelly:          {metrics['growth_rate_quarter']*10000:.2f} bps")
    
    lines.append(f"\nPortfolio Simulation (${metrics['initial_capital']:,} initial, {metrics['n_trades']:,} trades):")
    
    for label in ['quarter', 'half', 'full']:
        if label in metrics['risk_metrics']:
            rm = metrics['risk_metrics'][label]
            sim = metrics['simulations'][label]
            kelly_pct = metrics[f'kelly_{label}'] * 100
            
            lines.append(f"\n  {label.upper()} KELLY ({kelly_pct:.1f}% per trade):")
            lines.append(f"    Final equity:         ${sim['final_equity']:,.0f}")
            lines.append(f"    Total return:         {rm['total_return']*100:+.1f}%")
            lines.append(f"    Max drawdown:         {rm['max_drawdown']*100:.1f}%")
            lines.append(f"    Sharpe ratio:         {rm['sharpe']:.2f}")
            lines.append(f"    Sortino ratio:        {rm['sortino']:.2f}")
            lines.append(f"    Calmar ratio:         {rm['calmar']:.2f}")
    
    # Recommendation
    lines.append("\n" + "-" * 40)
    kelly_q = metrics['kelly_quarter']
    if kelly_q < 0.01:
        lines.append("⚠️  Marginal edge. Kelly suggests <1% position sizes.")
    elif kelly_q < 0.05:
        lines.append(f"✓  Viable edge. Recommend quarter Kelly ({kelly_q*100:.1f}% per trade).")
    else:
        lines.append(f"✓  Strong edge. Start with quarter Kelly ({kelly_q*100:.1f}%), scale up after validation.")
    
    if 'quarter' in metrics['risk_metrics']:
        dd = metrics['risk_metrics']['quarter']['max_drawdown']
        if dd < -0.25:
            lines.append(f"⚠️  Quarter Kelly still shows {dd*100:.0f}% max DD. Consider smaller sizing.")
    
    lines.append("")
    
    return "\n".join(lines)


# ==============================================================================
# EXAMPLE: How to integrate into your main() function
# ==============================================================================
"""
Add this after line 1412 in your script (after filtered_stats = compute_aggregate_stats(...)):

    # -------------------------------------------------------------------------
    # STEP 6b: Kelly Sizing and Risk Metrics
    # -------------------------------------------------------------------------
    
    if filtered_stats and len(filtered_df) > 0:
        log("\nComputing Kelly sizing and portfolio metrics...")
        
        kelly_metrics = compute_kelly_and_risk_metrics(
            filtered_df,
            trades_per_month=512,  # From capacity estimate
            initial_capital=10000,
            risk_free_rate=0.05
        )
        
        kelly_report_str = format_kelly_report(kelly_metrics)
        log(kelly_report_str)
    else:
        kelly_metrics = None
        kelly_report_str = ""

Then append kelly_report_str to your main report file.
"""


# ==============================================================================
# TEST: Run this file standalone to verify it works
# ==============================================================================

if __name__ == "__main__":
    import pandas as pd
    
    # Create synthetic data matching Strategy B distribution
    np.random.seed(42)
    n = 6150
    win_rate = 0.191
    
    n_wins = int(n * win_rate)
    n_losses = n - n_wins
    
    # Generate returns
    losses = np.random.triangular(-4100, -2000, -200, n_losses)
    wins = np.random.triangular(3000, 7500, 9900, n_wins)
    
    returns_bps = np.concatenate([losses, wins])
    np.random.shuffle(returns_bps)
    
    # Adjust to target mean of +104.8 bps
    returns_bps = returns_bps + (104.8 - np.mean(returns_bps))
    
    # Create DataFrame mimicking filtered_df structure
    filtered_df = pd.DataFrame({
        'return_bps': returns_bps,
        'is_winner': returns_bps > 0
    })
    
    print("Testing with synthetic Strategy B data...")
    print(f"  n = {len(filtered_df)}")
    print(f"  Win rate: {filtered_df['is_winner'].mean()*100:.1f}%")
    print(f"  Mean: {filtered_df['return_bps'].mean():+.1f} bps")
    print()
    
    # Run the analysis
    metrics = compute_kelly_and_risk_metrics(filtered_df)
    report = format_kelly_report(metrics)
    print(report)