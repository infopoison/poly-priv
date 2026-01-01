"""
Concurrent Position Kelly Simulation

Corrects the sequential simulation error by modeling overlapping trades.

KEY INSIGHT:
  With 512 trades/month and ~36-hour average holding period:
  - Concurrent positions ≈ 512 × (36/24) / 30 ≈ 25 positions
  - Kelly sizing applies to PORTFOLIO, not individual trades
  - Per-trade size = (Portfolio Kelly) / (Concurrent positions)

This simulation models trades arriving and resolving over calendar time,
with capital allocated across all open positions.
"""

import numpy as np
import pandas as pd
from collections import deque


def simulate_concurrent_portfolio(
    returns_bps,
    portfolio_kelly_fraction,
    initial_capital=10000,
    trades_per_month=512,
    avg_holding_hours=36,
    random_seed=42
):
    """
    Simulate portfolio with concurrent positions.
    
    Args:
        returns_bps: Array of trade returns in basis points
        portfolio_kelly_fraction: Target portfolio exposure (e.g., 0.25 for 25%)
        initial_capital: Starting capital
        trades_per_month: Expected trade frequency
        avg_holding_hours: Average time from entry to resolution
        random_seed: For reproducibility
    
    Returns:
        dict with equity curve, drawdowns, risk metrics
    """
    np.random.seed(random_seed)
    
    returns_decimal = np.array(returns_bps) / 10000
    n_trades = len(returns_decimal)
    
    # Shuffle returns to remove any ordering artifacts
    np.random.shuffle(returns_decimal)
    
    # Time parameters
    hours_per_month = 30 * 24
    trades_per_hour = trades_per_month / hours_per_month
    
    # Estimate simulation duration
    total_hours = int(n_trades / trades_per_hour) + int(avg_holding_hours * 2)
    
    # Expected concurrent positions
    expected_concurrent = trades_per_month * (avg_holding_hours / hours_per_month)
    
    # Per-position size to achieve target portfolio exposure
    per_position_fraction = portfolio_kelly_fraction / max(expected_concurrent, 1)
    
    print(f"  Expected concurrent positions: {expected_concurrent:.1f}")
    print(f"  Per-position size: {per_position_fraction*100:.2f}% of capital")
    print(f"  Target portfolio exposure: {portfolio_kelly_fraction*100:.1f}%")
    
    # Simulation state
    capital = initial_capital
    equity_curve = [capital]
    
    # Open positions: list of (entry_capital, return_decimal, hours_remaining)
    open_positions = deque()
    
    trade_idx = 0
    daily_returns = []
    current_day_start_capital = capital
    
    for hour in range(total_hours):
        if trade_idx >= n_trades and len(open_positions) == 0:
            break
        
        # --- POSITION ENTRY ---
        # Poisson arrival of new trades
        if trade_idx < n_trades:
            n_arrivals = np.random.poisson(trades_per_hour)
            
            for _ in range(min(n_arrivals, n_trades - trade_idx)):
                # Holding time: exponential around average
                holding_time = max(1, int(np.random.exponential(avg_holding_hours)))
                
                # Size based on CURRENT capital and target exposure
                # Adjust for current number of open positions
                current_open = len(open_positions)
                
                # Dynamic sizing: if we have fewer positions than expected, size up slightly
                # if more, size down. This keeps portfolio exposure near target.
                if current_open < expected_concurrent:
                    # Can take slightly larger position
                    size_multiplier = min(1.5, expected_concurrent / max(current_open + 1, 1))
                else:
                    # Scale down to avoid over-exposure
                    size_multiplier = expected_concurrent / (current_open + 1)
                
                position_size = capital * per_position_fraction * size_multiplier
                position_size = min(position_size, capital * 0.10)  # Cap at 10% per position
                
                open_positions.append({
                    'entry_capital': position_size,
                    'return': returns_decimal[trade_idx],
                    'hours_remaining': holding_time
                })
                
                trade_idx += 1
        
        # --- POSITION RESOLUTION ---
        # Decrement time and resolve completed positions
        positions_to_keep = deque()
        hour_pnl = 0
        
        for pos in open_positions:
            pos['hours_remaining'] -= 1
            
            if pos['hours_remaining'] <= 0:
                # Position resolves
                pnl = pos['entry_capital'] * pos['return']
                hour_pnl += pnl
            else:
                positions_to_keep.append(pos)
        
        open_positions = positions_to_keep
        
        # Update capital
        capital += hour_pnl
        capital = max(capital, 0)  # Can't go negative
        equity_curve.append(capital)
        
        # Track daily returns (every 24 hours)
        if hour > 0 and hour % 24 == 0:
            if current_day_start_capital > 0:
                daily_ret = (capital - current_day_start_capital) / current_day_start_capital
                daily_returns.append(daily_ret)
            current_day_start_capital = capital
    
    equity_curve = np.array(equity_curve)
    daily_returns = np.array(daily_returns)
    
    # Compute drawdowns
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / np.maximum(running_max, 1e-10)
    max_drawdown = np.min(drawdowns)
    
    # Total return
    total_return = (equity_curve[-1] - initial_capital) / initial_capital
    
    return {
        'equity_curve': equity_curve,
        'daily_returns': daily_returns,
        'final_equity': equity_curve[-1],
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'n_trades_processed': trade_idx,
        'expected_concurrent': expected_concurrent,
        'per_position_size': per_position_fraction
    }


def compute_risk_metrics_concurrent(sim_result, risk_free_rate=0.05):
    """
    Compute Sharpe, Sortino, Calmar from concurrent simulation.
    """
    daily_returns = sim_result['daily_returns']
    
    if len(daily_returns) < 10:
        return {'error': 'Insufficient daily returns for metrics'}
    
    # Annualize
    trading_days = 365  # Prediction markets trade every day
    rf_daily = (1 + risk_free_rate) ** (1/trading_days) - 1
    
    excess = daily_returns - rf_daily
    mean_excess = np.mean(excess)
    std_daily = np.std(daily_returns)
    
    # Sharpe
    sharpe = (mean_excess / std_daily) * np.sqrt(trading_days) if std_daily > 0 else 0
    
    # Sortino
    downside = daily_returns[daily_returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 0
    sortino = (mean_excess / downside_std) * np.sqrt(trading_days) if downside_std > 0 else np.inf
    
    # Calmar
    n_days = len(daily_returns)
    years = n_days / trading_days
    max_dd = sim_result['max_drawdown']
    
    if years > 0 and max_dd < 0:
        ann_return = (1 + sim_result['total_return']) ** (1/years) - 1
        calmar = ann_return / abs(max_dd)
    else:
        calmar = np.inf
    
    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'mean_daily_return': np.mean(daily_returns),
        'std_daily_return': std_daily,
        'n_positive_days': np.sum(daily_returns > 0),
        'n_negative_days': np.sum(daily_returns < 0),
        'n_days': n_days,
        'win_rate_daily': np.mean(daily_returns > 0)
    }


def run_full_analysis(returns_bps, initial_capital=10000, trades_per_month=512,
                      avg_holding_hours=36, risk_free_rate=0.05):
    """
    Run concurrent simulation at multiple portfolio Kelly levels.
    """
    returns_bps = np.array(returns_bps)
    
    # Basic stats
    n = len(returns_bps)
    win_rate = np.mean(returns_bps > 0)
    mean_bps = np.mean(returns_bps)
    std_bps = np.std(returns_bps)
    
    # Theoretical Kelly (for reference)
    mean_dec = mean_bps / 10000
    var_dec = (std_bps / 10000) ** 2
    theoretical_kelly = mean_dec / var_dec if var_dec > 0 else 0
    
    print("=" * 80)
    print("CONCURRENT POSITION KELLY ANALYSIS")
    print("=" * 80)
    
    print(f"\nTrade Distribution:")
    print(f"  n = {n:,}")
    print(f"  Win rate: {win_rate*100:.1f}%")
    print(f"  Mean return: {mean_bps:+.1f} bps")
    print(f"  Std dev: {std_bps:.1f} bps")
    print(f"  Theoretical full Kelly (sequential): {theoretical_kelly*100:.1f}%")
    
    print(f"\nPortfolio Parameters:")
    print(f"  Trades per month: {trades_per_month}")
    print(f"  Avg holding period: {avg_holding_hours} hours")
    
    # Test at different portfolio exposure levels
    results = {}
    
    for label, port_kelly in [('tenth', 0.10), ('quarter', 0.25), ('half', 0.50)]:
        print(f"\n{'-'*60}")
        print(f"Simulating {label.upper()} KELLY (portfolio exposure: {port_kelly*100:.0f}%)...")
        
        sim = simulate_concurrent_portfolio(
            returns_bps,
            portfolio_kelly_fraction=port_kelly,
            initial_capital=initial_capital,
            trades_per_month=trades_per_month,
            avg_holding_hours=avg_holding_hours
        )
        
        metrics = compute_risk_metrics_concurrent(sim, risk_free_rate)
        
        results[label] = {
            'simulation': sim,
            'metrics': metrics,
            'portfolio_kelly': port_kelly
        }
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Level':<12} {'Port %':<8} {'Pos %':<8} {'Return':<12} {'MaxDD':<10} {'Sharpe':<8} {'Sortino':<8} {'Calmar':<8}")
    print("-" * 80)
    
    for label in ['tenth', 'quarter', 'half']:
        r = results[label]
        sim = r['simulation']
        m = r['metrics']
        
        print(f"{label.upper():<12} "
              f"{r['portfolio_kelly']*100:>5.0f}%   "
              f"{sim['per_position_size']*100:>5.2f}%   "
              f"{sim['total_return']*100:>+8.1f}%   "
              f"{sim['max_drawdown']*100:>7.1f}%   "
              f"{m['sharpe']:>6.2f}   "
              f"{m['sortino']:>6.2f}   "
              f"{m['calmar']:>6.2f}")
    
    # Recommendations
    print("\n" + "-" * 80)
    print("INTERPRETATION")
    print("-" * 80)
    
    quarter = results['quarter']
    qm = quarter['metrics']
    qs = quarter['simulation']
    
    print(f"""
  At QUARTER KELLY (25% portfolio exposure):
    - Per-position size: {qs['per_position_size']*100:.2f}% of capital
    - You hold ~{qs['expected_concurrent']:.0f} concurrent positions
    - Daily win rate: {qm['win_rate_daily']*100:.1f}% of days profitable
    - Sharpe: {qm['sharpe']:.2f} (vs 2.2 in sequential simulation)
    
  The lower Sharpe reflects reality:
    - Concurrent positions mean correlated losses
    - Can't compound between every trade
    - Daily returns have actual variance
""")
    
    if qm['sharpe'] < 1.0:
        print("  ⚠️  Sharpe < 1.0 suggests strategy may not be viable after execution costs.")
    elif qm['sharpe'] < 1.5:
        print("  ✓  Sharpe 1.0-1.5 is reasonable for a retail/prop strategy.")
    else:
        print("  ✓  Sharpe > 1.5 is strong. Validate with paper trading.")
    
    return results


# ==============================================================================
# TEST WITH STRATEGY B DISTRIBUTION
# ==============================================================================

if __name__ == "__main__":
    # Generate synthetic returns matching Strategy B
    np.random.seed(42)
    n = 6150
    win_rate = 0.191
    target_mean = 104.8
    
    n_wins = int(n * win_rate)
    n_losses = n - n_wins
    
    # Distribution shape from the analysis
    losses = np.random.triangular(-4100, -2000, -200, n_losses)
    wins = np.random.triangular(3000, 7500, 9900, n_wins)
    
    returns_bps = np.concatenate([losses, wins])
    np.random.shuffle(returns_bps)
    returns_bps = returns_bps + (target_mean - np.mean(returns_bps))
    
    # Run analysis
    results = run_full_analysis(
        returns_bps,
        initial_capital=10000,
        trades_per_month=512,
        avg_holding_hours=36,  # ~1.5 days
        risk_free_rate=0.05
    )