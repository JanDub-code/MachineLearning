"""
Risk Metrics Module

Provides comprehensive risk analysis for backtest results:
- Equity curve calculation
- Drawdown analysis
- Tail risk metrics (VaR, Expected Shortfall)
- Trade-level statistics
"""

from typing import List, Dict
import numpy as np
import pandas as pd


def equity_curve(trades: List[Dict], initial_equity: float = 0.0) -> pd.Series:
    """
    Calculate cumulative equity curve from trades.
    
    Args:
        trades: List of trade dicts with 'pnl_pips' and optionally 'exit_time'
        initial_equity: Starting equity (default 0 for pips-based)
    
    Returns:
        Series with equity values indexed by trade number
    """
    if not trades:
        return pd.Series([initial_equity])
    
    pnls = [t['pnl_pips'] for t in trades]
    equity = np.cumsum([initial_equity] + pnls)
    return pd.Series(equity)


def max_drawdown(equity: pd.Series) -> float:
    """
    Calculate maximum drawdown in pips.
    
    Args:
        equity: Equity curve series
    
    Returns:
        Maximum drawdown (positive number = loss from peak)
    """
    if len(equity) == 0:
        return 0.0
    
    peak = equity.expanding().max()
    drawdown = peak - equity
    return float(drawdown.max())


def max_drawdown_pct(equity: pd.Series, initial_equity: float = 1000.0) -> float:
    """
    Calculate maximum drawdown as percentage.
    
    Args:
        equity: Equity curve (in pips)
        initial_equity: Starting equity in pips for percentage calculation
    
    Returns:
        Maximum drawdown as percentage (e.g., 0.15 = 15%)
    """
    if len(equity) == 0:
        return 0.0
    
    # Convert pips to equity value
    equity_value = initial_equity + equity
    peak = equity_value.expanding().max()
    drawdown_pct = (peak - equity_value) / peak
    return float(drawdown_pct.max())


def max_drawdown_duration(equity: pd.Series) -> int:
    """
    Calculate maximum drawdown duration in number of trades.
    
    Returns:
        Number of trades to recover from worst drawdown
    """
    if len(equity) == 0:
        return 0
    
    peak = equity.expanding().max()
    underwater = equity < peak
    
    if not underwater.any():
        return 0
    
    # Find longest consecutive underwater period
    max_duration = 0
    current_duration = 0
    
    for is_underwater in underwater:
        if is_underwater:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return max_duration


def worst_day(daily_pnl: List[float]) -> float:
    """
    Return the worst single day PnL.
    
    Args:
        daily_pnl: List of daily PnL values
    
    Returns:
        Worst (most negative) daily PnL
    """
    if not daily_pnl:
        return 0.0
    return float(min(daily_pnl))


def best_day(daily_pnl: List[float]) -> float:
    """Return the best single day PnL."""
    if not daily_pnl:
        return 0.0
    return float(max(daily_pnl))


def var_95(returns: List[float]) -> float:
    """
    Calculate 95% Value at Risk (VaR).
    
    Returns:
        The loss (positive number) at the 5th percentile
    """
    if not returns or len(returns) < 5:
        return 0.0
    return float(-np.percentile(returns, 5))


def expected_shortfall(returns: List[float], percentile: float = 5) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR).
    
    Average loss in the worst X% of cases.
    
    Args:
        returns: List of returns/PnL values
        percentile: Percentile threshold (default 5 = worst 5%)
    
    Returns:
        Expected shortfall (positive = loss)
    """
    if not returns or len(returns) < 5:
        return 0.0
    
    threshold = np.percentile(returns, percentile)
    tail_losses = [r for r in returns if r <= threshold]
    
    if not tail_losses:
        return 0.0
    
    return float(-np.mean(tail_losses))


def expectancy_per_trade(trades: List[Dict]) -> float:
    """
    Calculate expected PnL per trade.
    
    Args:
        trades: List of trade dicts with 'pnl_pips'
    
    Returns:
        Average PnL per trade
    """
    if not trades:
        return 0.0
    
    pnls = [t['pnl_pips'] for t in trades]
    return float(np.mean(pnls))


def profit_factor(trades: List[Dict]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        trades: List of trade dicts with 'pnl_pips'
    
    Returns:
        Profit factor (>1 = profitable)
    """
    if not trades:
        return 0.0
    
    pnls = [t['pnl_pips'] for t in trades]
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def win_rate(trades: List[Dict]) -> float:
    """Calculate percentage of winning trades."""
    if not trades:
        return 0.0
    
    wins = sum(1 for t in trades if t['pnl_pips'] > 0)
    return wins / len(trades)


def sharpe_ratio(daily_pnl: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio from daily returns.
    
    Args:
        daily_pnl: List of daily PnL values
        risk_free_rate: Daily risk-free rate (default 0)
    
    Returns:
        Annualized Sharpe ratio
    """
    if not daily_pnl or len(daily_pnl) < 2:
        return 0.0
    
    excess_returns = np.array(daily_pnl) - risk_free_rate
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    # Annualize (assuming 252 trading days)
    return float(mean_return / std_return * np.sqrt(252))


def calculate_all_metrics(trades: List[Dict], daily_pnl: List[float] = None) -> Dict:
    """
    Calculate all risk metrics for a backtest.
    
    Args:
        trades: List of trade dicts with 'pnl_pips'
        daily_pnl: Optional list of daily PnL (calculated from trades if not provided)
    
    Returns:
        Dict with all metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'total_pnl': 0.0,
            'expectancy': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'max_dd_duration': 0,
            'worst_day': 0.0,
            'best_day': 0.0,
            'var_95': 0.0,
            'expected_shortfall': 0.0,
            'sharpe_ratio': 0.0,
        }
    
    eq = equity_curve(trades)
    pnls = [t['pnl_pips'] for t in trades]
    
    # Use provided daily_pnl or fall back to trade pnls
    daily = daily_pnl if daily_pnl else pnls
    
    return {
        'total_trades': len(trades),
        'total_pnl': sum(pnls),
        'expectancy': expectancy_per_trade(trades),
        'win_rate': win_rate(trades),
        'profit_factor': profit_factor(trades),
        'max_drawdown': max_drawdown(eq),
        'max_dd_duration': max_drawdown_duration(eq),
        'worst_day': worst_day(daily),
        'best_day': best_day(daily),
        'var_95': var_95(pnls),
        'expected_shortfall': expected_shortfall(pnls),
        'sharpe_ratio': sharpe_ratio(daily),
    }


def format_metrics_report(metrics: Dict) -> str:
    """Format metrics as a readable string."""
    lines = [
        "=" * 50,
        "RISK METRICS REPORT",
        "=" * 50,
        f"Total Trades:       {metrics['total_trades']}",
        f"Total PnL:          {metrics['total_pnl']:+.1f} pips",
        f"Expectancy/Trade:   {metrics['expectancy']:+.2f} pips",
        f"Win Rate:           {metrics['win_rate']:.1%}",
        f"Profit Factor:      {metrics['profit_factor']:.2f}",
        "-" * 50,
        f"Max Drawdown:       {metrics['max_drawdown']:.1f} pips",
        f"DD Duration:        {metrics['max_dd_duration']} trades",
        f"Worst Day:          {metrics['worst_day']:+.1f} pips",
        f"Best Day:           {metrics['best_day']:+.1f} pips",
        "-" * 50,
        f"VaR (95%):          {metrics['var_95']:.1f} pips",
        f"Expected Shortfall: {metrics['expected_shortfall']:.1f} pips",
        f"Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}",
        "=" * 50,
    ]
    return "\n".join(lines)
