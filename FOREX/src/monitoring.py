"""Monitoring utilities for rolling performance checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .risk_metrics import calculate_all_metrics


@dataclass
class RollingMetrics:
    window: int
    metrics: Dict


def _to_trade_df(trades: List[Dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades).copy()
    if "exit_time" in df.columns:
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    return df


def rolling_trade_metrics(
    trades: List[Dict],
    window: int,
    daily_pnl: Optional[List[float]] = None,
) -> RollingMetrics:
    if not trades or len(trades) < window:
        return RollingMetrics(window=window, metrics={})
    last_trades = trades[-window:]
    metrics = calculate_all_metrics(last_trades, daily_pnl)
    return RollingMetrics(window=window, metrics=metrics)


def trade_frequency(trades: List[Dict]) -> float:
    """Trades per month based on first/last trade timestamps."""
    if not trades:
        return 0.0
    df = _to_trade_df(trades)
    if df.empty or "exit_time" not in df.columns:
        return 0.0
    df = df.dropna(subset=["exit_time"])
    if df.empty:
        return 0.0
    start = df["exit_time"].min()
    end = df["exit_time"].max()
    days = max((end - start).days, 1)
    months = days / 30.0
    return float(len(df) / months)


def recent_drawdown_ratio(
    recent_max_dd: float,
    baseline_max_dd: float,
) -> float:
    if baseline_max_dd <= 0:
        return 0.0
    return recent_max_dd / baseline_max_dd
