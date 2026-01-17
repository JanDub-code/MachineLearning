"""
Regime Detector

Simple volatility-based regime classification for forex data.
Separates high-volatility and low-volatility periods to analyze
strategy performance in different market conditions.
"""

import pandas as pd
import numpy as np


def detect_volatility_regime(df: pd.DataFrame, lookback: int = 60) -> pd.Series:
    """
    Classify each bar as high_vol or low_vol based on ATR.
    
    Args:
        df: DataFrame with 'atr' column
        lookback: Lookback period for median calculation
    
    Returns:
        Series with 'high_vol' or 'low_vol' for each row
    """
    if 'atr' not in df.columns:
        raise ValueError("DataFrame must have 'atr' column")
    
    atr = df['atr']
    atr_median = atr.rolling(lookback, min_periods=lookback).median()
    
    regime = np.where(atr > atr_median, 'high_vol', 'low_vol')
    return pd.Series(regime, index=df.index)


def detect_trend_regime(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Classify each bar as trending or ranging based on ADX-like metric.
    
    Args:
        df: DataFrame with price data
        lookback: Lookback period
    
    Returns:
        Series with 'trending' or 'ranging' for each row
    """
    close = df['close']
    
    # Simple trend detection: compare current price to SMA
    sma = close.rolling(lookback, min_periods=lookback).mean()
    std = close.rolling(lookback, min_periods=lookback).std()
    
    # Z-score from mean
    zscore = (close - sma) / (std + 1e-8)
    zscore = zscore.abs()
    
    # If z-score > 1, consider it trending
    regime = np.where(zscore > 1.0, 'trending', 'ranging')
    return pd.Series(regime, index=df.index)


def detect_session(df: pd.DataFrame) -> pd.Series:
    """
    Classify each bar by trading session.
    
    Returns:
        Series with session name for each row
    """
    if 'time' not in df.columns:
        raise ValueError("DataFrame must have 'time' column")
    
    hour = df['time'].dt.hour
    
    # Define sessions (UTC times)
    # Sydney: 22-07, Tokyo: 00-09, London: 08-17, New York: 13-22
    def get_session(h):
        if 0 <= h < 8:
            return 'asia'
        elif 8 <= h < 13:
            return 'london'
        elif 13 <= h < 17:
            return 'overlap'  # London + NY
        elif 17 <= h < 22:
            return 'new_york'
        else:
            return 'asia'
    
    return hour.apply(get_session)


def add_regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all regime columns to DataFrame.
    
    Args:
        df: DataFrame with price data and ATR
    
    Returns:
        DataFrame with added regime columns
    """
    df = df.copy()
    
    if 'atr' in df.columns:
        df['vol_regime'] = detect_volatility_regime(df)
    
    if 'close' in df.columns:
        df['trend_regime'] = detect_trend_regime(df)
    
    if 'time' in df.columns:
        df['session'] = detect_session(df)
    
    return df


def split_by_regime(trades: list, df: pd.DataFrame, regime_col: str = 'vol_regime') -> dict:
    """
    Split trades by regime at entry time.
    
    Args:
        trades: List of trade dicts with 'entry_time'
        df: DataFrame with regime columns and 'time'
        regime_col: Column to split by
    
    Returns:
        Dict mapping regime name to list of trades
    """
    if regime_col not in df.columns:
        return {'all': trades}
    
    result = {}
    
    for trade in trades:
        entry_time = trade.get('entry_time')
        if entry_time is None:
            continue
        
        # Find matching row
        mask = df['time'] == entry_time
        if mask.any():
            regime = df.loc[mask, regime_col].iloc[0]
        else:
            regime = 'unknown'
        
        if regime not in result:
            result[regime] = []
        result[regime].append(trade)
    
    return result
