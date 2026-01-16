import pandas as pd
import numpy as np


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def build_features(df: pd.DataFrame, horizon_minutes: int = 5) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    # Basic returns
    df["return_1"] = df["close"].pct_change().fillna(0)
    df["return_h"] = df["close"].pct_change(periods=horizon_minutes).shift(-horizon_minutes)

    # Volatility and ATR
    df["volatility"] = df["return_1"].rolling(60).std().fillna(0)
    df["atr"] = atr(df).bfill().fillna(0)

    # RSI and MACD
    df["rsi_14"] = rsi(df["close"]).bfill().fillna(50)
    macd_line, signal_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    # Moving averages
    for w in (3, 6, 12, 24, 48, 96):
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()

    # Time encodings
    df["minute"] = df["time"].dt.minute
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.dayofweek

    # Target: binary up/down
    df["target"] = (df["return_h"] > 0).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df
