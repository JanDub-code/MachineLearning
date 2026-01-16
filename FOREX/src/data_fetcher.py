import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
import yfinance as yf

from .configs import load_settings


@dataclass
class FetchResult:
    pair: str
    granularity: str
    candles: pd.DataFrame
    source: str = "oanda"


class OandaDataFetcher:
    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        self.cfg = self.settings.broker
        self.paths = self.settings.paths

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

    def fetch_historical(
        self,
        pair: str,
        granularity: str = "M1",
        start: Optional[dt.datetime] = None,
        end: Optional[dt.datetime] = None,
        max_candles: int = 10_000,
    ) -> FetchResult:
        """Download historical candles. Caller must handle pagination for >max_candles."""
        if not self.cfg.api_key:
            raise ValueError("api_key is empty; set BrokerConfig.api_key locally")

        params = {
            "price": "M",  # midpoint
            "granularity": granularity,
            "count": max_candles,
        }
        if start:
            params["from"] = start.isoformat()
        if end:
            params["to"] = end.isoformat()

        url = f"{self.cfg.rest_endpoint}/instruments/{pair}/candles"
        resp = requests.get(url, headers=self._auth_headers(), params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        candles = payload.get("candles", [])
        df = self._candles_to_df(candles)
        return FetchResult(pair=pair, granularity=granularity, candles=df)

    def _candles_to_df(self, candles) -> pd.DataFrame:
        records = []
        for c in candles:
            mid = c.get("mid", {})
            records.append(
                {
                    "time": pd.to_datetime(c.get("time")),
                    "open": float(mid.get("o")),
                    "high": float(mid.get("h")),
                    "low": float(mid.get("l")),
                    "close": float(mid.get("c")),
                    "complete": bool(c.get("complete", True)),
                    "volume": int(c.get("volume", 0)),
                }
            )
        df = pd.DataFrame.from_records(records)
        df = df.sort_values("time").reset_index(drop=True)
        return df

    def save_raw(self, result: FetchResult) -> Path:
        date_tag = dt.datetime.utcnow().strftime("%Y-%m-%d")
        out_dir = Path(self.paths.raw_dir) / result.pair / date_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"{result.granularity}.parquet"
        result.candles.to_parquet(fname, index=False)
        return fname


def fetch_and_store(pair: str = "EUR_USD", granularity: str = "M1", max_candles: int = 10_000) -> Path:
    fetcher = OandaDataFetcher()
    res = fetcher.fetch_historical(pair=pair, granularity=granularity, max_candles=max_candles)
    return fetcher.save_raw(res)


# ---------------------------------------------------------------------------
# Free fallback: yfinance (no account required, limited history on 1m)
# ---------------------------------------------------------------------------

def fetch_yfinance_fx(pair: str = "EURUSD", interval: str = "1m", period: str = "7d") -> Path:
    """Quick download of FX via Yahoo Finance. Pair format: EURUSD -> EURUSD=X."""
    settings = load_settings()
    ticker = f"{pair}=X"
    hist = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No data returned for {ticker} interval={interval} period={period}")

    # Handle multi-index columns from newer yfinance versions
    if isinstance(hist.columns, pd.MultiIndex):
        # Flatten multi-index columns, taking just the first level
        hist.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in hist.columns]
    else:
        hist.columns = [col.lower() for col in hist.columns]
    
    df = hist.reset_index()
    
    # Standardize the datetime column name
    datetime_col = None
    for col in df.columns:
        if 'datetime' in str(col).lower() or 'date' in str(col).lower():
            datetime_col = col
            break
    
    if datetime_col:
        df = df.rename(columns={datetime_col: "time"})
    
    # Ensure we have the standard column names
    col_mapping = {
        'adj close': 'adj_close',
        'adjclose': 'adj_close',
    }
    df = df.rename(columns=col_mapping)
    
    # Yahoo 1m limited to ~7d; higher intervals (5m, 15m, 1h, 1d) allow longer periods.
    df["time"] = pd.to_datetime(df["time"])
    # Remove timezone info if present for consistency
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)
    
    df = df.sort_values("time").reset_index(drop=True)

    date_tag = dt.datetime.utcnow().strftime("%Y-%m-%d")
    out_dir = Path(settings.paths.raw_dir) / pair / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"yf_{interval}.parquet"
    df.to_parquet(fname, index=False)
    return fname

