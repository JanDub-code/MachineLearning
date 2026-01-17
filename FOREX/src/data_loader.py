"""
Data Loader for Dukascopy and OANDA forex data.

Provides unified interface for loading historical data from multiple sources.
Supports efficient Parquet loading and on-the-fly resampling.
"""

from pathlib import Path
from typing import Optional, List, Literal
import pandas as pd

# Data directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DUKASCOPY_DIR = DATA_DIR / "dukascopy"
OANDA_DIR = DATA_DIR / "raw"


def load_pair(
    pair: str = "EURUSD",
    source: Literal["dukascopy", "oanda", "auto"] = "auto",
    start: Optional[str] = None,
    end: Optional[str] = None,
    resample: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data for a currency pair.
    
    Args:
        pair: Currency pair symbol (e.g., 'EURUSD')
        source: Data source ('dukascopy', 'oanda', or 'auto' to prefer dukascopy)
        start: Start date filter (e.g., '2023-01-01')
        end: End date filter (e.g., '2024-01-01')
        resample: Resample to different timeframe (e.g., '5min', '1h', '1D')
    
    Returns:
        DataFrame with columns: open, high, low, close, volume
        Index: DatetimeIndex
    
    Example:
        # Load EURUSD 1-minute data
        df = load_pair('EURUSD')
        
        # Load and resample to 5-minute
        df = load_pair('EURUSD', resample='5min')
        
        # Load specific date range
        df = load_pair('EURUSD', start='2024-01-01', end='2024-06-01')
    """
    df = None
    
    # Try Dukascopy first (better historical coverage)
    if source in ["dukascopy", "auto"]:
        duka_path = DUKASCOPY_DIR / pair
        if duka_path.exists():
            parquet_files = list(duka_path.glob("*.parquet"))
            if parquet_files:
                # Load most recent file
                latest = max(parquet_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_parquet(latest)
                print(f"📊 Loaded {pair} from Dukascopy: {len(df):,} rows")
    
    # Fallback to OANDA
    if df is None and source in ["oanda", "auto"]:
        oanda_path = OANDA_DIR / pair
        if oanda_path.exists():
            parquet_files = list(oanda_path.glob("**/*.parquet"))
            if parquet_files:
                dfs = [pd.read_parquet(f) for f in parquet_files]
                df = pd.concat(dfs).sort_index().drop_duplicates()
                print(f"📊 Loaded {pair} from OANDA: {len(df):,} rows")
    
    if df is None:
        raise FileNotFoundError(
            f"No data found for {pair}. "
            f"Run download_dukascopy.py first, or check data directory: {DATA_DIR}"
        )
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Filter by date range
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    
    # Resample if requested
    if resample:
        df = resample_ohlcv(df, resample)
    
    return df


def load_multiple_pairs(
    pairs: List[str] = None,
    source: Literal["dukascopy", "oanda", "auto"] = "auto",
    start: Optional[str] = None,
    end: Optional[str] = None,
    resample: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load data for multiple currency pairs into a single DataFrame.
    
    Each pair gets prefixed columns (e.g., EURUSD_close, GBPUSD_close).
    
    Args:
        pairs: List of pairs (default: all available)
        source: Data source
        start: Start date filter
        end: End date filter
        resample: Resample to different timeframe
    
    Returns:
        DataFrame with columns prefixed by pair name
    """
    if pairs is None:
        # Auto-detect available pairs
        pairs = []
        if DUKASCOPY_DIR.exists():
            pairs.extend([d.name for d in DUKASCOPY_DIR.iterdir() if d.is_dir()])
        if not pairs and OANDA_DIR.exists():
            pairs.extend([d.name for d in OANDA_DIR.iterdir() if d.is_dir()])
        pairs = list(set(pairs))
    
    if not pairs:
        raise ValueError("No pairs specified and no data found")
    
    all_data = {}
    for pair in pairs:
        try:
            df = load_pair(pair, source, start, end, resample)
            df.columns = [f"{pair}_{col}" for col in df.columns]
            all_data[pair] = df
        except FileNotFoundError:
            print(f"⚠️  Skipping {pair}: no data found")
    
    if not all_data:
        raise ValueError("No data loaded for any pair")
    
    # Combine with outer join
    combined = pd.concat(all_data.values(), axis=1)
    combined.sort_index(inplace=True)
    
    # Forward fill gaps (different market hours)
    combined.ffill(inplace=True)
    combined.dropna(inplace=True)
    
    print(f"📊 Combined dataset: {combined.shape[0]:,} rows, {combined.shape[1]} columns")
    
    return combined


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.
    
    Args:
        df: DataFrame with open, high, low, close, volume columns
        timeframe: Target timeframe (e.g., '5min', '15min', '1h', '4h', '1D')
    
    Returns:
        Resampled DataFrame
    """
    ohlcv_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    
    # Filter only existing columns
    rules = {k: v for k, v in ohlcv_rules.items() if k in df.columns}
    
    resampled = df.resample(timeframe).agg(rules)
    resampled.dropna(inplace=True)
    
    return resampled


def get_available_pairs() -> dict:
    """
    Get list of available currency pairs and their data info.
    
    Returns:
        Dict mapping pair name to info dict with rows, date_range, size_mb
    """
    pairs = {}
    
    # Check Dukascopy
    if DUKASCOPY_DIR.exists():
        for pair_dir in DUKASCOPY_DIR.iterdir():
            if pair_dir.is_dir():
                parquet_files = list(pair_dir.glob("*.parquet"))
                if parquet_files:
                    latest = max(parquet_files, key=lambda x: x.stat().st_mtime)
                    df = pd.read_parquet(latest)
                    pairs[pair_dir.name] = {
                        "source": "dukascopy",
                        "rows": len(df),
                        "date_range": f"{df.index.min()} to {df.index.max()}",
                        "size_mb": latest.stat().st_size / (1024 * 1024),
                    }
    
    # Check OANDA
    if OANDA_DIR.exists():
        for pair_dir in OANDA_DIR.iterdir():
            if pair_dir.is_dir() and pair_dir.name not in pairs:
                parquet_files = list(pair_dir.glob("**/*.parquet"))
                if parquet_files:
                    dfs = [pd.read_parquet(f) for f in parquet_files]
                    df = pd.concat(dfs)
                    total_size = sum(f.stat().st_size for f in parquet_files)
                    pairs[pair_dir.name] = {
                        "source": "oanda",
                        "rows": len(df),
                        "date_range": f"{df.index.min()} to {df.index.max()}",
                        "size_mb": total_size / (1024 * 1024),
                    }
    
    return pairs


def print_data_summary():
    """Print summary of all available data."""
    pairs = get_available_pairs()
    
    if not pairs:
        print("❌ No data found. Run download_dukascopy.py first.")
        return
    
    print("\n" + "="*70)
    print("📊 AVAILABLE FOREX DATA")
    print("="*70)
    
    total_rows = 0
    total_size = 0
    
    for pair, info in sorted(pairs.items()):
        print(f"{pair:8} | {info['source']:10} | {info['rows']:>12,} rows | {info['size_mb']:>6.1f} MB")
        total_rows += info['rows']
        total_size += info['size_mb']
    
    print("-"*70)
    print(f"{'TOTAL':8} | {'':10} | {total_rows:>12,} rows | {total_size:>6.1f} MB")
    print("="*70)


if __name__ == "__main__":
    print_data_summary()
