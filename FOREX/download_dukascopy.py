#!/usr/bin/env python3
"""
Dukascopy Direct Downloader - 1m candles

Downloads tick data directly from Dukascopy servers and converts to 1m OHLCV.
No external dependencies needed beyond requests and pandas.

Defaults: EURUSD, 2025-10-17 to 2026-01-17 (3 months)
"""

import argparse
import lzma
import struct
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import requests

from src.portfolio_baskets import get_basket

# Defaults
DEFAULT_PAIR = "EURUSD"
DEFAULT_START_DATE = datetime(2025, 10, 17)
DEFAULT_END_DATE = datetime(2026, 1, 17)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data" / "dukascopy"

# Dukascopy URL pattern for tick data
# Format: https://datafeed.dukascopy.com/datafeed/{PAIR}/{YYYY}/{MM-1}/{DD}/{HH}h_ticks.bi5
BASE_URL = "https://datafeed.dukascopy.com/datafeed"


def download_hour(pair: str, dt: datetime) -> list:
    """
    Download tick data for a specific hour.
    
    Dukascopy stores data in .bi5 files (LZMA compressed binary).
    Each tick is 20 bytes: time_ms(4), ask(4), bid(4), ask_vol(4), bid_vol(4)
    
    Returns list of tick dicts or empty list if no data.
    """
    # Dukascopy uses 0-indexed months!
    month = dt.month - 1
    url = f"{BASE_URL}/{pair}/{dt.year}/{month:02d}/{dt.day:02d}/{dt.hour:02d}h_ticks.bi5"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return []
        
        # Decompress LZMA
        try:
            data = lzma.decompress(response.content)
        except lzma.LZMAError:
            return []
        
        if len(data) == 0:
            return []
        
        # Parse binary tick data
        # Each tick: uint32 time_ms, uint32 ask, uint32 bid, float ask_vol, float bid_vol
        ticks = []
        tick_size = 20
        
        # Point value for EURUSD (5 decimal places)
        point = 0.00001
        
        for i in range(0, len(data), tick_size):
            if i + tick_size > len(data):
                break
            
            chunk = data[i:i + tick_size]
            time_ms, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack('>IIIff', chunk)
            
            # Convert to actual prices
            tick_time = dt.replace(minute=0, second=0, microsecond=0) + timedelta(milliseconds=time_ms)
            ask = ask_raw * point
            bid = bid_raw * point
            mid = (ask + bid) / 2
            
            ticks.append({
                'time': tick_time,
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'volume': ask_vol + bid_vol
            })
        
        return ticks
        
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return []


def ticks_to_ohlcv(ticks: list, timeframe: str = '1min') -> pd.DataFrame:
    """Convert tick data to OHLCV candles."""
    if not ticks:
        return pd.DataFrame()
    
    df = pd.DataFrame(ticks)
    df.set_index('time', inplace=True)
    
    # Resample to candles using mid price
    ohlcv = df['mid'].resample(timeframe).ohlc()
    ohlcv['volume'] = df['volume'].resample(timeframe).sum()
    
    ohlcv.dropna(inplace=True)
    return ohlcv


def download_day(pair: str, date: datetime, sleep_hour: float = 0.1) -> pd.DataFrame:
    """Download all hours for a specific day."""
    all_ticks = []
    
    for hour in range(24):
        dt = date.replace(hour=hour, minute=0, second=0, microsecond=0)
        ticks = download_hour(pair, dt)
        all_ticks.extend(ticks)
        
        # Small delay to be nice to servers
        time.sleep(sleep_hour)
    
    return ticks_to_ohlcv(all_ticks)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dukascopy downloader (1m candles)")
    parser.add_argument("--pairs", default=DEFAULT_PAIR, help="Comma-separated pairs (e.g., EURUSD,USDJPY)")
    parser.add_argument("--basket", default=None, help="Named basket (basket_6, basket_8). Overrides --pairs")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.date().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.date().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--output-dir", default=None, help="Base output dir (default: data/dukascopy)")
    parser.add_argument("--sleep-hour", type=float, default=0.1, help="Delay after each hour (seconds)")
    parser.add_argument("--sleep-day", type=float, default=0.3, help="Delay after each day (seconds)")
    return parser.parse_args()


def _resolve_pairs(args: argparse.Namespace) -> List[str]:
    if args.basket:
        return get_basket(args.basket)
    return [p.strip().upper() for p in args.pairs.split(",") if p.strip()]


def download_pair(
    pair: str,
    start_date: datetime,
    end_date: datetime,
    output_base: Path,
    sleep_hour: float,
    sleep_day: float,
) -> Path:
    output_dir = output_base / pair
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"🌐 DUKASCOPY DOWNLOADER: {pair}")
    print("=" * 60)
    print(f"Range: {start_date.date()} → {end_date.date()}")
    print("Timeframe: 1 minute candles")
    print(f"Output: {output_dir}")
    print("=" * 60)

    total_days = (end_date - start_date).days + 1
    daily_files: List[Path] = []

    current = start_date
    day_count = 0

    while current <= end_date:
        day_count += 1
        weekday = current.weekday()

        if weekday in [5, 6]:
            print(f"[{day_count:3}/{total_days}] {current.date()} - Weekend, skipping")
            current += timedelta(days=1)
            continue

        day_file = output_dir / f"{current.date()}.csv"
        if day_file.exists():
            print(f"[{day_count:3}/{total_days}] {current.date()} - Already exists, skipping")
            daily_files.append(day_file)
            current += timedelta(days=1)
            continue

        print(f"[{day_count:3}/{total_days}] {current.date()} - Downloading...", end=" ", flush=True)

        df = download_day(pair, current, sleep_hour=sleep_hour)

        if len(df) > 0:
            df.to_csv(day_file)
            daily_files.append(day_file)
            print(f"✅ {len(df)} candles saved")
        else:
            print("⚠️  No data (holiday?)")

        current += timedelta(days=1)
        time.sleep(sleep_day)

    if not daily_files:
        print("❌ No data downloaded!")
        return output_dir

    print(f"\n📊 Combining {len(daily_files)} daily files...")
    all_data = [pd.read_csv(f, index_col=0, parse_dates=True) for f in daily_files]
    combined = pd.concat(all_data)
    combined.sort_index(inplace=True)
    combined = combined[~combined.index.duplicated(keep="first")]

    parquet_path = output_dir / f"{pair}_1m_{start_date.date()}_{end_date.date()}.parquet"
    combined.to_parquet(parquet_path, compression="zstd")

    csv_path = output_dir / f"{pair}_1m_{start_date.date()}_{end_date.date()}.csv"
    combined.to_csv(csv_path)

    print("\n" + "=" * 60)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Total candles: {len(combined):,}")
    print(f"Date range: {combined.index.min()} → {combined.index.max()}")
    print(f"Parquet: {parquet_path} ({parquet_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"CSV: {csv_path} ({csv_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print("=" * 60)

    print("\nData preview:")
    print(combined.head(5))
    print("...")
    print(combined.tail(3))

    return parquet_path


def main():
    args = _parse_args()
    pairs = _resolve_pairs(args)
    start_date = datetime.fromisoformat(args.start_date)
    end_date = datetime.fromisoformat(args.end_date)
    output_base = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    for pair in pairs:
        download_pair(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            output_base=output_base,
            sleep_hour=args.sleep_hour,
            sleep_day=args.sleep_day,
        )


if __name__ == "__main__":
    main()
