#!/usr/bin/env python3
"""
Dukascopy Direct Downloader - EURUSD 1m candles

Downloads tick data directly from Dukascopy servers and converts to 1m OHLCV.
No external dependencies needed beyond requests and pandas.

Data range: 2025-10-17 to 2026-01-17 (3 months)
"""

import requests
import struct
import lzma
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from io import BytesIO
import time

# Configuration
PAIR = "EURUSD"
START_DATE = datetime(2025, 10, 17)
END_DATE = datetime(2026, 1, 17)
OUTPUT_DIR = Path(__file__).parent / "data" / "dukascopy" / PAIR

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


def download_day(pair: str, date: datetime) -> pd.DataFrame:
    """Download all hours for a specific day."""
    all_ticks = []
    
    for hour in range(24):
        dt = date.replace(hour=hour, minute=0, second=0, microsecond=0)
        ticks = download_hour(pair, dt)
        all_ticks.extend(ticks)
        
        # Small delay to be nice to servers
        time.sleep(0.1)
    
    return ticks_to_ohlcv(all_ticks)


def main():
    """Main download function."""
    print("=" * 60)
    print("🌐 DUKASCOPY EURUSD DOWNLOADER")
    print("=" * 60)
    print(f"Pair: {PAIR}")
    print(f"Range: {START_DATE.date()} → {END_DATE.date()}")
    print(f"Timeframe: 1 minute candles")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Calculate total days
    total_days = (END_DATE - START_DATE).days + 1
    daily_files = []
    
    current = START_DATE
    day_count = 0
    
    while current <= END_DATE:
        day_count += 1
        weekday = current.weekday()
        
        # Skip weekends (Sat=5, Sun=6) - forex market closed
        if weekday in [5, 6]:
            print(f"[{day_count:3}/{total_days}] {current.date()} - Weekend, skipping")
            current += timedelta(days=1)
            continue
        
        # Check if already downloaded
        day_file = OUTPUT_DIR / f"{current.date()}.csv"
        if day_file.exists():
            print(f"[{day_count:3}/{total_days}] {current.date()} - Already exists, skipping")
            daily_files.append(day_file)
            current += timedelta(days=1)
            continue
        
        print(f"[{day_count:3}/{total_days}] {current.date()} - Downloading...", end=" ", flush=True)
        
        df = download_day(PAIR, current)
        
        if len(df) > 0:
            # Save immediately!
            df.to_csv(day_file)
            daily_files.append(day_file)
            print(f"✅ {len(df)} candles saved")
        else:
            print(f"⚠️  No data (holiday?)")
        
        current += timedelta(days=1)
        
        # Rate limit
        time.sleep(0.3)
    
    if not daily_files:
        print("❌ No data downloaded!")
        return
    
    # Combine all days
    print(f"\n📊 Combining {len(daily_files)} daily files...")
    all_data = [pd.read_csv(f, index_col=0, parse_dates=True) for f in daily_files]
    combined = pd.concat(all_data)
    combined.sort_index(inplace=True)
    combined = combined[~combined.index.duplicated(keep='first')]
    
    # Save as Parquet (efficient)
    parquet_path = OUTPUT_DIR / f"{PAIR}_1m_{START_DATE.date()}_{END_DATE.date()}.parquet"
    combined.to_parquet(parquet_path, compression='zstd')
    
    # Also save combined CSV
    csv_path = OUTPUT_DIR / f"{PAIR}_1m_{START_DATE.date()}_{END_DATE.date()}.csv"
    combined.to_csv(csv_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Total candles: {len(combined):,}")
    print(f"Date range: {combined.index.min()} → {combined.index.max()}")
    print(f"Parquet: {parquet_path} ({parquet_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"CSV: {csv_path} ({csv_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print("=" * 60)
    
    # Quick data preview
    print("\n� Data preview:")
    print(combined.head(10))
    print("...")
    print(combined.tail(5))


if __name__ == "__main__":
    main()
