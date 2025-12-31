#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stažení 30 tickerů (10 per sektor) pro CleanSolution pipeline.
"""

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf

# === KONFIGURACE ===
START_DATE = "2015-01-01"
END_DATE = "2025-10-01"
DATA_DIR = r"c:\Users\Bc. Jan Dub\Desktop\GIT\MachineLearning\CleanSolution\data_10y"

# 30 tickerů: 10 per sektor (ověřené S&P 500 komponenty)
TICKERS = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", 
        "AVGO", "ORCL", "CSCO", "ADBE", "CRM"
    ],
    "Consumer": [
        "AMZN", "TSLA", "HD", "MCD", "NKE",
        "SBUX", "TGT", "LOW", "PG", "KO"
    ],
    "Industrials": [
        "CAT", "HON", "UPS", "BA", "GE",
        "RTX", "DE", "LMT", "MMM", "UNP"
    ]
}

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI indikátor"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series: pd.Series) -> pd.DataFrame:
    """MACD indikátor"""
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal_line
    return pd.DataFrame({
        'macd': macd,
        'macd_signal': signal_line,
        'macd_hist': histogram
    })

def download_ticker(ticker: str) -> pd.DataFrame:
    """Stáhne a zpracuje data pro jeden ticker"""
    try:
        yf_ticker = yf.Ticker(ticker)
        hist = yf_ticker.history(start=START_DATE, end=END_DATE, interval="1d")
        
        if hist.empty or len(hist) < 20:
            return pd.DataFrame()
        
        # Agregace na měsíční data
        monthly = hist.resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'mean',
            'Dividends': 'sum',
            'Stock Splits': lambda x: 1 if x.sum() > 0 else 0
        })
        
        monthly = monthly.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Volume': 'volume', 'Dividends': 'dividends', 'Stock Splits': 'split_occurred'
        })
        
        # Technické indikátory
        monthly['volatility'] = (monthly['high'] - monthly['low']) / monthly['close']
        monthly['returns'] = monthly['close'].pct_change()
        monthly['rsi_14'] = calculate_rsi(monthly['close'], period=14)
        
        macd_df = calculate_macd(monthly['close'])
        monthly['macd'] = macd_df['macd']
        monthly['macd_signal'] = macd_df['macd_signal']
        monthly['macd_hist'] = macd_df['macd_hist']
        
        monthly['sma_3'] = monthly['close'].rolling(window=3).mean()
        monthly['sma_6'] = monthly['close'].rolling(window=6).mean()
        monthly['sma_12'] = monthly['close'].rolling(window=12).mean()
        monthly['ema_3'] = monthly['close'].ewm(span=3, adjust=False).mean()
        monthly['ema_6'] = monthly['close'].ewm(span=6, adjust=False).mean()
        monthly['ema_12'] = monthly['close'].ewm(span=12, adjust=False).mean()
        monthly['volume_change'] = monthly['volume'].pct_change()
        
        monthly = monthly.reset_index()
        monthly.rename(columns={'Date': 'date'}, inplace=True)
        monthly['ticker'] = ticker
        
        return monthly
        
    except Exception as e:
        log(f"  ERROR {ticker}: {e}")
        return pd.DataFrame()

def main():
    log("=" * 60)
    log("STAŽENÍ 30 TICKERŮ PRO CLEANSOLUTION")
    log("=" * 60)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    all_data = []
    sector_data = {s: [] for s in TICKERS.keys()}
    
    total_tickers = sum(len(t) for t in TICKERS.values())
    current = 0
    
    for sector, tickers in TICKERS.items():
        log(f"\n--- {sector} ({len(tickers)} tickerů) ---")
        
        # Uložení tickerů do souboru
        ticker_file = os.path.join(DATA_DIR, f"{sector}_tickers.txt")
        with open(ticker_file, 'w') as f:
            f.write('\n'.join(tickers))
        
        for ticker in tickers:
            current += 1
            log(f"  [{current}/{total_tickers}] {ticker}...")
            
            df = download_ticker(ticker)
            
            if not df.empty:
                df['sector'] = sector
                all_data.append(df)
                sector_data[sector].append(df)
                log(f"    OK: {len(df)} měsíců")
            else:
                log(f"    SKIP: žádná data")
            
            time.sleep(0.5)  # Rate limiting
    
    # Uložení dat
    log("\n" + "=" * 60)
    log("UKLÁDÁNÍ DAT")
    log("=" * 60)
    
    if all_data:
        # Všechny sektory
        combined = pd.concat(all_data, ignore_index=True)
        combined_file = os.path.join(DATA_DIR, "all_sectors_full_10y.csv")
        combined.to_csv(combined_file, index=False)
        log(f"Uloženo: {combined_file}")
        log(f"  - {len(combined)} řádků, {combined['ticker'].nunique()} tickerů")
        
        # Per-sector
        for sector, dfs in sector_data.items():
            if dfs:
                sector_df = pd.concat(dfs, ignore_index=True)
                sector_file = os.path.join(DATA_DIR, f"{sector}_full_10y.csv")
                sector_df.to_csv(sector_file, index=False)
                log(f"Uloženo: {sector_file} ({len(sector_df)} řádků)")
    
    log("\n" + "=" * 60)
    log("HOTOVO!")
    log("=" * 60)

if __name__ == "__main__":
    main()
