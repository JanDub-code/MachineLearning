#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TESTOVACÍ VERZE - Příprava 10letých OHLCV dat s technickými indikátory.
Stahuje pouze 10 firem z každého sektoru pro rychlé testování.
"""

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict

# === KONFIGURACE ===
START_DATE = "2015-01-01"
END_DATE = "2025-10-01"
DATA_DIR = "./data_10y_test"  # Jiná složka pro test data

# PRO TESTOVÁNÍ: Počet firem na sektor
TEST_LIMIT = 10  # 10 firem/sektor = 30 firem celkem

# Sector mapping
SECTOR_BUCKET_MAP = {
    "Information Technology": "Technology",
    "Communication Services": "Technology",
    "Consumer Discretionary": "Consumer",
    "Consumer Staples": "Consumer",
    "Industrials": "Industrials",
    "Energy": "Industrials",
    "Materials": "Industrials",
}

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def get_sp500_constituents() -> pd.DataFrame:
    """Získat S&P 500 seznam z Wikipedie"""
    import io
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        df = tables[0].copy()
        df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        return df[["ticker", "name", "sector"]]
    except Exception as e:
        log(f"❌ Chyba při stahování S&P 500: {e}")
        return pd.DataFrame()

def filter_tickers_by_buckets(df: pd.DataFrame, target_buckets: List[str]) -> Dict[str, List[str]]:
    """Filtruje a rozděluje tickery podle target buckets"""
    df = df.copy()
    df["bucket"] = df["sector"].map(SECTOR_BUCKET_MAP)
    df = df[df["bucket"].isin(target_buckets)]
    
    result = {}
    for bucket in target_buckets:
        tickers = df[df["bucket"] == bucket]["ticker"].tolist()
        
        # Omezení pro testování
        if TEST_LIMIT is not None and len(tickers) > TEST_LIMIT:
            tickers = tickers[:TEST_LIMIT]
        
        result[bucket] = tickers
    
    return result

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Vypočítá RSI indikátor"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Vypočítá MACD indikátor"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'macd': macd,
        'macd_signal': signal_line,
        'macd_hist': histogram
    })

def download_and_process_ticker(ticker: str) -> pd.DataFrame:
    """
    Stáhne DENNÍ data pro ticker, agreguje na MĚSÍČNÍ a přidá technické indikátory.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        
        # Stažení denních dat
        hist = yf_ticker.history(start=START_DATE, end=END_DATE, interval="1d")
        
        if hist.empty or len(hist) < 20:
            return pd.DataFrame()
        
        # Agregace na měsíční data
        monthly = hist.resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',  # Adj Close (yfinance upravuje automaticky)
            'Volume': 'mean',
            'Dividends': 'sum',
            'Stock Splits': lambda x: 1 if x.sum() > 0 else 0
        })
        
        # Přejmenování sloupců
        monthly = monthly.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Dividends': 'dividends',
            'Stock Splits': 'split_occurred'
        })
        
        # === FEATURE ENGINEERING ===
        
        # 1. Volatilita (normalizovaný range)
        monthly['volatility'] = (monthly['high'] - monthly['low']) / monthly['close']
        
        # 2. Returns (měsíční % změna)
        monthly['returns'] = monthly['close'].pct_change()
        
        # 3. RSI (14 period)
        monthly['rsi_14'] = calculate_rsi(monthly['close'], period=14)
        
        # 4. MACD
        macd_df = calculate_macd(monthly['close'])
        monthly['macd'] = macd_df['macd']
        monthly['macd_signal'] = macd_df['macd_signal']
        monthly['macd_hist'] = macd_df['macd_hist']
        
        # 5. Simple Moving Averages
        monthly['sma_3'] = monthly['close'].rolling(window=3).mean()
        monthly['sma_6'] = monthly['close'].rolling(window=6).mean()
        monthly['sma_12'] = monthly['close'].rolling(window=12).mean()
        
        # 6. Exponential Moving Averages
        monthly['ema_3'] = monthly['close'].ewm(span=3, adjust=False).mean()
        monthly['ema_6'] = monthly['close'].ewm(span=6, adjust=False).mean()
        monthly['ema_12'] = monthly['close'].ewm(span=12, adjust=False).mean()
        
        # 7. Volume change
        monthly['volume_change'] = monthly['volume'].pct_change()
        
        # Reset indexu a přidání tickeru
        monthly = monthly.reset_index()
        monthly.rename(columns={'Date': 'date'}, inplace=True)
        monthly['ticker'] = ticker
        
        return monthly
        
    except Exception as e:
        log(f"  ❌ {ticker}: {e}")
        return pd.DataFrame()

def main():
    log("=" * 80)
    log("🧪 TESTOVACÍ RUN - 10 FIREM/SEKTOR")
    log("=" * 80)
    
    ensure_dir(DATA_DIR)
    
    # 1. Získat S&P 500 společnosti
    log("Stahuji seznam S&P 500...")
    sp500 = get_sp500_constituents()
    if sp500.empty:
        log("❌ Nepodařilo se získat S&P 500 data")
        return
    
    log(f"✓ Získáno {len(sp500)} společností")
    
    # 2. Filtrovat tickery podle sektorů
    target_buckets = ["Technology", "Consumer", "Industrials"]
    bucket_tickers = filter_tickers_by_buckets(sp500, target_buckets)
    
    ticker_counts = {b: len(t) for b, t in bucket_tickers.items()}
    log(f"✓ Vybráno {sum(ticker_counts.values())} tickerů: {ticker_counts}")
    log(f"⚠ TESTOVACÍ REŽIM: {TEST_LIMIT} firem/sektor (celkem {sum(ticker_counts.values())} firem)")
    
    # 3. Stahování a zpracování dat
    all_data = []
    total_tickers = sum(len(tickers) for tickers in bucket_tickers.values())
    processed = 0
    
    start_time = time.time()
    
    for bucket, tickers in bucket_tickers.items():
        log(f"\n📊 {bucket} ({len(tickers)} tickerů)...")
        
        for ticker in tickers:
            processed += 1
            log(f"  [{processed}/{total_tickers}] {ticker}...")
            
            df = download_and_process_ticker(ticker)
            if not df.empty:
                df['sector'] = bucket
                all_data.append(df)
    
    elapsed = time.time() - start_time
    log(f"\n✓ Staženo za {elapsed:.1f}s ({elapsed/total_tickers:.1f}s/ticker)")
    
    # 4. Spojení všech dat
    if not all_data:
        log("❌ Žádná data k uložení")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    log(f"✓ Celkem {len(combined)} záznamů")
    
    # 5. Uložení
    # Celý dataset
    all_path = os.path.join(DATA_DIR, "all_sectors_full_10y.csv")
    combined.to_csv(all_path, index=False)
    log(f"✓ {all_path}")
    
    # Po sektorech
    for bucket in target_buckets:
        sector_df = combined[combined['sector'] == bucket].copy()
        if not sector_df.empty:
            sector_path = os.path.join(DATA_DIR, f"{bucket}_full_10y.csv")
            sector_df.to_csv(sector_path, index=False)
            
            n_months = sector_df['date'].nunique()
            n_tickers = sector_df['ticker'].nunique()
            log(f"✓ {bucket}_full_10y.csv ({n_months} měsíců × {n_tickers} tickerů)")
            
            # Uložení seznamu tickerů
            ticker_list_path = os.path.join(DATA_DIR, f"{bucket}_tickers.txt")
            with open(ticker_list_path, 'w') as f:
                for t in sorted(sector_df['ticker'].unique()):
                    f.write(f"{t}\n")
            log(f"  → {bucket}_tickers.txt")
    
    # 6. Analýza chybějících dat
    log("\n" + "=" * 80)
    log("ANALÝZA CHYBĚJÍCÍCH DAT")
    log("=" * 80)
    
    missing_tickers = []
    for ticker in combined['ticker'].unique():
        ticker_df = combined[combined['ticker'] == ticker]
        missing_pct = ticker_df['close'].isna().sum() / len(ticker_df) * 100
        if missing_pct > 50:
            missing_tickers.append((ticker, missing_pct))
    
    if missing_tickers:
        log(f"⚠ {len(missing_tickers)} tickerů s >50% chybějících dat:")
        for ticker, pct in sorted(missing_tickers, key=lambda x: x[1], reverse=True):
            log(f"  • {ticker}: {pct:.1f}%")
    else:
        log("✓ Všechny tickery mají <50% chybějících dat")
    
    log("\n✅ TEST HOTOVO!")
    log(f"⏱ Celkový čas: {elapsed/60:.2f} minut ({elapsed:.1f}s)")
    log(f"📊 Testovací data: {DATA_DIR}/")
    log(f"🔍 Zkontroluj výsledky před spuštěním plného runu")

if __name__ == "__main__":
    main()
