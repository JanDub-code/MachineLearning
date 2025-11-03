#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Příprava 10letých cenových dat pro sektory: Technology, Consumer, Industrials
"""

import os
import time
import pandas as pd
import yfinance as yf
from typing import List

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# Sector mapping
SECTOR_BUCKET_MAP = {
    "Information Technology": "Technology",
    "Communication Services": "Communication",
    "Consumer Discretionary": "Consumer",
    "Consumer Staples": "Consumer",
    "Industrials": "Industrials",
    "Health Care": "HealthCare",
    "Financials": "Financials",
    "Energy": "Energy",
    "Materials": "Materials",
    "Real Estate": "RealEstate",
    "Utilities": "Utilities",
}

def get_sp500_constituents() -> pd.DataFrame:
    """Získat S&P 500 seznam z Wikipedie"""
    import io
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GPTResearchBot/1.0; +https://openai.com)"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        df = tables[0].copy()
        df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        return df[["ticker", "name", "sector"]]
    except Exception as e:
        log(f"WARN: Wikipedia failed ({e}), using fallback")
        csv_fallback = """ticker,name,sector
AAPL,Apple Inc,Information Technology
MSFT,Microsoft Corp,Information Technology
GOOGL,Alphabet Inc,Communication Services
AMZN,Amazon.com Inc,Consumer Discretionary
META,Meta Platforms Inc,Communication Services
NVDA,NVIDIA Corp,Information Technology
JPM,JP Morgan Chase & Co,Financials
UNH,UnitedHealth Group Inc,Health Care
XOM,Exxon Mobil Corp,Energy
PG,Procter & Gamble Co,Consumer Staples
HD,Home Depot Inc,Consumer Discretionary
CAT,Caterpillar Inc,Industrials
"""
        df = pd.read_csv(io.StringIO(csv_fallback))
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        return df

def filter_tickers_by_buckets(df_const: pd.DataFrame, buckets: List[str], per_sector: int) -> pd.DataFrame:
    """Filtrovat tickery podle sektorů"""
    df_const = df_const.copy()
    df_const["bucket"] = df_const["sector"].map(SECTOR_BUCKET_MAP)
    df_const = df_const[df_const["bucket"].isin(buckets)]
    out = []
    for b in buckets:
        g = df_const[df_const["bucket"] == b].head(per_sector)
        out.append(g)
    return pd.concat(out, axis=0).reset_index(drop=True)

def download_prices_10y(tickers: List[str], start: str = "2015-01-01") -> pd.DataFrame:
    """Stáhnout 10 let měsíčních cen"""
    log(f"Stahuji 10 let cenových dat pro {len(tickers)} tickerů...")
    try:
        prices = yf.download(tickers, start=start, end=None, auto_adjust=True, progress=False, interval="1mo")["Close"]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        prices = prices.sort_index()
        log(f"  ✓ Staženo {len(prices)} měsíců (od {prices.index[0]} do {prices.index[-1]})")
        return prices
    except Exception as e:
        log(f"  ✗ Chyba při stahování: {e}")
        return pd.DataFrame()

def main():
    log("="*80)
    log("PŘÍPRAVA 10LETÝCH DAT PRO SEKTORY")
    log("="*80)
    
    # Nastavení
    wanted_buckets = ["Technology", "Consumer", "Industrials"]
    per_sector = 50
    out_dir = "./data_10y"
    ensure_dir(out_dir)
    
    # 1. Získat S&P 500 seznam
    log("\n1. Získávám S&P 500 constituents...")
    sp500 = get_sp500_constituents()
    sp500["bucket"] = sp500["sector"].map(SECTOR_BUCKET_MAP)
    log(f"  ✓ Získáno {len(sp500)} společností")
    
    # 2. Filtrovat podle sektorů
    log(f"\n2. Filtr pro sektory: {wanted_buckets}")
    picks = filter_tickers_by_buckets(sp500, wanted_buckets, per_sector)
    log(f"  ✓ Vybráno {len(picks)} tickerů:")
    for bucket in wanted_buckets:
        count = len(picks[picks["bucket"] == bucket])
        log(f"    - {bucket}: {count} tickerů")
    
    # 3. Stáhnout ceny pro všechny tickery najednou
    log(f"\n3. Stahování cenových dat (10 let, měsíčně)...")
    all_tickers = sorted(picks["ticker"].unique().tolist())
    prices_all = download_prices_10y(all_tickers)
    
    if prices_all.empty:
        log("✗ Žádná data stažena, končím")
        return
    
    # 4. Uložit celý dataset
    log(f"\n4. Ukládání dat...")
    
    # Celkový soubor
    prices_all.to_csv(os.path.join(out_dir, "all_sectors_prices_10y.csv"))
    log(f"  ✓ {out_dir}/all_sectors_prices_10y.csv")
    
    # Rozdělit podle sektorů
    for bucket in wanted_buckets:
        bucket_tickers = picks[picks["bucket"] == bucket]["ticker"].tolist()
        bucket_prices = prices_all[bucket_tickers]
        
        # Uložit CSV
        out_file = os.path.join(out_dir, f"{bucket}_prices_10y.csv")
        bucket_prices.to_csv(out_file)
        log(f"  ✓ {out_file} ({len(bucket_prices)} měsíců × {len(bucket_tickers)} tickerů)")
        
        # Uložit seznam tickerů
        ticker_file = os.path.join(out_dir, f"{bucket}_tickers.txt")
        with open(ticker_file, "w") as f:
            f.write("\n".join(bucket_tickers))
        log(f"  ✓ {ticker_file}")
    
    # 5. Statistiky
    log(f"\n5. Souhrn:")
    log(f"  Časové období: {prices_all.index[0]} až {prices_all.index[-1]}")
    log(f"  Počet měsíců: {len(prices_all)}")
    log(f"  Celkem tickerů: {len(all_tickers)}")
    
    # Missing data analýza
    log(f"\n6. Analýza chybějících dat:")
    missing_pct = (prices_all.isna().sum() / len(prices_all) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 50]
    if len(high_missing) > 0:
        log(f"  ⚠ Tickery s >50% chybějících dat:")
        for ticker, pct in high_missing.head(10).items():
            log(f"    {ticker}: {pct:.1f}% chybí")
    else:
        log(f"  ✓ Všechny tickery mají <50% chybějících dat")
    
    log(f"\n{'='*80}")
    log("✅ HOTOVO! Data připravena v adresáři: ./data_10y/")
    log("="*80)

if __name__ == "__main__":
    main()
