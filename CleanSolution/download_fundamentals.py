#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stažení fundamentálních dat pro 30 tickerů.
"""

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

DATA_DIR = r"c:\Users\Bc. Jan Dub\Desktop\GIT\MachineLearning\CleanSolution\data_10y"
OUTPUT_DIR = r"c:\Users\Bc. Jan Dub\Desktop\GIT\MachineLearning\CleanSolution\data\fundamentals"

# Fundamentální metriky které chceme
FUNDAMENTAL_METRICS = [
    'trailingPE', 'forwardPE', 'priceToBook', 'priceToSalesTrailing12Months',
    'enterpriseToEbitda', 'pegRatio', 'returnOnEquity', 'returnOnAssets',
    'profitMargins', 'operatingMargins', 'grossMargins', 'debtToEquity',
    'currentRatio', 'quickRatio', 'revenueGrowth', 'earningsGrowth'
]

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def get_fundamentals(ticker_str: str) -> dict:
    """Stáhne fundamentální data pro ticker"""
    try:
        ticker = yf.Ticker(ticker_str)
        info = ticker.info
        
        result = {'ticker': ticker_str}
        
        # Základní metriky
        for metric in FUNDAMENTAL_METRICS:
            value = info.get(metric)
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                result[metric] = value
            else:
                result[metric] = np.nan
        
        # Přidat další užitečné metriky
        result['marketCap'] = info.get('marketCap', np.nan)
        result['trailingEps'] = info.get('trailingEps', np.nan)
        result['forwardEps'] = info.get('forwardEps', np.nan)
        result['bookValue'] = info.get('bookValue', np.nan)
        result['dividendYield'] = info.get('dividendYield', np.nan)
        result['beta'] = info.get('beta', np.nan)
        
        return result
        
    except Exception as e:
        log(f"  ERROR {ticker_str}: {e}")
        return {'ticker': ticker_str}

def main():
    log("=" * 60)
    log("STAŽENÍ FUNDAMENTÁLNÍCH DAT")
    log("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Načíst tickery z OHLCV dat
    ohlcv_file = os.path.join(DATA_DIR, "all_sectors_full_10y.csv")
    df_ohlcv = pd.read_csv(ohlcv_file)
    
    tickers = df_ohlcv['ticker'].unique().tolist()
    sectors = df_ohlcv.groupby('ticker')['sector'].first().to_dict()
    
    log(f"Nalezeno {len(tickers)} tickerů")
    
    all_fundamentals = []
    
    for i, ticker in enumerate(tickers, 1):
        log(f"[{i}/{len(tickers)}] {ticker}...")
        
        fund_data = get_fundamentals(ticker)
        fund_data['sector'] = sectors.get(ticker, 'Unknown')
        fund_data['download_date'] = datetime.now().strftime('%Y-%m-%d')
        
        all_fundamentals.append(fund_data)
        
        time.sleep(0.3)  # Rate limiting
    
    # Vytvořit DataFrame
    df_fund = pd.DataFrame(all_fundamentals)
    
    log("\n" + "=" * 60)
    log("VÝSLEDKY")
    log("=" * 60)
    
    # Statistiky
    log(f"Tickerů: {len(df_fund)}")
    log(f"Sloupců: {len(df_fund.columns)}")
    
    # Pokrytí metrik
    log("\nPokrytí metrik:")
    for col in FUNDAMENTAL_METRICS:
        if col in df_fund.columns:
            valid = df_fund[col].notna().sum()
            pct = valid / len(df_fund) * 100
            log(f"  {col}: {valid}/{len(df_fund)} ({pct:.0f}%)")
    
    # Uložit
    output_file = os.path.join(OUTPUT_DIR, "all_sectors_fundamentals.csv")
    df_fund.to_csv(output_file, index=False)
    log(f"\nUloženo: {output_file}")
    
    # Per-sector
    for sector in df_fund['sector'].unique():
        sector_df = df_fund[df_fund['sector'] == sector]
        sector_file = os.path.join(OUTPUT_DIR, f"{sector}_fundamentals.csv")
        sector_df.to_csv(sector_file, index=False)
        log(f"Uloženo: {sector_file} ({len(sector_df)} tickerů)")
    
    log("\n" + "=" * 60)
    log("HOTOVO!")
    log("=" * 60)
    
    return df_fund

if __name__ == "__main__":
    df = main()
    print("\nNáhled dat:")
    print(df[['ticker', 'sector', 'trailingPE', 'returnOnEquity', 'debtToEquity']].head(10))
