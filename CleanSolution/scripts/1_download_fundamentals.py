#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FÁZE 2: Stažení Fundamentálních Dat (1.5 roku)
================================================

Tento skript stahuje fundamentální metriky pro S&P 500 firmy za období 2024-2025.

Fundamentální metriky:
- Valuační: P/E, P/B, P/S, EV/EBITDA, PEG
- Profitabilita: ROE, ROA, Profit Margin, Operating Margin, Gross Margin
- Finanční zdraví: Debt-to-Equity, Current Ratio, Quick Ratio
- Růst: Revenue Growth YoY, Earnings Growth YoY

Výstup: data/fundamentals/all_sectors_fundamentals.csv
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime

warnings.filterwarnings('ignore')

# === KONFIGURACE ===
START_DATE = "2024-01-01"
END_DATE = "2025-10-31"
OUTPUT_DIR = "../data/fundamentals"
OHLCV_DIR = "../data_10y"  # Cesta k OHLCV datům z nadřazeného projektu

# Sektory
SECTORS = ["Technology", "Consumer", "Industrials"]

def log(msg: str):
    """Logování s časovou značkou"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_dir(path: str):
    """Vytvoří složku pokud neexistuje"""
    os.makedirs(path, exist_ok=True)

def load_tickers_from_ohlcv() -> Dict[str, List[str]]:
    """
    Načte seznam tickerů z OHLCV souborů.
    Vrací: {sector: [ticker1, ticker2, ...]}
    """
    result = {}
    
    for sector in SECTORS:
        ticker_file = os.path.join(OHLCV_DIR, f"{sector}_tickers.txt")
        
        if not os.path.exists(ticker_file):
            log(f"⚠️  {ticker_file} neexistuje, zkouším načíst z CSV...")
            csv_file = os.path.join(OHLCV_DIR, f"{sector}_full_10y.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                tickers = df['ticker'].unique().tolist()
                result[sector] = sorted(tickers)
                log(f"✓ {sector}: {len(tickers)} tickerů z CSV")
            else:
                log(f"❌ {sector}: CSV soubor nenalezen")
                result[sector] = []
        else:
            with open(ticker_file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            result[sector] = sorted(tickers)
            log(f"✓ {sector}: {len(tickers)} tickerů")
    
    return result

def safe_get_info(ticker: yf.Ticker, key: str, default=None):
    """Bezpečně získá hodnotu z ticker.info"""
    try:
        info = ticker.info
        return info.get(key, default)
    except Exception:
        return default

def calculate_quarterly_fundamentals(ticker_str: str) -> pd.DataFrame:
    """
    Stáhne a vypočítá fundamentální metriky z quarterly dat.
    Vrací DataFrame s časovými řádky (quarters).
    """
    try:
        ticker = yf.Ticker(ticker_str)
        
        # === Základní info z ticker.info (snapshot) ===
        market_cap = safe_get_info(ticker, 'marketCap')
        shares_outstanding = safe_get_info(ticker, 'sharesOutstanding')
        
        # === Quarterly financial statements ===
        try:
            financials = ticker.quarterly_financials.T  # Transpozice: řádky = quarters
            balance_sheet = ticker.quarterly_balance_sheet.T
            cashflow = ticker.quarterly_cashflow.T
        except Exception as e:
            log(f"  ❌ {ticker_str}: Chyba při stahování financial statements ({e})")
            return pd.DataFrame()
        
        if financials.empty or balance_sheet.empty:
            return pd.DataFrame()
        
        # Align indexy
        financials.index = pd.to_datetime(financials.index)
        balance_sheet.index = pd.to_datetime(balance_sheet.index)
        cashflow.index = pd.to_datetime(cashflow.index)
        
        # Merge všechny statements
        df = financials.join(balance_sheet, how='outer', rsuffix='_bs')
        df = df.join(cashflow, how='outer', rsuffix='_cf')
        
        # Filtrovat pouze období 2024-2025
        df = df[(df.index >= START_DATE) & (df.index <= END_DATE)]
        
        if df.empty:
            return pd.DataFrame()
        
        # === Helper funkce pro extrakci sloupců ===
        def get_col(df, candidates: List[str]):
            """Vrátí první existující sloupec z candidates"""
            for col in candidates:
                if col in df.columns:
                    return df[col]
            return pd.Series(np.nan, index=df.index)
        
        # === Extrakce dat ===
        total_revenue = get_col(df, ['Total Revenue', 'TotalRevenue'])
        net_income = get_col(df, ['Net Income', 'NetIncome'])
        ebitda = get_col(df, ['EBITDA', 'Ebitda'])
        operating_income = get_col(df, ['Operating Income', 'OperatingIncome'])
        gross_profit = get_col(df, ['Gross Profit', 'GrossProfit'])
        
        total_equity = get_col(df, ['Total Stockholder Equity', 'Stockholders Equity', 'Total Equity Gross Minority Interest'])
        total_assets = get_col(df, ['Total Assets', 'TotalAssets'])
        total_debt = get_col(df, ['Total Debt', 'Long Term Debt', 'LongTermDebt'])
        current_assets = get_col(df, ['Current Assets', 'CurrentAssets'])
        current_liabilities = get_col(df, ['Current Liabilities', 'CurrentLiabilities'])
        cash = get_col(df, ['Cash And Cash Equivalents', 'Cash'])
        
        # === Vypočítané metriky ===
        result = pd.DataFrame(index=df.index)
        result['ticker'] = ticker_str
        
        # TTM (Trailing Twelve Months) pro flow metriky
        revenue_ttm = total_revenue.rolling(4, min_periods=1).sum()
        net_income_ttm = net_income.rolling(4, min_periods=1).sum()
        ebitda_ttm = ebitda.rolling(4, min_periods=1).sum()
        operating_income_ttm = operating_income.rolling(4, min_periods=1).sum()
        gross_profit_ttm = gross_profit.rolling(4, min_periods=1).sum()
        
        # Valuační ratios (potřebujeme cenu)
        # Pro zjednodušení použijeme market cap / shares = price approx
        if market_cap and shares_outstanding:
            approx_price = market_cap / shares_outstanding
            eps_ttm = net_income_ttm / shares_outstanding
            result['PE'] = approx_price / eps_ttm.replace(0, np.nan)
            result['PS'] = market_cap / revenue_ttm.replace(0, np.nan)
        else:
            result['PE'] = np.nan
            result['PS'] = np.nan
        
        result['PB'] = (market_cap if market_cap else np.nan) / total_equity.replace(0, np.nan)
        
        # EV/EBITDA
        if market_cap:
            enterprise_value = market_cap + total_debt.fillna(0) - cash.fillna(0)
            result['EV_EBITDA'] = enterprise_value / ebitda_ttm.replace(0, np.nan)
        else:
            result['EV_EBITDA'] = np.nan
        
        # PEG (simplified - without forward growth rate)
        result['PEG'] = np.nan  # Potřebujeme forecast growth, který yfinance neposkytuje dobře
        
        # Profitabilita
        result['ROE'] = net_income_ttm / total_equity.replace(0, np.nan)
        result['ROA'] = net_income_ttm / total_assets.replace(0, np.nan)
        result['Profit_Margin'] = net_income_ttm / revenue_ttm.replace(0, np.nan)
        result['Operating_Margin'] = operating_income_ttm / revenue_ttm.replace(0, np.nan)
        result['Gross_Margin'] = gross_profit_ttm / revenue_ttm.replace(0, np.nan)
        
        # Finanční zdraví
        result['Debt_to_Equity'] = total_debt / total_equity.replace(0, np.nan)
        result['Current_Ratio'] = current_assets / current_liabilities.replace(0, np.nan)
        result['Quick_Ratio'] = (current_assets - get_col(df, ['Inventory'])) / current_liabilities.replace(0, np.nan)
        
        # Růst (YoY)
        result['Revenue_Growth_YoY'] = revenue_ttm.pct_change(periods=4)  # 4 quarters = 1 year
        result['Earnings_Growth_YoY'] = net_income_ttm.pct_change(periods=4)
        
        return result
        
    except Exception as e:
        log(f"  ❌ {ticker_str}: {e}")
        return pd.DataFrame()

def download_fundamentals_for_sector(sector: str, tickers: List[str]) -> pd.DataFrame:
    """
    Stáhne fundamentální data pro všechny tickery v sektoru.
    """
    log(f"\n{'='*80}")
    log(f"📊 {sector} ({len(tickers)} tickerů)")
    log(f"{'='*80}")
    
    all_data = []
    
    for i, ticker in enumerate(tickers, 1):
        log(f"  [{i}/{len(tickers)}] {ticker}...")
        
        df = calculate_quarterly_fundamentals(ticker)
        
        if not df.empty:
            df['sector'] = sector
            all_data.append(df)
            log(f"    ✓ {len(df)} quarters získáno")
        else:
            log(f"    ⚠️  Žádná data")
        
        # Rate limiting
        time.sleep(0.5)
    
    if not all_data:
        log(f"  ❌ Žádná data pro {sector}")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=False)
    combined = combined.reset_index().rename(columns={'index': 'date'})
    
    return combined

def main():
    log("="*80)
    log("FÁZE 2: STAŽENÍ FUNDAMENTÁLNÍCH DAT")
    log("="*80)
    log(f"Období: {START_DATE} → {END_DATE}")
    log(f"Sektory: {', '.join(SECTORS)}")
    
    # Vytvoření výstupní složky
    ensure_dir(OUTPUT_DIR)
    
    # Načtení tickerů z OHLCV dat
    log("\n📂 Načítám seznam tickerů...")
    sector_tickers = load_tickers_from_ohlcv()
    
    total_tickers = sum(len(tickers) for tickers in sector_tickers.values())
    log(f"✓ Celkem {total_tickers} tickerů načteno")
    
    # Stahování dat po sektorech
    start_time = time.time()
    all_sector_data = []
    
    for sector in SECTORS:
        tickers = sector_tickers.get(sector, [])
        if not tickers:
            log(f"⚠️  {sector}: Žádné tickery")
            continue
        
        df = download_fundamentals_for_sector(sector, tickers)
        
        if not df.empty:
            all_sector_data.append(df)
            
            # Uložení po sektorech
            sector_path = os.path.join(OUTPUT_DIR, f"{sector}_fundamentals.csv")
            df.to_csv(sector_path, index=False)
            log(f"✓ Uloženo: {sector_path}")
    
    elapsed = time.time() - start_time
    
    # Spojení všech sektorů
    if all_sector_data:
        combined = pd.concat(all_sector_data, ignore_index=True)
        
        # Uložení kompletního datasetu
        output_path = os.path.join(OUTPUT_DIR, "all_sectors_fundamentals.csv")
        combined.to_csv(output_path, index=False)
        
        log("\n" + "="*80)
        log("✅ HOTOVO!")
        log("="*80)
        log(f"⏱  Čas: {elapsed/60:.1f} minut")
        log(f"📊 Celkem: {len(combined)} záznamů")
        log(f"💾 Uloženo: {output_path}")
        
        # Statistiky
        log("\n📈 STATISTIKY:")
        log(f"  • Tickery: {combined['ticker'].nunique()}")
        log(f"  • Časové období: {combined['date'].min()} → {combined['date'].max()}")
        log(f"  • Průměrně quarters/ticker: {len(combined) / combined['ticker'].nunique():.1f}")
        
        # Analýza chybějících dat
        log("\n⚠️  CHYBĚJÍCÍ DATA:")
        missing_pct = (combined.isnull().sum() / len(combined) * 100).sort_values(ascending=False)
        for col, pct in missing_pct.head(10).items():
            if pct > 0:
                log(f"  • {col}: {pct:.1f}%")
    else:
        log("\n❌ Žádná data k uložení")
    
    log("\n" + "="*80)
    log("Další krok: python scripts/2_train_fundamental_predictor.py")
    log("="*80)

if __name__ == "__main__":
    main()
