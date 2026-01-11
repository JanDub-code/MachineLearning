#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ML PIPELINE PRO 150 TICKERŮ (5 sektorů × 30 firem)
=============================================================================

Experiment: Klasifikace cenových pohybů akcií
- Data: 10 let měsíčních OHLCV dat + technické indikátory
- Fundamenty: P/E, ROE, ROA, atd. (imputované pomocí RF Regressor)
- Klasifikace: DOWN / HOLD / UP (±3% threshold)

Kroky:
1. Stažení OHLCV dat + technických indikátorů
2. Stažení fundamentálních dat
3. Trénink RF Regressor pro imputaci fundamentů
4. Trénink RF Classifier (DOWN/HOLD/UP)
5. Hyperparameter tuning (GridSearchCV + TimeSeriesSplit)
6. Finální evaluace + grafy

Autor: Bc. Jan Dub
Datum: 2026-01-01
=============================================================================
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

# Matplotlib/Seaborn - import later to avoid GUI issues
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# KONFIGURACE
# =============================================================================

BASE_DIR = r"c:\Users\Hans\Desktop\cv\MachineLearning\CleanSolution"

# Výstupní adresáře pro experiment
DATA_DIR = os.path.join(BASE_DIR, "data", "150_tickers")
OHLCV_DIR = os.path.join(DATA_DIR, "ohlcv")
FUND_DIR = os.path.join(DATA_DIR, "fundamentals")
COMPLETE_DIR = os.path.join(DATA_DIR, "complete")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "models", "150_tickers")

# Časové rozmezí
START_DATE = "2015-01-01"
END_DATE = "2025-12-01"

# Klasifikační threshold
THRESHOLD = 0.03  # ±3%

# 150 tickerů: 5 sektorů × 30 firem
TICKERS = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CSCO", "ADBE", "CRM",
        "AMD", "INTC", "IBM", "NOW", "QCOM", "TXN", "AMAT", "INTU", "PANW", "MU",
        "SNPS", "CDNS", "KLAC", "LRCX", "ADI", "MRVL", "FTNT", "CTSH", "HPQ", "DELL"
    ],
    "Consumer": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "PG", "KO",
        "PEP", "COST", "WMT", "DIS", "NFLX", "BKNG", "CMG", "YUM", "ORLY", "AZO",
        "ROST", "TJX", "DHI", "LEN", "F", "GM", "MAR", "HLT", "EXPE", "EBAY"
    ],
    "Industrials": [
        "CAT", "HON", "UPS", "BA", "GE", "RTX", "DE", "LMT", "MMM", "UNP",
        "WM", "ETN", "ITW", "EMR", "NSC", "CSX", "FDX", "PH", "JCI", "ROK",
        "CMI", "PCAR", "GD", "NOC", "TT", "IR", "CARR", "OTIS", "AME", "DOV"
    ],
    "Healthcare": [
        "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
        "AMGN", "GILD", "MDT", "CVS", "ELV", "CI", "ISRG", "REGN", "VRTX", "SYK",
        "ZTS", "BDX", "BSX", "HUM", "HCA", "IQV", "DXCM", "IDXX", "MTD", "WST"
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SPGI", "C", "AXP", "USB",
        "PNC", "SCHW", "TFC", "BK", "COF", "ICE", "CME", "MCO", "AON", "MMC",
        "MET", "PRU", "AFL", "ALL", "TRV", "CB", "AIG", "CINF", "L", "WRB"
    ]
}

# Fundamentální metriky
FUNDAMENTAL_METRICS = [
    'trailingPE', 'forwardPE', 'priceToBook', 'priceToSalesTrailing12Months',
    'enterpriseToEbitda', 'pegRatio', 'returnOnEquity', 'returnOnAssets',
    'profitMargins', 'operatingMargins', 'grossMargins', 'debtToEquity',
    'currentRatio', 'quickRatio', 'revenueGrowth', 'earningsGrowth'
]

# OHLCV features pro ML
OHLCV_FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'volatility', 'returns', 'rsi_14',
    'macd', 'macd_signal', 'macd_hist',
    'sma_3', 'sma_6', 'sma_12',
    'ema_3', 'ema_6', 'ema_12',
    'volume_change'
]

# Fundamentální targets pro regressor
FUND_TARGETS = [
    'trailingPE', 'forwardPE', 'priceToBook',
    'returnOnEquity', 'returnOnAssets',
    'profitMargins', 'operatingMargins', 'grossMargins',
    'debtToEquity', 'currentRatio', 'beta'
]

# Všechny features pro classifier
ALL_FEATURES = OHLCV_FEATURES + FUND_TARGETS

# =============================================================================
# HELPER FUNKCE
# =============================================================================

def log(msg: str):
    """Logging s časovým razítkem"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def log_section(title: str):
    """Logging sekce"""
    print("\n" + "=" * 70, flush=True)
    print(f" {title}", flush=True)
    print("=" * 70, flush=True)

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

def create_target(df: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Vytvoří target variable: budoucí měsíční return klasifikace"""
    df = df.copy()
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    df['future_close'] = df.groupby('ticker')['close'].shift(-1)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    
    def classify(ret):
        if pd.isna(ret):
            return np.nan
        elif ret < -threshold:
            return 0  # DOWN
        elif ret > threshold:
            return 2  # UP
        else:
            return 1  # HOLD
    
    df['target'] = df['future_return'].apply(classify)
    return df

# =============================================================================
# KROK 1: STAŽENÍ OHLCV DAT
# =============================================================================

def download_ohlcv():
    """Stáhne OHLCV data + technické indikátory pro všech 150 tickerů"""
    log_section("KROK 1: STAŽENÍ OHLCV DAT + TECHNICKÝCH INDIKÁTORŮ")
    
    os.makedirs(OHLCV_DIR, exist_ok=True)
    
    all_data = []
    sector_data = {s: [] for s in TICKERS.keys()}
    failed_tickers = []
    
    total_tickers = sum(len(t) for t in TICKERS.values())
    current = 0
    
    for sector, tickers in TICKERS.items():
        log(f"\n--- {sector} ({len(tickers)} tickerů) ---")
        
        # Uložení tickerů do souboru
        ticker_file = os.path.join(OHLCV_DIR, f"{sector}_tickers.txt")
        with open(ticker_file, 'w') as f:
            f.write('\n'.join(tickers))
        
        for ticker in tickers:
            current += 1
            log(f"  [{current}/{total_tickers}] {ticker}...")
            
            try:
                yf_ticker = yf.Ticker(ticker)
                hist = yf_ticker.history(start=START_DATE, end=END_DATE, interval="1d")
                
                if hist.empty or len(hist) < 20:
                    log(f"    SKIP: nedostatek dat")
                    failed_tickers.append(ticker)
                    continue
                
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
                monthly['sector'] = sector
                
                all_data.append(monthly)
                sector_data[sector].append(monthly)
                
                log(f"    OK: {len(monthly)} měsíců")
                
            except Exception as e:
                log(f"    ERROR: {e}")
                failed_tickers.append(ticker)
            
            time.sleep(0.5)  # Rate limiting
    
    # Uložení dat
    log("\n" + "-" * 50)
    log("Ukládání dat...")
    
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        all_file = os.path.join(OHLCV_DIR, "all_sectors_full_10y.csv")
        df_all.to_csv(all_file, index=False)
        log(f"  Uloženo: {all_file}")
        log(f"  Celkem: {len(df_all)} řádků, {df_all['ticker'].nunique()} tickerů")
        
        # Per-sector
        for sector, data_list in sector_data.items():
            if data_list:
                df_sector = pd.concat(data_list, ignore_index=True)
                sector_file = os.path.join(OHLCV_DIR, f"{sector}_full_10y.csv")
                df_sector.to_csv(sector_file, index=False)
                log(f"  {sector}: {len(df_sector)} řádků, {df_sector['ticker'].nunique()} tickerů")
    
    if failed_tickers:
        log(f"\n  VAROVÁNÍ: {len(failed_tickers)} tickerů selhalo: {failed_tickers[:10]}...")
    
    return df_all if all_data else None

# =============================================================================
# KROK 2: STAŽENÍ FUNDAMENTÁLNÍCH DAT
# =============================================================================

def download_fundamentals():
    """Stáhne fundamentální data pro všechny tickery"""
    log_section("KROK 2: STAŽENÍ FUNDAMENTÁLNÍCH DAT")
    
    os.makedirs(FUND_DIR, exist_ok=True)
    
    # Načíst tickery z OHLCV dat
    ohlcv_file = os.path.join(OHLCV_DIR, "all_sectors_full_10y.csv")
    
    if not os.path.exists(ohlcv_file):
        log("ERROR: OHLCV data nenalezena! Spusťte nejdřív Krok 1.")
        return None
    
    df_ohlcv = pd.read_csv(ohlcv_file)
    tickers = df_ohlcv['ticker'].unique().tolist()
    sectors = df_ohlcv.groupby('ticker')['sector'].first().to_dict()
    
    log(f"Nalezeno {len(tickers)} tickerů")
    
    all_fundamentals = []
    
    for i, ticker in enumerate(tickers, 1):
        log(f"[{i}/{len(tickers)}] {ticker}...")
        
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            result = {'ticker': ticker}
            
            # Základní metriky
            for metric in FUNDAMENTAL_METRICS:
                value = info.get(metric)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    result[metric] = value
                else:
                    result[metric] = np.nan
            
            # Přidat další metriky
            result['marketCap'] = info.get('marketCap', np.nan)
            result['trailingEps'] = info.get('trailingEps', np.nan)
            result['forwardEps'] = info.get('forwardEps', np.nan)
            result['bookValue'] = info.get('bookValue', np.nan)
            result['dividendYield'] = info.get('dividendYield', np.nan)
            result['beta'] = info.get('beta', np.nan)
            result['sector'] = sectors.get(ticker, 'Unknown')
            result['download_date'] = datetime.now().strftime('%Y-%m-%d')
            
            all_fundamentals.append(result)
            
        except Exception as e:
            log(f"  ERROR: {e}")
            all_fundamentals.append({'ticker': ticker, 'sector': sectors.get(ticker, 'Unknown')})
        
        time.sleep(0.3)  # Rate limiting
    
    # Vytvořit DataFrame
    df_fund = pd.DataFrame(all_fundamentals)
    
    log("\n" + "-" * 50)
    log("Výsledky:")
    log(f"  Tickerů: {len(df_fund)}")
    log(f"  Sloupců: {len(df_fund.columns)}")
    
    # Pokrytí metrik
    log("\nPokrytí metrik:")
    for col in FUNDAMENTAL_METRICS[:8]:  # Top 8
        if col in df_fund.columns:
            valid = df_fund[col].notna().sum()
            pct = valid / len(df_fund) * 100
            log(f"  {col}: {valid}/{len(df_fund)} ({pct:.0f}%)")
    
    # Uložit
    output_file = os.path.join(FUND_DIR, "all_sectors_fundamentals.csv")
    df_fund.to_csv(output_file, index=False)
    log(f"\nUloženo: {output_file}")
    
    # Per-sector
    for sector in df_fund['sector'].unique():
        sector_df = df_fund[df_fund['sector'] == sector]
        sector_file = os.path.join(FUND_DIR, f"{sector}_fundamentals.csv")
        sector_df.to_csv(sector_file, index=False)
    
    return df_fund

# =============================================================================
# KROK 3: TRÉNINK RF REGRESSOR PRO IMPUTACI
# =============================================================================

def train_rf_regressor():
    """Natrénuje RF Regressor pro imputaci fundamentálních dat"""
    log_section("KROK 3: TRÉNINK RF REGRESSOR PRO IMPUTACI FUNDAMENTŮ")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(COMPLETE_DIR, exist_ok=True)
    
    # Načíst data
    ohlcv_file = os.path.join(OHLCV_DIR, "all_sectors_full_10y.csv")
    fund_file = os.path.join(FUND_DIR, "all_sectors_fundamentals.csv")
    
    if not os.path.exists(ohlcv_file) or not os.path.exists(fund_file):
        log("ERROR: Data nenalezena! Spusťte nejdřív Kroky 1 a 2.")
        return None
    
    df_ohlcv = pd.read_csv(ohlcv_file)
    df_fund = pd.read_csv(fund_file)
    
    log(f"OHLCV: {len(df_ohlcv)} řádků, {df_ohlcv['ticker'].nunique()} tickerů")
    log(f"Fundamenty: {len(df_fund)} tickerů")
    
    # Spojit data
    log("\nSpojování dat...")
    fund_cols = ['ticker'] + [c for c in FUND_TARGETS if c in df_fund.columns]
    df_fund_subset = df_fund[fund_cols].copy()
    df_merged = df_ohlcv.merge(df_fund_subset, on='ticker', how='left')
    
    log(f"Merged: {len(df_merged)} řádků")
    
    # Připravit trénovací data (posledních 24 měsíců)
    df_merged['date'] = pd.to_datetime(df_merged['date'], utc=True)
    cutoff_date = df_merged['date'].max() - pd.DateOffset(months=24)
    
    df_recent = df_merged[df_merged['date'] >= cutoff_date].copy()
    df_historical = df_merged[df_merged['date'] < cutoff_date].copy()
    
    log(f"Recent (training): {len(df_recent)} řádků")
    log(f"Historical (to predict): {len(df_historical)} řádků")
    
    # Dostupné features a targets
    available_features = [f for f in OHLCV_FEATURES if f in df_recent.columns]
    available_targets = [t for t in FUND_TARGETS if t in df_recent.columns]
    
    log(f"Features: {len(available_features)}")
    log(f"Targets: {len(available_targets)}")
    
    # Příprava dat
    df_train = df_recent.dropna(subset=available_features + available_targets)
    
    if len(df_train) < 50:
        log("VAROVÁNÍ: Málo trénovacích vzorků!")
    
    X = df_train[available_features].values
    y = df_train[available_targets].values
    
    log(f"Training samples: {len(X)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model
    log("\nTrénink modelu...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluace
    y_pred = model.predict(X_test_scaled)
    
    log("\nVýsledky (test set):")
    for i, target in enumerate(available_targets):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        log(f"  {target}: MAE={mae:.3f}, R²={r2:.3f}")
    
    # Imputace historických dat
    log("\nImputace historických dat...")
    df_hist_clean = df_historical.dropna(subset=available_features).copy()
    
    if len(df_hist_clean) > 0:
        X_hist = df_hist_clean[available_features].values
        X_hist_scaled = scaler.transform(X_hist)
        y_hist_pred = model.predict(X_hist_scaled)
        
        for i, target in enumerate(available_targets):
            df_hist_clean[target] = y_hist_pred[:, i]
        
        df_hist_clean['data_source'] = 'predicted'
    
    # Recent data
    df_recent_clean = df_recent.dropna(subset=available_features + available_targets).copy()
    df_recent_clean['data_source'] = 'real'
    
    # Spojit
    df_complete = pd.concat([df_hist_clean, df_recent_clean], ignore_index=True)
    df_complete = df_complete.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    log(f"\nKompletní dataset: {len(df_complete)} řádků")
    log(f"  - Predicted: {(df_complete['data_source'] == 'predicted').sum()}")
    log(f"  - Real: {(df_complete['data_source'] == 'real').sum()}")
    
    # Uložit
    model_file = os.path.join(MODEL_DIR, "fundamental_predictor.pkl")
    scaler_file = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    complete_file = os.path.join(COMPLETE_DIR, "all_sectors_complete_10y.csv")
    
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    df_complete.to_csv(complete_file, index=False)
    
    log(f"\nUloženo:")
    log(f"  Model: {model_file}")
    log(f"  Scaler: {scaler_file}")
    log(f"  Data: {complete_file}")
    
    # Metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'n_tickers': df_complete['ticker'].nunique(),
        'n_samples': len(df_complete),
        'features': available_features,
        'targets': available_targets,
        'model_params': model.get_params()
    }
    
    with open(os.path.join(MODEL_DIR, "regressor_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return df_complete

# =============================================================================
# KROK 4: TRÉNINK RF CLASSIFIER
# =============================================================================

def train_rf_classifier():
    """Natrénuje RF Classifier pro klasifikaci DOWN/HOLD/UP"""
    log_section("KROK 4: TRÉNINK RF CLASSIFIER (DOWN/HOLD/UP)")
    
    # Načíst data
    data_file = os.path.join(COMPLETE_DIR, "all_sectors_complete_10y.csv")
    
    if not os.path.exists(data_file):
        log("ERROR: Kompletní data nenalezena! Spusťte nejdřív Krok 3.")
        return None
    
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    log(f"Načteno: {len(df)} řádků, {df['ticker'].nunique()} tickerů")
    
    # Vytvořit target
    df = create_target(df)
    
    target_counts = df['target'].value_counts().sort_index()
    log(f"\nTarget distribuce:")
    log(f"  DOWN (0): {target_counts.get(0, 0)}")
    log(f"  HOLD (1): {target_counts.get(1, 0)}")
    log(f"  UP (2):   {target_counts.get(2, 0)}")
    
    # Dostupné features
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    log(f"\nFeatures: {len(available_features)}")
    
    # Odstranit NaN
    df_clean = df.dropna(subset=available_features + ['target']).copy()
    df_clean['target'] = df_clean['target'].astype(int)
    
    log(f"Clean samples: {len(df_clean)}")
    
    # Chronologický split
    df_clean = df_clean.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df_clean) * 0.8)
    
    df_train = df_clean.iloc[:split_idx]
    df_test = df_clean.iloc[split_idx:]
    
    log(f"Train: {len(df_train)}, Test: {len(df_test)}")
    
    X_train = df_train[available_features].values
    y_train = df_train['target'].values
    X_test = df_test[available_features].values
    y_test = df_test['target'].values
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model
    log("\nTrénink modelu...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluace
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    log(f"\n{'='*40}")
    log(f"VÝSLEDKY:")
    log(f"  Accuracy:  {accuracy:.4f}")
    log(f"  Precision: {precision:.4f}")
    log(f"  Recall:    {recall:.4f}")
    log(f"  F1-Score:  {f1:.4f}")
    log(f"{'='*40}")
    
    # Classification report
    log("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['DOWN', 'HOLD', 'UP'])
    for line in report.split('\n'):
        if line.strip():
            log(f"  {line}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    log("\nConfusion Matrix:")
    log(f"            DOWN  HOLD    UP")
    log(f"  DOWN     {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
    log(f"  HOLD     {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
    log(f"  UP       {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
    
    # Per-sector analýza
    log("\nPer-sector výsledky:")
    for sector in df_test['sector'].unique():
        sector_mask = df_test['sector'] == sector
        if sector_mask.sum() > 0:
            sector_indices = df_test[sector_mask].index - split_idx
            sector_indices = sector_indices[sector_indices >= 0]
            sector_indices = sector_indices[sector_indices < len(y_test)]
            
            if len(sector_indices) > 0:
                y_sector = y_test[sector_indices]
                y_pred_sector = y_pred[sector_indices]
                acc = accuracy_score(y_sector, y_pred_sector)
                f1_s = f1_score(y_sector, y_pred_sector, average='weighted')
                log(f"  {sector}: Accuracy={acc:.3f}, F1={f1_s:.3f} (n={len(sector_indices)})")
    
    # Feature importance
    log("\nFeature Importance (Top 10):")
    importance = model.feature_importances_
    feat_imp = sorted(zip(available_features, importance), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:10]:
        log(f"  {feat}: {imp:.4f}")
    
    # Uložit
    model_file = os.path.join(MODEL_DIR, "rf_classifier.pkl")
    scaler_file = os.path.join(MODEL_DIR, "classifier_scaler.pkl")
    
    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)
    
    log(f"\nUloženo:")
    log(f"  Model: {model_file}")
    log(f"  Scaler: {scaler_file}")
    
    # Metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'threshold': THRESHOLD,
        'n_train': len(df_train),
        'n_test': len(df_test),
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'features': available_features,
        'model_params': model.get_params()
    }
    
    with open(os.path.join(MODEL_DIR, "classifier_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return model, scaler, df_test, y_test, y_pred

# =============================================================================
# KROK 5: HYPERPARAMETER TUNING
# =============================================================================

def hyperparameter_tuning():
    """GridSearchCV s TimeSeriesSplit pro optimalizaci hyperparametrů"""
    log_section("KROK 5: HYPERPARAMETER TUNING (GridSearchCV + TimeSeriesSplit)")
    
    # Načíst data
    data_file = os.path.join(COMPLETE_DIR, "all_sectors_complete_10y.csv")
    
    if not os.path.exists(data_file):
        log("ERROR: Data nenalezena!")
        return None
    
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = create_target(df)
    
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    df_clean = df.dropna(subset=available_features + ['target']).copy()
    df_clean['target'] = df_clean['target'].astype(int)
    df_clean = df_clean.sort_values('date').reset_index(drop=True)
    
    log(f"Samples: {len(df_clean)}")
    log(f"Features: {len(available_features)}")
    
    X = df_clean[available_features].values
    y = df_clean['target'].values
    
    # Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Grid Search
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'class_weight': ['balanced']
    }
    
    n_combinations = (
        len(param_grid['n_estimators']) *
        len(param_grid['max_depth']) *
        len(param_grid['min_samples_split']) *
        len(param_grid['min_samples_leaf'])
    )
    
    log(f"\nGrid Search: {n_combinations} kombinací")
    log("Toto může trvat několik minut...")
    
    tscv = TimeSeriesSplit(n_splits=5)
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_scaled, y)
    
    # Výsledky
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    log(f"\n{'='*40}")
    log("NEJLEPŠÍ PARAMETRY:")
    for param, value in best_params.items():
        log(f"  {param}: {value}")
    log(f"\nNejlepší CV F1 Score: {best_score:.4f}")
    log(f"{'='*40}")
    
    # Finální model
    log("\nTrénink finálního modelu s nejlepšími parametry...")
    
    split_idx = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:split_idx]
    y_train = y[:split_idx]
    X_test = X_scaled[split_idx:]
    y_test = y[split_idx:]
    
    final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train, y_train)
    
    y_pred = final_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    log(f"\nFinální výsledky:")
    log(f"  Test Accuracy: {accuracy:.4f}")
    log(f"  Test F1 Score: {f1:.4f}")
    
    # Classification report
    log("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['DOWN', 'HOLD', 'UP'])
    for line in report.split('\n'):
        if line.strip():
            log(f"  {line}")
    
    # Uložit
    model_file = os.path.join(MODEL_DIR, "rf_classifier_tuned.pkl")
    scaler_file = os.path.join(MODEL_DIR, "classifier_scaler_tuned.pkl")
    
    joblib.dump(final_model, model_file)
    joblib.dump(scaler, scaler_file)
    
    log(f"\nUloženo:")
    log(f"  Model: {model_file}")
    log(f"  Scaler: {scaler_file}")
    
    # Metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'best_params': best_params,
        'best_cv_score': float(best_score),
        'test_accuracy': float(accuracy),
        'test_f1': float(f1),
        'n_cv_splits': 5
    }
    
    with open(os.path.join(MODEL_DIR, "tuning_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return final_model, best_params

# =============================================================================
# KROK 6: FINÁLNÍ EVALUACE + GRAFY
# =============================================================================

def final_evaluation():
    """Generuje finální vizualizace a metriky - PREMIUM verze"""
    log_section("KROK 6: FINÁLNÍ EVALUACE + PREMIUM VIZUALIZACE")
    
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Nastavení pro profesionální grafy
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['figure.facecolor'] = 'white'
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Načíst model a data
    model_file = os.path.join(MODEL_DIR, "rf_classifier_tuned.pkl")
    if not os.path.exists(model_file):
        model_file = os.path.join(MODEL_DIR, "rf_classifier.pkl")
    
    scaler_file = os.path.join(MODEL_DIR, "classifier_scaler_tuned.pkl")
    if not os.path.exists(scaler_file):
        scaler_file = os.path.join(MODEL_DIR, "classifier_scaler.pkl")
    
    data_file = os.path.join(COMPLETE_DIR, "all_sectors_complete_10y.csv")
    
    if not os.path.exists(model_file) or not os.path.exists(data_file):
        log("ERROR: Model nebo data nenalezena!")
        return
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    
    df = pd.read_csv(data_file)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = create_target(df)
    
    available_features = [f for f in ALL_FEATURES if f in df.columns]
    df_clean = df.dropna(subset=available_features + ['target']).copy()
    df_clean['target'] = df_clean['target'].astype(int)
    df_clean = df_clean.sort_values('date').reset_index(drop=True)
    
    # Train/Test set
    split_idx = int(len(df_clean) * 0.8)
    df_train = df_clean.iloc[:split_idx].copy()
    df_test = df_clean.iloc[split_idx:].copy()
    
    X_test = df_test[available_features].values
    y_test = df_test['target'].values
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # Přidat predikce do test DataFrame pro backtesting
    df_test = df_test.reset_index(drop=True)
    df_test['prediction'] = y_pred
    df_test['pred_proba_down'] = y_proba[:, 0]
    df_test['pred_proba_hold'] = y_proba[:, 1]
    df_test['pred_proba_up'] = y_proba[:, 2]
    df_test['pred_confidence'] = y_proba.max(axis=1)
    
    log(f"Test samples: {len(y_test)}")
    
    # 1. Confusion Matrix
    log("\n1. Generovani Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['DOWN', 'HOLD', 'UP'],
                yticklabels=['DOWN', 'HOLD', 'UP'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - 150 Tickers Classification')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    log("  Ulozeno: confusion_matrix.png")
    
    # 2. ROC Curves
    log("\n2. Generovani ROC krivek...")
    n_classes = 3
    classes = ['DOWN', 'HOLD', 'UP']
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    plt.figure(figsize=(10, 8))
    
    for i, (class_name, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - 150 Tickers Classification')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=150)
    plt.close()
    log("  Ulozeno: roc_curves.png")
    
    # 3. Feature Importance
    log("\n3. Generovani Feature Importance...")
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': available_features,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 12))
    plt.barh(feat_imp['feature'], feat_imp['importance'], color='steelblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance - 150 Tickers Classification')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    log("  Ulozeno: feature_importance.png")
    
    # 4. Per-sector comparison
    log("\n4. Generovani Per-sector porovnani...")
    sector_metrics = {}
    
    for sector in df_test['sector'].unique():
        sector_mask = df_test['sector'] == sector
        if sector_mask.sum() > 0:
            sector_indices = df_test[sector_mask].index - split_idx
            sector_indices = sector_indices[(sector_indices >= 0) & (sector_indices < len(y_test))]
            
            if len(sector_indices) > 0:
                y_s = y_test[sector_indices]
                y_p = y_pred[sector_indices]
                
                sector_metrics[sector] = {
                    'accuracy': accuracy_score(y_s, y_p),
                    'precision': precision_score(y_s, y_p, average='weighted', zero_division=0),
                    'recall': recall_score(y_s, y_p, average='weighted', zero_division=0),
                    'f1': f1_score(y_s, y_p, average='weighted', zero_division=0),
                    'n_samples': len(sector_indices)
                }
    
    if sector_metrics:
        sectors = list(sector_metrics.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(sectors))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 6))
        colors_bar = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
        
        for i, (metric, color) in enumerate(zip(metrics, colors_bar)):
            values = [sector_metrics[s][metric] for s in sectors]
            ax.bar(x + i * width, values, width, label=metric.capitalize(), color=color)
        
        ax.set_xlabel('Sector')
        ax.set_ylabel('Score')
        ax.set_title('Per-Sector Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(sectors, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "sector_comparison.png"), dpi=300)
        plt.close()
        log("  Ulozeno: sector_comparison.png")
    
    # =========================================================================
    # PREMIUM VIZUALIZACE (Nové grafy pro dokumentaci)
    # =========================================================================
    
    # 5. EQUITY CURVE (Backtest) - Klíčový graf!
    log("\n5. Generovani Equity Curve (Backtest)...")
    
    # Strategie: Long když UP, Short když DOWN, nic když HOLD
    df_backtest = df_test.copy()
    df_backtest['strategy_return'] = 0.0
    
    # UP predikce -> long pozice -> výnos = future_return
    up_mask = df_backtest['prediction'] == 2
    df_backtest.loc[up_mask, 'strategy_return'] = df_backtest.loc[up_mask, 'future_return'].fillna(0)
    
    # DOWN predikce -> short pozice -> výnos = -future_return
    down_mask = df_backtest['prediction'] == 0
    df_backtest.loc[down_mask, 'strategy_return'] = -df_backtest.loc[down_mask, 'future_return'].fillna(0)
    
    # Buy & Hold benchmark
    df_backtest['buyhold_return'] = df_backtest['future_return'].fillna(0)
    
    # Kumulativní výnosy
    df_backtest['cumulative_strategy'] = (1 + df_backtest['strategy_return']).cumprod()
    df_backtest['cumulative_buyhold'] = (1 + df_backtest['buyhold_return']).cumprod()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Horní graf: Equity Curve
    ax1 = axes[0]
    ax1.plot(range(len(df_backtest)), df_backtest['cumulative_strategy'], 
             label='ML Strategie', linewidth=2, color='#2ecc71')
    ax1.plot(range(len(df_backtest)), df_backtest['cumulative_buyhold'], 
             label='Buy & Hold', linewidth=2, color='#3498db', alpha=0.7)
    ax1.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Portfolio Value (počáteční = 1.0)', fontweight='bold')
    ax1.set_xlabel('Čas (měsíce v testovacím období)')
    ax1.set_title('Equity Curve: ML Strategie vs Buy & Hold\n(150 tickerů, 5 sektorů)', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Spodní graf: Drawdown
    ax2 = axes[1]
    rolling_max = df_backtest['cumulative_strategy'].expanding().max()
    drawdown = (df_backtest['cumulative_strategy'] - rolling_max) / rolling_max * 100
    ax2.fill_between(range(len(df_backtest)), drawdown, 0, alpha=0.5, color='#e74c3c')
    ax2.set_ylabel('Drawdown (%)', fontweight='bold')
    ax2.set_xlabel('Čas (měsíce)')
    ax2.set_title('Drawdown analýza', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "equity_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    log("  Ulozeno: equity_curve.png")
    
    # 6. NORMALIZED CONFUSION MATRIX (Procentuální)
    log("\n6. Generovani Normalized Confusion Matrix...")
    
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Absolutní
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['DOWN', 'HOLD', 'UP'],
                yticklabels=['DOWN', 'HOLD', 'UP'])
    ax1.set_xlabel('Predikované', fontweight='bold')
    ax1.set_ylabel('Skutečné', fontweight='bold')
    ax1.set_title('Confusion Matrix (absolutní počty)', fontweight='bold')
    
    # Normalizovaná
    ax2 = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax2,
                xticklabels=['DOWN', 'HOLD', 'UP'],
                yticklabels=['DOWN', 'HOLD', 'UP'],
                vmin=0, vmax=100)
    ax2.set_xlabel('Predikované', fontweight='bold')
    ax2.set_ylabel('Skutečné', fontweight='bold')
    ax2.set_title('Confusion Matrix (% správnosti per třída)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_normalized.png"), dpi=300, bbox_inches='tight')
    plt.close()
    log("  Ulozeno: confusion_matrix_normalized.png")
    
    # 7. CLASS DISTRIBUTION (Train vs Test)
    log("\n7. Generovani Class Distribution...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    train_counts = df_train['target'].value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    
    class_names = ['DOWN\n(< -3%)', 'HOLD\n(±3%)', 'UP\n(> +3%)']
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    # Train
    ax1 = axes[0]
    bars1 = ax1.bar(class_names, [train_counts.get(i, 0) for i in range(3)], color=colors)
    ax1.set_ylabel('Počet vzorků', fontweight='bold')
    ax1.set_title(f'Trénovací data (n={len(df_train):,})', fontweight='bold')
    for bar, val in zip(bars1, [train_counts.get(i, 0) for i in range(3)]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{val:,}', ha='center', fontweight='bold')
    
    # Test
    ax2 = axes[1]
    bars2 = ax2.bar(class_names, [test_counts.get(i, 0) for i in range(3)], color=colors)
    ax2.set_ylabel('Počet vzorků', fontweight='bold')
    ax2.set_title(f'Testovací data (n={len(y_test):,})', fontweight='bold')
    for bar, val in zip(bars2, [test_counts.get(i, 0) for i in range(3)]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{val:,}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    log("  Ulozeno: class_distribution.png")
    
    # 8. PREDICTION CONFIDENCE ANALYSIS
    log("\n8. Generovani Prediction Confidence Analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confidence distribution (normalized to compare shapes, not absolute counts)
    ax1 = axes[0]
    correct_mask = y_test == y_pred
    ax1.hist(df_test.loc[correct_mask, 'pred_confidence'], bins=20, alpha=0.6, 
             label='Spravne predikce', color='#27ae60', density=True)
    ax1.hist(df_test.loc[~correct_mask, 'pred_confidence'], bins=20, alpha=0.6, 
             label='Spatne predikce', color='#e74c3c', density=True)
    ax1.set_xlabel('Confidence (pravdepodobnost nejistsi tridy)', fontweight='bold')
    ax1.set_ylabel('Hustota (relativni cetnost)', fontweight='bold')
    ax1.set_title('Distribuce Confidence: Spravne vs Spatne', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy by confidence bin
    ax2 = axes[1]
    df_test['confidence_bin'] = pd.cut(df_test['pred_confidence'], bins=5)
    df_test['correct'] = (y_test == y_pred).astype(int)
    
    accuracy_by_conf = df_test.groupby('confidence_bin', observed=True)['correct'].agg(['mean', 'count'])
    accuracy_by_conf = accuracy_by_conf[accuracy_by_conf['count'] > 10]  # Min 10 vzorků
    
    if len(accuracy_by_conf) > 0:
        x_labels = [f'{interval.left:.2f}-{interval.right:.2f}' for interval in accuracy_by_conf.index]
        bars = ax2.bar(range(len(accuracy_by_conf)), accuracy_by_conf['mean'] * 100, color='#3498db')
        ax2.set_xticks(range(len(accuracy_by_conf)))
        ax2.set_xticklabels(x_labels, rotation=45, ha='right')
        ax2.set_xlabel('Confidence rozsah', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Accuracy podle Confidence urovne', fontweight='bold')
        ax2.axhline(y=33.3, color='r', linestyle='--', label='Random baseline')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "prediction_confidence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    log("  Ulozeno: prediction_confidence.png")
    
    # 9. MONTHLY RETURNS HISTOGRAM
    log("\n9. Generovani Monthly Returns Histogram...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    returns = df_test['future_return'].dropna() * 100
    
    ax.hist(returns, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax.axvline(x=-3, color='#e74c3c', linestyle='--', linewidth=2, label='DOWN threshold (-3%)')
    ax.axvline(x=3, color='#27ae60', linestyle='--', linewidth=2, label='UP threshold (+3%)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Měsíční výnos (%)', fontweight='bold')
    ax.set_ylabel('Počet pozorování', fontweight='bold')
    ax.set_title('Distribuce měsíčních výnosů v testovacím období\n(ukazuje přirozené rozložení tříd)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "returns_histogram.png"), dpi=300, bbox_inches='tight')
    plt.close()
    log("  Ulozeno: returns_histogram.png")
    
    # =========================================================================
    # BACKTEST METRIKY
    # =========================================================================
    
    total_trades = (df_backtest['prediction'] != 1).sum()
    long_trades = (df_backtest['prediction'] == 2).sum()
    short_trades = (df_backtest['prediction'] == 0).sum()
    
    strategy_final = df_backtest['cumulative_strategy'].iloc[-1]
    buyhold_final = df_backtest['cumulative_buyhold'].iloc[-1]
    
    strategy_return_pct = (strategy_final - 1) * 100
    buyhold_return_pct = (buyhold_final - 1) * 100
    
    winning_trades = ((df_backtest['prediction'] != 1) & (df_backtest['strategy_return'] > 0)).sum()
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    log("\n" + "=" * 50)
    log("BACKTEST VYSLEDKY")
    log("=" * 50)
    log(f"  Celkem obchodu: {total_trades:,}")
    log(f"  Long pozice: {long_trades:,}")
    log(f"  Short pozice: {short_trades:,}")
    log(f"  Win Rate: {win_rate:.1f}%")
    log(f"  Strategy Return: {strategy_return_pct:.2f}%")
    log(f"  Buy & Hold Return: {buyhold_return_pct:.2f}%")
    log(f"  Outperformance: {strategy_return_pct - buyhold_return_pct:.2f}%")
    
    # 10. Finalni report
    log("\n" + "=" * 50)
    log("FINALNI VYSLEDKY")
    log("=" * 50)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    log(f"\nCelkove metriky:")
    log(f"  Accuracy:  {accuracy:.4f}")
    log(f"  Precision: {precision:.4f}")
    log(f"  Recall:    {recall:.4f}")
    log(f"  F1-Score:  {f1:.4f}")
    
    log(f"\nPer-sector vysledky:")
    for sector, metrics in sector_metrics.items():
        log(f"  {sector}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f} (n={metrics['n_samples']})")
    
    # Ulozit finalni report
    report = {
        'timestamp': datetime.now().isoformat(),
        'experiment': '150_tickers_5_sectors',
        'threshold': THRESHOLD,
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'sector_metrics': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                               for kk, vv in v.items()} 
                          for k, v in sector_metrics.items()},
        'n_test_samples': len(y_test),
        'backtest': {
            'total_trades': int(total_trades),
            'long_trades': int(long_trades),
            'short_trades': int(short_trades),
            'win_rate': float(win_rate),
            'strategy_return_pct': float(strategy_return_pct),
            'buyhold_return_pct': float(buyhold_return_pct),
            'outperformance_pct': float(strategy_return_pct - buyhold_return_pct)
        },
        'figures': [
            'confusion_matrix.png',
            'confusion_matrix_normalized.png',
            'roc_curves.png',
            'feature_importance.png',
            'sector_comparison.png',
            'equity_curve.png',
            'class_distribution.png',
            'prediction_confidence.png',
            'returns_histogram.png'
        ]
    }
    
    with open(os.path.join(FIGURES_DIR, "final_report.json"), 'w') as f:
        json.dump(report, f, indent=2)
    
    log(f"\nReport uložen: {os.path.join(FIGURES_DIR, 'final_report.json')}")
    
    return report

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Spustí celou pipeline"""
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print(" ML PIPELINE PRO KLASIFIKACI CENOVYCH POHYBU - 150 TICKERU")
    print(" 5 sektoru x 30 firem | 10 let dat | DOWN/HOLD/UP klasifikace")
    print("=" * 70)
    print(f"\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Výstup: {DATA_DIR}")
    print(f"Modely: {MODEL_DIR}")
    
    # Krok 1: Stažení OHLCV dat
    df_ohlcv = download_ohlcv()
    
    if df_ohlcv is None:
        log("Pipeline přerušena - chyba při stahování OHLCV dat.")
        return
    
    # Krok 2: Stažení fundamentů
    df_fund = download_fundamentals()
    
    if df_fund is None:
        log("Pipeline přerušena - chyba při stahování fundamentálních dat.")
        return
    
    # Krok 3: Trénink RF Regressor
    df_complete = train_rf_regressor()
    
    if df_complete is None:
        log("Pipeline přerušena - chyba při tréninku regressoru.")
        return
    
    # Krok 4: Trénink RF Classifier
    result = train_rf_classifier()
    
    if result is None:
        log("Pipeline přerušena - chyba při tréninku classifieru.")
        return
    
    # Krok 5: Hyperparameter tuning
    tuning_result = hyperparameter_tuning()
    
    # Krok 6: Finální evaluace
    final_report = final_evaluation()
    
    # Souhrn
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(" PIPELINE DOKONČENA!")
    print("=" * 70)
    print(f"\nČas běhu: {elapsed/60:.1f} minut")
    print(f"Konec: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nVýstupy:")
    print(f"  Data:   {DATA_DIR}")
    print(f"  Modely: {MODEL_DIR}")
    print(f"  Grafy:  {FIGURES_DIR}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
