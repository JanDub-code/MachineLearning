#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FÁZE 4: Doplnění Historických Fundamentálních Dat (2015-2024)
===============================================================

Tento skript používá natrénovaný AI model k predikci fundamentálních metrik
pro období, kde nejsou dostupné (2015-2024).

Proces:
1. Načte OHLCV data (2015-2025)
2. Načte reálné fundamenty (2024-2025)
3. Použije AI model k predikci fundamentů pro 2015-2024
4. Spojí predikované a reálné fundamenty
5. Vytvoří kompletní 10letý dataset

Výstup: data/complete/all_sectors_complete_10y.csv
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from joblib import load

warnings.filterwarnings('ignore')

# === KONFIGURACE ===
OHLCV_DIR = "../data_10y"
FUNDAMENTALS_DIR = "../data/fundamentals"
MODELS_DIR = "../models"
OUTPUT_DIR = "../data/complete"

# Features z OHLCV dat
OHLCV_FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'volatility', 'returns',
    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'sma_3', 'sma_6', 'sma_12',
    'ema_3', 'ema_6', 'ema_12',
    'volume_change'
]

# Target fundamentální metriky
FUNDAMENTAL_TARGETS = [
    'PE', 'PB', 'PS', 'EV_EBITDA',
    'ROE', 'ROA', 'Profit_Margin', 'Operating_Margin', 'Gross_Margin',
    'Debt_to_Equity', 'Current_Ratio', 'Quick_Ratio',
    'Revenue_Growth_YoY', 'Earnings_Growth_YoY'
]

# Hranice mezi predikovanými a reálnými daty
SPLIT_DATE = "2024-01-01"

def log(msg: str):
    """Logování s časovou značkou"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_dir(path: str):
    """Vytvoří složku pokud neexistuje"""
    os.makedirs(path, exist_ok=True)

def load_model_and_scaler():
    """Načte natrénovaný AI model a scaler"""
    log("🤖 Načítám AI model...")
    
    model_path = os.path.join(MODELS_DIR, "fundamental_predictor.pkl")
    scaler_path = os.path.join(MODELS_DIR, "feature_scaler.pkl")
    
    if not os.path.exists(model_path):
        log(f"❌ Model nenalezen: {model_path}")
        log("   Nejprve spusťte: python scripts/2_train_fundamental_predictor.py")
        sys.exit(1)
    
    if not os.path.exists(scaler_path):
        log(f"❌ Scaler nenalezen: {scaler_path}")
        sys.exit(1)
    
    model = load(model_path)
    scaler = load(scaler_path)
    
    log("✓ Model a scaler načteny")
    
    return model, scaler

def load_ohlcv_data():
    """Načte kompletní OHLCV data (2015-2025)"""
    log("📂 Načítám OHLCV data...")
    
    ohlcv_path = os.path.join(OHLCV_DIR, "all_sectors_full_10y.csv")
    
    if not os.path.exists(ohlcv_path):
        log(f"❌ Soubor nenalezen: {ohlcv_path}")
        sys.exit(1)
    
    df = pd.read_csv(ohlcv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    log(f"✓ Načteno {len(df)} záznamů")
    log(f"  • Období: {df['date'].min()} → {df['date'].max()}")
    log(f"  • Tickery: {df['ticker'].nunique()}")
    
    return df

def load_real_fundamentals():
    """Načte reálné fundamentální data (2024-2025)"""
    log("📂 Načítám reálné fundamentální data...")
    
    fund_path = os.path.join(FUNDAMENTALS_DIR, "all_sectors_fundamentals.csv")
    
    if not os.path.exists(fund_path):
        log(f"⚠️  Reálné fundamenty nenalezeny: {fund_path}")
        log("   Pokračuji bez nich (všechna data budou predikovaná)")
        return pd.DataFrame()
    
    df = pd.read_csv(fund_path)
    df['date'] = pd.to_datetime(df['date'])
    
    log(f"✓ Načteno {len(df)} záznamů")
    log(f"  • Období: {df['date'].min()} → {df['date'].max()}")
    
    return df

def predict_historical_fundamentals(ohlcv: pd.DataFrame, model, scaler):
    """
    Predikuje fundamenty pro historické období (2015-2024).
    """
    log("\n🔮 Predikuji historické fundamenty (2015-2024)...")
    
    # Filtrovat pouze historické období
    historical = ohlcv[ohlcv['date'] < SPLIT_DATE].copy()
    
    log(f"  • {len(historical)} záznamů k predikci")
    log(f"  • Období: {historical['date'].min()} → {historical['date'].max()}")
    
    # Příprava features
    X = historical[OHLCV_FEATURES].copy()
    
    # Odstranit nekonečné hodnoty a NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Najít platné řádky
    valid_mask = ~X.isna().any(axis=1)
    valid_indices = historical[valid_mask].index
    X_valid = X[valid_mask]
    
    log(f"  • Validních vzorků: {len(X_valid)} / {len(X)} ({len(X_valid)/len(X)*100:.1f}%)")
    
    # Standardizace
    X_scaled = scaler.transform(X_valid)
    
    # Predikce
    log("  • Spouštím predikci...")
    start_time = time.time()
    
    y_pred = model.predict(X_scaled)
    
    elapsed = time.time() - start_time
    log(f"  ✓ Predikce dokončena za {elapsed:.1f}s")
    
    # Vytvoření DataFrame s predikcemi
    pred_df = pd.DataFrame(y_pred, columns=FUNDAMENTAL_TARGETS, index=valid_indices)
    
    # Přidat k původním datům
    result = historical.copy()
    result[FUNDAMENTAL_TARGETS] = np.nan
    result.loc[valid_indices, FUNDAMENTAL_TARGETS] = pred_df
    result['data_source'] = 'predicted'
    
    return result

def merge_with_real_fundamentals(predicted: pd.DataFrame, real_fund: pd.DataFrame, ohlcv: pd.DataFrame):
    """
    Spojí predikované fundamenty s reálnými daty.
    """
    log("\n🔗 Spojuji predikované a reálné fundamenty...")
    
    if real_fund.empty:
        log("  ⚠️  Žádné reálné fundamenty, používám pouze predikce")
        return predicted
    
    # Reálná data pro období 2024-2025
    recent_ohlcv = ohlcv[ohlcv['date'] >= SPLIT_DATE].copy()
    
    # Merge s reálnými fundamenty
    # Pro každý ticker zvlášť s forward-fill
    recent_parts = []
    
    for ticker in recent_ohlcv['ticker'].unique():
        ohlcv_ticker = recent_ohlcv[recent_ohlcv['ticker'] == ticker].copy()
        ohlcv_ticker = ohlcv_ticker.sort_values('date').set_index('date')
        
        # Reálné fundamenty pro ticker
        fund_ticker = real_fund[real_fund['ticker'] == ticker].copy()
        
        if fund_ticker.empty:
            # Žádné reálné fundamenty pro tento ticker
            ohlcv_ticker[FUNDAMENTAL_TARGETS] = np.nan
            ohlcv_ticker['data_source'] = 'none'
        else:
            fund_ticker = fund_ticker.sort_values('date').set_index('date')
            
            # Merge s forward-fill
            merged = ohlcv_ticker.join(fund_ticker[FUNDAMENTAL_TARGETS], how='left')
            merged[FUNDAMENTAL_TARGETS] = merged[FUNDAMENTAL_TARGETS].fillna(method='ffill')
            merged['data_source'] = 'real'
            
            ohlcv_ticker = merged
        
        ohlcv_ticker = ohlcv_ticker.reset_index()
        recent_parts.append(ohlcv_ticker)
    
    recent_with_fundamentals = pd.concat(recent_parts, ignore_index=True)
    
    log(f"  • Reálných záznamů: {len(recent_with_fundamentals)}")
    
    # Spojit historické (predikované) a recentní (reálné)
    complete = pd.concat([predicted, recent_with_fundamentals], ignore_index=True)
    complete = complete.sort_values(['ticker', 'date'])
    
    log(f"  ✓ Kompletní dataset: {len(complete)} záznamů")
    log(f"    • Predikované: {(complete['data_source'] == 'predicted').sum()}")
    log(f"    • Reálné: {(complete['data_source'] == 'real').sum()}")
    
    return complete

def validate_predictions(complete: pd.DataFrame):
    """
    Validuje predikce - kontroluje rozumnost hodnot.
    """
    log("\n✅ Validace predikcí...")
    
    # Statistiky pro predikované vs. reálné
    pred_data = complete[complete['data_source'] == 'predicted']
    real_data = complete[complete['data_source'] == 'real']
    
    if real_data.empty:
        log("  ⚠️  Žádné reálné data k porovnání")
        return
    
    log("\n  📊 Srovnání predikovaných vs. reálných hodnot:")
    log(f"  {'Metrika':<25} {'Predikované (mean)':<20} {'Reálné (mean)':<20} {'Rozdíl %'}")
    log("  " + "-"*85)
    
    for col in FUNDAMENTAL_TARGETS:
        pred_mean = pred_data[col].mean()
        real_mean = real_data[col].mean()
        
        if pd.notna(pred_mean) and pd.notna(real_mean) and real_mean != 0:
            diff_pct = abs(pred_mean - real_mean) / abs(real_mean) * 100
            log(f"  {col:<25} {pred_mean:<20.4f} {real_mean:<20.4f} {diff_pct:>6.1f}%")
        else:
            log(f"  {col:<25} {pred_mean:<20.4f} {real_mean:<20.4f} {'N/A':>6}")

def save_complete_dataset(complete: pd.DataFrame):
    """Uloží kompletní dataset"""
    log("\n💾 Ukládám kompletní dataset...")
    
    ensure_dir(OUTPUT_DIR)
    
    # Celý dataset
    output_path = os.path.join(OUTPUT_DIR, "all_sectors_complete_10y.csv")
    complete.to_csv(output_path, index=False)
    log(f"✓ {output_path}")
    
    # Po sektorech
    for sector in complete['sector'].unique():
        sector_df = complete[complete['sector'] == sector]
        sector_path = os.path.join(OUTPUT_DIR, f"{sector}_complete_10y.csv")
        sector_df.to_csv(sector_path, index=False)
        log(f"✓ {sector_path} ({len(sector_df)} záznamů)")
    
    # Statistiky
    log("\n📈 STATISTIKY:")
    log(f"  • Celkem záznamů: {len(complete)}")
    log(f"  • Tickery: {complete['ticker'].nunique()}")
    log(f"  • Období: {complete['date'].min()} → {complete['date'].max()}")
    log(f"  • Sektory: {', '.join(complete['sector'].unique())}")
    log(f"  • Predikované: {(complete['data_source'] == 'predicted').sum()}")
    log(f"  • Reálné: {(complete['data_source'] == 'real').sum()}")
    
    # Chybějící data
    log("\n⚠️  CHYBĚJÍCÍ DATA:")
    missing_pct = (complete[FUNDAMENTAL_TARGETS].isnull().sum() / len(complete) * 100).sort_values(ascending=False)
    for col, pct in missing_pct.items():
        if pct > 0:
            log(f"  • {col}: {pct:.1f}%")

def main():
    log("="*80)
    log("FÁZE 4: DOPLNĚNÍ HISTORICKÝCH FUNDAMENTÁLNÍCH DAT")
    log("="*80)
    
    start_time = time.time()
    
    # 1. Načtení modelu
    model, scaler = load_model_and_scaler()
    
    # 2. Načtení dat
    ohlcv = load_ohlcv_data()
    real_fundamentals = load_real_fundamentals()
    
    # 3. Predikce historických fundamentů (2015-2024)
    predicted = predict_historical_fundamentals(ohlcv, model, scaler)
    
    # 4. Spojení s reálnými daty (2024-2025)
    complete = merge_with_real_fundamentals(predicted, real_fundamentals, ohlcv)
    
    # 5. Validace
    validate_predictions(complete)
    
    # 6. Uložení
    save_complete_dataset(complete)
    
    elapsed = time.time() - start_time
    
    log("\n" + "="*80)
    log("✅ HOTOVO!")
    log("="*80)
    log(f"⏱  Celkový čas: {elapsed/60:.1f} minut")
    log(f"📊 Vytvořen kompletní 10letý dataset s OHLCV + Fundamenty")
    
    log("\n" + "="*80)
    log("Další krok: python scripts/4_train_price_predictor.py")
    log("="*80)

if __name__ == "__main__":
    main()
