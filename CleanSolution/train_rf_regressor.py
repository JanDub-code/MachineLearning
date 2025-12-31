#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trénink RF Regressor pro imputaci fundamentálních dat.

Pipeline:
1. Načíst OHLCV data (2015-2025)
2. Načíst fundamenty (aktuální snapshot)
3. Spojit data (match na ticker)
4. Natrénovat RF Regressor (OHLCV features → Fundamental targets)
5. Použít model k imputaci fundamentů pro historická data
6. Uložit kompletní dataset
"""

import os
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# === CESTY ===
BASE_DIR = r"c:\Users\Bc. Jan Dub\Desktop\GIT\MachineLearning\CleanSolution"
OHLCV_FILE = os.path.join(BASE_DIR, "data_10y", "all_sectors_full_10y.csv")
FUND_FILE = os.path.join(BASE_DIR, "data", "fundamentals", "all_sectors_fundamentals.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "complete")

# === KONFIGURACE ===
# OHLCV features pro predikci
OHLCV_FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'volatility', 'returns', 'rsi_14', 
    'macd', 'macd_signal', 'macd_hist',
    'sma_3', 'sma_6', 'sma_12',
    'ema_3', 'ema_6', 'ema_12',
    'volume_change'
]

# Fundamentální targets pro predikci
FUND_TARGETS = [
    'trailingPE', 'forwardPE', 'priceToBook', 
    'returnOnEquity', 'returnOnAssets',
    'profitMargins', 'operatingMargins', 'grossMargins',
    'debtToEquity', 'currentRatio',
    'beta'
]

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    log("=" * 60)
    log("TRÉNINK RF REGRESSOR PRO IMPUTACI FUNDAMENTŮ")
    log("=" * 60)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # === 1. NAČÍST DATA ===
    log("\n1. Načítání dat...")
    
    df_ohlcv = pd.read_csv(OHLCV_FILE)
    df_fund = pd.read_csv(FUND_FILE)
    
    log(f"   OHLCV: {len(df_ohlcv)} řádků, {df_ohlcv['ticker'].nunique()} tickerů")
    log(f"   Fundamenty: {len(df_fund)} tickerů, {len(df_fund.columns)} sloupců")
    
    # === 2. SPOJIT DATA ===
    log("\n2. Spojování dat...")
    
    # Pro každý ticker přiřadíme jeho fundamenty ke všem měsíčním záznamům
    # (fundamenty jsou snapshot, předpokládáme že jsou relativně stabilní)
    
    # Vyber pouze potřebné sloupce z fundamentů
    fund_cols = ['ticker'] + [c for c in FUND_TARGETS if c in df_fund.columns]
    df_fund_subset = df_fund[fund_cols].copy()
    
    # Merge
    df_merged = df_ohlcv.merge(df_fund_subset, on='ticker', how='left')
    
    log(f"   Merged: {len(df_merged)} řádků")
    
    # === 3. PŘIPRAVIT TRÉNOVACÍ DATA ===
    log("\n3. Příprava trénovacích dat...")
    
    # Vezmi pouze posledních 24 měsíců pro každý ticker (kde máme "reálné" fundamenty)
    df_merged['date'] = pd.to_datetime(df_merged['date'], utc=True)
    cutoff_date = df_merged['date'].max() - pd.DateOffset(months=24)
    
    df_recent = df_merged[df_merged['date'] >= cutoff_date].copy()
    df_historical = df_merged[df_merged['date'] < cutoff_date].copy()
    
    log(f"   Recent (training): {len(df_recent)} řádků")
    log(f"   Historical (to predict): {len(df_historical)} řádků")
    
    # Odstraň řádky s NaN v features nebo targets
    available_features = [f for f in OHLCV_FEATURES if f in df_recent.columns]
    available_targets = [t for t in FUND_TARGETS if t in df_recent.columns]
    
    log(f"   Features: {len(available_features)}")
    log(f"   Targets: {len(available_targets)}")
    
    # Příprava X, y
    df_train = df_recent.dropna(subset=available_features + available_targets)
    
    X = df_train[available_features].values
    y = df_train[available_targets].values
    
    log(f"   Training samples: {len(X)}")
    
    # === 4. TRÉNINK MODELU ===
    log("\n4. Trénink RF Regressor...")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model
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
    
    log("\n   Výsledky (test set):")
    for i, target in enumerate(available_targets):
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        log(f"     {target}: MAE={mae:.3f}, R²={r2:.3f}")
    
    # === 5. IMPUTACE HISTORICKÝCH DAT ===
    log("\n5. Imputace historických dat...")
    
    # Připrav historická data
    df_hist_clean = df_historical.dropna(subset=available_features).copy()
    X_hist = df_hist_clean[available_features].values
    X_hist_scaled = scaler.transform(X_hist)
    
    # Predikce
    y_hist_pred = model.predict(X_hist_scaled)
    
    # Přiřaď predikované hodnoty
    for i, target in enumerate(available_targets):
        df_hist_clean[target] = y_hist_pred[:, i]
    
    df_hist_clean['data_source'] = 'predicted'
    
    # Recent data - označit jako 'real'
    df_recent_clean = df_recent.dropna(subset=available_features + available_targets).copy()
    df_recent_clean['data_source'] = 'real'
    
    # Spojit
    df_complete = pd.concat([df_hist_clean, df_recent_clean], ignore_index=True)
    df_complete = df_complete.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    log(f"   Kompletní dataset: {len(df_complete)} řádků")
    log(f"   - Predicted: {(df_complete['data_source'] == 'predicted').sum()}")
    log(f"   - Real: {(df_complete['data_source'] == 'real').sum()}")
    
    # === 6. ULOŽIT ===
    log("\n6. Ukládání...")
    
    # Model
    model_file = os.path.join(MODEL_DIR, "fundamental_predictor.pkl")
    joblib.dump(model, model_file)
    log(f"   Model: {model_file}")
    
    # Scaler
    scaler_file = os.path.join(MODEL_DIR, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_file)
    log(f"   Scaler: {scaler_file}")
    
    # Kompletní dataset
    complete_file = os.path.join(OUTPUT_DIR, "all_sectors_complete_10y.csv")
    df_complete.to_csv(complete_file, index=False)
    log(f"   Data: {complete_file}")
    
    # Per-sector
    for sector in df_complete['sector'].unique():
        sector_df = df_complete[df_complete['sector'] == sector]
        sector_file = os.path.join(OUTPUT_DIR, f"{sector}_complete_10y.csv")
        sector_df.to_csv(sector_file, index=False)
        log(f"   {sector}: {len(sector_df)} řádků")
    
    # === 7. FEATURE IMPORTANCE ===
    log("\n7. Feature Importance:")
    
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': available_features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    for _, row in feat_imp.head(10).iterrows():
        log(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Uložit
    feat_imp_file = os.path.join(MODEL_DIR, "feature_importance.csv")
    feat_imp.to_csv(feat_imp_file, index=False)
    
    log("\n" + "=" * 60)
    log("HOTOVO!")
    log("=" * 60)
    
    return df_complete

if __name__ == "__main__":
    df = main()
