#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FÁZE 3: Trénování AI Modelu pro Predikci Fundamentů
====================================================

Tento skript trénuje Random Forest model, který se naučí predikovat
fundamentální metriky z OHLCV dat a technických indikátorů.

Input: OHLCV + technické indikátory (2024-2025)
Output: 15 fundamentálních metrik (P/E, ROE, atd.)

Model: Multi-output Random Forest Regressor

Výstup: models/fundamental_predictor.pkl
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# === KONFIGURACE ===
OHLCV_DIR = "../data_10y"
FUNDAMENTALS_DIR = "../data/fundamentals"
OUTPUT_DIR = "../models"
ANALYSIS_DIR = "../data/analysis"

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
    'PE', 'PB', 'PS', 'EV_EBITDA',  # Valuační
    'ROE', 'ROA', 'Profit_Margin', 'Operating_Margin', 'Gross_Margin',  # Profitabilita
    'Debt_to_Equity', 'Current_Ratio', 'Quick_Ratio',  # Finanční zdraví
    'Revenue_Growth_YoY', 'Earnings_Growth_YoY'  # Růst
]

# Hyperparametry Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': 0
}

def log(msg: str):
    """Logování s časovou značkou"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_dir(path: str):
    """Vytvoří složku pokud neexistuje"""
    os.makedirs(path, exist_ok=True)

def load_ohlcv_data() -> pd.DataFrame:
    """Načte OHLCV data s technickými indikátory (2015-2025)"""
    log("📂 Načítám OHLCV data...")
    
    ohlcv_path = os.path.join(OHLCV_DIR, "all_sectors_full_10y.csv")
    
    if not os.path.exists(ohlcv_path):
        log(f"❌ Soubor nenalezen: {ohlcv_path}")
        sys.exit(1)
    
    df = pd.read_csv(ohlcv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    log(f"✓ Načteno {len(df)} záznamů ({df['date'].min()} → {df['date'].max()})")
    
    return df

def load_fundamentals() -> pd.DataFrame:
    """Načte fundamentální data (2024-2025)"""
    log("📂 Načítám fundamentální data...")
    
    fund_path = os.path.join(FUNDAMENTALS_DIR, "all_sectors_fundamentals.csv")
    
    if not os.path.exists(fund_path):
        log(f"❌ Soubor nenalezen: {fund_path}")
        log("   Nejprve spusťte: python scripts/1_download_fundamentals.py")
        sys.exit(1)
    
    df = pd.read_csv(fund_path)
    df['date'] = pd.to_datetime(df['date'])
    
    log(f"✓ Načteno {len(df)} záznamů ({df['date'].min()} → {df['date'].max()})")
    
    return df

def merge_ohlcv_fundamentals(ohlcv: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Spojí OHLCV data s fundamentálními daty.
    Používá forward-fill pro fundamenty (quarterly → monthly).
    """
    log("🔗 Spojuji OHLCV a fundamentální data...")
    
    # Převést fundamenty na měsíční frekvenci (forward-fill)
    fundamentals = fundamentals.sort_values('date')
    
    # Pro každý ticker zvlášť
    merged_parts = []
    
    for ticker in fundamentals['ticker'].unique():
        # OHLCV pro ticker
        ohlcv_ticker = ohlcv[ohlcv['ticker'] == ticker].copy()
        ohlcv_ticker = ohlcv_ticker.sort_values('date').set_index('date')
        
        # Fundamenty pro ticker
        fund_ticker = fundamentals[fundamentals['ticker'] == ticker].copy()
        fund_ticker = fund_ticker.sort_values('date').set_index('date')
        
        # Merge s forward-fill
        merged = ohlcv_ticker.join(fund_ticker[FUNDAMENTAL_TARGETS], how='left')
        merged[FUNDAMENTAL_TARGETS] = merged[FUNDAMENTAL_TARGETS].fillna(method='ffill')
        
        merged = merged.reset_index()
        merged_parts.append(merged)
    
    result = pd.concat(merged_parts, ignore_index=True)
    
    # Filtrovat pouze období kde máme fundamenty (2024-2025)
    result = result[result['date'] >= '2024-01-01'].copy()
    
    # Odstranit řádky s chybějícími daty
    result = result.dropna(subset=OHLCV_FEATURES + FUNDAMENTAL_TARGETS)
    
    log(f"✓ Spojeno: {len(result)} záznamů")
    log(f"  • Tickery: {result['ticker'].nunique()}")
    log(f"  • Období: {result['date'].min()} → {result['date'].max()}")
    
    return result

def prepare_training_data(df: pd.DataFrame):
    """
    Připraví data pro trénování.
    Vrací: X_train, X_test, y_train, y_test, scaler
    """
    log("🔧 Příprava trénovacích dat...")
    
    # Features a targets
    X = df[OHLCV_FEATURES].copy()
    y = df[FUNDAMENTAL_TARGETS].copy()
    
    # Odstranit nekonečné hodnoty
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    
    # Dropnout NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]
    
    log(f"✓ Validních vzorků: {len(X)}")
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    log(f"  • Train: {len(X_train)} vzorků")
    log(f"  • Test: {len(X_test)} vzorků")
    
    # Standardizace features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler

def train_random_forest(X_train, y_train):
    """Trénuje Multi-output Random Forest model"""
    log("🤖 Trénování Random Forest modelu...")
    log(f"  • Parametry: {RF_PARAMS}")
    
    start_time = time.time()
    
    # Multi-output Random Forest
    model = MultiOutputRegressor(
        RandomForestRegressor(**RF_PARAMS)
    )
    
    model.fit(X_train, y_train)
    
    elapsed = time.time() - start_time
    log(f"✓ Trénování dokončeno za {elapsed:.1f}s")
    
    return model

def evaluate_model(model, X_test, y_test, feature_names, target_names):
    """Evaluace modelu na testovacích datech"""
    log("\n📊 Evaluace modelu...")
    
    # Predikce
    y_pred = model.predict(X_test)
    
    # Metriky pro každý target
    results = []
    
    for i, target in enumerate(target_names):
        y_true_i = y_test[:, i]
        y_pred_i = y_pred[:, i]
        
        mae = mean_absolute_error(y_true_i, y_pred_i)
        rmse = np.sqrt(mean_squared_error(y_true_i, y_pred_i))
        r2 = r2_score(y_true_i, y_pred_i)
        
        # Relativní MAE (%)
        mean_val = np.abs(y_true_i).mean()
        mae_pct = (mae / mean_val * 100) if mean_val > 0 else np.nan
        
        results.append({
            'target': target,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mae_pct': mae_pct
        })
        
        log(f"  • {target:20s}: MAE={mae:8.3f} ({mae_pct:5.1f}%)  RMSE={rmse:8.3f}  R²={r2:6.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Celkový průměr
    log(f"\n  📈 PRŮMĚR:")
    log(f"     MAE: {results_df['mae'].mean():.3f}")
    log(f"     MAE%: {results_df['mae_pct'].mean():.1f}%")
    log(f"     RMSE: {results_df['rmse'].mean():.3f}")
    log(f"     R²: {results_df['r2'].mean():.3f}")
    
    return results_df, y_pred

def extract_feature_importance(model, feature_names, target_names):
    """
    Extrahuje feature importance pro každý target.
    """
    log("\n🔍 Analýza Feature Importance...")
    
    importance_data = []
    
    for i, estimator in enumerate(model.estimators_):
        target = target_names[i]
        importances = estimator.feature_importances_
        
        for j, feature in enumerate(feature_names):
            importance_data.append({
                'target': target,
                'feature': feature,
                'importance': importances[j]
            })
    
    importance_df = pd.DataFrame(importance_data)
    
    # Top 5 features pro každý target
    for target in target_names:
        target_imp = importance_df[importance_df['target'] == target].sort_values('importance', ascending=False)
        top5 = target_imp.head(5)
        
        log(f"\n  {target}:")
        for _, row in top5.iterrows():
            log(f"    • {row['feature']:15s}: {row['importance']:.4f}")
    
    return importance_df

def save_results(model, scaler, metrics_df, importance_df, y_test, y_pred, target_names):
    """Uložení modelu, metrik a analýz"""
    log("\n💾 Ukládám výsledky...")
    
    ensure_dir(OUTPUT_DIR)
    ensure_dir(ANALYSIS_DIR)
    
    # 1. Model a scaler
    model_path = os.path.join(OUTPUT_DIR, "fundamental_predictor.pkl")
    scaler_path = os.path.join(OUTPUT_DIR, "feature_scaler.pkl")
    
    dump(model, model_path)
    dump(scaler, scaler_path)
    
    log(f"✓ Model: {model_path}")
    log(f"✓ Scaler: {scaler_path}")
    
    # 2. Metriky
    metrics_path = os.path.join(ANALYSIS_DIR, "fundamental_predictor_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    log(f"✓ Metriky: {metrics_path}")
    
    # 3. Feature importance
    importance_path = os.path.join(ANALYSIS_DIR, "feature_importance_fundamentals.csv")
    importance_df.to_csv(importance_path, index=False)
    log(f"✓ Feature Importance: {importance_path}")
    
    # 4. Predictions vs Actual
    pred_df = pd.DataFrame(y_test, columns=[f"{t}_true" for t in target_names])
    pred_df_pred = pd.DataFrame(y_pred, columns=[f"{t}_pred" for t in target_names])
    pred_df = pd.concat([pred_df, pred_df_pred], axis=1)
    
    pred_path = os.path.join(ANALYSIS_DIR, "fundamental_predictions_vs_actual.csv")
    pred_df.to_csv(pred_path, index=False)
    log(f"✓ Predictions: {pred_path}")

def main():
    log("="*80)
    log("FÁZE 3: TRÉNOVÁNÍ AI MODELU PRO PREDIKCI FUNDAMENTŮ")
    log("="*80)
    
    start_time = time.time()
    
    # 1. Načtení dat
    ohlcv = load_ohlcv_data()
    fundamentals = load_fundamentals()
    
    # 2. Spojení dat
    merged = merge_ohlcv_fundamentals(ohlcv, fundamentals)
    
    # 3. Příprava trénovacích dat
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(merged)
    
    # 4. Trénování modelu
    model = train_random_forest(X_train, y_train)
    
    # 5. Evaluace
    metrics_df, y_pred = evaluate_model(model, X_test, y_test, OHLCV_FEATURES, FUNDAMENTAL_TARGETS)
    
    # 6. Feature importance
    importance_df = extract_feature_importance(model, OHLCV_FEATURES, FUNDAMENTAL_TARGETS)
    
    # 7. Uložení výsledků
    save_results(model, scaler, metrics_df, importance_df, y_test, y_pred, FUNDAMENTAL_TARGETS)
    
    elapsed = time.time() - start_time
    
    log("\n" + "="*80)
    log("✅ HOTOVO!")
    log("="*80)
    log(f"⏱  Celkový čas: {elapsed/60:.1f} minut")
    log(f"🎯 Průměrná přesnost: {metrics_df['mae_pct'].mean():.1f}% MAE")
    log(f"📊 R² score: {metrics_df['r2'].mean():.3f}")
    
    # Doporučení
    avg_mae_pct = metrics_df['mae_pct'].mean()
    if avg_mae_pct < 15:
        log("\n✨ Výborně! Model dosáhl cílové přesnosti (<15% MAE)")
    elif avg_mae_pct < 20:
        log("\n👍 Dobře! Model je použitelný (15-20% MAE)")
    else:
        log("\n⚠️  Model má vyšší chybu (>20% MAE). Zvažte:")
        log("   • Více dat (delší období)")
        log("   • Hyperparameter tuning")
        log("   • Feature engineering")
    
    log("\n" + "="*80)
    log("Další krok: python scripts/3_complete_historical_data.py")
    log("="*80)

if __name__ == "__main__":
    main()
