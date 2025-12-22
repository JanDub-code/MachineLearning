#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FÁZE 5: Trénování Modelu pro Predikci Cen
==========================================

Tento skript trénuje Ridge Regression model, který predikuje budoucí cenu akcie
na základě fundamentálních metrik a technických indikátorů.

Model je trénován samostatně pro každý sektor (Technology, Consumer, Industrials).

Input: Fundamentální metriky + technické indikátory
Output: log_price_next_month (logaritmovaná cena za měsíc)

Výstup: models/{Sector}_price_model.pkl
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# === KONFIGURACE ===
COMPLETE_DATA_DIR = "../data/complete"
MODELS_DIR = "../models"
ANALYSIS_DIR = "../data/analysis"

# Features pro predikci ceny
FUNDAMENTAL_FEATURES = [
    'PE', 'PB', 'PS', 'EV_EBITDA',
    'ROE', 'ROA', 'Profit_Margin', 'Operating_Margin', 'Gross_Margin',
    'Debt_to_Equity', 'Current_Ratio', 'Quick_Ratio',
    'Revenue_Growth_YoY', 'Earnings_Growth_YoY'
]

TECHNICAL_FEATURES = [
    'volatility', 'returns',
    'rsi_14', 'macd',
    'volume_change'
]

ALL_FEATURES = FUNDAMENTAL_FEATURES + TECHNICAL_FEATURES

# Hyperparametry Ridge Regression
RIDGE_ALPHA = 1.0

# Sektory
SECTORS = ["Technology", "Consumer", "Industrials"]

def log(msg: str):
    """Logování s časovou značkou"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def ensure_dir(path: str):
    """Vytvoří složku pokud neexistuje"""
    os.makedirs(path, exist_ok=True)

def load_complete_data():
    """Načte kompletní dataset s fundamenty"""
    log("📂 Načítám kompletní dataset...")
    
    data_path = os.path.join(COMPLETE_DATA_DIR, "all_sectors_complete_10y.csv")
    
    if not os.path.exists(data_path):
        log(f"❌ Soubor nenalezen: {data_path}")
        log("   Nejprve spusťte: python scripts/3_complete_historical_data.py")
        sys.exit(1)
    
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    log(f"✓ Načteno {len(df)} záznamů")
    log(f"  • Období: {df['date'].min()} → {df['date'].max()}")
    log(f"  • Tickery: {df['ticker'].nunique()}")
    log(f"  • Sektory: {', '.join(df['sector'].unique())}")
    
    return df

def prepare_price_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Připraví data pro predikci ceny:
    - Vytvoří target: log_price_next_month
    - Odstraní chybějící hodnoty
    """
    log("🔧 Příprava dat pro predikci ceny...")
    
    # Seřadit podle tickeru a data
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Vytvoření target: cena za měsíc (log)
    df['price_t'] = df['close']
    df['log_price_t'] = np.log(df['price_t'].replace(0, np.nan))
    
    # Shift price o 1 měsíc vpřed (pro každý ticker zvlášť)
    df['log_price_next_month'] = df.groupby('ticker')['log_price_t'].shift(-1)
    
    # Odstranit poslední měsíc (nemá target)
    df = df.dropna(subset=['log_price_next_month'])
    
    # Odstranit řádky s chybějícími features
    df = df.dropna(subset=ALL_FEATURES)
    
    # Odstranit nekonečné hodnoty
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=ALL_FEATURES + ['log_price_next_month'])
    
    log(f"✓ Připraveno {len(df)} validních vzorků")
    log(f"  • Features: {len(ALL_FEATURES)}")
    log(f"  • Target: log_price_next_month")
    
    return df

def train_sector_model(sector_df: pd.DataFrame, sector_name: str):
    """
    Trénuje Ridge Regression model pro daný sektor.
    """
    log(f"\n{'='*80}")
    log(f"📊 SEKTOR: {sector_name}")
    log(f"{'='*80}")
    
    # Chronologický split (80/20)
    sector_df = sector_df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(sector_df) * 0.8)
    
    train_df = sector_df.iloc[:split_idx]
    test_df = sector_df.iloc[split_idx:]
    
    log(f"  • Train: {len(train_df)} vzorků ({train_df['date'].min()} → {train_df['date'].max()})")
    log(f"  • Test: {len(test_df)} vzorků ({test_df['date'].min()} → {test_df['date'].max()})")
    
    # Příprava X, y
    X_train = train_df[ALL_FEATURES].values
    y_train = train_df['log_price_next_month'].values
    
    X_test = test_df[ALL_FEATURES].values
    y_test = test_df['log_price_next_month'].values
    
    # Standardizace
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Trénování Ridge Regression
    log(f"  🤖 Trénuji Ridge Regression (alpha={RIDGE_ALPHA})...")
    
    model = Ridge(alpha=RIDGE_ALPHA, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    log(f"  ✓ Trénování dokončeno")
    
    # Predikce
    y_train_pred_log = model.predict(X_train_scaled)
    y_test_pred_log = model.predict(X_test_scaled)
    
    # Převod z log zpět na ceny
    y_train_true = np.exp(y_train)
    y_train_pred = np.exp(y_train_pred_log)
    
    y_test_true = np.exp(y_test)
    y_test_pred = np.exp(y_test_pred_log)
    
    # Metriky
    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
    train_r2 = r2_score(y_train_true, y_train_pred)
    
    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    test_r2 = r2_score(y_test_true, y_test_pred)
    
    log(f"\n  📈 VÝSLEDKY:")
    log(f"     Train:  MAE=${train_mae:8.2f}  RMSE=${train_rmse:8.2f}  R²={train_r2:.4f}")
    log(f"     Test:   MAE=${test_mae:8.2f}  RMSE=${test_rmse:8.2f}  R²={test_r2:.4f}")
    
    # Feature importance (absolutní hodnoty koeficientů)
    coef_df = pd.DataFrame({
        'feature': ALL_FEATURES,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    log(f"\n  🔍 TOP 10 FEATURES:")
    for _, row in coef_df.head(10).iterrows():
        sign = '+' if row['coefficient'] > 0 else '-'
        log(f"     {sign} {row['feature']:25s}: {row['coefficient']:8.4f}")
    
    # Uložení modelu
    model_path = os.path.join(MODELS_DIR, f"{sector_name}_price_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{sector_name}_price_scaler.pkl")
    
    dump(model, model_path)
    dump(scaler, scaler_path)
    
    log(f"\n  💾 Uloženo:")
    log(f"     • {model_path}")
    log(f"     • {scaler_path}")
    
    # Uložení predictions
    pred_df = test_df[['date', 'ticker', 'sector']].copy()
    pred_df['price_true'] = y_test_true
    pred_df['price_pred'] = y_test_pred
    pred_df['error'] = pred_df['price_pred'] - pred_df['price_true']
    pred_df['error_pct'] = (pred_df['error'] / pred_df['price_true'] * 100)
    
    pred_path = os.path.join(ANALYSIS_DIR, f"{sector_name}_price_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    
    # Uložení koeficientů
    coef_path = os.path.join(ANALYSIS_DIR, f"{sector_name}_price_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    
    return {
        'sector': sector_name,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'n_train': len(train_df),
        'n_test': len(test_df),
        'model_path': model_path,
        'pred_path': pred_path,
        'coef_path': coef_path
    }

def create_summary_visualizations(metrics_df: pd.DataFrame):
    """Vytvoří souhrnné vizualizace"""
    log("\n📊 Vytvářím vizualizace...")
    
    ensure_dir(ANALYSIS_DIR)
    
    # 1. Srovnání MAE po sektorech
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sectors = metrics_df['sector']
    x = np.arange(len(sectors))
    width = 0.35
    
    ax.bar(x - width/2, metrics_df['train_mae'], width, label='Train MAE', alpha=0.8)
    ax.bar(x + width/2, metrics_df['test_mae'], width, label='Test MAE', alpha=0.8)
    
    ax.set_xlabel('Sektor')
    ax.set_ylabel('MAE ($)')
    ax.set_title('Mean Absolute Error po Sektorech')
    ax.set_xticks(x)
    ax.set_xticklabels(sectors)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'sector_mae_comparison.png'), dpi=150)
    plt.close()
    
    log("  ✓ sector_mae_comparison.png")
    
    # 2. R² score srovnání
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width/2, metrics_df['train_r2'], width, label='Train R²', alpha=0.8)
    ax.bar(x + width/2, metrics_df['test_r2'], width, label='Test R²', alpha=0.8)
    
    ax.set_xlabel('Sektor')
    ax.set_ylabel('R² Score')
    ax.set_title('R² Score po Sektorech')
    ax.set_xticks(x)
    ax.set_xticklabels(sectors)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.75, color='r', linestyle='--', alpha=0.5, label='Target (0.75)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'sector_r2_comparison.png'), dpi=150)
    plt.close()
    
    log("  ✓ sector_r2_comparison.png")

def main():
    log("="*80)
    log("FÁZE 5: TRÉNOVÁNÍ MODELU PRO PREDIKCI CEN")
    log("="*80)
    
    start_time = time.time()
    
    ensure_dir(MODELS_DIR)
    ensure_dir(ANALYSIS_DIR)
    
    # 1. Načtení dat
    df = load_complete_data()
    
    # 2. Příprava dat
    df_prepared = prepare_price_prediction_data(df)
    
    # 3. Trénování modelů po sektorech
    all_metrics = []
    
    for sector in SECTORS:
        sector_df = df_prepared[df_prepared['sector'] == sector].copy()
        
        if sector_df.empty:
            log(f"⚠️  {sector}: Žádná data, přeskakuji")
            continue
        
        if len(sector_df) < 100:
            log(f"⚠️  {sector}: Málo dat ({len(sector_df)} vzorků), přeskakuji")
            continue
        
        metrics = train_sector_model(sector_df, sector)
        all_metrics.append(metrics)
    
    # 4. Souhrnné metriky
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        # Uložení
        summary_path = os.path.join(ANALYSIS_DIR, "price_prediction_metrics_summary.csv")
        metrics_df.to_csv(summary_path, index=False)
        
        log("\n" + "="*80)
        log("📊 SOUHRNNÉ VÝSLEDKY")
        log("="*80)
        
        for _, row in metrics_df.iterrows():
            log(f"\n{row['sector']}:")
            log(f"  Test MAE:  ${row['test_mae']:.2f}")
            log(f"  Test RMSE: ${row['test_rmse']:.2f}")
            log(f"  Test R²:   {row['test_r2']:.4f}")
        
        avg_mae = metrics_df['test_mae'].mean()
        avg_r2 = metrics_df['test_r2'].mean()
        
        log("\n📈 PRŮMĚR VŠECH SEKTORŮ:")
        log(f"  • MAE:  ${avg_mae:.2f}")
        log(f"  • R²:   {avg_r2:.4f}")
        
        # Vizualizace
        create_summary_visualizations(metrics_df)
        
        # Hodnocení
        log("\n" + "="*80)
        log("💡 HODNOCENÍ:")
        log("="*80)
        
        if avg_mae < 15:
            log("✨ Výborně! Model dosáhl cílové přesnosti (MAE < $15)")
        elif avg_mae < 20:
            log("👍 Dobře! Model má slušnou přesnost (MAE < $20)")
        else:
            log("⚠️  Model má vyšší chybu (MAE > $20). Zvažte:")
            log("   • Hyperparameter tuning (grid search)")
            log("   • Feature selection")
            log("   • Ensemble metody")
        
        if avg_r2 > 0.75:
            log("✨ Výborně! Model vysvětluje >75% variance")
        elif avg_r2 > 0.60:
            log("👍 Dobře! Model vysvětluje >60% variance")
        else:
            log("⚠️  Model vysvětluje <60% variance")
    
    elapsed = time.time() - start_time
    
    log("\n" + "="*80)
    log("✅ HOTOVO!")
    log("="*80)
    log(f"⏱  Celkový čas: {elapsed/60:.1f} minut")
    log(f"📊 Natrénováno {len(all_metrics)} modelů")
    log(f"💾 Výstupy: {MODELS_DIR}/ a {ANALYSIS_DIR}/")
    
    log("\n" + "="*80)
    log("🎉 KOMPLETNÍ PIPELINE DOKONČEN!")
    log("="*80)
    log("\nDalší kroky:")
    log("  • Otevřete notebooks/ pro analýzu v Google Colab")
    log("  • Použijte natrénované modely k predikci nových dat")
    log("  • Prozkoumejte analýzy v data/analysis/")

if __name__ == "__main__":
    main()
