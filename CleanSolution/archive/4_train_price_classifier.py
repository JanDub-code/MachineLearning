#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FÁZE 5: Klasifikace Cenových Pohybů (Ternární Klasifikace)
============================================================

Tento skript implementuje klasifikační přístup k predikci cenových pohybů
místo regresní predikce přesné ceny.

Klasifikační Problém:
- Třída 0 (DOWN):  return < -3%
- Třída 1 (HOLD):  -3% ≤ return ≤ +3%  
- Třída 2 (UP):    return > +3%

Model: Random Forest Classifier (per-sector)

Výhody oproti regresi:
1. Praktická interpretace (BUY/HOLD/SELL signály)
2. Robustnost vůči outlierům
3. Jasné evaluační metriky (Precision, Recall, F1)
4. Přímá aplikovatelnost pro trading strategie

Výstup: models/{Sector}_price_classifier.pkl
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV
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
PREDICTIONS_DIR = "../data/predictions"

# Threshold pro klasifikaci (3% = minimální profitabilní pohyb po transakčních nákladech)
CLASSIFICATION_THRESHOLD = 0.03

# Features pro predikci
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

# Hyperparametry Random Forest Classifier
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': 0
}

# Sektory
SECTORS = ["Technology", "Consumer", "Industrials"]

# Názvy tříd
CLASS_NAMES = ['DOWN', 'HOLD', 'UP']


def log(msg: str):
    """Logování s časovou značkou"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dir(path: str):
    """Vytvoří složku pokud neexistuje"""
    os.makedirs(path, exist_ok=True)


def load_complete_data() -> pd.DataFrame:
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


def create_classification_target(df: pd.DataFrame, threshold: float = 0.03) -> pd.DataFrame:
    """
    Vytvoří ternární klasifikační target.
    
    Třídy:
    - 0 (DOWN):  return < -threshold
    - 1 (HOLD):  -threshold ≤ return ≤ +threshold
    - 2 (UP):    return > +threshold
    
    Args:
        df: DataFrame s cenami
        threshold: Hranice pro DOWN/UP (default 3%)
    
    Returns:
        DataFrame s přidaným sloupcem 'target'
    """
    log(f"🎯 Vytvářím klasifikační target (threshold = ±{threshold*100:.1f}%)...")
    
    # Seřadit podle tickeru a data
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Výpočet měsíčního výnosu (return pro NÁSLEDUJÍCÍ měsíc)
    df['return_next_month'] = df.groupby('ticker')['close'].pct_change(-1) * -1
    
    # Klasifikace do tříd
    conditions = [
        df['return_next_month'] < -threshold,  # DOWN
        df['return_next_month'] > threshold    # UP
    ]
    choices = [0, 2]
    
    df['target'] = np.select(conditions, choices, default=1)  # HOLD je default
    
    # Odstranit poslední měsíc pro každý ticker (nemá target)
    df = df.dropna(subset=['return_next_month'])
    
    # Odstranit řádky s chybějícími features
    df = df.dropna(subset=ALL_FEATURES)
    
    # Odstranit nekonečné hodnoty
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=ALL_FEATURES + ['target'])
    
    # Statistiky tříd
    class_counts = df['target'].value_counts().sort_index()
    total = len(df)
    
    log(f"✓ Připraveno {total} vzorků")
    log(f"  • Třída 0 (DOWN): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/total*100:.1f}%)")
    log(f"  • Třída 1 (HOLD): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/total*100:.1f}%)")
    log(f"  • Třída 2 (UP):   {class_counts.get(2, 0):,} ({class_counts.get(2, 0)/total*100:.1f}%)")
    
    return df


def train_sector_classifier(sector_df: pd.DataFrame, sector_name: str) -> dict:
    """
    Trénuje Random Forest Classifier pro daný sektor.
    
    Používá chronologický split pro validní evaluaci.
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
    log(f"  • Test:  {len(test_df)} vzorků ({test_df['date'].min()} → {test_df['date'].max()})")
    
    # Distribuce tříd v train/test
    log(f"\n  📊 Distribuce tříd:")
    for name, data in [('Train', train_df), ('Test', test_df)]:
        dist = data['target'].value_counts(normalize=True).sort_index() * 100
        log(f"     {name}: DOWN={dist.get(0, 0):.1f}% | HOLD={dist.get(1, 0):.1f}% | UP={dist.get(2, 0):.1f}%")
    
    # Příprava X, y
    X_train = train_df[ALL_FEATURES].values
    y_train = train_df['target'].values.astype(int)
    
    X_test = test_df[ALL_FEATURES].values
    y_test = test_df['target'].values.astype(int)
    
    # Standardizace
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Trénování Random Forest Classifier
    log(f"\n  🤖 Trénuji Random Forest Classifier...")
    log(f"     Parametry: n_estimators={RF_PARAMS['n_estimators']}, max_depth={RF_PARAMS['max_depth']}")
    
    start_time = time.time()
    
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train_scaled, y_train)
    
    elapsed = time.time() - start_time
    log(f"  ✓ Trénování dokončeno za {elapsed:.1f}s")
    
    # Predikce
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)
    
    # === EVALUAČNÍ METRIKY ===
    log(f"\n  📈 EVALUAČNÍ METRIKY:")
    
    # Accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    log(f"\n     ACCURACY:")
    log(f"       Train: {train_acc:.4f} ({train_acc*100:.1f}%)")
    log(f"       Test:  {test_acc:.4f} ({test_acc*100:.1f}%)")
    
    # Baseline (random guess) = 1/3 = 33.3%
    baseline_acc = 1/3
    improvement = (test_acc - baseline_acc) / baseline_acc * 100
    log(f"       Baseline (random): {baseline_acc:.4f}")
    log(f"       Zlepšení: {improvement:+.1f}%")
    
    # Classification Report
    log(f"\n     CLASSIFICATION REPORT (Test Set):")
    report = classification_report(y_test, y_test_pred, target_names=CLASS_NAMES, output_dict=True)
    report_str = classification_report(y_test, y_test_pred, target_names=CLASS_NAMES)
    
    for line in report_str.split('\n'):
        if line.strip():
            log(f"       {line}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    log(f"\n     CONFUSION MATRIX:")
    log(f"                    Predicted")
    log(f"                    DOWN  HOLD   UP")
    log(f"       Actual DOWN  {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
    log(f"       Actual HOLD  {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
    log(f"       Actual UP    {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
    
    # Feature Importance
    importance_df = pd.DataFrame({
        'feature': ALL_FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    log(f"\n     TOP 10 FEATURES:")
    for i, row in importance_df.head(10).iterrows():
        log(f"       {row['feature']:25s}: {row['importance']:.4f}")
    
    # === UKLÁDÁNÍ ===
    # Model a scaler
    model_path = os.path.join(MODELS_DIR, f"{sector_name}_price_classifier.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{sector_name}_classifier_scaler.pkl")
    
    dump(model, model_path)
    dump(scaler, scaler_path)
    
    log(f"\n  💾 Uloženo:")
    log(f"     • {model_path}")
    log(f"     • {scaler_path}")
    
    # Predictions
    pred_df = test_df[['date', 'ticker', 'sector', 'close', 'return_next_month', 'target']].copy()
    pred_df['predicted_class'] = y_test_pred
    pred_df['predicted_class_name'] = pred_df['predicted_class'].map({0: 'DOWN', 1: 'HOLD', 2: 'UP'})
    pred_df['actual_class_name'] = pred_df['target'].map({0: 'DOWN', 1: 'HOLD', 2: 'UP'})
    pred_df['correct'] = (pred_df['target'] == pred_df['predicted_class']).astype(int)
    
    # Pravděpodobnosti
    pred_df['prob_DOWN'] = y_test_proba[:, 0]
    pred_df['prob_HOLD'] = y_test_proba[:, 1]
    pred_df['prob_UP'] = y_test_proba[:, 2]
    pred_df['confidence'] = y_test_proba.max(axis=1)
    
    pred_path = os.path.join(PREDICTIONS_DIR, f"{sector_name}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    
    # Feature importance
    importance_path = os.path.join(ANALYSIS_DIR, f"{sector_name}_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    
    # Confusion matrix jako CSV
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_path = os.path.join(ANALYSIS_DIR, f"{sector_name}_confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    
    return {
        'sector': sector_name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'baseline_accuracy': baseline_acc,
        'improvement_pct': improvement,
        'precision_macro': report['macro avg']['precision'],
        'recall_macro': report['macro avg']['recall'],
        'f1_macro': report['macro avg']['f1-score'],
        'precision_up': report['UP']['precision'],
        'recall_up': report['UP']['recall'],
        'f1_up': report['UP']['f1-score'],
        'precision_down': report['DOWN']['precision'],
        'recall_down': report['DOWN']['recall'],
        'f1_down': report['DOWN']['f1-score'],
        'n_train': len(train_df),
        'n_test': len(test_df),
        'model_path': model_path,
        'pred_path': pred_path
    }


def create_visualizations(metrics_df: pd.DataFrame, complete_df: pd.DataFrame):
    """Vytvoří souhrnné vizualizace"""
    log("\n📊 Vytvářím vizualizace...")
    
    ensure_dir(ANALYSIS_DIR)
    
    # 1. Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sectors = metrics_df['sector']
    x = np.arange(len(sectors))
    width = 0.25
    
    ax.bar(x - width, metrics_df['train_accuracy'], width, label='Train Accuracy', alpha=0.8, color='steelblue')
    ax.bar(x, metrics_df['test_accuracy'], width, label='Test Accuracy', alpha=0.8, color='darkorange')
    ax.bar(x + width, [metrics_df['baseline_accuracy'].iloc[0]]*len(sectors), width, 
           label='Baseline (Random)', alpha=0.5, color='gray')
    
    ax.set_xlabel('Sektor', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Klasifikátoru po Sektorech', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(sectors)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'sector_accuracy_comparison.png'), dpi=150)
    plt.close()
    
    log("  ✓ sector_accuracy_comparison.png")
    
    # 2. F1 Score per Class
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, class_name in enumerate(['DOWN', 'HOLD', 'UP']):
        ax = axes[i]
        col = f'f1_{class_name.lower()}' if class_name != 'HOLD' else 'f1_macro'
        
        if f'f1_{class_name.lower()}' in metrics_df.columns:
            values = metrics_df[f'f1_{class_name.lower()}']
        else:
            # Pro HOLD použijeme macro F1 jako proxy
            values = metrics_df['f1_macro']
        
        ax.bar(sectors, values if class_name != 'HOLD' else metrics_df['f1_macro'], 
               color=['red', 'gray', 'green'][i], alpha=0.7)
        ax.set_title(f'F1-Score: {class_name}', fontsize=12)
        ax.set_ylabel('F1-Score')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('F1-Score po Třídách a Sektorech', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'sector_f1_per_class.png'), dpi=150)
    plt.close()
    
    log("  ✓ sector_f1_per_class.png")
    
    # 3. Confusion Matrix Heatmaps
    fig, axes = plt.subplots(1, len(SECTORS), figsize=(15, 5))
    
    for i, sector in enumerate(SECTORS):
        ax = axes[i]
        cm_path = os.path.join(ANALYSIS_DIR, f"{sector}_confusion_matrix.csv")
        if os.path.exists(cm_path):
            cm = pd.read_csv(cm_path, index_col=0)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
            ax.set_title(f'{sector}', fontsize=12)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
    
    plt.suptitle('Confusion Matrices po Sektorech', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'confusion_matrices_all.png'), dpi=150)
    plt.close()
    
    log("  ✓ confusion_matrices_all.png")
    
    # 4. Feature Importance Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    all_importance = []
    for sector in SECTORS:
        imp_path = os.path.join(ANALYSIS_DIR, f"{sector}_feature_importance.csv")
        if os.path.exists(imp_path):
            imp_df = pd.read_csv(imp_path)
            imp_df['sector'] = sector
            all_importance.append(imp_df)
    
    if all_importance:
        combined_imp = pd.concat(all_importance)
        pivot_imp = combined_imp.pivot(index='feature', columns='sector', values='importance')
        pivot_imp['mean'] = pivot_imp.mean(axis=1)
        pivot_imp = pivot_imp.sort_values('mean', ascending=True)
        
        pivot_imp[SECTORS].plot(kind='barh', ax=ax, alpha=0.8)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Feature Importance po Sektorech', fontsize=14)
        ax.legend(title='Sektor')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, 'feature_importance_comparison.png'), dpi=150)
    plt.close()
    
    log("  ✓ feature_importance_comparison.png")


def analyze_trading_strategy(metrics_df: pd.DataFrame):
    """
    Analyzuje potenciální trading strategii založenou na predikcích.
    """
    log("\n💹 ANALÝZA TRADING STRATEGIE:")
    log("="*80)
    
    for sector in SECTORS:
        pred_path = os.path.join(PREDICTIONS_DIR, f"{sector}_predictions.csv")
        if not os.path.exists(pred_path):
            continue
        
        pred_df = pd.read_csv(pred_path)
        
        log(f"\n  {sector}:")
        
        # Strategie: Kup když model predikuje UP
        up_predictions = pred_df[pred_df['predicted_class'] == 2]
        if len(up_predictions) > 0:
            avg_return_when_up = up_predictions['return_next_month'].mean()
            hit_rate = (up_predictions['return_next_month'] > 0).mean()
            
            log(f"    Strategie 'BUY when UP predicted':")
            log(f"      • Počet signálů: {len(up_predictions)}")
            log(f"      • Průměrný return: {avg_return_when_up*100:+.2f}%")
            log(f"      • Hit rate (return > 0): {hit_rate*100:.1f}%")
        
        # Strategie: Prodej/Short když model predikuje DOWN
        down_predictions = pred_df[pred_df['predicted_class'] == 0]
        if len(down_predictions) > 0:
            avg_return_when_down = down_predictions['return_next_month'].mean()
            hit_rate = (down_predictions['return_next_month'] < 0).mean()
            
            log(f"    Strategie 'SELL when DOWN predicted':")
            log(f"      • Počet signálů: {len(down_predictions)}")
            log(f"      • Průměrný return akcie: {avg_return_when_down*100:+.2f}%")
            log(f"      • Hit rate (return < 0): {hit_rate*100:.1f}%")
        
        # High confidence predictions
        high_conf = pred_df[pred_df['confidence'] > 0.6]
        if len(high_conf) > 0:
            high_conf_acc = (high_conf['target'] == high_conf['predicted_class']).mean()
            log(f"    High Confidence (>60%) predictions:")
            log(f"      • Počet: {len(high_conf)}")
            log(f"      • Accuracy: {high_conf_acc*100:.1f}%")


def main():
    log("="*80)
    log("FÁZE 5: KLASIFIKACE CENOVÝCH POHYBŮ (TERNÁRNÍ)")
    log("="*80)
    log(f"Threshold: ±{CLASSIFICATION_THRESHOLD*100:.1f}%")
    log(f"Třídy: DOWN (<-{CLASSIFICATION_THRESHOLD*100:.0f}%), HOLD, UP (>+{CLASSIFICATION_THRESHOLD*100:.0f}%)")
    log("="*80)
    
    start_time = time.time()
    
    ensure_dir(MODELS_DIR)
    ensure_dir(ANALYSIS_DIR)
    ensure_dir(PREDICTIONS_DIR)
    
    # 1. Načtení dat
    df = load_complete_data()
    
    # 2. Vytvoření klasifikačního targetu
    df_prepared = create_classification_target(df, threshold=CLASSIFICATION_THRESHOLD)
    
    # 3. Trénování klasifikátorů po sektorech
    all_metrics = []
    
    for sector in SECTORS:
        sector_df = df_prepared[df_prepared['sector'] == sector].copy()
        
        if sector_df.empty:
            log(f"⚠️  {sector}: Žádná data, přeskakuji")
            continue
        
        if len(sector_df) < 100:
            log(f"⚠️  {sector}: Málo dat ({len(sector_df)} vzorků), přeskakuji")
            continue
        
        metrics = train_sector_classifier(sector_df, sector)
        all_metrics.append(metrics)
    
    # 4. Souhrnné výsledky
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        # Uložení
        summary_path = os.path.join(ANALYSIS_DIR, "classification_metrics_summary.csv")
        metrics_df.to_csv(summary_path, index=False)
        
        log("\n" + "="*80)
        log("📊 SOUHRNNÉ VÝSLEDKY")
        log("="*80)
        
        for _, row in metrics_df.iterrows():
            log(f"\n{row['sector']}:")
            log(f"  Accuracy:     {row['test_accuracy']:.4f} ({row['test_accuracy']*100:.1f}%)")
            log(f"  vs Baseline:  {row['improvement_pct']:+.1f}% zlepšení")
            log(f"  Macro F1:     {row['f1_macro']:.4f}")
            log(f"  UP Precision: {row['precision_up']:.4f}")
            log(f"  UP Recall:    {row['recall_up']:.4f}")
        
        avg_acc = metrics_df['test_accuracy'].mean()
        avg_f1 = metrics_df['f1_macro'].mean()
        avg_improvement = metrics_df['improvement_pct'].mean()
        
        log("\n📈 PRŮMĚR VŠECH SEKTORŮ:")
        log(f"  • Accuracy: {avg_acc:.4f} ({avg_acc*100:.1f}%)")
        log(f"  • Macro F1: {avg_f1:.4f}")
        log(f"  • Zlepšení vs. baseline: {avg_improvement:+.1f}%")
        
        # Vizualizace
        create_visualizations(metrics_df, df_prepared)
        
        # Trading analýza
        analyze_trading_strategy(metrics_df)
        
        # Hodnocení
        log("\n" + "="*80)
        log("💡 HODNOCENÍ:")
        log("="*80)
        
        if avg_acc > 0.45:
            log("✨ Výborně! Model překonává baseline o >35%")
        elif avg_acc > 0.40:
            log("👍 Dobře! Model má solidní prediktivní schopnost")
        else:
            log("⚠️  Model má omezenou prediktivní schopnost. Zvažte:")
            log("   • Feature engineering")
            log("   • Hyperparameter tuning")
            log("   • Jiný threshold pro klasifikaci")
        
        if metrics_df['precision_up'].mean() > 0.5:
            log("✨ UP Precision > 50%: Když model říká BUY, je spolehlivý")
        
        if metrics_df['precision_down'].mean() > 0.5:
            log("✨ DOWN Precision > 50%: Když model říká SELL, je spolehlivý")
    
    elapsed = time.time() - start_time
    
    log("\n" + "="*80)
    log("✅ HOTOVO!")
    log("="*80)
    log(f"⏱  Celkový čas: {elapsed/60:.1f} minut")
    log(f"📊 Natrénováno {len(all_metrics)} klasifikátorů")
    log(f"💾 Výstupy:")
    log(f"   • Modely: {MODELS_DIR}/")
    log(f"   • Analýzy: {ANALYSIS_DIR}/")
    log(f"   • Predikce: {PREDICTIONS_DIR}/")
    
    log("\n" + "="*80)
    log("🎉 KOMPLETNÍ KLASIFIKAČNÍ PIPELINE DOKONČEN!")
    log("="*80)
    log("\nDalší kroky:")
    log("  • Prozkoumejte predikce v data/predictions/")
    log("  • Vizualizace v data/analysis/")
    log("  • Použijte modely pro nové predikce")


if __name__ == "__main__":
    main()
