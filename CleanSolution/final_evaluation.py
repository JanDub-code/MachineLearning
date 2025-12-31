#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finální evaluace ML pipeline.

Generuje:
- Confusion matrix
- ROC křivky
- Feature importance
- Per-sector analýza
"""

import os
import time
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# === CESTY ===
BASE_DIR = r"c:\Users\Bc. Jan Dub\Desktop\GIT\MachineLearning\CleanSolution"
DATA_FILE = os.path.join(BASE_DIR, "data", "complete", "all_sectors_complete_10y.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "data", "figures")

# === KONFIGURACE ===
THRESHOLD = 0.03

FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'volatility', 'returns', 'rsi_14',
    'macd', 'macd_signal', 'macd_hist',
    'sma_3', 'sma_6', 'sma_12',
    'ema_3', 'ema_6', 'ema_12',
    'volume_change',
    'trailingPE', 'forwardPE', 'priceToBook',
    'returnOnEquity', 'returnOnAssets',
    'profitMargins', 'operatingMargins', 'grossMargins',
    'debtToEquity', 'currentRatio', 'beta'
]

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def create_target(df):
    df = df.copy()
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    df['future_close'] = df.groupby('ticker')['close'].shift(-1)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    
    def classify(ret):
        if pd.isna(ret):
            return np.nan
        elif ret < -THRESHOLD:
            return 0
        elif ret > THRESHOLD:
            return 2
        else:
            return 1
    
    df['target'] = df['future_return'].apply(classify)
    return df

def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Plot a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['DOWN', 'HOLD', 'UP'],
                yticklabels=['DOWN', 'HOLD', 'UP'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    
    return cm

def plot_roc_curves(y_true, y_proba, title, filename):
    """Plot ROC curves for multiclass classification."""
    n_classes = 3
    classes = ['DOWN', 'HOLD', 'UP']
    
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    for i, (class_name, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_feature_importance(model, features, title, filename):
    """Plot feature importance."""
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 12))
    plt.barh(feat_imp['feature'], feat_imp['importance'], color='steelblue')
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    
    return feat_imp

def plot_sector_comparison(sector_metrics, filename):
    """Plot sector comparison."""
    sectors = list(sector_metrics.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    x = np.arange(len(sectors))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [sector_metrics[s][metric] for s in sectors]
        ax.bar(x + i * width, values, width, label=metric.capitalize(), color=color)
    
    ax.set_xlabel('Sector')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance by Sector')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(sectors)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.axhline(y=0.333, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    log("=" * 60)
    log("FINÁLNÍ EVALUACE ML PIPELINE")
    log("=" * 60)
    
    # === SETUP ===
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # === 1. NAČÍST DATA ===
    log("\n1. Načítání dat a modelu...")
    
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = create_target(df)
    
    available_features = [f for f in FEATURES if f in df.columns]
    df_clean = df.dropna(subset=available_features + ['target']).copy()
    df_clean['target'] = df_clean['target'].astype(int)
    df_clean = df_clean.sort_values('date').reset_index(drop=True)
    
    # Chronologický split
    split_idx = int(len(df_clean) * 0.8)
    df_test = df_clean.iloc[split_idx:].copy()
    
    X_test = df_test[available_features].values
    y_test = df_test['target'].values
    
    log(f"   Test samples: {len(df_test)}")
    
    # Model a scaler
    # Zkusíme tuned model, pokud existuje, jinak základní
    tuned_model_path = os.path.join(MODEL_DIR, "rf_classifier_tuned.pkl")
    base_model_path = os.path.join(MODEL_DIR, "rf_classifier_all_sectors.pkl")
    
    tuned_scaler_path = os.path.join(MODEL_DIR, "classifier_scaler_tuned.pkl")
    base_scaler_path = os.path.join(MODEL_DIR, "classifier_scaler.pkl")
    
    if os.path.exists(tuned_model_path):
        model = joblib.load(tuned_model_path)
        scaler = joblib.load(tuned_scaler_path)
        model_type = "tuned"
        log("   Používám tuned model")
    else:
        model = joblib.load(base_model_path)
        scaler = joblib.load(base_scaler_path)
        model_type = "base"
        log("   Používám base model")
    
    X_test_scaled = scaler.transform(X_test)
    
    # === 2. PREDICTIONS ===
    log("\n2. Generování predikcí...")
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # === 3. CELKOVÉ METRIKY ===
    log("\n3. Celkové metriky:")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    log(f"\n   Accuracy:  {accuracy:.4f}")
    log(f"   Precision: {precision:.4f}")
    log(f"   Recall:    {recall:.4f}")
    log(f"   F1-Score:  {f1:.4f}")
    
    log("\n   Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['DOWN', 'HOLD', 'UP'])
    for line in report.split('\n'):
        if line.strip():
            log(f"   {line}")
    
    # === 4. CONFUSION MATRIX ===
    log("\n4. Confusion Matrix...")
    
    cm = plot_confusion_matrix(
        y_test, y_pred,
        f'Confusion Matrix - {model_type.upper()} Model',
        os.path.join(FIGURES_DIR, 'confusion_matrix.png')
    )
    log(f"   Uloženo: confusion_matrix.png")
    
    # === 5. ROC CURVES ===
    log("\n5. ROC Curves...")
    
    plot_roc_curves(
        y_test, y_proba,
        f'ROC Curves - {model_type.upper()} Model',
        os.path.join(FIGURES_DIR, 'roc_curves.png')
    )
    log(f"   Uloženo: roc_curves.png")
    
    # === 6. FEATURE IMPORTANCE ===
    log("\n6. Feature Importance...")
    
    feat_imp = plot_feature_importance(
        model, available_features,
        f'Feature Importance - {model_type.upper()} Model',
        os.path.join(FIGURES_DIR, 'feature_importance.png')
    )
    log(f"   Uloženo: feature_importance.png")
    
    log("\n   Top 10 Features:")
    for _, row in feat_imp.tail(10).iloc[::-1].iterrows():
        log(f"   {row['feature']}: {row['importance']:.4f}")
    
    # === 7. PER-SECTOR ANALÝZA ===
    log("\n7. Per-sector analýza...")
    
    sector_metrics = {}
    
    for sector in df_test['sector'].unique():
        sector_mask = df_test['sector'] == sector
        sector_indices = np.where(sector_mask)[0]
        
        y_sector = y_test[sector_indices]
        y_pred_sector = y_pred[sector_indices]
        
        sector_metrics[sector] = {
            'accuracy': accuracy_score(y_sector, y_pred_sector),
            'precision': precision_score(y_sector, y_pred_sector, average='weighted', zero_division=0),
            'recall': recall_score(y_sector, y_pred_sector, average='weighted', zero_division=0),
            'f1': f1_score(y_sector, y_pred_sector, average='weighted', zero_division=0),
            'samples': len(sector_indices)
        }
        
        log(f"\n   {sector}:")
        log(f"      Accuracy:  {sector_metrics[sector]['accuracy']:.4f}")
        log(f"      Precision: {sector_metrics[sector]['precision']:.4f}")
        log(f"      Recall:    {sector_metrics[sector]['recall']:.4f}")
        log(f"      F1-Score:  {sector_metrics[sector]['f1']:.4f}")
        log(f"      Samples:   {sector_metrics[sector]['samples']}")
    
    # Sector comparison plot
    plot_sector_comparison(
        sector_metrics,
        os.path.join(FIGURES_DIR, 'sector_comparison.png')
    )
    log(f"\n   Uloženo: sector_comparison.png")
    
    # === 8. ULOŽIT VÝSLEDKY ===
    log("\n8. Ukládání výsledků...")
    
    results = {
        'model_type': model_type,
        'overall': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'test_samples': len(df_test)
        },
        'per_class': {
            'DOWN': {
                'precision': cm[0,0] / max(cm[:,0].sum(), 1),
                'recall': cm[0,0] / max(cm[0,:].sum(), 1),
            },
            'HOLD': {
                'precision': cm[1,1] / max(cm[:,1].sum(), 1),
                'recall': cm[1,1] / max(cm[1,:].sum(), 1),
            },
            'UP': {
                'precision': cm[2,2] / max(cm[:,2].sum(), 1),
                'recall': cm[2,2] / max(cm[2,:].sum(), 1),
            }
        },
        'per_sector': sector_metrics,
        'confusion_matrix': cm.tolist()
    }
    
    results_file = os.path.join(MODEL_DIR, "final_evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"   Výsledky: {results_file}")
    
    log("\n" + "=" * 60)
    log("FINÁLNÍ EVALUACE DOKONČENA!")
    log("=" * 60)
    log(f"\nVýstupní soubory v: {FIGURES_DIR}")
    log(f"  - confusion_matrix.png")
    log(f"  - roc_curves.png")
    log(f"  - feature_importance.png")
    log(f"  - sector_comparison.png")
    
    return results

if __name__ == "__main__":
    results = main()
