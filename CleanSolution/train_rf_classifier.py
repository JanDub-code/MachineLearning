#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trénink RF Classifier pro klasifikaci cenových pohybů.

Klasifikace: DOWN / HOLD / UP (±3% threshold)
"""

import os
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# === CESTY ===
BASE_DIR = r"c:\Users\Bc. Jan Dub\Desktop\GIT\MachineLearning\CleanSolution"
DATA_FILE = os.path.join(BASE_DIR, "data", "complete", "all_sectors_complete_10y.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# === KONFIGURACE ===
THRESHOLD = 0.03  # ±3% pro klasifikaci

# Features pro klasifikaci
FEATURES = [
    # OHLCV
    'open', 'high', 'low', 'close', 'volume',
    # Technické indikátory
    'volatility', 'returns', 'rsi_14',
    'macd', 'macd_signal', 'macd_hist',
    'sma_3', 'sma_6', 'sma_12',
    'ema_3', 'ema_6', 'ema_12',
    'volume_change',
    # Fundamenty
    'trailingPE', 'forwardPE', 'priceToBook',
    'returnOnEquity', 'returnOnAssets',
    'profitMargins', 'operatingMargins', 'grossMargins',
    'debtToEquity', 'currentRatio', 'beta'
]

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vytvoří target variable: budoucí měsíční return.
    Klasifikace: 0=DOWN, 1=HOLD, 2=UP
    """
    df = df.copy()
    
    # Seřadit podle ticker a datum
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # Budoucí cena (shift -1 = next month)
    df['future_close'] = df.groupby('ticker')['close'].shift(-1)
    
    # Budoucí return
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    
    # Klasifikace
    def classify(ret):
        if pd.isna(ret):
            return np.nan
        elif ret < -THRESHOLD:
            return 0  # DOWN
        elif ret > THRESHOLD:
            return 2  # UP
        else:
            return 1  # HOLD
    
    df['target'] = df['future_return'].apply(classify)
    
    return df

def main():
    log("=" * 60)
    log("TRÉNINK RF CLASSIFIER PRO KLASIFIKACI CENOVÝCH POHYBŮ")
    log("=" * 60)
    
    # === 1. NAČÍST DATA ===
    log("\n1. Načítání dat...")
    
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    log(f"   Načteno: {len(df)} řádků, {df['ticker'].nunique()} tickerů")
    
    # === 2. VYTVOŘIT TARGET ===
    log("\n2. Vytváření target variable...")
    
    df = create_target(df)
    
    # Statistiky target
    target_counts = df['target'].value_counts().sort_index()
    log(f"   DOWN (0): {target_counts.get(0, 0)}")
    log(f"   HOLD (1): {target_counts.get(1, 0)}")
    log(f"   UP (2): {target_counts.get(2, 0)}")
    
    # === 3. PŘIPRAVIT DATA ===
    log("\n3. Příprava dat...")
    
    # Dostupné features
    available_features = [f for f in FEATURES if f in df.columns]
    log(f"   Features: {len(available_features)}")
    
    # Odstranit NaN
    df_clean = df.dropna(subset=available_features + ['target']).copy()
    df_clean['target'] = df_clean['target'].astype(int)
    
    log(f"   Clean samples: {len(df_clean)}")
    
    # Chronologický split (poslední 20% jako test)
    df_clean = df_clean.sort_values('date').reset_index(drop=True)
    split_idx = int(len(df_clean) * 0.8)
    
    df_train = df_clean.iloc[:split_idx]
    df_test = df_clean.iloc[split_idx:]
    
    log(f"   Train: {len(df_train)}, Test: {len(df_test)}")
    
    X_train = df_train[available_features].values
    y_train = df_train['target'].values
    X_test = df_test[available_features].values
    y_test = df_test['target'].values
    
    # === 4. SCALER ===
    log("\n4. Scaling...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # === 5. TRÉNINK ===
    log("\n5. Trénink RF Classifier...")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Pro nevyvážené třídy
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # === 6. EVALUACE ===
    log("\n6. Evaluace...")
    
    y_pred = model.predict(X_test_scaled)
    
    # Metriky
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    log(f"\n   Accuracy:  {accuracy:.4f}")
    log(f"   Precision: {precision:.4f}")
    log(f"   Recall:    {recall:.4f}")
    log(f"   F1-Score:  {f1:.4f}")
    
    # Classification report
    log("\n   Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['DOWN', 'HOLD', 'UP'])
    for line in report.split('\n'):
        if line.strip():
            log(f"   {line}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    log("\n   Confusion Matrix:")
    log(f"              DOWN  HOLD    UP")
    log(f"   DOWN      {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
    log(f"   HOLD      {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
    log(f"   UP        {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
    
    # === 7. PER-SECTOR ANALÝZA ===
    log("\n7. Per-sector analýza...")
    
    for sector in df_test['sector'].unique():
        sector_mask = df_test['sector'] == sector
        sector_idx = df_test[sector_mask].index - split_idx  # Adjust index
        
        if len(sector_idx) > 0:
            y_sector = y_test[sector_idx]
            y_pred_sector = y_pred[sector_idx]
            acc = accuracy_score(y_sector, y_pred_sector)
            f1_s = f1_score(y_sector, y_pred_sector, average='weighted')
            log(f"   {sector}: Accuracy={acc:.3f}, F1={f1_s:.3f} (n={len(sector_idx)})")
    
    # === 8. FEATURE IMPORTANCE ===
    log("\n8. Feature Importance (Top 10):")
    
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': available_features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    for _, row in feat_imp.head(10).iterrows():
        log(f"   {row['feature']}: {row['importance']:.4f}")
    
    # === 9. ULOŽIT ===
    log("\n9. Ukládání...")
    
    # Model
    model_file = os.path.join(MODEL_DIR, "rf_classifier_all_sectors.pkl")
    joblib.dump(model, model_file)
    log(f"   Model: {model_file}")
    
    # Scaler
    scaler_file = os.path.join(MODEL_DIR, "classifier_scaler.pkl")
    joblib.dump(scaler, scaler_file)
    log(f"   Scaler: {scaler_file}")
    
    # Feature importance
    feat_imp_file = os.path.join(MODEL_DIR, "classifier_feature_importance.csv")
    feat_imp.to_csv(feat_imp_file, index=False)
    
    # Metadata
    metadata = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_samples': len(df_train),
        'test_samples': len(df_test),
        'features': available_features,
        'threshold': THRESHOLD
    }
    
    import json
    meta_file = os.path.join(MODEL_DIR, "classifier_metadata.json")
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    log("\n" + "=" * 60)
    log("HOTOVO!")
    log("=" * 60)
    
    return model, accuracy, f1

if __name__ == "__main__":
    model, acc, f1 = main()
