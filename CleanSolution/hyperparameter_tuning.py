#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter tuning pro RF Classifier.

Použití TimeSeriesSplit pro časově konzistentní validaci.
"""

import os
import time
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# === CESTY ===
BASE_DIR = r"c:\Users\Bc. Jan Dub\Desktop\GIT\MachineLearning\CleanSolution"
DATA_FILE = os.path.join(BASE_DIR, "data", "complete", "all_sectors_complete_10y.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

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

def main():
    log("=" * 60)
    log("HYPERPARAMETER TUNING - RF CLASSIFIER")
    log("=" * 60)
    
    # === 1. NAČÍST DATA ===
    log("\n1. Načítání dat...")
    
    df = pd.read_csv(DATA_FILE)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = create_target(df)
    
    available_features = [f for f in FEATURES if f in df.columns]
    df_clean = df.dropna(subset=available_features + ['target']).copy()
    df_clean['target'] = df_clean['target'].astype(int)
    df_clean = df_clean.sort_values('date').reset_index(drop=True)
    
    log(f"   Samples: {len(df_clean)}")
    log(f"   Features: {len(available_features)}")
    
    X = df_clean[available_features].values
    y = df_clean['target'].values
    
    # === 2. SCALER ===
    log("\n2. Scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # === 3. GRID SEARCH ===
    log("\n3. Grid Search s TimeSeriesSplit...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }
    
    # Počet kombinací
    n_combinations = (
        len(param_grid['n_estimators']) *
        len(param_grid['max_depth']) *
        len(param_grid['min_samples_split']) *
        len(param_grid['min_samples_leaf']) *
        len(param_grid['class_weight'])
    )
    log(f"   Kombinací: {n_combinations}")
    log(f"   Toto může trvat delší dobu...")
    
    # Menší grid pro rychlý test
    param_grid_small = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'class_weight': ['balanced']
    }
    
    log(f"   Používám zredukovaný grid ({len(param_grid_small['n_estimators'])*len(param_grid_small['max_depth'])*len(param_grid_small['min_samples_split'])*len(param_grid_small['min_samples_leaf'])} kombinací)")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid_small,
        cv=tscv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    log("\n   Spouštím Grid Search...")
    grid_search.fit(X_scaled, y)
    
    # === 4. VÝSLEDKY ===
    log("\n4. Výsledky Grid Search:")
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    log(f"\n   Nejlepší parametry:")
    for param, value in best_params.items():
        log(f"   - {param}: {value}")
    log(f"\n   Nejlepší CV F1 Score: {best_score:.4f}")
    
    # === 5. FINÁLNÍ MODEL ===
    log("\n5. Trénink finálního modelu...")
    
    # Chronologický split
    split_idx = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:split_idx]
    y_train = y[:split_idx]
    X_test = X_scaled[split_idx:]
    y_test = y[split_idx:]
    
    final_model = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)
    
    # === 6. EVALUACE ===
    log("\n6. Finální evaluace:")
    
    y_pred = final_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    log(f"\n   Test Accuracy: {accuracy:.4f}")
    log(f"   Test F1 Score: {f1:.4f}")
    
    log("\n   Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['DOWN', 'HOLD', 'UP'])
    for line in report.split('\n'):
        if line.strip():
            log(f"   {line}")
    
    cm = confusion_matrix(y_test, y_pred)
    log("\n   Confusion Matrix:")
    log(f"              DOWN  HOLD    UP")
    log(f"   DOWN      {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
    log(f"   HOLD      {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
    log(f"   UP        {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")
    
    # === 7. ULOŽIT ===
    log("\n7. Ukládání...")
    
    # Model
    model_file = os.path.join(MODEL_DIR, "rf_classifier_tuned.pkl")
    joblib.dump(final_model, model_file)
    log(f"   Model: {model_file}")
    
    # Scaler
    scaler_file = os.path.join(MODEL_DIR, "classifier_scaler_tuned.pkl")
    joblib.dump(scaler, scaler_file)
    
    # Best params
    params_file = os.path.join(MODEL_DIR, "optimal_hyperparameters.json")
    with open(params_file, 'w') as f:
        json.dump({
            'best_params': best_params,
            'cv_score': best_score,
            'test_accuracy': accuracy,
            'test_f1': f1
        }, f, indent=2)
    log(f"   Params: {params_file}")
    
    # Grid Search results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    results_file = os.path.join(MODEL_DIR, "grid_search_results.csv")
    results_df.to_csv(results_file, index=False)
    log(f"   Results: {results_file}")
    
    log("\n" + "=" * 60)
    log("HOTOVO!")
    log("=" * 60)
    
    return final_model, best_params

if __name__ == "__main__":
    model, params = main()
