#!/usr/bin/env python3
"""
Model Comparison Framework

Tests multiple ML models with different train windows in walk-forward fashion:
- Train on last N days → Trade 1 day → Record result → Retrain → Repeat

Models tested:
1. Random Forest
2. XGBoost  
3. LightGBM
4. Logistic Regression (baseline)
5. Simple Momentum Rule (non-ML baseline)

Train windows: 5 days, 10 days
Test period: 10 consecutive days

Usage:
    python run_model_comparison.py
"""

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Callable
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not installed")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️  LightGBM not installed")

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features
from src.strategy_configs import get_all_configs, adapt_to_1m_regime, StrategyConfig

# =============================================================================
# CONFIG
# =============================================================================

DATA_FILE = PROJECT_ROOT / "data" / "dukascopy" / "EURUSD" / "EURUSD_prepared.parquet"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "model_comparison"

TRAIN_WINDOWS = [8, 10, 12]  # Days - sweet spot around 10
TEST_CONSECUTIVE_DAYS = 10  # How many days to test walk-forward
END_DATE = "2026-01-17"

# Model-specific probability thresholds and gaps
# Slightly reduced from previous run (-0.01)
# LogReg and MomentumRule need very low thresholds to generate trades
MODEL_THRESHOLDS = {
    'RandomForest': 0.67,  # Was 0.68
    'XGBoost': 0.74,       # Was 0.75
    'LightGBM': 0.74,      # Was 0.75
    'LogisticReg': 0.52,   # VERY LOW - LogReg outputs near 0.5
    'MomentumRule': 0.51,  # VERY LOW - simple rule outputs near 0.5
}

# Per-model minimum probability gap (distance from 0.5)
MODEL_MIN_GAPS = {
    'RandomForest': 0.15,
    'XGBoost': 0.20,
    'LightGBM': 0.20,
    'LogisticReg': 0.02,   # Very small gap for conservative model
    'MomentumRule': 0.01,  # Almost no gap for simple rule
}

# Create STRICT config for high-confidence trades
# probability_threshold=0.75, higher gaps, better R:R
from dataclasses import replace

_base = adapt_to_1m_regime(get_all_configs()[0])
BASE_CONFIG = StrategyConfig(
    name="Strict High-Confidence",
    version="6.0",
    description="Strict: prob>=0.75, gap>=0.20, SL 0.8x ATR, TP 2.5x ATR",
    interval="1m",
    test_week_rows=1440,
    # STRICT THRESHOLDS - fewer but higher quality trades
    probability_threshold=0.75,  # Was 0.58-0.60
    min_probability_gap=0.20,    # Was 0.08-0.10
    # BETTER RISK/REWARD
    sl_atr_multiplier=0.8,       # Tighter SL
    tp_atr_multiplier=2.5,       # Good reward
    max_holding_bars=30,         # 30 minutes max
    # ATR FILTERS
    min_atr_pips=0.3,
    max_atr_pips=8.0,
    use_session_filter=False,
    allowed_sessions=[],
    # COSTS
    spread_pips=0.2,
    slippage_pips=0.1,
    horizon_bars=1,
    confidence_level=0.95,
    pip_value=0.0001,
)


# =============================================================================
# MODEL FACTORIES
# =============================================================================

def create_random_forest():
    """Random Forest - current baseline."""
    return RandomForestClassifier(
        n_estimators=100,  # Reduced for speed
        max_depth=8,
        min_samples_leaf=30,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )


def create_xgboost():
    """XGBoost - gradient boosted trees."""
    if not HAS_XGB:
        return None
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # Will adjust in training
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )


def create_lightgbm():
    """LightGBM - fast gradient boosting."""
    if not HAS_LGB:
        return None
    return lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def create_logistic():
    """Logistic Regression - simple baseline."""
    return LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )


class MomentumRule:
    """Simple momentum rule - non-ML baseline."""
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.classes_ = [0, 1]
    
    def fit(self, X, y):
        # No training needed
        return self
    
    def predict_proba(self, X):
        # Use return features if available, else random
        if hasattr(X, 'values'):
            X = X.values
        
        # Simple rule: if last returns are positive, predict up
        probs = np.ones((len(X), 2)) * 0.5
        
        # Look for return-like columns
        if X.shape[1] > 0:
            # Average of all features as simple signal
            signal = np.mean(X, axis=1)
            signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            probs[:, 1] = 0.5 + 0.1 * np.clip(signal_normalized, -2, 2)
            probs[:, 0] = 1 - probs[:, 1]
        
        return probs


MODELS = {
    'RandomForest': create_random_forest,
    'XGBoost': create_xgboost,
    'LightGBM': create_lightgbm,
    'LogisticReg': create_logistic,
    'MomentumRule': lambda: MomentumRule(),
}


# =============================================================================
# TRADE SIMULATION (copied from holdout script)
# =============================================================================

def simulate_trade(df, entry_idx, direction, probability, config):
    entry_row = df.iloc[entry_idx]
    entry_price = entry_row["close"]
    entry_time = entry_row["time"]
    atr_pips = entry_row["atr"] / config.pip_value

    sl_distance = atr_pips * config.sl_atr_multiplier
    tp_distance = atr_pips * config.tp_atr_multiplier
    cost = config.spread_pips + config.slippage_pips

    outcome = "TIME_EXIT"
    pnl_pips = 0
    exit_idx = entry_idx

    for i in range(1, min(config.max_holding_bars + 1, len(df) - entry_idx)):
        future_idx = entry_idx + i
        if future_idx >= len(df):
            break

        future_row = df.iloc[future_idx]
        high, low = future_row["high"], future_row["low"]

        if direction == "long":
            sl_price = entry_price - (sl_distance * config.pip_value)
            tp_price = entry_price + (tp_distance * config.pip_value)
            if low <= sl_price:
                pnl_pips = -sl_distance - cost
                outcome = "SL_HIT"
                exit_idx = future_idx
                break
            if high >= tp_price:
                pnl_pips = tp_distance - cost
                outcome = "TP_HIT"
                exit_idx = future_idx
                break
        else:
            sl_price = entry_price + (sl_distance * config.pip_value)
            tp_price = entry_price - (tp_distance * config.pip_value)
            if high >= sl_price:
                pnl_pips = -sl_distance - cost
                outcome = "SL_HIT"
                exit_idx = future_idx
                break
            if low <= tp_price:
                pnl_pips = tp_distance - cost
                outcome = "TP_HIT"
                exit_idx = future_idx
                break

    if outcome == "TIME_EXIT":
        exit_idx = min(entry_idx + config.max_holding_bars, len(df) - 1)
        exit_price = df.iloc[exit_idx]["close"]
        if direction == "long":
            pnl_pips = (exit_price - entry_price) / config.pip_value - cost
        else:
            pnl_pips = (entry_price - exit_price) / config.pip_value - cost

    return {
        "entry_time": entry_time,
        "exit_time": df.iloc[exit_idx]["time"],
        "direction": direction,
        "pnl_pips": pnl_pips,
        "outcome": outcome,
    }


def run_day_backtest(model, train_df, test_df, config, needs_scaling=False, custom_threshold=None, custom_min_gap=None):
    """Run backtest for single day with given model."""
    feature_cols = [c for c in train_df.columns if c not in {"time", "target", "return_h", "date"}]
    X_train = train_df[feature_cols].copy()
    y_train = train_df["target"].copy()
    X_test = test_df[feature_cols].copy()
    
    # Handle NaN/Inf
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Scale if needed (for Logistic Regression)
    if needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict
    probas = model.predict_proba(X_test)[:, 1]
    
    # Simulate trades
    trades = []
    skip_until = 0
    
    # Use custom min_gap if provided
    min_gap = custom_min_gap if custom_min_gap is not None else config.min_probability_gap
    
    for i, prob in enumerate(probas):
        if i < skip_until:
            continue
        
        atr_pips = test_df.iloc[i]["atr"] / config.pip_value
        if atr_pips < config.min_atr_pips or atr_pips > config.max_atr_pips:
            continue
        
        prob_gap = abs(prob - 0.5)
        if prob_gap < min_gap:
            continue
        
        # Use custom threshold if provided, else config default
        threshold = custom_threshold if custom_threshold else config.probability_threshold
        
        if prob >= threshold:
            direction = "long"
        elif prob <= (1 - threshold):
            direction = "short"
        else:
            continue
        
        trade = simulate_trade(test_df, i, direction, prob, config)
        trades.append(trade)
        skip_until = i + 1  # Simplified - move to next bar
    
    return trades


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    print("=" * 80)
    print("🔬 MODEL COMPARISON EXPERIMENT")
    print(f"   Models: {list(MODELS.keys())}")
    print(f"   Train windows: {TRAIN_WINDOWS} days")
    print(f"   Test period: {TEST_CONSECUTIVE_DAYS} consecutive days")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    # Load data
    print("\n📊 Loading data...")
    if not DATA_FILE.exists():
        print(f"❌ Data not found: {DATA_FILE}")
        return
    
    df_raw = pd.read_parquet(DATA_FILE)
    df_raw["time"] = pd.to_datetime(df_raw["time"])
    df_raw = df_raw.sort_values("time").reset_index(drop=True)
    
    end_dt = pd.to_datetime(END_DATE)
    df_raw = df_raw[df_raw["time"] <= end_dt].copy()
    
    print(f"   Rows: {len(df_raw):,}")
    print(f"   Range: {df_raw['time'].min()} → {df_raw['time'].max()}")
    
    # Build features
    print("\n🔧 Building features...")
    df = build_features(df_raw, horizon_minutes=1)
    df["date"] = df["time"].dt.date
    
    unique_dates = sorted(df["date"].unique())
    print(f"   Feature rows: {len(df):,}")
    print(f"   Unique dates: {len(unique_dates)}")
    
    # Results storage
    all_results = []
    
    # Run experiments
    for train_days in TRAIN_WINDOWS:
        print(f"\n{'='*60}")
        print(f"📈 TRAIN WINDOW: {train_days} DAYS")
        print(f"{'='*60}")
        
        # Calculate start index for test period
        test_start_idx = max(train_days, len(unique_dates) - TEST_CONSECUTIVE_DAYS - train_days)
        
        for model_name, model_factory in MODELS.items():
            model_template = model_factory()
            if model_template is None:
                print(f"   ⏭️  {model_name}: skipped (not installed)")
                continue
            
            print(f"\n   🤖 Testing {model_name}...")
            
            needs_scaling = model_name == 'LogisticReg'
            daily_results = []
            
            # Walk-forward: test each day
            for test_idx in range(test_start_idx, min(test_start_idx + TEST_CONSECUTIVE_DAYS, len(unique_dates))):
                train_dates = unique_dates[test_idx - train_days:test_idx]
                test_date = unique_dates[test_idx]
                
                train_df = df[df["date"].isin(train_dates)].copy()
                test_df = df[df["date"] == test_date].copy()
                
                if len(train_df) < 500 or len(test_df) < 100:
                    continue
                
                # Create fresh model instance
                model = model_factory()
                
                # Get model-specific threshold and min_gap
                model_threshold = MODEL_THRESHOLDS.get(model_name, 0.75)
                model_min_gap = MODEL_MIN_GAPS.get(model_name, 0.20)
                
                trades = run_day_backtest(model, train_df, test_df, BASE_CONFIG, needs_scaling, model_threshold, model_min_gap)
                day_pnl = sum(t["pnl_pips"] for t in trades)
                
                daily_results.append({
                    "date": test_date,
                    "trades": len(trades),
                    "pnl_pips": day_pnl,
                    "wins": sum(1 for t in trades if t["pnl_pips"] > 0),
                })
                
                status = "✅" if day_pnl > 0 else "❌"
                print(f"      {test_date}: {len(trades)} trades, {day_pnl:+.1f} pips {status}")
            
            # Aggregate results
            if daily_results:
                total_pnl = sum(r["pnl_pips"] for r in daily_results)
                total_trades = sum(r["trades"] for r in daily_results)
                profitable_days = sum(1 for r in daily_results if r["pnl_pips"] > 0)
                
                result = {
                    "model": model_name,
                    "train_days": train_days,
                    "total_pnl_pips": total_pnl,
                    "total_trades": total_trades,
                    "test_days": len(daily_results),
                    "profitable_days": profitable_days,
                    "avg_daily_pnl": total_pnl / len(daily_results) if daily_results else 0,
                    "win_rate_days": profitable_days / len(daily_results) if daily_results else 0,
                }
                all_results.append(result)
                
                print(f"      ─────────────────────────────")
                print(f"      TOTAL: {total_pnl:+.1f} pips | {profitable_days}/{len(daily_results)} profitable days")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 80)
    
    if not all_results:
        print("❌ No results generated!")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Pivot table
    print(f"\n{'Model':<15}", end="")
    for td in TRAIN_WINDOWS:
        print(f"  {td}d train      ", end="")
    print()
    print("-" * 60)
    
    for model_name in MODELS.keys():
        model_results = results_df[results_df["model"] == model_name]
        if model_results.empty:
            continue
        
        print(f"{model_name:<15}", end="")
        for td in TRAIN_WINDOWS:
            row = model_results[model_results["train_days"] == td]
            if not row.empty:
                pnl = row["total_pnl_pips"].values[0]
                days = row["profitable_days"].values[0]
                total = row["test_days"].values[0]
                marker = "✅" if pnl > 0 else "❌"
                print(f"  {pnl:>+7.1f} ({days}/{total}){marker}", end="")
            else:
                print(f"  {'N/A':>15}", end="")
        print()
    
    print("-" * 60)
    
    # Find best combination
    if len(results_df) > 0:
        best = results_df.loc[results_df["total_pnl_pips"].idxmax()]
        print(f"\n🏆 BEST: {best['model']} with {best['train_days']}d train → {best['total_pnl_pips']:+.1f} pips")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    print(f"\n✅ COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    return results_df


if __name__ == "__main__":
    run_experiment()
