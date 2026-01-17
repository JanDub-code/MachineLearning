#!/usr/bin/env python3
"""
Individual Model Optimization

Each model gets tailored treatment based on its strengths:

1. RandomForest: Robust to noise, needs more trees for 1m data
   - Lower threshold (can handle uncertainty)
   - Longer holding (less noise-sensitive)
   - Higher min_samples_leaf (regularization)

2. XGBoost: Great at patterns, prone to overfit
   - Higher threshold (confident signals only)
   - Early stopping
   - Lower learning rate

3. LightGBM: Fast, feature-selective
   - Feature importance pruning
   - Histogram-based for speed
   - Balanced precision/recall

4. LogisticReg: Linear baseline
   - Heavy feature preprocessing
   - Momentum-based features only
   - Very high threshold

5. GradientBoosting (sklearn): Alternative ensemble
   - Different boosting approach
   - More regularization
"""

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features
from src.risk_metrics import calculate_all_metrics, equity_curve
from src.strategy_configs import StrategyConfig

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# =============================================================================
# PER-MODEL CONFIGURATIONS
# =============================================================================

@dataclass
class ModelConfig:
    """Individual model configuration."""
    name: str
    model_factory: callable
    probability_threshold: float
    min_probability_gap: float
    sl_atr_multiplier: float
    tp_atr_multiplier: float
    max_holding_bars: int
    train_days: int
    needs_scaling: bool = False
    description: str = ""


def get_model_configs() -> List[ModelConfig]:
    """Get optimized config for each model type."""
    
    configs = []
    
    # 1. RANDOM FOREST - Robust, but needs lower threshold on 1m
    configs.append(ModelConfig(
        name="RF-Balanced",
        model_factory=lambda: RandomForestClassifier(
            n_estimators=200,      # More trees for stability
            max_depth=6,           # Shallower to prevent overfit
            min_samples_leaf=50,   # High regularization
            max_features='sqrt',
            n_jobs=-1,
            class_weight='balanced',
            random_state=42,
        ),
        probability_threshold=0.55,  # LOWERED - RF needs low threshold
        min_probability_gap=0.04,    # LOWERED - RF is conservative
        sl_atr_multiplier=1.2,       # Wider SL for noise
        tp_atr_multiplier=2.0,       # Moderate TP
        max_holding_bars=45,         # Hold longer
        train_days=12,               # More training data
        description="Robust RF with regularization"
    ))
    
    # 2. XGBOOST - Pattern hunter, early stopping
    if HAS_XGB:
        configs.append(ModelConfig(
            name="XGB-Precise",
            model_factory=lambda: xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,           # Shallow
                learning_rate=0.05,    # Slow learning
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=10,   # Regularization
                reg_alpha=0.1,         # L1
                reg_lambda=1.0,        # L2
                scale_pos_weight=1,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
            ),
            probability_threshold=0.58,  # LOWERED from 0.72
            min_probability_gap=0.06,    # LOWERED from 0.18
            sl_atr_multiplier=0.8,       # Tight SL
            tp_atr_multiplier=2.5,       # High reward
            max_holding_bars=25,
            train_days=10,
            description="Precise XGB with regularization"
        ))
    
    # 3. LIGHTGBM - Fast, feature-selective
    if HAS_LGB:
        configs.append(ModelConfig(
            name="LGB-Fast",
            model_factory=lambda: lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.6,    # Less features
                min_child_samples=30,
                reg_alpha=0.05,
                reg_lambda=0.5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            probability_threshold=0.70,
            min_probability_gap=0.16,
            sl_atr_multiplier=0.9,
            tp_atr_multiplier=2.2,
            max_holding_bars=30,
            train_days=10,
            description="Fast LGB with feature selection"
        ))
    
    # 4. LOGISTIC REGRESSION - Momentum features only
    configs.append(ModelConfig(
        name="LogReg-Momentum",
        model_factory=lambda: LogisticRegression(
            max_iter=2000,
            C=0.1,                  # Strong regularization
            class_weight='balanced',
            solver='saga',
            penalty='l1',           # Sparse features
            random_state=42,
            n_jobs=-1,
        ),
        probability_threshold=0.58,  # Low - LogReg is conservative
        min_probability_gap=0.06,
        sl_atr_multiplier=1.5,
        tp_atr_multiplier=2.0,
        max_holding_bars=40,
        train_days=15,              # More data for linear model
        needs_scaling=True,
        description="Momentum-based LogReg"
    ))
    
    # 5. GRADIENT BOOSTING (sklearn) - Alternative ensemble
    configs.append(ModelConfig(
        name="GBoost-Stable",
        model_factory=lambda: GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=30,
            subsample=0.8,
            random_state=42,
        ),
        probability_threshold=0.68,
        min_probability_gap=0.14,
        sl_atr_multiplier=1.0,
        tp_atr_multiplier=2.2,
        max_holding_bars=30,
        train_days=10,
        description="Stable sklearn GBoost"
    ))
    
    return configs


# =============================================================================
# DATA & CONSTANTS
# =============================================================================

DATA_FILE = PROJECT_ROOT / "data" / "dukascopy" / "EURUSD" / "EURUSD_prepared.parquet"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "individual_models"
TEST_DAYS = 15  # Longer test period
COOLDOWN_BARS = 3
PIP_VALUE = 0.0001


# =============================================================================
# TRADE SIMULATION  
# =============================================================================

def simulate_trade(df, entry_idx, direction, probability, config: ModelConfig):
    entry_row = df.iloc[entry_idx]
    entry_price = entry_row["close"]
    entry_time = entry_row["time"]
    atr_pips = entry_row["atr"] / PIP_VALUE

    sl_distance = atr_pips * config.sl_atr_multiplier
    tp_distance = atr_pips * config.tp_atr_multiplier
    cost = 0.3  # spread + slippage

    outcome = "TIME_EXIT"
    pnl_pips = 0.0
    exit_idx = entry_idx

    for i in range(1, min(config.max_holding_bars + 1, len(df) - entry_idx)):
        future_idx = entry_idx + i
        if future_idx >= len(df):
            break

        future_row = df.iloc[future_idx]
        high, low = future_row["high"], future_row["low"]

        if direction == "long":
            sl_price = entry_price - (sl_distance * PIP_VALUE)
            tp_price = entry_price + (tp_distance * PIP_VALUE)
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
            sl_price = entry_price + (sl_distance * PIP_VALUE)
            tp_price = entry_price - (tp_distance * PIP_VALUE)
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
            pnl_pips = (exit_price - entry_price) / PIP_VALUE - cost
        else:
            pnl_pips = (entry_price - exit_price) / PIP_VALUE - cost

    return {
        "entry_time": entry_time,
        "exit_time": df.iloc[exit_idx]["time"],
        "direction": direction,
        "pnl_pips": pnl_pips,
        "outcome": outcome,
        "holding_bars": exit_idx - entry_idx,
    }


def run_day_backtest(model, train_df, test_df, config: ModelConfig):
    """Run backtest with model-specific config."""
    feature_cols = [c for c in train_df.columns if c not in {"time", "target", "return_h", "date"}]
    X_train = train_df[feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df["target"].copy()
    X_test = test_df[feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
    
    if config.needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    model.fit(X_train, y_train)
    probas = model.predict_proba(X_test)[:, 1]
    
    trades = []
    skip_until = 0
    
    for i, prob in enumerate(probas):
        if i < skip_until:
            continue
        
        atr_pips = test_df.iloc[i]["atr"] / PIP_VALUE
        if atr_pips < 0.3 or atr_pips > 8.0:
            continue
        
        prob_gap = abs(prob - 0.5)
        if prob_gap < config.min_probability_gap:
            continue
        
        if prob >= config.probability_threshold:
            direction = "long"
        elif prob <= (1 - config.probability_threshold):
            direction = "short"
        else:
            continue
        
        trade = simulate_trade(test_df, i, direction, prob, config)
        trades.append(trade)
        skip_until = i + trade["holding_bars"] + COOLDOWN_BARS
    
    return trades


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("🎯 INDIVIDUAL MODEL OPTIMIZATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Load data
    print("\n📊 Loading data...")
    if not DATA_FILE.exists():
        print(f"❌ Data not found: {DATA_FILE}")
        return
    
    df_raw = pd.read_parquet(DATA_FILE)
    df_raw["time"] = pd.to_datetime(df_raw["time"])
    df_raw = df_raw.sort_values("time").reset_index(drop=True)
    
    end_dt = pd.to_datetime("2026-01-17")
    df_raw = df_raw[df_raw["time"] <= end_dt].copy()
    
    print(f"   Rows: {len(df_raw):,}")
    
    # Build features
    print("\n🔧 Building features...")
    df = build_features(df_raw, horizon_minutes=1)
    df["date"] = df["time"].dt.date
    unique_dates = sorted(df["date"].unique())
    print(f"   Unique dates: {len(unique_dates)}")
    
    # Get model configs
    model_configs = get_model_configs()
    print(f"\n📋 Testing {len(model_configs)} optimized models...")
    
    all_results = []
    
    for config in model_configs:
        print(f"\n{'='*60}")
        print(f"🤖 {config.name}")
        print(f"   {config.description}")
        print(f"   Threshold: {config.probability_threshold}, Gap: {config.min_probability_gap}")
        print(f"   SL: {config.sl_atr_multiplier}x, TP: {config.tp_atr_multiplier}x, Train: {config.train_days}d")
        print(f"{'='*60}")
        
        # Calculate test start index
        test_start_idx = max(config.train_days, len(unique_dates) - TEST_DAYS - config.train_days)
        
        all_trades = []
        daily_pnl = []
        
        for test_idx in range(test_start_idx, min(test_start_idx + TEST_DAYS, len(unique_dates))):
            train_dates = unique_dates[test_idx - config.train_days:test_idx]
            test_date = unique_dates[test_idx]
            
            train_df = df[df["date"].isin(train_dates)].copy()
            test_df = df[df["date"] == test_date].copy()
            
            if len(train_df) < 500 or len(test_df) < 100:
                continue
            
            model = config.model_factory()
            trades = run_day_backtest(model, train_df, test_df, config)
            
            day_pnl = sum(t["pnl_pips"] for t in trades)
            all_trades.extend(trades)
            daily_pnl.append(day_pnl)
            
            status = "✅" if day_pnl > 0 else "❌" if day_pnl < 0 else "➖"
            print(f"   {test_date}: {len(trades):3} trades, {day_pnl:+7.1f} pips {status}")
        
        # Calculate metrics
        if all_trades:
            metrics = calculate_all_metrics(all_trades, daily_pnl)
            
            result = {
                "model": config.name,
                "total_pnl": metrics['total_pnl'],
                "trades": metrics['total_trades'],
                "win_rate": metrics['win_rate'],
                "profit_factor": metrics['profit_factor'],
                "max_drawdown": metrics['max_drawdown'],
                "sharpe": metrics['sharpe_ratio'],
                "expectancy": metrics['expectancy'],
                "profitable_days": sum(1 for p in daily_pnl if p > 0),
                "total_days": len(daily_pnl),
            }
            all_results.append(result)
            
            print(f"\n   📊 TOTAL: {metrics['total_pnl']:+.1f} pips | PF: {metrics['profit_factor']:.2f} | WR: {metrics['win_rate']:.1%}")
            print(f"      DD: {metrics['max_drawdown']:.1f} | Exp: {metrics['expectancy']:+.2f}/trade")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 INDIVIDUAL MODEL COMPARISON")
    print("=" * 80)
    
    if all_results:
        print(f"\n{'Model':<18} {'PnL':>10} {'Trades':>8} {'WR':>8} {'PF':>8} {'DD':>8} {'Exp':>8}")
        print("-" * 80)
        
        for r in sorted(all_results, key=lambda x: x['total_pnl'], reverse=True):
            marker = "✅" if r['total_pnl'] > 0 else "❌"
            print(f"{r['model']:<18} {r['total_pnl']:>+9.1f} {r['trades']:>8} {r['win_rate']:>7.1%} "
                  f"{r['profit_factor']:>7.2f} {r['max_drawdown']:>7.1f} {r['expectancy']:>+7.2f} {marker}")
        
        print("-" * 80)
        
        # Best model
        best = max(all_results, key=lambda x: x['total_pnl'])
        print(f"\n🏆 BEST: {best['model']} → {best['total_pnl']:+.1f} pips")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_path = OUTPUT_DIR / f"individual_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    print(f"\n✅ COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
