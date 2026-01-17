#!/usr/bin/env python3
"""
Strict Hold-out Test

CRITICAL: This test runs ONCE with frozen config on reserved data.
- Development period: Oct 17 - Dec 17 (tuning)
- Hold-out period: Dec 18 - Jan 16 (final validation)

DO NOT modify thresholds after seeing hold-out results!
"""

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features
from src.risk_metrics import calculate_all_metrics, format_metrics_report, equity_curve
from src.regime_detector import add_regime_columns, split_by_regime
from src.strategy_configs import StrategyConfig

# Only test LightGBM - the winner from development
try:
    import lightgbm as lgb
except ImportError:
    print("❌ LightGBM required for this test")
    exit(1)


# =============================================================================
# FROZEN CONFIG - DO NOT CHANGE AFTER DEVELOPMENT
# =============================================================================

FROZEN_CONFIG = StrategyConfig(
    name="LightGBM-10d-Strict",
    version="7.0",
    description="Frozen config from development phase",
    interval="1m",
    test_week_rows=1440,
    probability_threshold=0.74,  # Frozen from dev
    min_probability_gap=0.20,    # Frozen from dev
    sl_atr_multiplier=0.8,
    tp_atr_multiplier=2.5,
    max_holding_bars=30,
    min_atr_pips=0.3,
    max_atr_pips=8.0,
    use_session_filter=False,
    allowed_sessions=[],
    spread_pips=0.2,
    slippage_pips=0.1,
    horizon_bars=1,
    confidence_level=0.95,
    pip_value=0.0001,
)

# Data settings
DATA_FILE = PROJECT_ROOT / "data" / "dukascopy" / "EURUSD" / "EURUSD_prepared.parquet"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "holdout_final"

# Split dates
DEV_END = "2025-12-17"      # Development period ends
HOLDOUT_START = "2025-12-18"  # Hold-out period starts
HOLDOUT_END = "2026-01-16"    # Hold-out period ends

TRAIN_DAYS = 10  # Frozen from development
COOLDOWN_BARS = 5  # Minimum bars between trades


# =============================================================================
# TRADE SIMULATION (with cooldown)
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
    pnl_pips = 0.0
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
        "probability": probability,
        "pnl_pips": pnl_pips,
        "outcome": outcome,
        "holding_bars": exit_idx - entry_idx,
    }


def run_day_backtest(model, train_df, test_df, config):
    """Run backtest for single day with LightGBM."""
    feature_cols = [c for c in train_df.columns if c not in {"time", "target", "return_h", "date", "vol_regime", "trend_regime", "session"}]
    X_train = train_df[feature_cols].copy()
    y_train = train_df["target"].copy()
    X_test = test_df[feature_cols].copy()
    
    # Handle NaN/Inf
    X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
    X_test = X_test.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict
    probas = model.predict_proba(X_test)[:, 1]
    
    # Simulate trades with cooldown
    trades = []
    skip_until = 0
    
    for i, prob in enumerate(probas):
        if i < skip_until:
            continue
        
        atr_pips = test_df.iloc[i]["atr"] / config.pip_value
        if atr_pips < config.min_atr_pips or atr_pips > config.max_atr_pips:
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
        
        # Cooldown: skip next N bars after trade
        skip_until = i + trade["holding_bars"] + COOLDOWN_BARS
    
    return trades


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("🔒 STRICT HOLD-OUT TEST")
    print("=" * 80)
    print(f"Config: {FROZEN_CONFIG.name}")
    print(f"Train window: {TRAIN_DAYS} days")
    print(f"Hold-out period: {HOLDOUT_START} → {HOLDOUT_END}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    # Load data
    print("\n📊 Loading data...")
    if not DATA_FILE.exists():
        print(f"❌ Data not found: {DATA_FILE}")
        return
    
    df_raw = pd.read_parquet(DATA_FILE)
    df_raw["time"] = pd.to_datetime(df_raw["time"])
    df_raw = df_raw.sort_values("time").reset_index(drop=True)
    
    # Filter to hold-out period only
    holdout_start = pd.to_datetime(HOLDOUT_START)
    holdout_end = pd.to_datetime(HOLDOUT_END)
    
    # Get training data for first hold-out day
    train_start = holdout_start - pd.Timedelta(days=TRAIN_DAYS + 5)  # buffer
    
    df_raw = df_raw[(df_raw["time"] >= train_start) & (df_raw["time"] <= holdout_end)].copy()
    
    print(f"   Rows: {len(df_raw):,}")
    print(f"   Range: {df_raw['time'].min()} → {df_raw['time'].max()}")
    
    # Build features
    print("\n🔧 Building features...")
    df = build_features(df_raw, horizon_minutes=1)
    df["date"] = df["time"].dt.date
    
    # Add regime columns
    df = add_regime_columns(df)
    
    unique_dates = sorted(df["date"].unique())
    holdout_dates = [d for d in unique_dates if pd.to_datetime(d) >= holdout_start]
    
    print(f"   Total dates: {len(unique_dates)}")
    print(f"   Hold-out dates: {len(holdout_dates)}")
    
    # Create model
    model = lgb.LGBMClassifier(
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
    
    all_trades = []
    daily_pnl = []
    
    print("\n📈 Running hold-out walk-forward...")
    print("-" * 60)
    
    for i, test_date in enumerate(holdout_dates):
        # Get train dates (TRAIN_DAYS before test)
        test_date_ts = pd.to_datetime(test_date)
        train_dates = [d for d in unique_dates 
                       if pd.to_datetime(d) < test_date_ts][-TRAIN_DAYS:]
        
        if len(train_dates) < TRAIN_DAYS:
            continue
        
        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"] == test_date].copy()
        
        if len(train_df) < 500 or len(test_df) < 100:
            continue
        
        trades = run_day_backtest(model, train_df, test_df, FROZEN_CONFIG)
        day_pnl = sum(t["pnl_pips"] for t in trades)
        
        all_trades.extend(trades)
        daily_pnl.append(day_pnl)
        
        status = "✅" if day_pnl > 0 else "❌" if day_pnl < 0 else "➖"
        print(f"[{i+1:2}/{len(holdout_dates)}] {test_date}: {len(trades):3} trades, {day_pnl:+7.1f} pips {status}")
    
    print("-" * 60)
    
    # Calculate all risk metrics
    print("\n📊 Calculating risk metrics...")
    metrics = calculate_all_metrics(all_trades, daily_pnl)
    
    print(format_metrics_report(metrics))
    
    # Regime analysis
    print("\n🔄 REGIME ANALYSIS")
    print("-" * 50)
    
    if 'vol_regime' in df.columns:
        regime_trades = split_by_regime(all_trades, df, 'vol_regime')
        for regime, trades in regime_trades.items():
            if trades:
                regime_metrics = calculate_all_metrics(trades)
                pnl = regime_metrics['total_pnl']
                count = len(trades)
                exp = regime_metrics['expectancy']
                print(f"{regime:10}: {count:4} trades, {pnl:+8.1f} pips, exp={exp:+.2f}")
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save trades
    trades_df = pd.DataFrame(all_trades)
    trades_path = OUTPUT_DIR / "holdout_trades.csv"
    trades_df.to_csv(trades_path, index=False)
    
    # Save metrics
    metrics_path = OUTPUT_DIR / "holdout_metrics.json"
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Save equity curve
    eq = equity_curve(all_trades)
    equity_path = OUTPUT_DIR / "equity_curve.csv"
    eq.to_csv(equity_path)
    
    print(f"\n💾 Results saved to: {OUTPUT_DIR}")
    print(f"   - {trades_path.name}")
    print(f"   - {metrics_path.name}")
    print(f"   - {equity_path.name}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏁 HOLD-OUT TEST COMPLETE")
    print("=" * 80)
    
    if metrics['total_pnl'] > 0:
        print("✅ POSITIVE RESULT - Strategy shows edge on unseen data")
    else:
        print("❌ NEGATIVE RESULT - Strategy does not show edge on hold-out")
    
    print(f"\nTotal PnL: {metrics['total_pnl']:+.1f} pips")
    print(f"Max Drawdown: {metrics['max_drawdown']:.1f} pips")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
