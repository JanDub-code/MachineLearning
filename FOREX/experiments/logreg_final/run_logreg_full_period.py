#!/usr/bin/env python3
"""
LogReg-Momentum Full Period Validation

Runs LogReg-Momentum on ENTIRE data period (Oct 17 - Jan 16)
to validate strategy across all market conditions.
"""

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import json

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import build_features
from src.risk_metrics import calculate_all_metrics, format_metrics_report, equity_curve
from src.regime_detector import add_regime_columns, split_by_regime


# =============================================================================
# FROZEN CONFIG
# =============================================================================

PROBABILITY_THRESHOLD = 0.58
MIN_GAP = 0.06
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0
MAX_HOLDING = 40
TRAIN_DAYS = 15
COOLDOWN = 3
PIP_VALUE = 0.0001

DATA_FILE = PROJECT_ROOT / "data" / "dukascopy" / "EURUSD" / "EURUSD_prepared.parquet"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "logreg_final"


# =============================================================================
# TRADE SIMULATION
# =============================================================================

def simulate_trade(df, entry_idx, direction, prob):
    entry_row = df.iloc[entry_idx]
    entry_price = entry_row["close"]
    entry_time = entry_row["time"]
    atr_pips = entry_row["atr"] / PIP_VALUE

    sl_distance = atr_pips * SL_ATR_MULT
    tp_distance = atr_pips * TP_ATR_MULT
    cost = 0.3

    outcome = "TIME_EXIT"
    pnl_pips = 0.0
    exit_idx = entry_idx

    for i in range(1, min(MAX_HOLDING + 1, len(df) - entry_idx)):
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
        exit_idx = min(entry_idx + MAX_HOLDING, len(df) - 1)
        exit_price = df.iloc[exit_idx]["close"]
        if direction == "long":
            pnl_pips = (exit_price - entry_price) / PIP_VALUE - cost
        else:
            pnl_pips = (entry_price - exit_price) / PIP_VALUE - cost

    return {
        "entry_time": entry_time,
        "exit_time": df.iloc[exit_idx]["time"],
        "direction": direction,
        "probability": prob,
        "pnl_pips": pnl_pips,
        "outcome": outcome,
        "holding_bars": exit_idx - entry_idx,
    }


def run_day_backtest(model, train_df, test_df, scaler):
    feature_cols = [c for c in train_df.columns if c not in {"time", "target", "return_h", "date", "vol_regime", "trend_regime", "session"}]
    
    X_train = train_df[feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
    y_train = train_df["target"].copy()
    X_test = test_df[feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model.fit(X_train_scaled, y_train)
    probas = model.predict_proba(X_test_scaled)[:, 1]
    
    trades = []
    skip_until = 0
    
    for i, prob in enumerate(probas):
        if i < skip_until:
            continue
        
        atr_pips = test_df.iloc[i]["atr"] / PIP_VALUE
        if atr_pips < 0.3 or atr_pips > 8.0:
            continue
        
        prob_gap = abs(prob - 0.5)
        if prob_gap < MIN_GAP:
            continue
        
        if prob >= PROBABILITY_THRESHOLD:
            direction = "long"
        elif prob <= (1 - PROBABILITY_THRESHOLD):
            direction = "short"
        else:
            continue
        
        trade = simulate_trade(test_df, i, direction, prob)
        trades.append(trade)
        skip_until = i + trade["holding_bars"] + COOLDOWN
    
    return trades


def main():
    print("=" * 80)
    print("🔬 LOGREG-MOMENTUM FULL PERIOD VALIDATION")
    print("=" * 80)
    print(f"Threshold: {PROBABILITY_THRESHOLD}, Gap: {MIN_GAP}")
    print(f"SL: {SL_ATR_MULT}x, TP: {TP_ATR_MULT}x, Train: {TRAIN_DAYS}d")
    print(f"Testing: ENTIRE DATASET (Oct 17 - Jan 16)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    # Load ALL data
    print("\n📊 Loading data...")
    df_raw = pd.read_parquet(DATA_FILE)
    df_raw["time"] = pd.to_datetime(df_raw["time"])
    df_raw = df_raw.sort_values("time").reset_index(drop=True)
    print(f"   Rows: {len(df_raw):,}")
    print(f"   Range: {df_raw['time'].min()} → {df_raw['time'].max()}")
    
    # Build features
    print("\n🔧 Building features...")
    df = build_features(df_raw, horizon_minutes=1)
    df["date"] = df["time"].dt.date
    df = add_regime_columns(df)
    
    unique_dates = sorted(df["date"].unique())
    # Start testing after TRAIN_DAYS
    test_dates = unique_dates[TRAIN_DAYS:]
    print(f"   Total dates: {len(unique_dates)}")
    print(f"   Test dates: {len(test_dates)} (after {TRAIN_DAYS}d warmup)")
    
    # Create model
    model = LogisticRegression(
        max_iter=2000,
        C=0.1,
        class_weight='balanced',
        solver='saga',
        penalty='l1',
        random_state=42,
        n_jobs=-1,
    )
    scaler = StandardScaler()
    
    all_trades = []
    daily_pnl = []
    weekly_pnl = []
    current_week_pnl = 0
    current_week = None
    
    print("\n📈 Running full period walk-forward...")
    print("-" * 70)
    
    for i, test_date in enumerate(test_dates):
        test_date_ts = pd.to_datetime(test_date)
        train_dates = [d for d in unique_dates 
                       if pd.to_datetime(d) < test_date_ts][-TRAIN_DAYS:]
        
        if len(train_dates) < TRAIN_DAYS:
            continue
        
        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"] == test_date].copy()
        
        if len(train_df) < 500 or len(test_df) < 50:
            continue
        
        trades = run_day_backtest(model, train_df, test_df, scaler)
        day_pnl = sum(t["pnl_pips"] for t in trades)
        
        all_trades.extend(trades)
        daily_pnl.append(day_pnl)
        
        # Track weekly
        week = test_date_ts.isocalendar()[1]
        if current_week is None:
            current_week = week
        if week != current_week:
            weekly_pnl.append(current_week_pnl)
            current_week_pnl = day_pnl
            current_week = week
        else:
            current_week_pnl += day_pnl
        
        status = "✅" if day_pnl > 0 else "❌" if day_pnl < 0 else "➖"
        print(f"[{i+1:2}/{len(test_dates)}] {test_date}: {len(trades):3} trades, {day_pnl:+7.1f} pips {status}")
    
    # Add last week
    if current_week_pnl != 0:
        weekly_pnl.append(current_week_pnl)
    
    print("-" * 70)
    
    # Calculate metrics
    print("\n📊 Risk Metrics:")
    metrics = calculate_all_metrics(all_trades, daily_pnl)
    print(format_metrics_report(metrics))
    
    # Weekly summary
    print("\n📅 WEEKLY BREAKDOWN")
    print("-" * 50)
    profitable_weeks = sum(1 for w in weekly_pnl if w > 0)
    print(f"   Weeks tested: {len(weekly_pnl)}")
    print(f"   Profitable weeks: {profitable_weeks}/{len(weekly_pnl)} ({profitable_weeks/len(weekly_pnl)*100:.0f}%)")
    print(f"   Best week: {max(weekly_pnl):+.1f} pips")
    print(f"   Worst week: {min(weekly_pnl):+.1f} pips")
    print(f"   Avg week: {np.mean(weekly_pnl):+.1f} pips")
    
    # Monthly estimate
    avg_daily = np.mean(daily_pnl) if daily_pnl else 0
    monthly_est = avg_daily * 22  # ~22 trading days
    print(f"\n   📈 Monthly projection: {monthly_est:+.1f} pips")
    
    # Regime analysis
    print("\n🔄 REGIME ANALYSIS")
    print("-" * 50)
    if 'vol_regime' in df.columns:
        regime_trades = split_by_regime(all_trades, df, 'vol_regime')
        for regime, trades in regime_trades.items():
            if trades:
                r_metrics = calculate_all_metrics(trades)
                pct = len(trades) / len(all_trades) * 100 if all_trades else 0
                print(f"{regime:10}: {len(trades):4} trades ({pct:.0f}%), {r_metrics['total_pnl']:+8.1f} pips, exp={r_metrics['expectancy']:+.2f}")
    
    # Save results
    trades_df = pd.DataFrame(all_trades)
    trades_df.to_csv(OUTPUT_DIR / "logreg_full_period_trades.csv", index=False)
    
    metrics['weekly_pnl'] = weekly_pnl
    metrics['monthly_projection'] = monthly_est
    with open(OUTPUT_DIR / "logreg_full_period_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    eq = equity_curve(all_trades)
    eq.to_csv(OUTPUT_DIR / "logreg_full_equity_curve.csv")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("🏁 FULL PERIOD VALIDATION COMPLETE")
    print("=" * 80)
    
    if metrics['total_pnl'] > 0 and metrics['profit_factor'] > 1.0:
        print("✅ PASSED - Positive expectancy across entire 3-month period")
    else:
        print("❌ FAILED - No edge over full period")
    
    print(f"\n   Period: Oct 17, 2025 → Jan 16, 2026 ({len(test_dates)} trading days)")
    print(f"   Total PnL: {metrics['total_pnl']:+.1f} pips")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Expectancy: {metrics['expectancy']:+.2f} pips/trade")
    print(f"   Max Drawdown: {metrics['max_drawdown']:.1f} pips")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    print(f"\n💾 Results saved to: {OUTPUT_DIR}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
