"""
FOREX Daily Retrain Pipeline

Walk-forward validation with DAILY retraining:
- Train on last 2 days
- Test on next 1 day  
- Slide forward and repeat

This mimics production where we retrain every 24 hours.

Usage:
    python run_daily_retrain.py
"""

import sys
import io
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data_fetcher import fetch_yfinance_fx
from src.strategy_configs import get_all_configs, adapt_to_1m_regime, StrategyConfig
from src.features import build_features
from src.configs import load_settings


# =============================================================================
# CONSTANTS
# =============================================================================

MINUTES_PER_DAY = 1440  # 24 * 60
TRAIN_DAYS = 2          # Train on 2 days of data
TEST_DAYS = 1           # Test on 1 day, then retrain


# =============================================================================
# TRADE SIMULATION
# =============================================================================

def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str, 
                   probability: float, config: StrategyConfig) -> dict:
    """Simulate a single trade with SL/TP logic."""
    
    entry_row = df.iloc[entry_idx]
    entry_price = entry_row["close"]
    entry_time = entry_row["time"]
    atr_pips = entry_row["atr"] / config.pip_value
    
    # Calculate SL and TP levels
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
            elif high >= tp_price:
                pnl_pips = tp_distance - cost
                outcome = "TP_HIT"
                exit_idx = future_idx
                break
        else:  # short
            sl_price = entry_price + (sl_distance * config.pip_value)
            tp_price = entry_price - (tp_distance * config.pip_value)
            
            if high >= sl_price:
                pnl_pips = -sl_distance - cost
                outcome = "SL_HIT"
                exit_idx = future_idx
                break
            elif low <= tp_price:
                pnl_pips = tp_distance - cost
                outcome = "TP_HIT"
                exit_idx = future_idx
                break
    
    # Time exit if neither SL nor TP hit
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


# =============================================================================
# SINGLE DAY BACKTEST
# =============================================================================

def run_single_day_backtest(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                            config: StrategyConfig) -> list:
    """Train model and test on one day of unseen data. Returns trades."""
    
    feature_cols = [c for c in train_df.columns if c not in {"time", "target", "return_h", "date"}]
    
    # Train
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    model.fit(X_train, y_train)
    
    # Predict on test
    X_test = test_df[feature_cols]
    probas = model.predict_proba(X_test)[:, 1]
    
    # Simulate trades
    trades = []
    skip_until = 0
    
    for i, prob in enumerate(probas):
        if i < skip_until:
            continue
        
        atr_pips = test_df.iloc[i]["atr"] / config.pip_value
        
        # Volatility filter
        if atr_pips < config.min_atr_pips or atr_pips > config.max_atr_pips:
            continue
        
        # Check session filter
        if config.use_session_filter and config.allowed_sessions:
            hour = test_df.iloc[i]["time"].hour
            in_session = any(start <= hour < end for start, end in config.allowed_sessions)
            if not in_session:
                continue
        
        # Probability gap from 0.5
        prob_gap = abs(prob - 0.5)
        if prob_gap < config.min_probability_gap:
            continue
        
        # Determine direction
        if prob >= config.probability_threshold:
            direction = "long"
        elif prob <= (1 - config.probability_threshold):
            direction = "short"
        else:
            continue
        
        # Simulate trade
        trade = simulate_trade(test_df, i, direction, prob, config)
        trades.append(trade)
        
        skip_until = i + trade["holding_bars"] + 1
    
    return trades


# =============================================================================
# MAIN PIPELINE - DAILY RETRAIN
# =============================================================================

def main():
    print("=" * 80)
    print("FOREX ML - DAILY RETRAIN WALK-FORWARD TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print()
    print(f"Method: Train on {TRAIN_DAYS} days -> Test on {TEST_DAYS} day -> Retrain -> Repeat")
    print()
    
    # 1. Fetch data
    print("Step 1: Fetching 1-minute EURUSD data (7 days)...")
    try:
        path_1m = fetch_yfinance_fx(pair="EURUSD", interval="1m", period="7d")
        df_raw = pd.read_parquet(path_1m)
        df_raw["time"] = pd.to_datetime(df_raw["time"])
        df_raw = df_raw.sort_values("time").reset_index(drop=True)
        print(f"  [OK] Loaded {len(df_raw):,} rows")
        print(f"  Range: {df_raw['time'].min()} -> {df_raw['time'].max()}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        return
    
    # 2. Build features
    print("\nStep 2: Building features...")
    df = build_features(df_raw, horizon_minutes=1)
    print(f"  [OK] {len(df):,} rows with features")
    
    # 3. Calculate day boundaries
    print("\nStep 3: Identifying day boundaries...")
    df["date"] = df["time"].dt.date
    unique_dates = sorted(df["date"].unique())
    print(f"  [OK] Found {len(unique_dates)} trading days: {unique_dates[0]} to {unique_dates[-1]}")
    
    # 4. Walk-forward with daily retrain
    print("\nStep 4: Running daily retrain walk-forward...")
    print("-" * 80)
    
    # Get configs adapted for 1m
    configs_1m = [adapt_to_1m_regime(c) for c in get_all_configs()]
    
    # Results storage
    all_results = {cfg.name: {"trades": [], "daily_pnl": []} for cfg in configs_1m}
    
    # Walk forward: train on day[i:i+2], test on day[i+2]
    for i in range(len(unique_dates) - TRAIN_DAYS):
        train_dates = unique_dates[i:i+TRAIN_DAYS]
        test_date = unique_dates[i+TRAIN_DAYS] if i+TRAIN_DAYS < len(unique_dates) else None
        
        if test_date is None:
            break
        
        # Get data for train and test
        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"] == test_date].copy()
        
        if len(train_df) < 500 or len(test_df) < 100:
            continue
        
        print(f"\n[DAY {i+1}] Train: {train_dates[0]} to {train_dates[-1]} | Test: {test_date}")
        
        for config in configs_1m:
            trades = run_single_day_backtest(train_df, test_df, config)
            daily_pnl = sum(t["pnl_pips"] for t in trades)
            
            all_results[config.name]["trades"].extend(trades)
            all_results[config.name]["daily_pnl"].append(daily_pnl)
            
            status = "[+]" if daily_pnl > 0 else "[-]" if daily_pnl < 0 else "[=]"
            print(f"  {config.name}: {len(trades)} trades, {daily_pnl:+.1f} pips {status}")
    
    # 5. Summary
    print("\n" + "=" * 80)
    print("DAILY RETRAIN RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    results_summary = []
    
    for config in configs_1m:
        name = config.name
        trades = all_results[name]["trades"]
        daily_pnls = all_results[name]["daily_pnl"]
        
        if not trades:
            continue
        
        pnls = [t["pnl_pips"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        total_pnl = sum(pnls)
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.0001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        win_rate = len(wins) / len(trades) if trades else 0
        
        tp_hits = sum(1 for t in trades if t["outcome"] == "TP_HIT")
        sl_hits = sum(1 for t in trades if t["outcome"] == "SL_HIT")
        time_exits = sum(1 for t in trades if t["outcome"] == "TIME_EXIT")
        
        profitable_days = sum(1 for p in daily_pnls if p > 0)
        total_days = len(daily_pnls)
        
        results_summary.append({
            "strategy": name,
            "trades": len(trades),
            "pnl_pips": total_pnl,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "tp_hits": tp_hits,
            "sl_hits": sl_hits,
            "time_exits": time_exits,
            "profitable_days": profitable_days,
            "total_days": total_days,
        })
    
    print(f"{'Strategy':<25} {'Trades':<8} {'PnL':<12} {'PF':<8} {'Win%':<8} {'Days +/-'}")
    print("-" * 80)
    
    for r in sorted(results_summary, key=lambda x: x["pnl_pips"], reverse=True):
        status = "[WINNER]" if r["pnl_pips"] > 0 and r["profit_factor"] > 1.1 else \
                 "[OK]" if r["pnl_pips"] > 0 else "[LOSS]"
        print(f"{r['strategy']:<25} {r['trades']:<8} {r['pnl_pips']:+10.1f}  "
              f"{r['profit_factor']:<8.2f} {r['win_rate']:<8.1%} {r['profitable_days']}/{r['total_days']} {status}")
    
    print("-" * 80)
    
    if results_summary:
        best = max(results_summary, key=lambda x: x["pnl_pips"])
        
        print(f"\n[BEST] {best['strategy']}")
        print(f"   Total PnL: {best['pnl_pips']:+.1f} pips over {best['total_days']} test days")
        print(f"   Profit Factor: {best['profit_factor']:.2f}")
        print(f"   Win Rate: {best['win_rate']:.1%}")
        print(f"   Profitable Days: {best['profitable_days']}/{best['total_days']}")
        print(f"   Trade Outcomes: TP={best['tp_hits']}, SL={best['sl_hits']}, Time={best['time_exits']}")
    
    # Save results
    if results_summary:
        results_df = pd.DataFrame(results_summary)
        results_path = Path(__file__).parent / "reports" / "daily_retrain_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
