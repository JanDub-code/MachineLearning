"""
FOREX Trading Pipeline - 7 Day Backtest

Time-series walk-forward validation:
- 7 days of 1-minute EURUSD data
- 5 days training / 2 days testing (strict time-series split)
- No lookahead bias

Usage:
    python run_backtest.py
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
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

from src.data_fetcher import fetch_yfinance_fx
from src.strategy_configs import get_all_configs, adapt_to_1m_regime, StrategyConfig
from src.features import build_features
from src.configs import load_settings


# =============================================================================
# CONSTANTS
# =============================================================================

MINUTES_PER_DAY = 1440  # 24 * 60
TRAIN_DAYS = 5
TEST_DAYS = 2
TOTAL_DAYS = 7


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
        "entry_price": entry_price,
        "probability": probability,
        "pnl_pips": pnl_pips,
        "outcome": outcome,
        "holding_bars": exit_idx - entry_idx,
    }


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                 config: StrategyConfig) -> dict:
    """Train model and test on unseen data."""
    
    feature_cols = [c for c in train_df.columns if c not in {"time", "target", "return_h"}]
    
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
        
        # Volatility filter (adapted for timeframe)
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
    
    # Calculate metrics
    if not trades:
        return {
            "strategy": config.name,
            "trades": 0,
            "pnl_pips": 0,
            "profit_factor": 0,
            "win_rate": 0,
            "tp_hits": 0,
            "sl_hits": 0,
            "time_exits": 0,
            "avg_holding": 0,
        }
    
    pnls = [t["pnl_pips"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.0001
    
    return {
        "strategy": config.name,
        "trades": len(trades),
        "pnl_pips": sum(pnls),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0,
        "win_rate": len(wins) / len(trades),
        "tp_hits": sum(1 for t in trades if t["outcome"] == "TP_HIT"),
        "sl_hits": sum(1 for t in trades if t["outcome"] == "SL_HIT"),
        "time_exits": sum(1 for t in trades if t["outcome"] == "TIME_EXIT"),
        "avg_holding": np.mean([t["holding_bars"] for t in trades]),
        "trades_detail": trades,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 80)
    print("FOREX ML BACKTEST - 7 DAY TIME-SERIES VALIDATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print()
    print(f"Setup: {TRAIN_DAYS} days training -> {TEST_DAYS} days testing")
    print(f"Expected: ~{TRAIN_DAYS * MINUTES_PER_DAY:,} train rows, ~{TEST_DAYS * MINUTES_PER_DAY:,} test rows")
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
    
    # Check ATR distribution
    atr_pips = df["atr"] / 0.0001
    print(f"  ATR stats: min={atr_pips.min():.2f}, median={atr_pips.median():.2f}, max={atr_pips.max():.2f} pips")
    
    # 3. Split: 5 days train / 2 days test
    print("\nStep 3: Time-series split...")
    
    # Calculate split based on actual data
    total_rows = len(df)
    train_rows = int(total_rows * (TRAIN_DAYS / TOTAL_DAYS))
    test_rows = total_rows - train_rows
    
    train_df = df.iloc[:train_rows].copy()
    test_df = df.iloc[train_rows:].copy()
    
    print(f"  Training: {len(train_df):,} rows ({train_df['time'].min()} -> {train_df['time'].max()})")
    print(f"  Testing:  {len(test_df):,} rows ({test_df['time'].min()} -> {test_df['time'].max()})")
    
    # 4. Run backtests for each strategy (adapted to 1m)
    print("\nStep 4: Running backtests...")
    print("-" * 80)
    
    results = []
    
    # Get configs and adapt them for 1m timeframe
    configs_1m = [adapt_to_1m_regime(c) for c in get_all_configs()]
    
    for config in configs_1m:
        print(f"\n[STRATEGY] {config.name}")
        print(f"   Threshold: {config.probability_threshold}, SL: {config.sl_atr_multiplier}xATR, TP: {config.tp_atr_multiplier}xATR")
        print(f"   ATR filter: {config.min_atr_pips}-{config.max_atr_pips} pips, Max hold: {config.max_holding_bars} bars")
        if config.use_session_filter:
            print(f"   Session filter: {config.allowed_sessions}")
        
        result = run_backtest(train_df, test_df, config)
        results.append(result)
        
        if result["trades"] > 0:
            status = "[WINNER]" if result["pnl_pips"] > 0 and result["profit_factor"] > 1.1 else \
                     "[OK]" if result["pnl_pips"] > 0 else "[LOSS]"
        else:
            status = "[NO TRADES]"
        
        print(f"   -> Trades: {result['trades']}, PnL: {result['pnl_pips']:+.1f} pips, "
              f"PF: {result['profit_factor']:.2f}, Win: {result['win_rate']:.1%} {status}")
        if result["trades"] > 0:
            print(f"   -> TP: {result['tp_hits']}, SL: {result['sl_hits']}, Time: {result['time_exits']}")
    
    # 5. Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Strategy':<30} {'Trades':<8} {'PnL':<12} {'PF':<8} {'Win%':<8} {'Status'}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x["pnl_pips"], reverse=True):
        if r["trades"] > 0:
            status = "[WINNER]" if r["pnl_pips"] > 0 and r["profit_factor"] > 1.1 else \
                     "[OK]" if r["pnl_pips"] > 0 else "[LOSS]"
        else:
            status = "[NO TRADES]"
        print(f"{r['strategy']:<30} {r['trades']:<8} {r['pnl_pips']:+10.1f}  "
              f"{r['profit_factor']:<8.2f} {r['win_rate']:<8.1%} {status}")
    
    print("-" * 80)
    
    # Filter results with trades
    results_with_trades = [r for r in results if r["trades"] > 0]
    
    if results_with_trades:
        # Best strategy
        best = max(results_with_trades, key=lambda x: x["pnl_pips"])
        
        print(f"\n[BEST] {best['strategy']}")
        print(f"   PnL: {best['pnl_pips']:+.1f} pips over {TEST_DAYS} days")
        print(f"   Profit Factor: {best['profit_factor']:.2f}")
        print(f"   Win Rate: {best['win_rate']:.1%}")
        
        # Generate report
        report_path = Path(__file__).parent / "reports" / "backtest_report.md"
        generate_report(results, train_df, test_df, report_path)
        print(f"\nReport saved to: {report_path}")
    else:
        print("\n[WARNING] No strategies generated trades!")
        print("Possible causes:")
        print("  - Probability threshold too high")
        print("  - ATR filter too restrictive for 1m data")
        print("  - Session filter excluding all test data")
    
    # Save results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "trades_detail"} for r in results])
    results_path = Path(__file__).parent / "reports" / "backtest_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


def generate_report(results: list, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                    output_path: Path):
    """Generate markdown report."""
    
    results_with_trades = [r for r in results if r["trades"] > 0]
    
    if not results_with_trades:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("# No trades generated\n\nAll strategies filtered out.", encoding="utf-8")
        return
    
    best = max(results_with_trades, key=lambda x: x["pnl_pips"])
    
    report = f"""# Backtest Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Test Configuration

| Parameter | Value |
|-----------|-------|
| **Training Period** | {train_df['time'].min().strftime('%Y-%m-%d %H:%M')} -> {train_df['time'].max().strftime('%Y-%m-%d %H:%M')} |
| **Testing Period** | {test_df['time'].min().strftime('%Y-%m-%d %H:%M')} -> {test_df['time'].max().strftime('%Y-%m-%d %H:%M')} |
| **Training Rows** | {len(train_df):,} (~{TRAIN_DAYS} days) |
| **Testing Rows** | {len(test_df):,} (~{TEST_DAYS} days) |
| **Pair** | EUR/USD |
| **Interval** | 1 minute |

## Results Summary

| Strategy | Trades | PnL (pips) | Profit Factor | Win Rate | Status |
|----------|--------|------------|---------------|----------|--------|
"""
    
    for r in sorted(results, key=lambda x: x["pnl_pips"], reverse=True):
        if r["trades"] > 0:
            status = "**WINNER**" if r["pnl_pips"] > 0 and r["profit_factor"] > 1.1 else \
                     "Profit" if r["pnl_pips"] > 0 else "Loss"
        else:
            status = "No trades"
        report += f"| {r['strategy']} | {r['trades']} | {r['pnl_pips']:+.1f} | {r['profit_factor']:.2f} | {r['win_rate']:.1%} | {status} |\n"
    
    if best["trades"] > 0:
        tp_pct = best['tp_hits']/best['trades']*100 if best['trades'] > 0 else 0
        sl_pct = best['sl_hits']/best['trades']*100 if best['trades'] > 0 else 0
        te_pct = best['time_exits']/best['trades']*100 if best['trades'] > 0 else 0
        
        report += f"""
## Best Strategy: {best['strategy']}

| Metric | Value |
|--------|-------|
| **Total PnL** | {best['pnl_pips']:+.1f} pips |
| **Profit Factor** | {best['profit_factor']:.2f} |
| **Win Rate** | {best['win_rate']:.1%} |
| **Total Trades** | {best['trades']} |
| **TP Hits** | {best['tp_hits']} ({tp_pct:.1f}%) |
| **SL Hits** | {best['sl_hits']} ({sl_pct:.1f}%) |
| **Time Exits** | {best['time_exits']} ({te_pct:.1f}%) |
| **Avg Holding** | {best['avg_holding']:.1f} bars |

## Interpretation

- **Profit Factor {best['profit_factor']:.2f}** means: For every 100 units lost, you gain {best['profit_factor']*100:.0f} units
- **{best['pnl_pips']:+.1f} pips in {TEST_DAYS} days** = projected ~{best['pnl_pips'] * 30 / TEST_DAYS:+.0f} pips/month (rough estimate)
- All costs (spread + slippage) are **already included**

## Important Notes

1. This is a **single backtest** on limited data - results may vary
2. Past performance does not guarantee future results
3. 7 days is a small sample - consider testing on longer periods
4. Paper trading recommended before going live

---
*Generated by FOREX ML Trading System*
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
