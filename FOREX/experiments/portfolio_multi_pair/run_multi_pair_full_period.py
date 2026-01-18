#!/usr/bin/env python3
"""
Multi-Pair LogReg-Momentum Full Period Validation

Runs the same high-vol momentum edge across a basket of FX pairs
using Dukascopy data. Outputs per-pair metrics and portfolio aggregate.
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_pair
from src.features import build_features
from src.portfolio_baskets import get_basket
from src.risk_metrics import calculate_all_metrics, format_metrics_report

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# DEFAULT CONFIG (aligned with logreg_final full period)
# ---------------------------------------------------------------------------

PROBABILITY_THRESHOLD = 0.58
MIN_GAP = 0.06
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0
MAX_HOLDING = 40
TRAIN_DAYS = 15
COOLDOWN = 3
HORIZON_MINUTES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-pair full period validation")
    parser.add_argument("--basket", default="basket_6", help="Basket name (basket_6, basket_8)")
    parser.add_argument("--pairs", default=None, help="Comma-separated pairs to override basket")
    parser.add_argument("--start-date", default=None, help="Start date filter YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="End date filter YYYY-MM-DD")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--min-atr", type=float, default=None, help="Min ATR filter in pips")
    parser.add_argument("--max-atr", type=float, default=None, help="Max ATR filter in pips")
    parser.add_argument("--cost-pips", type=float, default=None, help="Round-trip cost in pips")
    return parser.parse_args()


def resolve_pairs(args: argparse.Namespace) -> List[str]:
    if args.pairs:
        return [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    return get_basket(args.basket)


def pip_value_for_pair(pair: str, median_price: float) -> float:
    if pair.endswith("JPY"):
        return 0.01 if median_price >= 20 else 0.0001
    return 0.0001


def atr_bounds_for_pair(pair: str) -> Tuple[float, float]:
    if pair.endswith("JPY"):
        return 1.0, 120.0
    return 0.3, 10.0


def cost_pips_for_pair(pair: str) -> float:
    return 0.4 if pair.endswith("JPY") else 0.3


# ---------------------------------------------------------------------------
# TRADE SIMULATION
# ---------------------------------------------------------------------------

def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    prob: float,
    pip_value: float,
    cost_pips: float,
) -> Dict:
    entry_row = df.iloc[entry_idx]
    entry_price = entry_row["close"]
    entry_time = entry_row["time"]
    atr_pips = entry_row["atr"] / pip_value

    sl_distance = atr_pips * SL_ATR_MULT
    tp_distance = atr_pips * TP_ATR_MULT

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
            sl_price = entry_price - (sl_distance * pip_value)
            tp_price = entry_price + (tp_distance * pip_value)
            if low <= sl_price:
                pnl_pips = -sl_distance - cost_pips
                outcome = "SL_HIT"
                exit_idx = future_idx
                break
            if high >= tp_price:
                pnl_pips = tp_distance - cost_pips
                outcome = "TP_HIT"
                exit_idx = future_idx
                break
        else:
            sl_price = entry_price + (sl_distance * pip_value)
            tp_price = entry_price - (tp_distance * pip_value)
            if high >= sl_price:
                pnl_pips = -sl_distance - cost_pips
                outcome = "SL_HIT"
                exit_idx = future_idx
                break
            if low <= tp_price:
                pnl_pips = tp_distance - cost_pips
                outcome = "TP_HIT"
                exit_idx = future_idx
                break

    if outcome == "TIME_EXIT":
        exit_idx = min(entry_idx + MAX_HOLDING, len(df) - 1)
        exit_price = df.iloc[exit_idx]["close"]
        if direction == "long":
            pnl_pips = (exit_price - entry_price) / pip_value - cost_pips
        else:
            pnl_pips = (entry_price - exit_price) / pip_value - cost_pips

    return {
        "entry_time": entry_time,
        "exit_time": df.iloc[exit_idx]["time"],
        "direction": direction,
        "probability": prob,
        "pnl_pips": pnl_pips,
        "outcome": outcome,
        "holding_bars": exit_idx - entry_idx,
    }


def run_day_backtest(
    model: LogisticRegression,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: StandardScaler,
    pip_value: float,
    min_atr_pips: float,
    max_atr_pips: float,
    cost_pips: float,
) -> List[Dict]:
    feature_cols = [
        c
        for c in train_df.columns
        if c not in {"time", "target", "return_h", "date"}
    ]

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

        atr_pips = test_df.iloc[i]["atr"] / pip_value
        if atr_pips < min_atr_pips or atr_pips > max_atr_pips:
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

        trade = simulate_trade(test_df, i, direction, prob, pip_value, cost_pips)
        trades.append(trade)
        skip_until = i + trade["holding_bars"] + COOLDOWN

    return trades


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def backtest_pair(
    pair: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    min_atr_override: float,
    max_atr_override: float,
    cost_override: float,
) -> Dict:
    print("=" * 80)
    print(f"🔬 {pair} FULL PERIOD VALIDATION")
    print("=" * 80)

    df_raw = load_pair(pair, source="dukascopy", start=start_date, end=end_date)
    df_raw = df_raw.reset_index()
    if "time" not in df_raw.columns and "index" in df_raw.columns:
        df_raw = df_raw.rename(columns={"index": "time"})
    if "time" not in df_raw.columns:
        raise ValueError(f"Missing 'time' column after loading {pair}")
    df_raw["time"] = pd.to_datetime(df_raw["time"])
    df_raw = df_raw.sort_values("time").reset_index(drop=True)

    print(f"📊 Rows: {len(df_raw):,} | Range: {df_raw['time'].min()} → {df_raw['time'].max()}")

    df = build_features(df_raw, horizon_minutes=HORIZON_MINUTES)
    df["date"] = df["time"].dt.date

    unique_dates = sorted(df["date"].unique())
    test_dates = unique_dates[TRAIN_DAYS:]

    print(f"🔧 Total dates: {len(unique_dates)} | Test dates: {len(test_dates)}")

    model = LogisticRegression(
        max_iter=2000,
        C=0.1,
        class_weight="balanced",
        solver="saga",
        penalty="l1",
        random_state=42,
        n_jobs=-1,
    )
    scaler = StandardScaler()

    median_price = float(df["close"].median())
    pip_value = pip_value_for_pair(pair, median_price)
    min_atr_pips, max_atr_pips = atr_bounds_for_pair(pair)
    cost_pips = cost_pips_for_pair(pair)

    if min_atr_override is not None:
        min_atr_pips = min_atr_override
    if max_atr_override is not None:
        max_atr_pips = max_atr_override
    if cost_override is not None:
        cost_pips = cost_override

    all_trades: List[Dict] = []
    daily_pnl: Dict[str, float] = {}

    for test_date in test_dates:
        test_date_ts = pd.to_datetime(test_date)
        train_dates = [d for d in unique_dates if pd.to_datetime(d) < test_date_ts][-TRAIN_DAYS:]

        if len(train_dates) < TRAIN_DAYS:
            continue

        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"] == test_date].copy()

        if len(train_df) < 500 or len(test_df) < 50:
            continue

        trades = run_day_backtest(
            model,
            train_df,
            test_df,
            scaler,
            pip_value,
            min_atr_pips,
            max_atr_pips,
            cost_pips,
        )
        day_pnl = float(sum(t["pnl_pips"] for t in trades))
        all_trades.extend(trades)
        daily_pnl[str(test_date)] = day_pnl

    metrics = calculate_all_metrics(all_trades, list(daily_pnl.values()))
    report_text = format_metrics_report(metrics)

    pair_dir = output_dir / pair
    pair_dir.mkdir(parents=True, exist_ok=True)

    trades_df = pd.DataFrame(all_trades)
    trades_path = pair_dir / "trades.csv"
    trades_df.to_csv(trades_path, index=False)

    metrics_path = pair_dir / "metrics.json"
    metrics_path.write_text(pd.Series(metrics).to_json(), encoding="utf-8")

    daily_path = pair_dir / "daily_pnl.csv"
    pd.Series(daily_pnl).to_csv(daily_path, header=["pnl_pips"], index_label="date")

    report_path = pair_dir / "report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "pair": pair,
        "metrics": metrics,
        "daily_pnl": daily_pnl,
        "trades": all_trades,
        "report_path": report_path,
    }


def aggregate_portfolio(results: List[Dict]) -> Dict:
    daily_union: Dict[str, float] = {}
    all_trades: List[Dict] = []

    for result in results:
        daily = result["daily_pnl"]
        for date, pnl in daily.items():
            daily_union[date] = daily_union.get(date, 0.0) + pnl
        all_trades.extend(result.get("trades", []))

    daily_values = [daily_union[k] for k in sorted(daily_union.keys())]
    metrics = calculate_all_metrics(all_trades, daily_values)

    return {
        "metrics": metrics,
        "daily_pnl": daily_union,
    }


def write_portfolio_report(output_dir: Path, results: List[Dict], portfolio: Dict) -> Path:
    lines = []
    lines.append("# Multi-Pair Full Period Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("## Per-Pair Summary")
    lines.append("")

    for result in results:
        m = result["metrics"]
        lines.append(f"### {result['pair']}")
        lines.append(f"- Total Trades: {m.get('total_trades', 0)}")
        lines.append(f"- Total PnL: {m.get('total_pnl', 0):.1f} pips")
        lines.append(f"- Expectancy/Trade: {m.get('expectancy', 0):.2f} pips")
        lines.append(f"- Win Rate: {m.get('win_rate', 0) * 100:.1f}%")
        lines.append(f"- Profit Factor: {m.get('profit_factor', 0):.2f}")
        lines.append(f"- Max Drawdown: {m.get('max_drawdown', 0):.1f} pips")
        lines.append("")

    lines.append("## Portfolio Aggregate")
    lines.append("")
    pm = portfolio["metrics"]
    lines.append(f"- Total Trades: {pm.get('total_trades', 0)}")
    lines.append(f"- Total PnL: {pm.get('total_pnl', 0):.1f} pips")
    lines.append(f"- Expectancy/Trade: {pm.get('expectancy', 0):.2f} pips")
    lines.append(f"- Win Rate: {pm.get('win_rate', 0) * 100:.1f}%")
    lines.append(f"- Profit Factor: {pm.get('profit_factor', 0):.2f}")
    lines.append(f"- Max Drawdown: {pm.get('max_drawdown', 0):.1f} pips")

    report_path = output_dir / "portfolio_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    pairs = resolve_pairs(args)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / f"results_{run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("🚀 MULTI-PAIR FULL PERIOD VALIDATION")
    print("=" * 90)
    print(f"Pairs: {', '.join(pairs)}")
    if args.start_date or args.end_date:
        print(f"Date filter: {args.start_date or '...'} → {args.end_date or '...'}")
    print(f"Output: {output_dir}")
    print("=" * 90)

    results = []
    for pair in pairs:
        result = backtest_pair(
            pair=pair,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=output_dir,
            min_atr_override=args.min_atr,
            max_atr_override=args.max_atr,
            cost_override=args.cost_pips,
        )
        results.append(result)

    portfolio = aggregate_portfolio(results)
    report_path = write_portfolio_report(output_dir, results, portfolio)

    print("\n✅ COMPLETED")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
