"""
FOREX Holdout Backtest - 4 Week Window

Purpose:
- Evaluate the winning configs on older data with strict time ordering
- Daily retrain walk-forward with a small train window
- Purge training rows that overlap the test horizon to avoid label leakage

Usage:
    python run_holdout_4w.py
    python run_holdout_4w.py --end-date 2026-01-01 --weeks 4 --train-days 2
    python run_holdout_4w.py --parquet /path/to/data.parquet
"""

import argparse
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.configs import load_settings
from src.data_fetcher import OandaDataFetcher
from src.features import build_features
from src.strategy_configs import get_all_configs, adapt_to_1m_regime, StrategyConfig


# =============================================================================
# CONSTANTS
# =============================================================================

MIN_TRAIN_ROWS = 500
MIN_TEST_ROWS = 100


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4-week holdout backtest (no lookahead)")
    parser.add_argument("--pair", default="EUR_USD", help="OANDA instrument (default: EUR_USD)")
    parser.add_argument("--granularity", default="M1", help="OANDA granularity (default: M1)")
    parser.add_argument("--weeks", type=int, default=4, help="Weeks of data to evaluate (default: 4)")
    parser.add_argument("--train-days", type=int, default=2, help="Days used for each train window (default: 2)")
    parser.add_argument("--horizon-minutes", type=int, default=1, help="Target horizon in minutes (default: 1)")
    parser.add_argument("--end-date", default=None, help="End date/time (YYYY-MM-DD or ISO)")
    parser.add_argument("--embargo-days", type=int, default=7, help="Days to skip before end (default: 7)")
    parser.add_argument("--parquet", default=None, help="Path to local parquet file")
    parser.add_argument("--save-raw", action="store_true", help="Save fetched data to data/raw")
    parser.add_argument("--max-candles", type=int, default=5000, help="OANDA page size (default: 5000)")
    return parser.parse_args()


def _normalize_end_date(value: Optional[str], embargo_days: int) -> pd.Timestamp:
    if value:
        end = pd.to_datetime(value)
        if end.time() == time(0, 0):
            end = end + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        if end.tzinfo is not None:
            end = end.tz_localize(None)
        return end
    end = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=embargo_days)
    return end


def _granularity_step(granularity: str) -> pd.Timedelta:
    if granularity.startswith("M"):
        return pd.Timedelta(minutes=int(granularity[1:]))
    if granularity.startswith("H"):
        return pd.Timedelta(hours=int(granularity[1:]))
    if granularity == "D":
        return pd.Timedelta(days=1)
    if granularity == "W":
        return pd.Timedelta(weeks=1)
    raise ValueError(f"Unsupported granularity: {granularity}")


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_oanda_range(
    pair: str,
    granularity: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_candles: int,
) -> pd.DataFrame:
    fetcher = OandaDataFetcher()
    step = _granularity_step(granularity)
    frames = []
    current = start

    while current < end:
        res = fetcher.fetch_historical(
            pair=pair,
            granularity=granularity,
            start=current,
            end=end,
            max_candles=max_candles,
        )
        df = res.candles
        if df.empty:
            break

        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            if df["time"].dt.tz is not None:
                df["time"] = df["time"].dt.tz_localize(None)

        frames.append(df)

        last_time = df["time"].max()
        if last_time <= current:
            break
        current = last_time + step

        if len(df) < max_candles:
            break

    if not frames:
        raise ValueError("No data returned from OANDA.")

    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if "complete" in data.columns:
        data = data[data["complete"]].copy()
    return data


def save_raw_data(df: pd.DataFrame, pair: str, granularity: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    settings = load_settings()
    date_tag = datetime.utcnow().strftime("%Y-%m-%d")
    out_dir = Path(settings.paths.raw_dir) / pair / date_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"oanda_{granularity}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    df.to_parquet(fname, index=False)
    return fname


# =============================================================================
# TRADE SIMULATION
# =============================================================================

def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    probability: float,
    config: StrategyConfig,
) -> Dict:
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
        "entry_price": entry_price,
        "probability": probability,
        "pnl_pips": pnl_pips,
        "outcome": outcome,
        "holding_bars": exit_idx - entry_idx,
    }


def purge_training_tail(
    train_df: pd.DataFrame,
    test_start: pd.Timestamp,
    horizon_minutes: int,
) -> pd.DataFrame:
    cutoff = test_start - pd.Timedelta(minutes=horizon_minutes)
    return train_df[train_df["time"] < cutoff].copy()


def run_single_day_backtest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: StrategyConfig,
) -> List[Dict]:
    feature_cols = [c for c in train_df.columns if c not in {"time", "target", "return_h", "date"}]
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

    X_test = test_df[feature_cols]
    probas = model.predict_proba(X_test)[:, 1]

    trades = []
    skip_until = 0

    for i, prob in enumerate(probas):
        if i < skip_until:
            continue

        atr_pips = test_df.iloc[i]["atr"] / config.pip_value

        if atr_pips < config.min_atr_pips or atr_pips > config.max_atr_pips:
            continue

        if config.use_session_filter and config.allowed_sessions:
            hour = test_df.iloc[i]["time"].hour
            in_session = any(start <= hour < end for start, end in config.allowed_sessions)
            if not in_session:
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
        skip_until = i + trade["holding_bars"] + 1

    return trades


def max_drawdown(pnls: List[float]) -> float:
    if not pnls:
        return 0.0
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    return float(np.max(drawdown)) if len(drawdown) else 0.0


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("FOREX ML - 4 WEEK HOLDOUT WALK-FORWARD")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    print()

    end_dt = _normalize_end_date(args.end_date, args.embargo_days)
    start_dt = end_dt - pd.Timedelta(weeks=args.weeks)
    print(f"Holdout window: {start_dt} -> {end_dt} ({args.weeks} weeks)")

    if args.parquet:
        df_raw = pd.read_parquet(args.parquet)
        df_raw["time"] = pd.to_datetime(df_raw["time"])
        if df_raw["time"].dt.tz is not None:
            df_raw["time"] = df_raw["time"].dt.tz_localize(None)
        df_raw = df_raw.sort_values("time").reset_index(drop=True)
        df_raw = df_raw[(df_raw["time"] >= start_dt) & (df_raw["time"] <= end_dt)].copy()
        print(f"Loaded parquet: {args.parquet} ({len(df_raw):,} rows)")
    else:
        print("Fetching OANDA data...")
        df_raw = fetch_oanda_range(
            pair=args.pair,
            granularity=args.granularity,
            start=start_dt,
            end=end_dt,
            max_candles=args.max_candles,
        )
        print(f"Fetched {len(df_raw):,} rows from OANDA")
        if args.save_raw:
            saved = save_raw_data(df_raw, args.pair, args.granularity, start_dt, end_dt)
            print(f"Saved raw data to: {saved}")

    df_raw = df_raw.sort_values("time").reset_index(drop=True)
    if df_raw.empty:
        print("No data available for the selected window.")
        return
    df = build_features(df_raw, horizon_minutes=args.horizon_minutes)
    df["date"] = df["time"].dt.date

    unique_dates = sorted(df["date"].unique())
    print(f"Feature rows: {len(df):,} across {len(unique_dates)} days")

    configs_1m = [adapt_to_1m_regime(c) for c in get_all_configs()]

    results = {cfg.name: {"trades": [], "daily_pnl": []} for cfg in configs_1m}

    for i in range(args.train_days, len(unique_dates)):
        train_dates = unique_dates[i - args.train_days:i]
        test_date = unique_dates[i]

        train_df = df[df["date"].isin(train_dates)].copy()
        test_df = df[df["date"] == test_date].copy()

        if test_df.empty:
            continue

        train_df = purge_training_tail(train_df, test_df["time"].min(), args.horizon_minutes)

        if len(train_df) < MIN_TRAIN_ROWS or len(test_df) < MIN_TEST_ROWS:
            continue

        print(f"\n[DAY] Train: {train_dates[0]} -> {train_dates[-1]} | Test: {test_date}")

        for config in configs_1m:
            trades = run_single_day_backtest(train_df, test_df, config)
            daily_pnl = sum(t["pnl_pips"] for t in trades)

            results[config.name]["trades"].extend(trades)
            results[config.name]["daily_pnl"].append(daily_pnl)

            status = "[+]" if daily_pnl > 0 else "[-]" if daily_pnl < 0 else "[=]"
            print(f"  {config.name}: {len(trades)} trades, {daily_pnl:+.1f} pips {status}")

    summary = []

    for config in configs_1m:
        name = config.name
        trades = results[name]["trades"]
        daily_pnls = results[name]["daily_pnl"]

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
        avg_holding = float(np.mean([t["holding_bars"] for t in trades]))

        max_dd = max_drawdown(pnls)
        profitable_days = sum(1 for p in daily_pnls if p > 0)
        total_days = len(daily_pnls)
        avg_daily = float(np.mean(daily_pnls)) if daily_pnls else 0.0

        summary.append({
            "strategy": name,
            "trades": len(trades),
            "pnl_pips": total_pnl,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "max_drawdown_pips": max_dd,
            "avg_daily_pnl_pips": avg_daily,
            "tp_hits": tp_hits,
            "sl_hits": sl_hits,
            "time_exits": time_exits,
            "avg_holding_bars": avg_holding,
            "profitable_days": profitable_days,
            "total_days": total_days,
        })

    print("\n" + "=" * 80)
    print("HOLDOUT SUMMARY")
    print("=" * 80)
    if summary:
        print(f"{'Strategy':<25} {'Trades':<8} {'PnL':<12} {'PF':<8} {'Win%':<8} {'MaxDD'}")
        print("-" * 80)
        for r in sorted(summary, key=lambda x: x["pnl_pips"], reverse=True):
            print(f"{r['strategy']:<25} {r['trades']:<8} {r['pnl_pips']:+10.1f}  "
                  f"{r['profit_factor']:<8.2f} {r['win_rate']:<8.1%} {r['max_drawdown_pips']:.1f}")
        print("-" * 80)
    else:
        print("No trades generated in the holdout window.")

    if summary:
        results_df = pd.DataFrame(summary)
        results_path = Path(__file__).parent / "reports" / "holdout_4w_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
