#!/usr/bin/env python3
"""Portfolio monitoring report for rolling performance checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import rolling_trade_metrics, trade_frequency, recent_drawdown_ratio
from src.risk_metrics import calculate_all_metrics


@dataclass
class Thresholds:
    min_trades: int = 30
    min_expectancy: float = 0.0
    min_profit_factor: float = 1.05
    min_win_rate: float = 0.45
    max_dd_ratio: float = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitoring report for portfolio runs")
    parser.add_argument("--results-dir", required=True, help="Path to results_YYYYMMDD_HHMM folder")
    parser.add_argument("--window", type=int, default=30, help="Rolling trade window (default 30)")
    parser.add_argument("--window2", type=int, default=50, help="Secondary trade window (default 50)")
    parser.add_argument("--min-trades", type=int, default=30, help="Minimum trades for alerting")
    parser.add_argument("--min-exp", type=float, default=0.0, help="Min expectancy for alerting")
    parser.add_argument("--min-pf", type=float, default=1.05, help="Min profit factor for alerting")
    parser.add_argument("--min-wr", type=float, default=0.45, help="Min win rate for alerting")
    parser.add_argument("--max-dd-ratio", type=float, default=2.0, help="Max recent DD / baseline DD")
    return parser.parse_args()


def load_pair_outputs(pair_dir: Path) -> Tuple[List[Dict], Dict, Dict[str, float]]:
    trades_path = pair_dir / "trades.csv"
    metrics_path = pair_dir / "metrics.json"
    daily_path = pair_dir / "daily_pnl.csv"

    trades = []
    if trades_path.exists():
        trades = pd.read_csv(trades_path).to_dict(orient="records")

    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    daily_pnl = {}
    if daily_path.exists():
        daily_df = pd.read_csv(daily_path)
        if "date" in daily_df.columns:
            daily_pnl = dict(zip(daily_df["date"], daily_df["pnl_pips"]))
        else:
            daily_pnl = dict(zip(daily_df.iloc[:, 0], daily_df.iloc[:, 1]))

    return trades, metrics, daily_pnl


def evaluate_alerts(
    pair: str,
    trades: List[Dict],
    baseline_metrics: Dict,
    rolling_metrics: Dict,
    thresholds: Thresholds,
) -> List[str]:
    alerts = []
    if not rolling_metrics:
        return alerts

    total_trades = rolling_metrics.get("total_trades", 0)
    if total_trades < thresholds.min_trades:
        return alerts

    exp = rolling_metrics.get("expectancy", 0.0)
    pf = rolling_metrics.get("profit_factor", 0.0)
    wr = rolling_metrics.get("win_rate", 0.0)
    recent_dd = rolling_metrics.get("max_drawdown", 0.0)
    base_dd = baseline_metrics.get("max_drawdown", 0.0)
    dd_ratio = recent_drawdown_ratio(recent_dd, base_dd)

    if exp < thresholds.min_expectancy:
        alerts.append(f"{pair}: rolling expectancy {exp:+.2f} < {thresholds.min_expectancy:+.2f}")
    if pf < thresholds.min_profit_factor:
        alerts.append(f"{pair}: rolling PF {pf:.2f} < {thresholds.min_profit_factor:.2f}")
    if wr < thresholds.min_win_rate:
        alerts.append(f"{pair}: rolling WR {wr:.1%} < {thresholds.min_win_rate:.1%}")
    if dd_ratio > thresholds.max_dd_ratio:
        alerts.append(f"{pair}: DD ratio {dd_ratio:.2f} > {thresholds.max_dd_ratio:.2f}")

    return alerts


def build_report(
    results_dir: Path,
    windows: List[int],
    thresholds: Thresholds,
) -> Tuple[str, Dict]:
    pair_dirs = [p for p in results_dir.iterdir() if p.is_dir()]
    pair_dirs = [p for p in pair_dirs if (p / "trades.csv").exists()]

    report_lines = []
    alerts_all: List[str] = []

    report_lines.append("# Monitoring Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("")

    portfolio_daily: Dict[str, float] = {}
    portfolio_trades: List[Dict] = []

    report_lines.append("## Per-Pair Status")
    report_lines.append("")

    for pair_dir in sorted(pair_dirs):
        pair = pair_dir.name
        trades, baseline_metrics, daily_pnl = load_pair_outputs(pair_dir)
        portfolio_trades.extend(trades)
        for date, pnl in daily_pnl.items():
            portfolio_daily[date] = portfolio_daily.get(date, 0.0) + pnl

        report_lines.append(f"### {pair}")
        report_lines.append(f"- Total Trades: {baseline_metrics.get('total_trades', 0)}")
        report_lines.append(f"- Total PnL: {baseline_metrics.get('total_pnl', 0):+.1f} pips")
        report_lines.append(f"- Profit Factor: {baseline_metrics.get('profit_factor', 0):.2f}")
        report_lines.append(f"- Expectancy/Trade: {baseline_metrics.get('expectancy', 0):+.2f} pips")
        report_lines.append(f"- Trade Frequency (per month): {trade_frequency(trades):.1f}")

        for window in windows:
            rolling = rolling_trade_metrics(trades, window)
            if not rolling.metrics:
                continue
            rm = rolling.metrics
            report_lines.append(f"  - Rolling({window}) Exp: {rm.get('expectancy', 0):+.2f}")
            report_lines.append(f"  - Rolling({window}) PF: {rm.get('profit_factor', 0):.2f}")
            report_lines.append(f"  - Rolling({window}) WR: {rm.get('win_rate', 0):.1%}")

            alerts = evaluate_alerts(pair, trades, baseline_metrics, rm, thresholds)
            alerts_all.extend(alerts)

        report_lines.append("")

    report_lines.append("## Portfolio Summary")
    report_lines.append("")
    daily_values = [portfolio_daily[k] for k in sorted(portfolio_daily.keys())]
    portfolio_metrics = calculate_all_metrics(portfolio_trades, daily_values)
    report_lines.append(f"- Total Trades: {portfolio_metrics.get('total_trades', 0)}")
    report_lines.append(f"- Total PnL: {portfolio_metrics.get('total_pnl', 0):+.1f} pips")
    report_lines.append(f"- Profit Factor: {portfolio_metrics.get('profit_factor', 0):.2f}")
    report_lines.append(f"- Expectancy/Trade: {portfolio_metrics.get('expectancy', 0):+.2f} pips")

    report_lines.append("")
    report_lines.append("## Alerts")
    report_lines.append("")
    if alerts_all:
        report_lines.extend([f"- {a}" for a in sorted(set(alerts_all))])
    else:
        report_lines.append("- None")

    report = "\n".join(report_lines)
    payload = {
        "alerts": sorted(set(alerts_all)),
        "portfolio": portfolio_metrics,
    }
    return report, payload


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()

    thresholds = Thresholds(
        min_trades=args.min_trades,
        min_expectancy=args.min_exp,
        min_profit_factor=args.min_pf,
        min_win_rate=args.min_wr,
        max_dd_ratio=args.max_dd_ratio,
    )

    report, payload = build_report(results_dir, [args.window, args.window2], thresholds)

    report_path = results_dir / "monitoring_report.md"
    report_path.write_text(report, encoding="utf-8")

    alerts_path = results_dir / "monitoring_alerts.json"
    alerts_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"✅ Monitoring report saved: {report_path}")


if __name__ == "__main__":
    main()
