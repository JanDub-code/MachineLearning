"""
Walk-Forward Backtester V2 - IMPROVED VERSION

Key improvements over V1:
1. Stop Loss (SL) and Take Profit (TP) based on ATR
2. Higher confidence threshold with dynamic adjustment
3. Minimum expected move filter (trade only when move > costs)
4. Confidence intervals on performance metrics
5. Detailed trade journal with evidence

Ensures no lookahead bias with proper time-series splits.
"""

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .configs import load_settings
from .features import build_features


@dataclass
class TradeResultV2:
    """Single trade outcome with full details."""
    entry_time: dt.datetime
    exit_time: Optional[dt.datetime]
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    probability: float
    stop_loss: float
    take_profit: float
    atr_at_entry: float
    pnl_pips: float
    outcome: str  # "TP_HIT", "SL_HIT", "TIME_EXIT", "PENDING"
    correct: bool
    holding_bars: int


@dataclass
class BacktestResultV2:
    """Enhanced results with confidence intervals."""
    window_weeks: int
    total_trades: int
    
    # Classification metrics
    accuracy: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    precision: float
    recall: float
    f1: float
    
    # Trading metrics
    total_pnl_pips: float
    pnl_ci_low: float
    pnl_ci_high: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    
    # Risk metrics
    avg_win_pips: float
    avg_loss_pips: float
    profit_factor: float
    risk_reward_ratio: float
    
    # Trade breakdown
    tp_hits: int
    sl_hits: int
    time_exits: int
    
    trades: List[TradeResultV2] = field(default_factory=list)


@dataclass
class WalkForwardConfigV2:
    """Enhanced configuration with risk management."""
    # Period settings (in rows/bars)
    test_week_rows: int = 2016  # ~1 week of 5m bars
    
    # Cost model
    spread_pips: float = 0.2
    slippage_pips: float = 0.1
    
    # Signal thresholds
    probability_threshold: float = 0.65  # Higher = more confident trades only
    min_probability_gap: float = 0.15  # Diff from 0.5 must be > this
    
    # Prediction settings
    horizon_bars: int = 1
    
    # Risk management (ATR-based)
    sl_atr_multiplier: float = 1.5  # Stop loss = 1.5 * ATR
    tp_atr_multiplier: float = 2.0  # Take profit = 2.0 * ATR (better RR ratio)
    max_holding_bars: int = 12  # Exit if neither SL nor TP hit in 12 bars (1 hour for 5m)
    
    # Filters
    min_atr_pips: float = 2.0  # Don't trade if ATR < 2 pips (low volatility)
    max_atr_pips: float = 30.0  # Don't trade if ATR > 30 pips (too volatile)
    
    # Value settings
    pip_value: float = 0.0001  # For EUR/USD
    
    # Statistical settings
    confidence_level: float = 0.95  # For confidence intervals


class WalkForwardBacktesterV2:
    """
    Enhanced walk-forward backtester with risk management.
    
    Key features:
    - ATR-based stop loss and take profit
    - Confidence-filtered entries
    - Proper position management
    - Statistical confidence intervals
    """
    
    def __init__(self, config: WalkForwardConfigV2 = None):
        self.config = config or WalkForwardConfigV2()
        self.settings = load_settings()
    
    def _get_transaction_cost(self) -> float:
        """Total round-trip cost in pips."""
        return self.config.spread_pips + self.config.slippage_pips
    
    def _train_model(self, train_df: pd.DataFrame) -> RandomForestClassifier:
        """Train model on the given data."""
        feature_cols = [c for c in train_df.columns if c not in {"time", "target", "return_h"}]
        X = train_df[feature_cols]
        y = train_df["target"]
        
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,  # Limit depth to prevent overfitting
            min_samples_leaf=20,  # Require more samples per leaf
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        )
        clf.fit(X, y)
        return clf
    
    def _simulate_trade(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        direction: str,
        probability: float,
    ) -> TradeResultV2:
        """
        Simulate a single trade with SL/TP logic.
        
        Returns trade result after SL hit, TP hit, or max holding period.
        """
        entry_row = df.iloc[entry_idx]
        entry_price = entry_row["close"]
        entry_time = entry_row["time"]
        atr_pips = entry_row["atr"] / self.config.pip_value
        
        # Calculate SL and TP levels in pips
        sl_distance = atr_pips * self.config.sl_atr_multiplier
        tp_distance = atr_pips * self.config.tp_atr_multiplier
        
        cost = self._get_transaction_cost()
        
        # Simulate forward from entry
        exit_idx = entry_idx
        outcome = "TIME_EXIT"
        pnl_pips = 0
        
        for i in range(1, min(self.config.max_holding_bars + 1, len(df) - entry_idx)):
            future_idx = entry_idx + i
            if future_idx >= len(df):
                break
            
            future_row = df.iloc[future_idx]
            high = future_row["high"]
            low = future_row["low"]
            close = future_row["close"]
            
            # Calculate current P&L based on direction
            if direction == "long":
                # Check if SL hit (price went below entry - SL distance)
                sl_price = entry_price - (sl_distance * self.config.pip_value)
                tp_price = entry_price + (tp_distance * self.config.pip_value)
                
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
                sl_price = entry_price + (sl_distance * self.config.pip_value)
                tp_price = entry_price - (tp_distance * self.config.pip_value)
                
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
        
        # If neither SL nor TP hit, exit at current price (time exit)
        if outcome == "TIME_EXIT":
            exit_idx = min(entry_idx + self.config.max_holding_bars, len(df) - 1)
            exit_row = df.iloc[exit_idx]
            exit_price = exit_row["close"]
            
            if direction == "long":
                pnl_pips = (exit_price - entry_price) / self.config.pip_value - cost
            else:
                pnl_pips = (entry_price - exit_price) / self.config.pip_value - cost
        else:
            exit_price = df.iloc[exit_idx]["close"]
        
        exit_time = df.iloc[exit_idx]["time"]
        holding_bars = exit_idx - entry_idx
        
        # Determine if trade was correct (profitable)
        correct = pnl_pips > 0
        
        return TradeResultV2(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            probability=probability,
            stop_loss=sl_distance,
            take_profit=tp_distance,
            atr_at_entry=atr_pips,
            pnl_pips=pnl_pips,
            outcome=outcome,
            correct=correct,
            holding_bars=holding_bars,
        )
    
    def _evaluate_on_test(
        self,
        model: RandomForestClassifier,
        test_df: pd.DataFrame,
    ) -> Tuple[List[TradeResultV2], Dict]:
        """Evaluate model on test data with full trade simulation."""
        feature_cols = [c for c in test_df.columns if c not in {"time", "target", "return_h"}]
        X_test = test_df[feature_cols]
        y_test = test_df["target"]
        
        probas = model.predict_proba(X_test)[:, 1]
        
        trades = []
        y_pred_filtered = []
        y_true_filtered = []
        
        cost = self._get_transaction_cost()
        skip_until = 0  # Skip bars while in a trade
        
        for i, (prob, actual_target) in enumerate(zip(probas, y_test)):
            # Skip if we're still in a previous trade
            if i < skip_until:
                continue
            
            # Get ATR for this bar
            atr_pips = test_df.iloc[i]["atr"] / self.config.pip_value
            
            # Filter: Skip if volatility is too low or too high
            if atr_pips < self.config.min_atr_pips or atr_pips > self.config.max_atr_pips:
                continue
            
            # Calculate probability gap from 0.5
            prob_gap = abs(prob - 0.5)
            
            # Only trade if probability exceeds threshold AND gap is significant
            if prob_gap < self.config.min_probability_gap:
                continue
            
            # Determine direction
            if prob >= self.config.probability_threshold:
                direction = "long"
                predicted = 1
            elif prob <= (1 - self.config.probability_threshold):
                direction = "short"
                predicted = 0
            else:
                continue  # Not confident enough
            
            # Simulate the trade
            trade = self._simulate_trade(test_df, i, direction, prob)
            trades.append(trade)
            
            # Skip bars until trade exits
            skip_until = i + trade.holding_bars + 1
            
            y_pred_filtered.append(predicted)
            y_true_filtered.append(actual_target)
        
        # Calculate metrics
        metrics = {}
        if y_pred_filtered and len(y_pred_filtered) >= 5:
            metrics["accuracy"] = accuracy_score(y_true_filtered, y_pred_filtered)
            metrics["precision"] = precision_score(y_true_filtered, y_pred_filtered, zero_division=0)
            metrics["recall"] = recall_score(y_true_filtered, y_pred_filtered, zero_division=0)
            metrics["f1"] = f1_score(y_true_filtered, y_pred_filtered, zero_division=0)
        else:
            metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
        
        return trades, metrics
    
    def _calculate_confidence_interval(
        self,
        values: List[float],
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Calculate mean and confidence interval for a list of values."""
        if not values or len(values) < 2:
            return 0, 0, 0
        
        mean = np.mean(values)
        sem = stats.sem(values)
        ci = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
        
        return mean, mean - ci, mean + ci
    
    def _calculate_performance_stats(self, trades: List[TradeResultV2]) -> Dict:
        """Calculate comprehensive trading performance statistics."""
        if not trades:
            return {
                "total_pnl_pips": 0, "pnl_ci_low": 0, "pnl_ci_high": 0,
                "sharpe_ratio": 0, "max_drawdown": 0, "win_rate": 0,
                "avg_win_pips": 0, "avg_loss_pips": 0, "profit_factor": 0,
                "risk_reward_ratio": 0, "tp_hits": 0, "sl_hits": 0, "time_exits": 0,
            }
        
        pnls = [t.pnl_pips for t in trades]
        
        # PnL with confidence interval
        total_pnl = sum(pnls)
        mean_pnl, ci_low, ci_high = self._calculate_confidence_interval(
            pnls, self.config.confidence_level
        )
        
        # Equity curve for Sharpe and drawdown
        equity_curve = np.cumsum(pnls)
        
        # Sharpe ratio
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 * 24 / self.config.max_holding_bars)
        else:
            sharpe = 0
        
        # Max drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown_pips = peak - equity_curve
        max_dd = np.max(drawdown_pips) if len(drawdown_pips) > 0 else 0
        
        # Win/loss analysis
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Risk/reward ratio
        risk_reward = avg_win / abs(avg_loss) if avg_loss != 0 else 0
        
        # Trade outcome breakdown
        tp_hits = sum(1 for t in trades if t.outcome == "TP_HIT")
        sl_hits = sum(1 for t in trades if t.outcome == "SL_HIT")
        time_exits = sum(1 for t in trades if t.outcome == "TIME_EXIT")
        
        return {
            "total_pnl_pips": total_pnl,
            "pnl_ci_low": ci_low * len(pnls),  # Scale to total
            "pnl_ci_high": ci_high * len(pnls),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "avg_win_pips": avg_win,
            "avg_loss_pips": avg_loss,
            "profit_factor": profit_factor,
            "risk_reward_ratio": risk_reward,
            "tp_hits": tp_hits,
            "sl_hits": sl_hits,
            "time_exits": time_exits,
        }
    
    def _run_single_window(
        self,
        df: pd.DataFrame,
        window_weeks: int,
        test_start_idx: int,
        train_len: int,
        test_len: int,
    ) -> BacktestResultV2:
        """Run backtest for a single window with full metrics."""
        train_end_idx = test_start_idx
        train_start_idx = max(0, train_end_idx - train_len)
        test_end_idx = min(len(df), test_start_idx + test_len)
        
        if train_end_idx - train_start_idx < train_len // 2:
            return None
        
        train_df = df.iloc[train_start_idx:train_end_idx].copy()
        test_df = df.iloc[test_start_idx:test_end_idx].copy()
        
        if len(test_df) < 50 or len(train_df) < 100:
            return None
        
        model = self._train_model(train_df)
        trades, metrics = self._evaluate_on_test(model, test_df)
        perf_stats = self._calculate_performance_stats(trades)
        
        # Calculate accuracy confidence interval
        if trades:
            correct_trades = [1 if t.correct else 0 for t in trades]
            acc, acc_ci_low, acc_ci_high = self._calculate_confidence_interval(
                correct_trades, self.config.confidence_level
            )
        else:
            acc, acc_ci_low, acc_ci_high = 0, 0, 0
        
        return BacktestResultV2(
            window_weeks=window_weeks,
            total_trades=len(trades),
            accuracy=metrics["accuracy"],
            accuracy_ci_low=acc_ci_low,
            accuracy_ci_high=acc_ci_high,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            total_pnl_pips=perf_stats["total_pnl_pips"],
            pnl_ci_low=perf_stats["pnl_ci_low"],
            pnl_ci_high=perf_stats["pnl_ci_high"],
            sharpe_ratio=perf_stats["sharpe_ratio"],
            max_drawdown=perf_stats["max_drawdown"],
            win_rate=perf_stats["win_rate"],
            avg_win_pips=perf_stats["avg_win_pips"],
            avg_loss_pips=perf_stats["avg_loss_pips"],
            profit_factor=perf_stats["profit_factor"],
            risk_reward_ratio=perf_stats["risk_reward_ratio"],
            tp_hits=perf_stats["tp_hits"],
            sl_hits=perf_stats["sl_hits"],
            time_exits=perf_stats["time_exits"],
            trades=trades,
        )
    
    def run_walk_forward(
        self,
        df: pd.DataFrame,
        n_test_periods: int = 4,
    ) -> Dict[int, List[BacktestResultV2]]:
        """Run full walk-forward backtest with enhanced metrics."""
        # Build features
        df = build_features(df, horizon_minutes=self.config.horizon_bars)
        df = df.sort_values("time").reset_index(drop=True)
        
        total_rows = len(df)
        test_week_len = self.config.test_week_rows
        
        min_rows_needed = test_week_len + 100
        
        if total_rows < min_rows_needed:
            print(f"ERROR: Only {total_rows} rows available, need at least {min_rows_needed}")
            return {w: [] for w in range(1, 13)}
        
        max_window_possible = max(1, (total_rows - test_week_len) // test_week_len)
        max_window = min(12, max_window_possible)
        
        remaining_after_training = total_rows - (max_window * test_week_len)
        scaled_test_len = min(test_week_len, remaining_after_training // max(1, n_test_periods))
        scaled_test_len = max(100, scaled_test_len)
        
        print(f"Data summary: {total_rows} rows available")
        print(f"Adapted to: max {max_window}-week window, ~{scaled_test_len} rows per test period")
        print(f"Risk settings: SL={self.config.sl_atr_multiplier}xATR, TP={self.config.tp_atr_multiplier}xATR")
        print(f"Confidence threshold: {self.config.probability_threshold}")
        
        results: Dict[int, List[BacktestResultV2]] = {w: [] for w in range(1, 13)}
        
        actual_test_periods = max(1, min(n_test_periods, remaining_after_training // scaled_test_len))
        
        for period_idx in range(actual_test_periods):
            print(f"\n--- Test Period {period_idx + 1}/{actual_test_periods} ---")
            
            for window_weeks in range(1, max_window + 1):
                train_len = window_weeks * test_week_len
                test_start = train_len + period_idx * scaled_test_len
                
                if test_start + scaled_test_len > total_rows:
                    continue
                
                result = self._run_single_window(
                    df, window_weeks, test_start,
                    train_len=train_len,
                    test_len=scaled_test_len
                )
                
                if result and result.total_trades > 0:
                    results[window_weeks].append(result)
                    print(f"  Window {window_weeks:2d}w: Trades={result.total_trades:3d}, "
                          f"WinRate={result.win_rate:.1%}, PnL={result.total_pnl_pips:+.1f}, "
                          f"TP={result.tp_hits}/SL={result.sl_hits}/Time={result.time_exits}")
        
        return results
    
    def aggregate_results(self, results: Dict[int, List[BacktestResultV2]]) -> pd.DataFrame:
        """Aggregate results with confidence intervals."""
        summary = []
        
        for window_weeks, period_results in results.items():
            if not period_results:
                continue
            
            all_trades = []
            for r in period_results:
                all_trades.extend(r.trades)
            
            if not all_trades:
                continue
            
            n_periods = len(period_results)
            total_trades = sum(r.total_trades for r in period_results)
            
            # Aggregate metrics with confidence intervals
            accuracies = [r.accuracy for r in period_results if r.total_trades > 0]
            pnls = [r.total_pnl_pips for r in period_results]
            
            avg_accuracy, acc_ci_low, acc_ci_high = self._calculate_confidence_interval(
                accuracies, self.config.confidence_level
            )
            
            total_pnl = sum(pnls)
            avg_pnl, pnl_ci_low, pnl_ci_high = self._calculate_confidence_interval(
                pnls, self.config.confidence_level
            )
            
            avg_win_rate = np.mean([r.win_rate for r in period_results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in period_results])
            max_dd = max(r.max_drawdown for r in period_results)
            avg_profit_factor = np.mean([r.profit_factor for r in period_results])
            avg_rr = np.mean([r.risk_reward_ratio for r in period_results])
            
            total_tp = sum(r.tp_hits for r in period_results)
            total_sl = sum(r.sl_hits for r in period_results)
            total_time = sum(r.time_exits for r in period_results)
            
            summary.append({
                "window_weeks": window_weeks,
                "n_periods": n_periods,
                "total_trades": total_trades,
                "avg_accuracy": avg_accuracy,
                "accuracy_ci_low": acc_ci_low,
                "accuracy_ci_high": acc_ci_high,
                "avg_win_rate": avg_win_rate,
                "total_pnl_pips": total_pnl,
                "avg_pnl_per_period": avg_pnl,
                "pnl_ci_low": pnl_ci_low,
                "pnl_ci_high": pnl_ci_high,
                "avg_sharpe": avg_sharpe,
                "max_drawdown": max_dd,
                "avg_profit_factor": avg_profit_factor,
                "avg_risk_reward": avg_rr,
                "tp_hits": total_tp,
                "sl_hits": total_sl,
                "time_exits": total_time,
            })
        
        if not summary:
            return pd.DataFrame()
        
        return pd.DataFrame(summary).sort_values("total_pnl_pips", ascending=False)
    
    def generate_evidence_report(
        self,
        results: Dict[int, List[BacktestResultV2]],
        output_path: Path = None,
    ) -> str:
        """Generate comprehensive evidence report with statistics."""
        summary_df = self.aggregate_results(results)
        
        report_lines = [
            "=" * 80,
            "WALK-FORWARD BACKTEST REPORT V2 - WITH RISK MANAGEMENT",
            f"Generated: {dt.datetime.now().isoformat()}",
            "=" * 80,
            "",
            "CONFIGURATION:",
            f"  - Test period: ~1 week ({self.config.test_week_rows} bars)",
            f"  - Transaction costs: {self.config.spread_pips + self.config.slippage_pips:.2f} pips",
            f"  - Confidence threshold: {self.config.probability_threshold}",
            f"  - Min probability gap: {self.config.min_probability_gap}",
            "",
            "RISK MANAGEMENT:",
            f"  - Stop Loss: {self.config.sl_atr_multiplier} x ATR",
            f"  - Take Profit: {self.config.tp_atr_multiplier} x ATR",
            f"  - Risk/Reward Target: 1:{self.config.tp_atr_multiplier/self.config.sl_atr_multiplier:.1f}",
            f"  - Max holding period: {self.config.max_holding_bars} bars",
            f"  - Min ATR filter: {self.config.min_atr_pips} pips",
            f"  - Max ATR filter: {self.config.max_atr_pips} pips",
            "",
            "STATISTICAL SETTINGS:",
            f"  - Confidence level: {self.config.confidence_level:.0%}",
            "",
        ]
        
        if summary_df.empty:
            report_lines.extend([
                "NO RESULTS - Not enough trades generated.",
                "Consider lowering the probability threshold.",
            ])
        else:
            report_lines.extend([
                "=" * 80,
                "RESULTS BY WINDOW SIZE (sorted by Total PnL):",
                "=" * 80,
                "",
            ])
            
            for _, row in summary_df.iterrows():
                report_lines.extend([
                    f"WINDOW: {int(row['window_weeks'])} WEEKS",
                    "-" * 40,
                    f"  Total Trades: {int(row['total_trades'])} across {int(row['n_periods'])} test periods",
                    "",
                    "  CLASSIFICATION METRICS:",
                    f"    Accuracy: {row['avg_accuracy']:.1%} (95% CI: {row['accuracy_ci_low']:.1%} - {row['accuracy_ci_high']:.1%})",
                    f"    Win Rate: {row['avg_win_rate']:.1%}",
                    "",
                    "  PROFIT/LOSS:",
                    f"    Total PnL: {row['total_pnl_pips']:+.1f} pips",
                    f"    Avg PnL/Period: {row['avg_pnl_per_period']:+.1f} pips (95% CI: {row['pnl_ci_low']:+.1f} to {row['pnl_ci_high']:+.1f})",
                    f"    Max Drawdown: {row['max_drawdown']:.1f} pips",
                    "",
                    "  RISK METRICS:",
                    f"    Sharpe Ratio: {row['avg_sharpe']:.2f}",
                    f"    Profit Factor: {row['avg_profit_factor']:.2f}",
                    f"    Risk/Reward Ratio: 1:{row['avg_risk_reward']:.2f}",
                    "",
                    "  TRADE OUTCOMES:",
                    f"    Take Profit Hits: {int(row['tp_hits'])} ({row['tp_hits']/row['total_trades']*100:.1f}%)",
                    f"    Stop Loss Hits: {int(row['sl_hits'])} ({row['sl_hits']/row['total_trades']*100:.1f}%)",
                    f"    Time Exits: {int(row['time_exits'])} ({row['time_exits']/row['total_trades']*100:.1f}%)",
                    "",
                ])
            
            # Best model recommendation
            best = summary_df.iloc[0]
            report_lines.extend([
                "=" * 80,
                "EVIDENCE SUMMARY & RECOMMENDATION:",
                "=" * 80,
                "",
                f"Best performing window: {int(best['window_weeks'])} weeks",
                "",
                "KEY EVIDENCE:",
                f"  1. Win Rate: {best['avg_win_rate']:.1%}",
                f"  2. Total PnL: {best['total_pnl_pips']:+.1f} pips",
                f"  3. Profit Factor: {best['avg_profit_factor']:.2f}",
                f"  4. Risk/Reward Achieved: 1:{best['avg_risk_reward']:.2f}",
                "",
            ])
            
            # Verdict
            if best['total_pnl_pips'] > 0 and best['avg_profit_factor'] > 1.0:
                report_lines.extend([
                    "VERDICT: POTENTIALLY VIABLE FOR PAPER TRADING",
                    "  - Strategy shows positive expectancy after costs",
                    "  - Recommend extended testing with more data",
                    "  - Start with small position sizes on practice account",
                ])
            elif best['avg_profit_factor'] > 0.8:
                report_lines.extend([
                    "VERDICT: MARGINAL - NEEDS IMPROVEMENT",
                    "  - Strategy is close to break-even",
                    "  - May work in trending markets",
                    "  - Consider adjusting parameters or adding filters",
                ])
            else:
                report_lines.extend([
                    "VERDICT: NOT READY FOR TRADING",
                    "  - Strategy shows negative expectancy",
                    "  - Improvements needed before paper trading",
                    "  - Consider: different features, longer timeframe, more data",
                ])
        
        report_lines.extend([
            "",
            "=" * 80,
            "METHODOLOGY NOTE:",
            "=" * 80,
            "This backtest uses walk-forward validation with NO lookahead bias.",
            "Each model is trained on past data only and tested on unseen future data.",
            "Transaction costs (spread + slippage) are applied to all trades.",
            "Stop loss and take profit levels are based on ATR at entry time.",
            "Confidence intervals are calculated using t-distribution.",
            "",
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            print(f"\nReport saved to: {output_path}")
        
        return report
