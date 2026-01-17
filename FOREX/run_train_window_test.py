#!/usr/bin/env python3
"""
Train Window Comparison Test

Compares different training window sizes (2, 3, 5, 7 days) 
on 2-week holdout to find optimal train/trade cadence.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

DATA_FILE = Path(__file__).parent / "data" / "dukascopy" / "EURUSD" / "EURUSD_prepared.parquet"
TRAIN_WINDOWS = [2, 3, 5, 7]  # Days of training data
HOLDOUT_WEEKS = 4  # Test on 4 weeks for enough data
END_DATE = "2026-01-17"


def run_test(train_days: int) -> dict:
    """Run holdout test with specific train window."""
    print(f"\n{'='*60}")
    print(f"🔄 TESTING {train_days}-DAY TRAINING WINDOW")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "run_holdout_4w.py",
        "--parquet", str(DATA_FILE),
        "--weeks", str(HOLDOUT_WEEKS),
        "--train-days", str(train_days),
        "--end-date", END_DATE,
        "--embargo-days", "0",
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    # Load results
    results_path = Path(__file__).parent / "reports" / "holdout_4w_results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        # Rename to preserve
        new_path = Path(__file__).parent / "reports" / f"holdout_train{train_days}d_results.csv"
        results_path.rename(new_path)
        return df.to_dict('records')
    return []


def main():
    print("="*70)
    print("🔬 TRAIN WINDOW COMPARISON TEST")
    print(f"   Windows: {TRAIN_WINDOWS} days")
    print(f"   Holdout: {HOLDOUT_WEEKS} weeks")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)
    
    if not DATA_FILE.exists():
        print(f"❌ Data file not found: {DATA_FILE}")
        return
    
    all_results = {}
    
    for train_days in TRAIN_WINDOWS:
        results = run_test(train_days)
        all_results[train_days] = results
    
    # Comparison
    print("\n" + "="*80)
    print("📊 TRAIN WINDOW COMPARISON")
    print("="*80)
    print(f"{'Strategy':<25} ", end="")
    for td in TRAIN_WINDOWS:
        print(f" {td}d train ", end="")
    print()
    print("-"*80)
    
    strategies = set()
    for results in all_results.values():
        for r in results:
            strategies.add(r['strategy'])
    
    for strategy in sorted(strategies):
        print(f"{strategy:<25} ", end="")
        for td in TRAIN_WINDOWS:
            results = all_results.get(td, [])
            strat_result = [r for r in results if r['strategy'] == strategy]
            if strat_result:
                pnl = strat_result[0]['pnl_pips']
                marker = "✅" if pnl > 0 else "❌"
                print(f" {pnl:>+7.1f}{marker}", end="")
            else:
                print(f" {'N/A':>9}", end="")
        print()
    
    print("-"*80)
    
    # Find best train window per strategy
    print("\n🏆 BEST TRAIN WINDOW PER STRATEGY:")
    for strategy in sorted(strategies):
        best_pnl = float('-inf')
        best_td = None
        for td in TRAIN_WINDOWS:
            results = all_results.get(td, [])
            strat_result = [r for r in results if r['strategy'] == strategy]
            if strat_result and strat_result[0]['pnl_pips'] > best_pnl:
                best_pnl = strat_result[0]['pnl_pips']
                best_td = td
        if best_td:
            print(f"   {strategy}: {best_td} days → {best_pnl:+.1f} pips")
    
    print(f"\n✅ COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
