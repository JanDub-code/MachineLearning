#!/usr/bin/env python3
"""
Multi-Period Strategy Validation

Tests strategies on different time windows (2, 4, 8 weeks) using Dukascopy data.
Generates comparative report showing robustness across time periods.

Usage:
    python run_multi_period_test.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Dukascopy data file
DATA_FILE = Path(__file__).parent / "data" / "dukascopy" / "EURUSD" / "EURUSD_1m_2025-10-17_2026-01-17.parquet"

# Test periods (weeks)
PERIODS = [2, 4, 8]

# Training days per fold
TRAIN_DAYS = 2

# End date (most recent data available)
END_DATE = "2026-01-17"


def prepare_data():
    """Load and prepare the parquet data for the holdout script format."""
    print("📊 Loading Dukascopy data...")
    
    df = pd.read_parquet(DATA_FILE)
    
    # The holdout script expects 'time' column, not index
    if 'time' not in df.columns:
        df = df.reset_index()
        df.rename(columns={df.columns[0]: 'time'}, inplace=True)
    
    # Ensure proper format
    df['time'] = pd.to_datetime(df['time'])
    
    # Holdout script expects OHLCV columns
    expected_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in expected_cols if c in df.columns]]
    
    # Save in format expected by holdout script
    prepared_file = DATA_FILE.parent / "EURUSD_prepared.parquet"
    df.to_parquet(prepared_file, index=False)
    
    print(f"   Rows: {len(df):,}")
    print(f"   Range: {df['time'].min()} → {df['time'].max()}")
    print(f"   Saved to: {prepared_file}")
    
    return prepared_file


def run_holdout_test(parquet_path: Path, weeks: int, train_days: int, end_date: str) -> Path:
    """Run holdout backtest for specific number of weeks."""
    print(f"\n{'='*60}")
    print(f"🔄 RUNNING {weeks}-WEEK HOLDOUT TEST")
    print(f"{'='*60}")
    
    output_name = f"holdout_{weeks}w_results.csv"
    
    cmd = [
        sys.executable,
        "run_holdout_4w.py",
        "--parquet", str(parquet_path),
        "--weeks", str(weeks),
        "--train-days", str(train_days),
        "--end-date", end_date,
        "--embargo-days", "0",  # Use all data up to end date
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=False)
    
    if result.returncode != 0:
        print(f"⚠️  {weeks}-week test had issues")
    
    # Rename results file
    results_path = Path(__file__).parent / "reports" / "holdout_4w_results.csv"
    new_path = Path(__file__).parent / "reports" / f"holdout_{weeks}w_results.csv"
    
    if results_path.exists():
        results_path.rename(new_path)
        print(f"\n✅ Results saved to: {new_path}")
        return new_path
    
    return None


def compare_results():
    """Load all period results and create comparison report."""
    print("\n" + "="*80)
    print("📊 MULTI-PERIOD COMPARISON")
    print("="*80)
    
    reports_dir = Path(__file__).parent / "reports"
    
    all_results = {}
    
    for weeks in PERIODS:
        results_file = reports_dir / f"holdout_{weeks}w_results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            all_results[f"{weeks}w"] = df
            print(f"\n📁 Loaded {weeks}-week results: {len(df)} strategies")
    
    if not all_results:
        print("❌ No results files found!")
        return
    
    # Find strategies that appear in all periods
    all_strategies = set()
    for period, df in all_results.items():
        all_strategies.update(df['strategy'].tolist())
    
    print(f"\n📋 Total unique strategies: {len(all_strategies)}")
    
    # Create comparison DataFrame
    comparison = []
    
    for strategy in sorted(all_strategies):
        row = {'strategy': strategy}
        
        for period, df in all_results.items():
            strat_data = df[df['strategy'] == strategy]
            if len(strat_data) > 0:
                row[f'{period}_pnl'] = strat_data['pnl_pips'].values[0]
                row[f'{period}_trades'] = strat_data['trades'].values[0]
                row[f'{period}_pf'] = strat_data['profit_factor'].values[0]
                row[f'{period}_winrate'] = strat_data['win_rate'].values[0]
            else:
                row[f'{period}_pnl'] = None
                row[f'{period}_trades'] = None
                row[f'{period}_pf'] = None
                row[f'{period}_winrate'] = None
        
        # Check consistency (profitable across all periods?)
        pnls = [row.get(f'{p}_pnl') for p in all_results.keys()]
        pnls = [p for p in pnls if p is not None]
        
        if len(pnls) == len(all_results):
            row['consistent_profitable'] = all(p > 0 for p in pnls)
            row['avg_pnl'] = sum(pnls) / len(pnls)
            row['min_pnl'] = min(pnls)
            row['max_pnl'] = max(pnls)
        else:
            row['consistent_profitable'] = False
            row['avg_pnl'] = None
            row['min_pnl'] = None
            row['max_pnl'] = None
        
        comparison.append(row)
    
    comparison_df = pd.DataFrame(comparison)
    
    # Sort by average PnL
    comparison_df = comparison_df.sort_values('avg_pnl', ascending=False, na_position='last')
    
    # Print summary
    print("\n" + "-"*100)
    print(f"{'STRATEGY':<30}", end="")
    for period in all_results.keys():
        print(f"  {period.upper():>12}", end="")
    print(f"  {'AVG':>10}  {'CONSISTENT':>12}")
    print("-"*100)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['strategy']:<30}", end="")
        for period in all_results.keys():
            pnl = row.get(f'{period}_pnl')
            if pnl is not None:
                marker = "✅" if pnl > 0 else "❌"
                print(f"  {pnl:>+10.1f}{marker}", end="")
            else:
                print(f"  {'N/A':>12}", end="")
        
        avg = row.get('avg_pnl')
        if avg is not None:
            print(f"  {avg:>+10.1f}", end="")
        else:
            print(f"  {'N/A':>10}", end="")
        
        consistent = row.get('consistent_profitable', False)
        print(f"  {'✅ YES' if consistent else '❌ NO':>12}")
    
    print("-"*100)
    
    # Highlight consistently profitable strategies
    consistent_profitable = comparison_df[comparison_df['consistent_profitable'] == True]
    
    print(f"\n🏆 CONSISTENTLY PROFITABLE STRATEGIES: {len(consistent_profitable)}")
    
    if len(consistent_profitable) > 0:
        print("\nTop 5 by average PnL:")
        for i, (_, row) in enumerate(consistent_profitable.head().iterrows(), 1):
            print(f"  {i}. {row['strategy']}: avg {row['avg_pnl']:+.1f} pips (min: {row['min_pnl']:+.1f}, max: {row['max_pnl']:+.1f})")
    else:
        print("⚠️  No strategy was profitable across ALL tested periods!")
        print("   This suggests overfitting or market regime changes.")
    
    # Save comparison
    comparison_path = reports_dir / "multi_period_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n💾 Full comparison saved to: {comparison_path}")
    
    return comparison_df


def main():
    print("="*80)
    print("🔬 MULTI-PERIOD STRATEGY VALIDATION")
    print(f"   Periods: {PERIODS} weeks")
    print(f"   Data: {DATA_FILE.name}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)
    
    # Check data exists
    if not DATA_FILE.exists():
        print(f"❌ Data file not found: {DATA_FILE}")
        print("   Run download_dukascopy.py first!")
        return
    
    # Prepare data
    prepared_path = prepare_data()
    
    # Run tests for each period
    for weeks in PERIODS:
        run_holdout_test(prepared_path, weeks, TRAIN_DAYS, END_DATE)
    
    # Compare results
    compare_results()
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)


if __name__ == "__main__":
    main()
