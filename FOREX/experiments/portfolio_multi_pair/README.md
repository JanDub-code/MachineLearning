# Multi-Pair Full Period Validation

This folder contains a full-period, multi-pair validation for the LogReg-Momentum high-vol edge.

## Workflow

1) Download Dukascopy data for the basket:

- Basket 6 (recommended): `--basket basket_6`
- Basket 8 (extended): `--basket basket_8`

2) Run multi-pair validation:

- Produces per-pair outputs and a portfolio report in a timestamped folder.

3) Run monitoring report:

- Uses rolling windows to flag expectancy/PF/DD issues.

## Outputs

Each run creates:
- `results_YYYYMMDD_HHMM/PAIR/metrics.json`
- `results_YYYYMMDD_HHMM/PAIR/trades.csv`
- `results_YYYYMMDD_HHMM/PAIR/daily_pnl.csv`
- `results_YYYYMMDD_HHMM/portfolio_report.md`
- `results_YYYYMMDD_HHMM/monitoring_report.md`
- `results_YYYYMMDD_HHMM/monitoring_alerts.json`

## Notes

- Costs and ATR filters are in pips and auto-adjust for JPY pairs.
- Portfolio aggregation sums daily PnL across pairs.
- Use the same date range across pairs for fair comparison.
- Monitoring script: [FOREX/experiments/portfolio_multi_pair/run_monitoring.py](FOREX/experiments/portfolio_multi_pair/run_monitoring.py)
