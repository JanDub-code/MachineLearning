# Multi-Pair Summary (basket_6)

Period: 2025-10-17 → 2026-01-17 (66 trading days, 51 test days after 15d warmup)

## Key Results (Portfolio)
- Total Trades: 254
- Total PnL: +54.4 pips
- Expectancy/Trade: +0.21 pips
- Win Rate: 50.8%
- Profit Factor: 1.13
- Max Drawdown: 63.3 pips

## Per-Pair Highlights
- EURUSD: strongest and consistent (+33.9 pips, PF 1.79)
- USDCAD: negative edge (-16.7 pips, PF 0.82)
- AUDUSD: small positive edge (+13.2 pips, PF 1.19)
- EURGBP: small positive edge with higher trade count (+24.0 pips, PF 1.11)
- USDJPY / AUDJPY: 0 trades (filters too strict or regime mismatch)

## Interpretation
- Edge survives in EURUSD and partially in AUDUSD/EURGBP.
- USDCAD weakens the basket; JPY pairs show no triggers under current thresholds.
- Portfolio expectancy is positive but materially below the target 15–20% annualized.

## Next Steps
1) Recalibrate JPY filters (ATR bounds, cost) to restore trade frequency.
2) Drop or downweight USDCAD unless edge improves after recalibration.
3) Run basket_8 as a robustness check.
4) Add correlation-aware portfolio sizing (volatility-weighted) and re-run.
5) Validate stability across rolling 30–50 trade windows per pair.
