# Forex Pipeline Plan

Goal: pick lowest-cost major pair, select demo-capable broker/platform, and define a 12-horizon rolling-window model pipeline ready for paper trading.

## Pair and Broker
- Primary pair: **EUR/USD** (tightest typical spread 0.1–0.4 pips on ECN; deep liquidity; 24/5).
- Secondary pair (optional): GBP/USD for diversification if spread <0.7 pips on chosen broker.
- Broker/platform: **OANDA v20 API + practice account** (free tick/1m data, REST + streaming, native paper trading, no platform lock-in). If local spreads are worse than 0.5 pips, fallback: IC Markets demo via MT5 (requires bridge library). We will proceed assuming OANDA practice.

## Cost Model (paper baseline)
- Spread: assume 0.2 pip typical on EUR/USD practice; adjust from live quotes.
- Commission: OANDA practice = 0 per side; we still track effective spread cost.
- Slippage: model +0.1 pip on market orders; revisit after first paper tests.

## Architecture Overview
- **Data fetcher**: OANDA streaming for live ticks; REST for historical 1m/5m candles; store compressed Parquet under `FOREX/data/raw/YYYY-MM-DD/`.
- **Feature builder**: resample to 1m bars; compute returns, rolling volatility, ATR, RSI, MACD, moving averages, volume proxies (if available), session/time-of-day encodings.
- **Model trainer**: 12 parallel models `M_k`, each trained on the last `k` weeks of data (k=1..12). Weekly retrain on Sunday night UTC.
- **Model selector/ensemble**: track live performance; weight models by recent Sharpe/F1 over rolling holdout; optionally pick top-1.
- **Executor**: consume latest predictions, filter by confidence threshold, size positions with ATR-based sizing, enforce trailing drawdown circuit breaker.

## 12-Window Scheme
- Windows: weeks `k = 1..12`; each window uses only the most recent `k * 7 * 24 * 60` minutes.
- Target: next-bar direction or short-horizon return bucket; start with binary UP/DOWN and probability threshold.
- Training cadence: every Sunday run all 12 fits; store under `FOREX/models/w{k}/` with metrics and feature importance.
- Online weighting: weights updated daily from last N trades; ensemble prediction = weighted average of model probabilities.

## Risk and Execution Rules
- Confidence gate: place trades only if `p > 0.6` (tune later).
- Position sizing: fixed risk per trade `r = 0.25%` of equity; lot size uses ATR-based stop distance: $$\text{size} = \frac{r \times E}{\text{ATR}_n \times pip\_value}$$
- Trailing drawdown breaker: halt trading if equity drops 2–5% from the day high-water mark; require manual restart.
- Session filter: avoid low-liquidity Friday close / Sunday open; prefer London+NY overlap.

## Folder Structure (new)
- `FOREX/PLAN.md` (this doc)
- `FOREX/config/` broker keys, pair list, hyperparams (local, not committed when secrets added)
- `FOREX/src/` code:
  - `data_fetcher.py` (OANDA REST/stream stubs)
  - `features.py`
  - `trainer.py` (12-window loop)
  - `executor.py` (paper-trading loop)
  - `configs.py` (paths, constants)
- `FOREX/notebooks/` optional EDA/backtests
- `FOREX/data/`, `FOREX/models/`, `FOREX/reports/`

## Immediate Actions
1) Scaffold `src` modules with OANDA client placeholders and 12-window training loop interface.
2) Add config template for pair list, thresholds, and paths.
3) Implement historical download for EUR/USD 1m bars to seed initial training set.
4) Add simple backtest to validate signal/threshold logic before going live on paper.
