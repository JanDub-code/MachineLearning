# Forex Edge Workflow Plan

Goal: udržitelný **multi‑pair high‑vol momentum edge** s walk‑forward validací, monitoringem a konzervativním risk managementem.

## Pairs a data
- Primární koš: **EURUSD, USDJPY, AUDUSD, EURGBP, AUDJPY**
- Volitelné rozšíření: **NZDJPY**, **GBPUSD** (jen pokud edge drží)
- Data zdroj: **Dukascopy 1m** (parquet v `FOREX/data/dukascopy/PAIR/`)

## Core edge (baseline)
- Probability threshold: **0.58**
- Min gap: **0.06**
- SL/TP: **1.5× ATR / 2.0× ATR**
- Max holding: **40 barů**

## Validace a portfolio
- Walk‑forward per pair → agregace do portfolia
- Portfolio report: PnL, PF, expectancy, DD
- Preferovat koše s nízkou korelací faktorů

## Monitoring (rolling okna 30/50)
- Expectancy/Trade
- Profit Factor
- Win Rate
- Trade frequency
- Max drawdown + DD duration
- Alerting podle prahů

## Retrénink / rebalancing
- Monitoring týdně
- Rebalancing vah měsíčně (na základě 3–6 měsíců)
- Retrénink modelu 1× za 1–3 měsíce

## Struktura
- Data downloader: [FOREX/download_dukascopy.py](FOREX/download_dukascopy.py)
- Multi‑pair backtest: [FOREX/experiments/portfolio_multi_pair/run_multi_pair_full_period.py](FOREX/experiments/portfolio_multi_pair/run_multi_pair_full_period.py)
- Monitoring: [FOREX/experiments/portfolio_multi_pair/run_monitoring.py](FOREX/experiments/portfolio_multi_pair/run_monitoring.py)
- Retrénink guide: [FOREX/experiments/portfolio_multi_pair/model_retraining_guide.md](FOREX/experiments/portfolio_multi_pair/model_retraining_guide.md)

## Immediate Actions
1) Udržovat koš bez negativního edge (drop/downweight slabé páry).
2) Spouštět monitoring report po každém runu.
3) Měsíčně re‑fit + jemná kalibrace thresholdů.
