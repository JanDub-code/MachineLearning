# FOREX ML Edge System

Praktický systém pro **multi‑pair high‑vol momentum edge** s walk‑forward validací, portfolio agregací a průběžným monitoringem.

## Cíl
- Udržet malý, ale stabilní edge
- Zvýšit frekvenci signálů přes několik nízko‑korelovaných párů
- Řídit riziko konzervativně (bez hype projekcí)

## Aktuální workflow

### 1) Data (Dukascopy)
- 1m data pro koše párů (basket_6 / basket_8)

### 2) Multi‑pair validace
- Plný periodický walk‑forward test s per‑pair výstupy a portfolio agregací

### 3) Monitoring
- Rolling expectancy / PF / WR / drawdown / trade frequency
- Alerting podle prahů

## Doporučený koš
- basket_6: EURUSD, USDJPY, AUDUSD, EURGBP, AUDJPY (+/‑ podle edge)

## Klíčové skripty
- Data: [FOREX/download_dukascopy.py](FOREX/download_dukascopy.py)
- Multi‑pair backtest: [FOREX/experiments/portfolio_multi_pair/run_multi_pair_full_period.py](FOREX/experiments/portfolio_multi_pair/run_multi_pair_full_period.py)
- Monitoring: [FOREX/experiments/portfolio_multi_pair/run_monitoring.py](FOREX/experiments/portfolio_multi_pair/run_monitoring.py)
- Holdout test: [FOREX/run_holdout_4w.py](FOREX/run_holdout_4w.py)
- Live paper: [FOREX/run_live_paper.py](FOREX/run_live_paper.py)

## Dokumentace
- Edge a retrénink: [FOREX/experiments/portfolio_multi_pair/model_retraining_guide.md](FOREX/experiments/portfolio_multi_pair/model_retraining_guide.md)
- Strategie detailně: [FOREX/STRATEGY_DOCUMENTATION.md](FOREX/STRATEGY_DOCUMENTATION.md)

Aktuální výsledky a reporty jsou v [FOREX/experiments/portfolio_multi_pair/](FOREX/experiments/portfolio_multi_pair/).
