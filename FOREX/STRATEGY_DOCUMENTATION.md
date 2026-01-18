# 📚 Dokumentace aktuální edge strategie

## Přehled

Aktuální přístup je **multi‑pair high‑vol momentum edge** validovaný přes walk‑forward režim a řízený monitoringem. Cíl je **stabilita a udržitelnost**, ne krátkodobé výstřely.

## Edge – princip
- Krátkodobé momentum pouze v **high‑vol režimu**.
- Vstupy na základě **probability threshold** a **min gap**.
- SL/TP jsou **ATR‑based** s konzervativním R:R.

## Core parametry (baseline)
- Probability threshold: **0.58**
- Min gap: **0.06**
- SL: **1.5× ATR**
- TP: **2.0× ATR**
- Max holding: **40 barů**

Poznámka: JPY páry mají odlišnou pip scale; v kódu je automatická detekce podle price scale.

## Doporučené koše
- **basket_6**: EURUSD, USDJPY, AUDUSD, EURGBP, AUDJPY (USDCAD jen pokud obnoví edge)
- **basket_8**: + GBPUSD, NZDJPY (aktuálně slabší)

## Monitoring (rolling okna)
Sleduj:
- Expectancy/Trade
- Profit Factor
- Win Rate
- Trade Frequency
- Max Drawdown a DD duration

## Rozhodovací pravidla
- Pokud expectancy < 0 na 30–50 trades → snížit váhu.
- Pokud PF < 1.05 po 2 měsících → re‑fit + retune thresholdů.
- Pokud DD > 2× historický průměr → snížit risk/strategii.

## Implementace
- Multi‑pair backtest: [FOREX/experiments/portfolio_multi_pair/run_multi_pair_full_period.py](FOREX/experiments/portfolio_multi_pair/run_multi_pair_full_period.py)
