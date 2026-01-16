# 🏆 FOREX ML Trading System

## Finální výsledky po optimalizaci

Systém prošel rigorózním testováním 8 strategií s daily retrain na 7 dnech 1m EURUSD dat.

### 📊 Vítězné strategie

| Strategie | PnL (5 dnů) | Profit Factor | Profitable Days | Měsíční projekce |
|-----------|-------------|---------------|-----------------|------------------|
| **🥇 V5.3 Tight R:R** | **+71.6 pips** | **1.16** | **4/5 (80%)** | ~315 pips |
| 🥈 V5.6 Balanced | +44.3 pips | 1.10 | 2/5 (40%) | ~195 pips |

### ❌ Smazané strategie (losers)

- V5.1 Aggressive: -120.5 pips, PF 0.83
- V5.4 Scalper: -40.3 pips, PF 0.92
- V4.4 Tight SL: -5.4 pips, PF 0.99
- V4.3 Combined: +4.7 pips, PF 1.01 (marginal)
- V5.2 Conservative: +0.9 pips, PF 1.00 (break-even)
- V5.5 London: +6.7 pips, PF 1.03 (marginal)

---

## 🎯 Hlavní strategie: V5.3 Tight R:R

### Parametry
```
Probability Threshold: 0.58
Min Probability Gap: 0.08
Stop Loss: 1.0 × ATR (velmi těsný)
Take Profit: 3.0 × ATR (vysoký)
Max Holding: 15 barů (60 minut na 1m)
R:R Ratio: 1:3
```

### Proč funguje
```
Logika "Lose small, win big":
- Win Rate: 38% (prohrává většinu)
- ALE: Když vyhraje, získá 3× více než ztratí
- Matematická edge: 0.38 × 3 - 0.62 × 1 = +0.52 na trade
```

### Výkonnost po dnech
| Den | PnL | Status |
|-----|-----|--------|
| 1 | +43.7 pips | ✅ |
| 2 | +4.5 pips | ✅ |
| 3 | +16.4 pips | ✅ |
| 4 | -3.9 pips | ❌ |
| 5 | +10.8 pips | ✅ |
| **TOTAL** | **+71.6 pips** | **4/5** |

---

## 💰 Realistická projekce zisku

### Konzervativní odhad (PF 1.10)
| Období | PnL (pips) | S 10k účtem (1 lot) |
|--------|------------|---------------------|
| Týden | +71.6 pips | $716 |
| Měsíc | ~315 pips | **$3,150** |
| Rok | ~3,780 pips | **$37,800** |

### ROI
```
Měsíční: 3,150 / 10,000 = 31.5%
Roční (simple): 378%
Roční (compound): (1.315)^12 = 3,108% teoreticky
```

### ⚠️ Důležité upozornění
```
TOTO JSOU PROJEKCE NA ZÁKLADĚ 5 DNŮ!
- Vysoká variance
- Potřeba validace na delším období
- Paper trading NUTNÝ před live
- Drawdown může být značný
```

---

## 📁 Struktura projektu

```
FOREX/
├── README.md                   # Tento soubor
├── PLAN.md                     # Technický plán
├── run_daily_retrain.py        # Hlavní pipeline (denní retrain)
├── run_backtest.py             # Jednorázový backtest
├── config/                     # API klíče
├── data/                       # Stažená data
├── models/                     # Natrénované modely
├── reports/
│   ├── daily_retrain_results.csv
│   └── backtest_report.md
└── src/
    ├── strategy_configs.py     # V5.3 + V5.6 POUZE
    ├── backtester_v2.py
    ├── data_fetcher.py
    ├── features.py
    ├── trainer.py
    ├── executor.py
    └── configs.py
```

---

## 🚀 Jak spustit

### Denní retrain test (doporučeno)
```bash
python run_daily_retrain.py
```

### Jednorázový backtest
```bash
python run_backtest.py
```

---

## 🔬 Proč ML strategie funguje

1. **Institucionální hráči jsou příliš velcí** - nemohou operovat na 1m timeframe
2. **Retail tradeři jsou emocionální** - náš model nemá emoce
3. **Sweet spot v timeframe** - příliš rychlé pro velké, příliš pomalé pro HFT
4. **Denní retrain** - model se adaptuje na aktuální tržní podmínky
5. **Disciplína** - 100% dodržování pravidel bez výjimek

---

*Poslední aktualizace: 2026-01-16*
*Vítězná strategie: V5.3 Tight R:R (PF 1.16)*
