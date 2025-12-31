# 📂 Data - Datové soubory

Tato složka obsahuje všechna data pro ML pipeline klasifikace cenových pohybů.

**Každý experiment má vlastní podsložku** (např. `30_tickers/`, `50_tickers/`, `100_tickers/`).

---

## 📁 Struktura

```
data/
├── 30_tickers/          # Experiment: 30 tickerů (10 per sektor)
│   ├── ohlcv/           # Surová OHLCV data z yfinance
│   ├── fundamentals/    # Fundamentální metriky
│   ├── complete/        # Kompletní dataset (OHLCV + fundamenty)
│   └── figures/         # Vizualizace výsledků
│
├── 50_tickers/          # (budoucí experiment)
├── 100_tickers/         # (budoucí experiment)
└── README.md
```

---

## 📂 30_tickers/

### Statistiky experimentu

| Metrika | Hodnota |
|---------|---------|
| Tickerů | 30 (10 per sektor) |
| Období | 10.7 let (2014-2024) |
| Celkem vzorků | 3,380 |
| Accuracy | 32.1% |

### ohlcv/

Surová cenová data stažená z yfinance API.

| Soubor | Popis |
|--------|-------|
| `all_sectors_ohlcv_10y.csv` | OHLCV pro všechny sektory |

### fundamentals/

Fundamentální metriky stažené z yfinance .info.

### complete/

🎯 **HLAVNÍ DATASET** - Kompletní data s imputovanými fundamenty.

| Soubor | Řádků |
|--------|-------|
| `all_sectors_complete_10y.csv` | 3,380 |
| `Technology_complete_10y.csv` | ~1,100 |
| `Consumer_complete_10y.csv` | ~1,100 |
| `Industrials_complete_10y.csv` | ~1,100 |

### figures/

📈 **Vizualizace výsledků:**
- `confusion_matrix.png`
- `roc_curves.png`
- `feature_importance.png`
- `sector_comparison.png`

---

## 📊 Tickery v 30_tickers experimentu

### Technology (10)
AAPL, MSFT, NVDA, GOOGL, META, AVGO, ORCL, CSCO, ADBE, CRM

### Consumer (10)
AMZN, TSLA, HD, MCD, NKE, SBUX, TGT, LOW, PG, KO

### Industrials (10)
CAT, HON, UPS, BA, GE, RTX, DE, LMT, MMM, UNP

---

*Vytvořeno: 31. prosince 2025*
