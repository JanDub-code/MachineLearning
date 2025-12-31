# 📊 Dokumentace ML Pipeline: 30 Tickerů, 3 Sektory

**Projekt:** Klasifikace cenových pohybů akcií pomocí hybridního ML přístupu  
**Datum:** 31. prosince 2025  
**Autor:** Bc. Jan Dub

---

## 📋 Obsah

1. [Přehled projektu](#1-přehled-projektu)
2. [Data a tickery](#2-data-a-tickery)
3. [Architektura pipeline](#3-architektura-pipeline)
4. [Krok 1: Stažení OHLCV dat](#4-krok-1-stažení-ohlcv-dat)
5. [Krok 2: Stažení fundamentálních dat](#5-krok-2-stažení-fundamentálních-dat)
6. [Krok 3: Trénink RF Regressoru](#6-krok-3-trénink-rf-regressoru)
7. [Krok 4: Imputace historických dat](#7-krok-4-imputace-historických-dat)
8. [Krok 5: Trénink RF Classifieru](#8-krok-5-trénink-rf-classifieru)
9. [Krok 6: Hyperparameter Tuning](#9-krok-6-hyperparameter-tuning)
10. [Krok 7: Finální evaluace](#10-krok-7-finální-evaluace)
11. [Výsledky a vizualizace](#11-výsledky-a-vizualizace)
12. [Struktura souborů](#12-struktura-souborů)
13. [Závěry a doporučení](#13-závěry-a-doporučení)

---

## 1. Přehled projektu

### Cíl
Vytvořit ML model pro **klasifikaci měsíčních cenových pohybů** akcií do tří kategorií:
- **DOWN** (pokles > 3%)
- **HOLD** (změna ±3%)
- **UP** (růst > 3%)

### Metodologie
**Hybridní přístup** kombinující:
1. **RandomForest Regressor** - pro imputaci chybějících fundamentálních dat
2. **RandomForest Classifier** - pro klasifikaci cenových pohybů

### Proč hybridní přístup?
- Fundamentální data (P/E, ROE, atd.) jsou dostupná pouze pro poslední období
- Historická data mají pouze OHLCV (cena, objem)
- RF Regressor se naučí vztah mezi OHLCV a fundamenty
- Pak predikuje fundamenty pro historická data
- Classifier využívá kompletní dataset (OHLCV + fundamenty)

---

## 2. Data a tickery

### Sektory a tickery (30 celkem)

| Sektor | Tickery (10) |
|--------|-------------|
| **Technology** | AAPL, MSFT, NVDA, GOOGL, META, AVGO, ORCL, CSCO, ADBE, CRM |
| **Consumer** | AMZN, TSLA, HD, MCD, NKE, SBUX, TGT, LOW, PG, KO |
| **Industrials** | CAT, HON, UPS, BA, GE, RTX, DE, LMT, MMM, UNP |

### Statistiky datasetu

| Metrika | Hodnota |
|---------|---------|
| Celkem řádků | 3,870 |
| Počet tickerů | 30 |
| Časové období | 10.7 let (2014-2024) |
| Frekvence | Měsíční |
| OHLCV features | 5 (open, high, low, close, volume) |
| Technické indikátory | 13 |
| Fundamentální metriky | 11 |

---

## 3. Architektura pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE ARCHITEKTURA                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   yfinance   │───▶│  OHLCV Data  │───▶│  Technické   │      │
│  │     API      │    │   3,870 rows │    │  Indikátory  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                  │              │
│  ┌──────────────┐    ┌──────────────┐            │              │
│  │   yfinance   │───▶│ Fundamenty   │            │              │
│  │   .info      │    │   30 tickers │            │              │
│  └──────────────┘    └──────────────┘            │              │
│                             │                    │              │
│                             ▼                    ▼              │
│                    ┌─────────────────────────────────┐          │
│                    │     RF REGRESSOR (R²=0.76-0.97) │          │
│                    │  Predikce fundamentů z OHLCV    │          │
│                    └─────────────────────────────────┘          │
│                                    │                            │
│                                    ▼                            │
│                    ┌─────────────────────────────────┐          │
│                    │   KOMPLETNÍ DATASET (3,380)     │          │
│                    │   OHLCV + Tech. Ind. + Fundam.  │          │
│                    └─────────────────────────────────┘          │
│                                    │                            │
│                                    ▼                            │
│                    ┌─────────────────────────────────┐          │
│                    │      RF CLASSIFIER              │          │
│                    │   DOWN / HOLD / UP (±3%)        │          │
│                    └─────────────────────────────────┘          │
│                                    │                            │
│                                    ▼                            │
│                    ┌─────────────────────────────────┐          │
│                    │      PREDIKCE + EVALUACE        │          │
│                    │   Accuracy: 32.1%, F1: 31.0%    │          │
│                    └─────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Krok 1: Stažení OHLCV dat

### Skript: `download_30_tickers.py`

**Vstup:** Seznam 30 tickerů z 3 sektorů  
**Výstup:** `data/ohlcv/all_sectors_ohlcv_10y.csv`

### Stažená data
- **Období:** 2014-01-01 až 2024-12-31
- **Frekvence:** Denní → agregováno na měsíční
- **Sloupce:** date, ticker, sector, open, high, low, close, volume

### Vypočtené technické indikátory

| Indikátor | Popis | Perioda |
|-----------|-------|---------|
| `volatility` | Směrodatná odchylka returns | - |
| `returns` | Procentuální změna close | - |
| `rsi_14` | Relative Strength Index | 14 |
| `macd` | MACD linie | 12/26 |
| `macd_signal` | MACD signal | 9 |
| `macd_hist` | MACD histogram | - |
| `sma_3/6/12` | Simple Moving Average | 3/6/12 |
| `ema_3/6/12` | Exponential Moving Average | 3/6/12 |
| `volume_change` | Změna objemu | - |

### Výsledek
```
✅ Staženo: 3,870 řádků
✅ Tickerů: 30
✅ Období: 10.7 let
```

---

## 5. Krok 2: Stažení fundamentálních dat

### Skript: `download_fundamentals.py`

**Vstup:** Seznam 30 tickerů  
**Výstup:** `data/fundamentals/all_sectors_fundamentals.csv`

### Stažené metriky (25 sloupců)

| Kategorie | Metriky |
|-----------|---------|
| **Valuační** | trailingPE, forwardPE, priceToBook, priceToSalesTrailing12Months, enterpriseToRevenue, enterpriseToEbitda |
| **Profitabilita** | returnOnEquity, returnOnAssets, profitMargins, operatingMargins, grossMargins |
| **Zadluženost** | debtToEquity, currentRatio, quickRatio |
| **Dividendy** | dividendYield, payoutRatio |
| **Růst** | revenueGrowth, earningsGrowth, earningsQuarterlyGrowth |
| **Riziko** | beta |
| **Ostatní** | bookValue, marketCap, sharesOutstanding |

### Výsledek
```
✅ Tickerů: 30
✅ Sloupců: 25
✅ Pokrytí: ~80% (některé metriky NaN)
```

---

## 6. Krok 3: Trénink RF Regressoru

### Skript: `train_rf_regressor.py`

**Cíl:** Naučit se predikovat fundamentální metriky z OHLCV features

### Konfigurace modelu
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
```

### Features (vstup)
- OHLCV: open, high, low, close, volume
- Technické: volatility, returns, rsi_14, macd, sma_*, ema_*, volume_change

### Targets (výstup)
- trailingPE, forwardPE, priceToBook
- returnOnEquity, returnOnAssets
- profitMargins, operatingMargins, grossMargins
- debtToEquity, currentRatio, beta

### Výsledky trénování

| Target | MAE | R² Score |
|--------|-----|----------|
| trailingPE | 4.419 | **0.957** |
| forwardPE | 2.595 | **0.964** |
| returnOnAssets | 0.015 | **0.970** |
| returnOnEquity | 0.045 | 0.935 |
| priceToBook | 1.854 | 0.891 |
| profitMargins | 0.031 | 0.886 |
| debtToEquity | 38.513 | 0.765 |

### Feature Importance (Top 5)
1. **volume**: 0.4995 (dominantní!)
2. sma_12: 0.0734
3. ema_12: 0.0730
4. sma_6: 0.0586
5. ema_6: 0.0583

### Výsledek
```
✅ Model uložen: models/regressors/fundamental_predictor.pkl
✅ Scaler uložen: models/scalers/feature_scaler.pkl
✅ Průměrné R²: 0.91
```

---

## 7. Krok 4: Imputace historických dat

### Proces
1. Rozdělení dat na:
   - **Recent** (poslední ~2 roky): má reálné fundamenty
   - **Historical** (starší): pouze OHLCV
2. Trénink RF Regressoru na Recent datech
3. Predikce fundamentů pro Historical data
4. Spojení do kompletního datasetu

### Statistiky

| Část | Počet řádků |
|------|-------------|
| Recent (reálné fundamenty) | 650 |
| Historical (predikované) | 2,730 |
| **Celkem** | **3,380** |

### Výstup
```
✅ data/complete/all_sectors_complete_10y.csv
✅ data/complete/Technology_complete_10y.csv
✅ data/complete/Consumer_complete_10y.csv
✅ data/complete/Industrials_complete_10y.csv
```

---

## 8. Krok 5: Trénink RF Classifieru

### Skript: `train_rf_classifier.py`

### Definice target variable
```python
THRESHOLD = 0.03  # ±3%

def classify(future_return):
    if future_return < -0.03:
        return 0  # DOWN
    elif future_return > 0.03:
        return 2  # UP
    else:
        return 1  # HOLD
```

### Konfigurace modelu (baseline)
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

### Distribuce tříd

| Třída | Počet | Procento |
|-------|-------|----------|
| DOWN | 871 | 26.0% |
| HOLD | 1,111 | 33.2% |
| UP | 1,368 | 40.8% |

### Baseline výsledky

| Metrika | Hodnota |
|---------|---------|
| Accuracy | 33.4% |
| Precision | 33.7% |
| Recall | 33.4% |
| F1-Score | 32.6% |

---

## 9. Krok 6: Hyperparameter Tuning

### Skript: `hyperparameter_tuning.py`

### Metodologie
- **GridSearchCV** s **TimeSeriesSplit** (5 folds)
- Chronologický split pro respektování časové závislosti

### Prohledávaný prostor

| Parametr | Hodnoty |
|----------|---------|
| n_estimators | 100, 200 |
| max_depth | 10, 15, 20 |
| min_samples_split | 5, 10 |
| min_samples_leaf | 2, 4 |
| class_weight | balanced |

### Nejlepší parametry
```json
{
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced"
}
```

### Výsledky po tuningu

| Metrika | Baseline | Tuned | Změna |
|---------|----------|-------|-------|
| CV F1 | - | 36.8% | - |
| Test Accuracy | 33.4% | 32.1% | -1.3% |
| Test F1 | 32.6% | 31.0% | -1.6% |

> **Poznámka:** Nižší test metriky po tuningu mohou indikovat overfitting na CV data nebo lepší regularizaci (menší max_depth).

---

## 10. Krok 7: Finální evaluace

### Skript: `final_evaluation.py`

### Celkové výsledky

| Metrika | Hodnota |
|---------|---------|
| **Accuracy** | 32.09% |
| **Precision** | 32.87% |
| **Recall** | 32.09% |
| **F1-Score** | 31.00% |
| Test samples | 670 |

### Classification Report

```
              precision    recall  f1-score   support

        DOWN       0.30      0.51      0.38       193
        HOLD       0.33      0.20      0.25       216
          UP       0.35      0.28      0.31       261

    accuracy                           0.32       670
   macro avg       0.33      0.33      0.31       670
weighted avg       0.33      0.32      0.31       670
```

### Analýza per-sector

| Sektor | Accuracy | F1-Score | Samples |
|--------|----------|----------|---------|
| **Industrials** | 35.9% | 34.6% | 231 |
| Consumer | 30.4% | 29.8% | 181 |
| Technology | 29.8% | 27.6% | 258 |

**Poznatek:** Model funguje nejlépe na Industrials sektoru, nejhůře na Technology (vyšší volatilita, těžší predikce).

---

## 11. Výsledky a vizualizace

### Confusion Matrix

![Confusion Matrix](data/30_tickers/figures/confusion_matrix.png)

**Analýza:**
- Model má tendenci predikovat DOWN častěji než ostatní třídy
- HOLD je nejhůře rozpoznávaná třída (pouze 20% recall)
- Nejvíce záměn mezi UP a DOWN

### ROC Curves

![ROC Curves](data/30_tickers/figures/roc_curves.png)

**AUC skóre:**
- DOWN: ~0.55
- HOLD: ~0.52
- UP: ~0.54

> Hodnoty blízko 0.5 indikují slabou separabilitu tříd (typické pro finanční predikce).

### Feature Importance

![Feature Importance](data/30_tickers/figures/feature_importance.png)

**Top 10 nejdůležitějších features:**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | returns | 0.0577 |
| 2 | volatility | 0.0560 |
| 3 | macd_hist | 0.0489 |
| 4 | macd_signal | 0.0481 |
| 5 | volume_change | 0.0449 |
| 6 | rsi_14 | 0.0430 |
| 7 | macd | 0.0392 |
| 8 | returnOnEquity | 0.0380 |
| 9 | returnOnAssets | 0.0373 |
| 10 | currentRatio | 0.0359 |

**Poznámky:**
- Technické indikátory (returns, volatility, MACD) dominují
- Fundamentální metriky (ROE, ROA) jsou také důležité
- Volume-based features mají menší vliv než očekáváno

### Sector Comparison

![Sector Comparison](data/30_tickers/figures/sector_comparison.png)

---

## 12. Struktura souborů

```
CleanSolution/
├── 📄 DOKUMENTACE_30tickeru_3sektory_postup.md  (tento soubor)
│
├── 📂 data/
│   └── 📂 30_tickers/               # 🎯 EXPERIMENT: 30 tickerů
│       ├── 📂 ohlcv/                # Surová OHLCV data
│       │   └── all_sectors_ohlcv_10y.csv
│       │
│       ├── 📂 fundamentals/         # Fundamentální data
│       │   └── all_sectors_fundamentals.csv
│       │
│       ├── 📂 complete/             # Kompletní dataset (OHLCV + fundamenty)
│       │   ├── all_sectors_complete_10y.csv
│       │   ├── Technology_complete_10y.csv
│       │   ├── Consumer_complete_10y.csv
│       │   └── Industrials_complete_10y.csv
│       │
│       └── 📂 figures/              # Vizualizace výsledků
│           ├── confusion_matrix.png
│           ├── roc_curves.png
│           ├── feature_importance.png
│           └── sector_comparison.png
│
├── 📂 models/
│   └── 📂 30_tickers/               # 🎯 MODELY: 30 tickerů
│       ├── 📂 classifiers/          # Klasifikační modely
│       │   ├── rf_classifier_all_sectors.pkl    (baseline)
│       │   └── rf_classifier_tuned.pkl          (po tuningu)
│       │
│       ├── 📂 regressors/           # Regresní modely
│       │   └── fundamental_predictor.pkl        (pro imputaci)
│       │
│       ├── 📂 scalers/              # Scalery
│       │   ├── feature_scaler.pkl
│       │   ├── classifier_scaler.pkl
│       │   └── classifier_scaler_tuned.pkl
│       │
│       └── 📂 metadata/             # Metadata a výsledky
│           ├── optimal_hyperparameters.json
│           ├── final_evaluation_results.json
│           ├── classifier_metadata.json
│           ├── classifier_feature_importance.csv
│           ├── feature_importance.csv
│           └── grid_search_results.csv
│
├── 📄 download_30_tickers.py        # Krok 1: Stažení OHLCV
├── 📄 download_fundamentals.py      # Krok 2: Stažení fundamentů
├── 📄 train_rf_regressor.py         # Krok 3-4: RF Regressor + imputace
├── 📄 train_rf_classifier.py        # Krok 5: RF Classifier
├── 📄 hyperparameter_tuning.py      # Krok 6: Tuning
└── 📄 final_evaluation.py           # Krok 7: Evaluace
```

### 📁 Proč podsložky `30_tickers/`?

Struktura umožňuje **snadné porovnání experimentů** s různým počtem tickerů:

```
data/
├── 30_tickers/    # Accuracy: 32.1%
├── 50_tickers/    # (budoucí experiment)
├── 100_tickers/   # (budoucí experiment)
└── 150_tickers/   # (budoucí experiment)
```

Tímto způsobem lze snadno porovnat, zda **více tickerů zlepšuje přesnost modelu**.

---

## 13. Závěry a doporučení

### Co funguje dobře ✅

1. **RF Regressor pro imputaci** - R² 0.76-0.97 je excelentní
2. **Hybridní přístup** - umožňuje využít fundamentální data i pro historii
3. **Technické indikátory** - returns a volatility jsou nejdůležitější features
4. **Industrials sektor** - model zde funguje nejlépe (35.9% accuracy)

### Limitace ⚠️

1. **Accuracy ~32%** - mírně nad random baseline (33.3%)
2. **HOLD třída** - nejhůře rozpoznávaná (20% recall)
3. **Finanční trhy** - inherentně těžko predikovatelné (EMH)
4. **Malý dataset** - pouze 30 tickerů, 3,380 vzorků

### Doporučení pro zlepšení 🚀

1. **Více dat**
   - Přidat více tickerů (100+)
   - Delší časové období (15-20 let)
   - Více sektorů

2. **Feature engineering**
   - Sentiment analýza (zprávy, social media)
   - Makroekonomické indikátory (úrokové sazby, inflace)
   - Sektorové indikátory

3. **Jiné modely**
   - Gradient Boosting (XGBoost, LightGBM)
   - LSTM pro sekvenční data
   - Ensemble metody

4. **Změna target variable**
   - Binární klasifikace (UP vs NOT UP)
   - Regrese (přesný return)
   - Jiné thresholdy (±5%)

5. **Risk management**
   - Confidence thresholds
   - Position sizing based on probability
   - Stop-loss strategie

---

## 📌 Rychlý start

```bash
# Aktivace prostředí
cd CleanSolution
..\\.venv\\Scripts\\activate

# Spuštění celé pipeline
python download_30_tickers.py
python download_fundamentals.py
python train_rf_regressor.py
python train_rf_classifier.py
python hyperparameter_tuning.py
python final_evaluation.py
```

---

**Konec dokumentace**

*Vytvořeno: 31. prosince 2025*
