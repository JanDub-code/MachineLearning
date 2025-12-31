# 📚 KOMPLETNÍ DOKUMENTACE DIPLOMOVÉ PRÁCE

# Klasifikace Cenových Pohybů Akcií pomocí Strojového Učení

**Autor:** Bc. Jan Dub  
**Typ práce:** Diplomová práce - Ing. Informatika  
**Datum:** Prosinec 2025  
**Instituce:** [Název univerzity]

---

# OBSAH

1. [ÚVOD A MOTIVACE](#1-úvod-a-motivace)
2. [TEORETICKÝ RÁMEC](#2-teoretický-rámec)
3. [MATEMATICKÉ ZÁKLADY](#3-matematické-základy)
4. [VÝBĚR ALGORITMŮ](#4-výběr-algoritmů)
5. [ARCHITEKTURA ŘEŠENÍ](#5-architektura-řešení)
6. [IMPLEMENTACE PIPELINE](#6-implementace-pipeline)
7. [EXPERIMENT: 30 TICKERŮ](#7-experiment-30-tickerů)
8. [VÝSLEDKY A ANALÝZA](#8-výsledky-a-analýza)
9. [VIZUALIZACE](#9-vizualizace)
10. [OMEZENÍ A BUDOUCÍ PRÁCE](#10-omezení-a-budoucí-práce)
11. [ZÁVĚR](#11-závěr)
12. [REFERENCE](#12-reference)
13. [PŘÍLOHY](#13-přílohy)

---

# 1. ÚVOD A MOTIVACE

## 1.1 Kontext Problému

Predikce pohybů akciových trhů představuje jeden z nejnáročnějších problémů kvantitativních financí. Tato diplomová práce se zaměřuje na vývoj a evaluaci ML systému pro klasifikaci měsíčních cenových pohybů akcií z indexu S&P 500.

### Hypotéza Efektivních Trhů (EMH)

Podle Eugene Famy (1970) existují tři formy tržní efektivity:

| Forma | Dostupné informace | Implikace |
|-------|-------------------|-----------|
| **Slabá** | Historické ceny | Technická analýza nefunguje |
| **Polo-silná** | Veřejné informace | Fundamentální analýza nefunguje |
| **Silná** | Veškeré informace | Žádná strategie nepřekoná trh |

**Naše pozice:** Pokud existují tržní neefektivity, ML modely mohou tyto neefektivity identifikovat a využít. Práce testuje hypotézu, že kombinace fundamentálních a technických faktorů může poskytnout prediktivní signál.

## 1.2 Cíle Práce

1. **Primární cíl:** Vyvinout ML model pro klasifikaci měsíčních cenových pohybů
2. **Sekundární cíl:** Řešit problém chybějících historických fundamentálních dat
3. **Terciární cíl:** Analyzovat prediktivní sílu různých typů features

## 1.3 Klíčová Inovace

Projekt řeší fundamentální problém v kvantitativních financích: **neúplnost historických fundamentálních dat**. Zatímco cenová data (OHLCV) jsou dostupná za 10+ let, fundamentální metriky (P/E, ROE, atd.) jsou typicky dostupné pouze za 1-2 roky.

**Navrhované řešení - Hybridní přístup:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRIDNÍ ML ARCHITEKTURA                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Random Forest Regressor (Imputace)                         │
│     Input: OHLCV + Technické indikátory                        │
│     Output: Fundamentální metriky (P/E, ROE, atd.)             │
│                                                                 │
│  2. Random Forest Classifier (Predikce)                        │
│     Input: OHLCV + Technické + Fundamenty (reálné/imputované)  │
│     Output: Třída pohybu (DOWN / HOLD / UP)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# 2. TEORETICKÝ RÁMEC

## 2.1 Fundamentální vs. Technická Analýza

### 2.1.1 Fundamentální Analýza

Fundamentální analýza se zaměřuje na vnitřní hodnotu aktiva na základě finančních výkazů, ekonomických podmínek a konkurenčního postavení firmy.

**Klíčové metriky používané v této práci:**

| Kategorie | Metriky | Interpretace |
|-----------|---------|--------------|
| **Valuační** | P/E, P/B, P/S, EV/EBITDA | Nadhodnocení/podhodnocení |
| **Profitabilita** | ROE, ROA, Marže | Efektivita generování zisku |
| **Finanční zdraví** | Debt/Equity, Current Ratio | Schopnost splácet závazky |
| **Růst** | Revenue Growth, Earnings Growth | Dynamika růstu |

**Teoretické zdůvodnění:**

Benjamin Graham a David Dodd ve své práci "Security Analysis" (1934) argumentují, že dlouhodobě cena akcie konverguje k její vnitřní hodnotě. Tato práce testuje, zda ML model může identifikovat tuto konvergenci na měsíčním horizontu.

### 2.1.2 Technická Analýza

Technická analýza předpokládá, že veškeré informace jsou zahrnuty v ceně a objemu obchodování.

**Používané indikátory:**

| Indikátor | Formule | Interpretace |
|-----------|---------|--------------|
| **RSI (14)** | $100 - \frac{100}{1 + RS}$ | Překoupenost/přeprodanost |
| **MACD** | $EMA_{12} - EMA_{26}$ | Momentum, změna trendu |
| **SMA/EMA** | Klouzavé průměry | Trend, support/resistance |
| **Volatilita** | $\sigma = \frac{High - Low}{Close}$ | Míra rizika |

### 2.1.3 Proč kombinace obou přístupů?

| Aspekt | Fundamentální | Technická | Kombinace |
|--------|--------------|-----------|-----------|
| Horizont | Dlouhodobý | Krátkodobý | Střední |
| Data | Kvartální | Denní/měsíční | Oba zdroje |
| Lag | Vysoký (reporting) | Nízký | Vyvážený |
| Noise | Nízký | Vysoký | Střední |

## 2.2 Klasifikace vs. Regrese

### 2.2.1 Proč klasifikace?

V původním návrhu byl použit regresní přístup pro predikci přesné hodnoty ceny. Přechod na klasifikaci je motivován:

| Aspekt | Regrese | Klasifikace |
|--------|---------|-------------|
| **Output** | Přesná cena/výnos | Třída pohybu |
| **Interpretace** | "Cena bude $152.34" | "Cena vzroste o >3%" |
| **Praktické využití** | Obtížné (chyba $5 = profit nebo ztráta?) | Přímé trading signály |
| **Robustnost** | Citlivá na outliers | Robustní |
| **Evaluace** | R², MAE (co znamená MAE=$12?) | Accuracy, Precision (72% správných BUY) |

### 2.2.2 Definice Tříd

```
Třída 0 (DOWN):  return < -3%   → Signifikantní pokles
Třída 1 (HOLD):  -3% ≤ return ≤ +3%  → Stagnace
Třída 2 (UP):    return > +3%   → Signifikantní růst
```

**Zdůvodnění prahu ±3%:**
- Typické transakční náklady: 0.1-0.5% (bid-ask spread, poplatky)
- Minimální pohyb pro profitabilní obchod: ~1%
- 3% poskytuje dostatečnou "bezpečnostní rezervu"
- Historicky ~30% měsíců má pohyb > ±3%

## 2.3 Problém Chybějících Dat

### 2.3.1 Klasifikace Missing Data Mechanismů

**Definice (Rubin, 1976):**

| Mechanismus | Definice | V našem případě |
|-------------|----------|-----------------|
| **MCAR** | Chybění nezávisí na žádných hodnotách | API limit, neexistující data |
| **MAR** | Chybění závisí na pozorovaných hodnotách | - |
| **MNAR** | Chybění závisí na nepozorovaných hodnotách | - |

**V našem datasetu:** Fundamentální data chybí primárně kvůli omezení API (MCAR) - mechanismus chybění nesouvisí s hodnotami samotných fundamentů.

### 2.3.2 Přístup k Imputaci

**Regresní imputace pomocí Random Forest:**

$$\hat{F}_t = RF(OHLCV_t, TechIndicators_t)$$

Kde:
- $\hat{F}_t$ = predikované fundamentální metriky v čase $t$
- $RF$ = Random Forest regressor
- $OHLCV_t$ = cenová data v čase $t$
- $TechIndicators_t$ = technické indikátory v čase $t$

**Zdůvodnění přístupu:**
1. Fundamenty nejsou náhodné - existuje vztah s cenou/objemem
2. P/E = Price / Earnings → Price je v OHLCV
3. ROE závisí na tržní kapitalizaci (Price × Shares)
4. Volatilita koreluje s rizikovými metrikami

---

# 3. MATEMATICKÉ ZÁKLADY

## 3.1 Random Forest

### 3.1.1 Definice

**Random Forest** je ensemble metoda kombinující více rozhodovacích stromů:

$$\hat{f}_{RF}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

Kde:
- $B$ = počet stromů (n_estimators)
- $T_b$ = b-tý rozhodovací strom
- $x$ = vstupní vektor features

### 3.1.2 Konstrukce Stromu

Pro každý uzel $t$ s daty $D_t$:

1. Náhodně vyber $m$ features z celkových $p$ (typicky $m = \sqrt{p}$)
2. Najdi nejlepší split $(j^*, s^*)$:

$$
(j^*, s^*) = \arg\min_{j \in M} \arg\min_{s} [L(D_{left}) + L(D_{right})]
$$

Kde $L$ je loss funkce:
- **Klasifikace:** Gini impurity nebo Entropy
- **Regrese:** MSE

### 3.1.3 Gini Impurity (pro klasifikaci)

$$
Gini(t) = 1 - \sum_{k=1}^{K} p_{tk}^2
$$

Kde $p_{tk}$ je proporce třídy $k$ v uzlu $t$.

**Interpretace:**
- $Gini = 0$: Čistý uzel (všechny vzorky jedné třídy)
- $Gini = 0.5$: Maximální impurity pro binární klasifikaci

### 3.1.4 Feature Importance

**Mean Decrease in Impurity (MDI):**

$$
Importance(X_j) = \sum_{t \in T} \frac{n_t}{n} \cdot \Delta impurity(t, X_j)
$$

Kde:
- $T$ = množina uzlů, kde se splituje na $X_j$
- $n_t$ = počet vzorků v uzlu $t$
- $\Delta impurity$ = pokles impurity po splitu

## 3.2 Evaluační Metriky

### 3.2.1 Klasifikační Metriky

**Confusion Matrix:**

```
                    Predicted
                 DOWN  HOLD  UP
Actual   DOWN     TP₀   E₀₁   E₀₂
         HOLD     E₁₀   TP₁   E₁₂
         UP       E₂₀   E₂₁   TP₂
```

**Per-class metriky:**

$$Precision_k = \frac{TP_k}{TP_k + FP_k}$$

$$Recall_k = \frac{TP_k}{TP_k + FN_k}$$

$$F1_k = 2 \cdot \frac{Precision_k \cdot Recall_k}{Precision_k + Recall_k}$$

### 3.2.2 Agregované Metriky

$$Accuracy = \frac{\sum_k TP_k}{N}$$

$$Macro\ F1 = \frac{1}{K} \sum_{k=1}^{K} F1_k$$

$$Weighted\ F1 = \sum_{k=1}^{K} \frac{n_k}{N} \cdot F1_k$$

### 3.2.3 ROC a AUC

**ROC Curve:**
- True Positive Rate: $TPR = \frac{TP}{TP + FN}$
- False Positive Rate: $FPR = \frac{FP}{FP + TN}$

**AUC (Area Under Curve):**

$$AUC = \int_0^1 TPR(FPR^{-1}(x)) dx$$

**Interpretace:**
- AUC = 0.5: Náhodný klasifikátor
- AUC = 1.0: Perfektní klasifikátor
- AUC > 0.5: Lepší než náhodný

### 3.2.4 Regresní Metriky (pro imputaci)

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

## 3.3 Cross-Validation pro Časové Řady

### 3.3.1 TimeSeriesSplit

Pro časové řady nelze použít náhodnou cross-validaci (data leakage). TimeSeriesSplit zajišťuje, že trénovací data jsou vždy před testovacími:

```
Fold 1: Train [1, ..., n₁]     Test [n₁+1, ..., n₂]
Fold 2: Train [1, ..., n₂]     Test [n₂+1, ..., n₃]
Fold 3: Train [1, ..., n₃]     Test [n₃+1, ..., n₄]
...
```

---

# 4. VÝBĚR ALGORITMŮ

## 4.1 Proč Random Forest?

### 4.1.1 Srovnání s Alternativami

| Algoritmus | Výhody | Nevýhody | Vhodnost |
|------------|--------|----------|----------|
| **Random Forest** | Interpretovatelný, robustní, nativní feature importance | Pomalejší než boosting | ⭐⭐⭐⭐⭐ |
| **XGBoost/LightGBM** | Rychlý, vysoká přesnost | Méně interpretovatelný, náchylný k overfittingu | ⭐⭐⭐⭐ |
| **Neural Networks** | Zachycuje komplexní vzory | Black-box, potřebuje hodně dat | ⭐⭐⭐ |
| **SVM** | Dobrý pro malé datasety | Pomalý trénink, obtížná interpretace | ⭐⭐ |
| **Logistic Regression** | Velmi interpretovatelný | Lineární, omezená kapacita | ⭐⭐ |

### 4.1.2 Zdůvodnění Volby RF

1. **Konzistence:** Stejný algoritmus pro imputaci i klasifikaci
2. **Interpretovatelnost:** Feature importance pro analýzu
3. **Robustnost:** Ensemble metoda odolná vůči šumu
4. **Flexibilita:** Nativní podpora multi-class klasifikace
5. **Class balancing:** `class_weight='balanced'`

## 4.2 Hyperparametry RF

| Parametr | Hodnota | Zdůvodnění |
|----------|---------|------------|
| `n_estimators` | 100-200 | Více stromů = stabilnější predikce |
| `max_depth` | 10-15 | Prevence overfittingu |
| `min_samples_split` | 5-10 | Regularizace |
| `min_samples_leaf` | 2-4 | Zajištění robustních listů |
| `class_weight` | 'balanced' | Kompenzace nevyvážených tříd |
| `random_state` | 42 | Reprodukovatelnost |

---

# 5. ARCHITEKTURA ŘEŠENÍ

## 5.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA COLLECTION PHASE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  OHLCV Data (2015-2025)          Fundamental Data (2024-2025)               │
│  ├── Open, High, Low, Close      ├── P/E, P/B, P/S, EV/EBITDA               │
│  ├── Volume                       ├── ROE, ROA, Margins                      │
│  └── Technical Indicators        └── Debt ratios, Growth                    │
│       (RSI, MACD, SMA, EMA)                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMPUTATION MODEL (Random Forest)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  Training: OHLCV (2024-2025) → Fundamentals (2024-2025)                     │
│  Inference: OHLCV (2015-2024) → Predicted Fundamentals (2015-2024)          │
│                                                                              │
│  Input Features (18):              Output Targets (11):                      │
│  ├── OHLCV (5)                     ├── Valuation (3)                        │
│  ├── Technical (8)                 ├── Profitability (5)                    │
│  └── Derived (5)                   └── Health (3)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMPLETE DATASET (2015-2025)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  2015-2024: OHLCV + Predicted Fundamentals (data_source='predicted')        │
│  2024-2025: OHLCV + Real Fundamentals (data_source='real')                  │
│                                                                              │
│  Total: ~3,380 records × 30 tickers × 3 sectors                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CLASSIFICATION MODEL (Random Forest)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Target Definition:                                                          │
│  ├── Class 0 (DOWN):  return < -3%                                          │
│  ├── Class 1 (HOLD):  -3% ≤ return ≤ +3%                                    │
│  └── Class 2 (UP):    return > +3%                                          │
│                                                                              │
│  Features: OHLCV (5) + Technical (13) + Fundamental (11) = 29 features      │
│  Training: Chronological split (80% train / 20% test)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT & EVALUATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Metrics:                          Outputs:                                  │
│  ├── Accuracy                      ├── Trained models (.pkl)                │
│  ├── Precision, Recall, F1         ├── Predictions                          │
│  ├── Confusion Matrix              ├── Feature Importance                   │
│  └── AUC-ROC (per class)           └── Visualizations                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 5.2 Datový Tok

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Krok 1     │────▶│   Krok 2     │────▶│   Krok 3     │────▶│   Krok 4     │
│  Download    │     │  Download    │     │    Train     │     │   Complete   │
│   OHLCV      │     │ Fundamentals │     │ RF Regressor │     │  Historical  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
   data/ohlcv/         data/               models/               data/
   └── all_            fundamentals/       └── fundamental_     complete/
       sectors_        └── all_               predictor.pkl     └── all_sectors_
       ohlcv.csv          sectors_         └── feature_            complete_
                          fundamentals.       scaler.pkl           10y.csv
                          csv
                                                                     │
                                                                     ▼
                                              ┌──────────────┐  ┌──────────────┐
                                              │   Krok 5     │──│   Krok 6-7   │
                                              │    Train     │  │   Tuning +   │
                                              │ RF Classifier│  │  Evaluation  │
                                              └──────────────┘  └──────────────┘
```

---

# 6. IMPLEMENTACE PIPELINE

## 6.1 Krok 1: Stažení OHLCV Dat

### Skript: `download_30_tickers.py`

```python
#!/usr/bin/env python3
"""Stažení 30 tickerů (10 per sektor) pro pipeline."""

import yfinance as yf
import pandas as pd

# Konfigurace
TICKERS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", 
                   "AVGO", "ORCL", "CSCO", "ADBE", "CRM"],
    "Consumer": ["AMZN", "TSLA", "HD", "MCD", "NKE",
                 "SBUX", "TGT", "LOW", "PG", "KO"],
    "Industrials": ["CAT", "HON", "UPS", "BA", "GE",
                    "RTX", "DE", "LMT", "MMM", "UNP"]
}

def calculate_rsi(series, period=14):
    """RSI indikátor"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    """MACD indikátor"""
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def download_ticker(ticker, start, end):
    """Stáhne a zpracuje data pro jeden ticker"""
    hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
    
    # Agregace na měsíční data
    monthly = hist.resample('ME').agg({
        'Open': 'first', 'High': 'max', 
        'Low': 'min', 'Close': 'last', 'Volume': 'mean'
    })
    
    # Technické indikátory
    monthly['volatility'] = (monthly['High'] - monthly['Low']) / monthly['Close']
    monthly['returns'] = monthly['Close'].pct_change()
    monthly['rsi_14'] = calculate_rsi(monthly['Close'])
    
    macd, signal, hist = calculate_macd(monthly['Close'])
    monthly['macd'], monthly['macd_signal'], monthly['macd_hist'] = macd, signal, hist
    
    for n in [3, 6, 12]:
        monthly[f'sma_{n}'] = monthly['Close'].rolling(n).mean()
        monthly[f'ema_{n}'] = monthly['Close'].ewm(span=n).mean()
    
    monthly['volume_change'] = monthly['Volume'].pct_change()
    
    return monthly
```

**Výstup:** 
- Soubor: `data/ohlcv/all_sectors_ohlcv_10y.csv`
- 3,870 řádků, 30 tickerů, 10.7 let historie

## 6.2 Krok 2: Stažení Fundamentálních Dat

### Skript: `download_fundamentals.py`

```python
def get_fundamentals(ticker):
    """Stáhne fundamentální metriky pro ticker"""
    info = yf.Ticker(ticker).info
    
    return {
        # Valuační
        'trailingPE': info.get('trailingPE'),
        'forwardPE': info.get('forwardPE'),
        'priceToBook': info.get('priceToBook'),
        
        # Profitabilita
        'returnOnEquity': info.get('returnOnEquity'),
        'returnOnAssets': info.get('returnOnAssets'),
        'profitMargins': info.get('profitMargins'),
        'operatingMargins': info.get('operatingMargins'),
        'grossMargins': info.get('grossMargins'),
        
        # Finanční zdraví
        'debtToEquity': info.get('debtToEquity'),
        'currentRatio': info.get('currentRatio'),
        'beta': info.get('beta')
    }
```

**Stažené metriky (25 sloupců):**

| Kategorie | Metriky |
|-----------|---------|
| Valuační | trailingPE, forwardPE, priceToBook, enterpriseToEbitda |
| Profitabilita | returnOnEquity, returnOnAssets, profitMargins, operatingMargins |
| Zadluženost | debtToEquity, currentRatio, quickRatio |
| Riziko | beta |

## 6.3 Krok 3: Trénink RF Regressoru

### Skript: `train_rf_regressor.py`

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Features pro predikci fundamentů
OHLCV_FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'volatility', 'returns', 'rsi_14', 
    'macd', 'macd_signal', 'macd_hist',
    'sma_3', 'sma_6', 'sma_12',
    'ema_3', 'ema_6', 'ema_12',
    'volume_change'
]

# Targets
FUND_TARGETS = [
    'trailingPE', 'forwardPE', 'priceToBook', 
    'returnOnEquity', 'returnOnAssets',
    'profitMargins', 'operatingMargins', 'grossMargins',
    'debtToEquity', 'currentRatio', 'beta'
]

# Model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Trénink
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
```

**Výsledky imputace:**

| Target | MAE | R² Score |
|--------|-----|----------|
| trailingPE | 4.419 | **0.957** |
| forwardPE | 2.595 | **0.964** |
| returnOnAssets | 0.015 | **0.970** |
| returnOnEquity | 0.045 | 0.935 |
| priceToBook | 1.854 | 0.891 |
| profitMargins | 0.031 | 0.886 |
| debtToEquity | 38.513 | 0.765 |

**Průměrné R²: 0.91** - Excelentní kvalita imputace

## 6.4 Krok 4: Kompletace Historických Dat

```python
# Rozdělení dat
cutoff_date = df['date'].max() - pd.DateOffset(months=24)
df_recent = df[df['date'] >= cutoff_date]      # Reálné fundamenty
df_historical = df[df['date'] < cutoff_date]   # K imputaci

# Imputace
X_hist = df_historical[OHLCV_FEATURES]
X_hist_scaled = scaler.transform(X_hist)
predicted_funds = model.predict(X_hist_scaled)

# Označení zdroje dat
df_recent['data_source'] = 'real'
df_historical['data_source'] = 'predicted'

# Spojení
df_complete = pd.concat([df_historical, df_recent])
```

**Statistiky:**

| Část | Počet řádků |
|------|-------------|
| Recent (reálné) | 650 |
| Historical (predikované) | 2,730 |
| **Celkem** | **3,380** |

## 6.5 Krok 5: Trénink RF Classifieru

### Skript: `train_rf_classifier.py`

```python
from sklearn.ensemble import RandomForestClassifier

# Definice target variable
THRESHOLD = 0.03  # ±3%

def create_target(df):
    """Vytvoří klasifikační target"""
    df['future_close'] = df.groupby('ticker')['close'].shift(-1)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    
    def classify(ret):
        if ret < -THRESHOLD:
            return 0  # DOWN
        elif ret > THRESHOLD:
            return 2  # UP
        else:
            return 1  # HOLD
    
    df['target'] = df['future_return'].apply(classify)
    return df

# Features
FEATURES = OHLCV_FEATURES + FUND_TARGETS  # 18 + 11 = 29 features

# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Chronologický split
df_sorted = df.sort_values('date')
split_idx = int(len(df_sorted) * 0.8)
train, test = df_sorted[:split_idx], df_sorted[split_idx:]
```

**Distribuce tříd:**

| Třída | Počet | Procento |
|-------|-------|----------|
| DOWN | 871 | 26.0% |
| HOLD | 1,111 | 33.2% |
| UP | 1,368 | 40.8% |

## 6.6 Krok 6: Hyperparameter Tuning

### Skript: `hyperparameter_tuning.py`

```python
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Grid search prostor
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'class_weight': ['balanced']
}

# TimeSeriesSplit pro časovou konzistenci
tscv = TimeSeriesSplit(n_splits=5)

# Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=tscv,
    scoring='f1_weighted',
    n_jobs=-1
)

grid_search.fit(X_scaled, y)
```

**Nejlepší parametry:**

```json
{
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced"
}
```

## 6.7 Krok 7: Finální Evaluace

### Skript: `final_evaluation.py`

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['DOWN', 'HOLD', 'UP'],
                yticklabels=['DOWN', 'HOLD', 'UP'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)

def plot_roc_curves(y_true, y_proba, filename):
    for i, class_name in enumerate(['DOWN', 'HOLD', 'UP']):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC={roc_auc:.2f})')
    plt.savefig(filename)

def plot_feature_importance(model, features, filename):
    importance = model.feature_importances_
    plt.barh(features, importance)
    plt.savefig(filename)
```

---

# 7. EXPERIMENT: 30 TICKERŮ

## 7.1 Konfigurace Experimentu

| Parametr | Hodnota |
|----------|---------|
| **Počet tickerů** | 30 |
| **Počet sektorů** | 3 |
| **Tickerů per sektor** | 10 |
| **Období** | 2014-01-01 až 2024-12-31 |
| **Frekvence** | Měsíční |
| **Target threshold** | ±3% |

## 7.2 Vybrané Tickery

| Sektor | Tickery |
|--------|---------|
| **Technology** | AAPL, MSFT, NVDA, GOOGL, META, AVGO, ORCL, CSCO, ADBE, CRM |
| **Consumer** | AMZN, TSLA, HD, MCD, NKE, SBUX, TGT, LOW, PG, KO |
| **Industrials** | CAT, HON, UPS, BA, GE, RTX, DE, LMT, MMM, UNP |

## 7.3 Statistiky Datasetu

| Metrika | Hodnota |
|---------|---------|
| Celkem řádků | 3,870 |
| Po čištění | 3,380 |
| Časové období | 10.7 let |
| OHLCV features | 5 |
| Technické indikátory | 13 |
| Fundamentální metriky | 11 |
| **Celkem features** | **29** |

---

# 8. VÝSLEDKY A ANALÝZA

## 8.1 RF Regressor (Imputace)

### Výsledky per-target

| Target | MAE | R² Score | Kvalita |
|--------|-----|----------|---------|
| trailingPE | 4.419 | 0.957 | ⭐⭐⭐⭐⭐ |
| forwardPE | 2.595 | 0.964 | ⭐⭐⭐⭐⭐ |
| returnOnAssets | 0.015 | 0.970 | ⭐⭐⭐⭐⭐ |
| returnOnEquity | 0.045 | 0.935 | ⭐⭐⭐⭐ |
| priceToBook | 1.854 | 0.891 | ⭐⭐⭐⭐ |
| profitMargins | 0.031 | 0.886 | ⭐⭐⭐⭐ |
| debtToEquity | 38.513 | 0.765 | ⭐⭐⭐ |

**Průměrné R²: 0.91**

### Feature Importance (Regressor)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **volume** | 0.4995 |
| 2 | sma_12 | 0.0734 |
| 3 | ema_12 | 0.0730 |
| 4 | sma_6 | 0.0586 |
| 5 | ema_6 | 0.0583 |

**Poznatek:** Volume je dominantní prediktor fundamentálních metrik (korelace s tržní kapitalizací a likviditou).

## 8.2 RF Classifier (Klasifikace)

### Celkové Výsledky

| Metrika | Hodnota |
|---------|---------|
| **Accuracy** | 32.09% |
| **Precision** | 32.87% |
| **Recall** | 32.09% |
| **F1-Score** | 31.00% |
| Random baseline | 33.33% |
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

### Per-Sector Analýza

| Sektor | Accuracy | F1-Score | Samples |
|--------|----------|----------|---------|
| **Industrials** | 35.9% | 34.6% | 231 |
| Consumer | 30.4% | 29.8% | 181 |
| Technology | 29.8% | 27.6% | 258 |

**Poznatek:** Industrials sektor je nejlépe predikovatelný. Technology má nejvyšší volatilitu a je nejtěžší k predikci.

### Feature Importance (Classifier)

| Rank | Feature | Importance | Typ |
|------|---------|------------|-----|
| 1 | returns | 0.0577 | Technický |
| 2 | volatility | 0.0560 | Technický |
| 3 | macd_hist | 0.0489 | Technický |
| 4 | macd_signal | 0.0481 | Technický |
| 5 | volume_change | 0.0449 | Technický |
| 6 | rsi_14 | 0.0430 | Technický |
| 7 | macd | 0.0392 | Technický |
| 8 | returnOnEquity | 0.0380 | Fundamentální |
| 9 | returnOnAssets | 0.0373 | Fundamentální |
| 10 | currentRatio | 0.0359 | Fundamentální |

**Poznatky:**
- Technické indikátory dominují (7 z top 10)
- Momentum features (returns, MACD) jsou nejdůležitější
- Fundamenty (ROE, ROA) jsou stále významné (top 10)

## 8.3 Interpretace Výsledků

### Accuracy vs. Random Baseline

- **Model accuracy:** 32.1%
- **Random baseline (3 třídy):** 33.3%
- **Rozdíl:** -1.2%

**Interpretace:** Model dosahuje accuracy blízké náhodnému klasifikátoru. Toto je typické pro finanční predikce a odráží vysokou efektivitu trhů.

### Analýza Confusion Matrix

```
              DOWN  HOLD    UP
   DOWN       98    39      56    (51% recall)
   HOLD       72    44      100   (20% recall)
   UP         84    85      92    (35% recall)
```

**Poznatky:**
1. Model má tendenci predikovat DOWN častěji
2. HOLD je nejhůře rozpoznávaná třída (pouze 20% recall)
3. Nejvíce záměn mezi UP a HOLD

### AUC Skóre

| Třída | AUC |
|-------|-----|
| DOWN | ~0.55 |
| HOLD | ~0.52 |
| UP | ~0.54 |

Hodnoty AUC blízko 0.5 indikují slabou separabilitu tříd.

---

# 9. VIZUALIZACE

## 9.1 Confusion Matrix

![Confusion Matrix](data/30_tickers/figures/confusion_matrix.png)

**Popis:** Matice záměn ukazuje distribuci skutečných vs. predikovaných tříd. Diagonála reprezentuje správné predikce.

## 9.2 ROC Křivky

![ROC Curves](data/30_tickers/figures/roc_curves.png)

**Popis:** ROC křivky pro každou třídu. Čím blíže křivka k levému hornímu rohu, tím lepší separabilita.

## 9.3 Feature Importance

![Feature Importance](data/30_tickers/figures/feature_importance.png)

**Popis:** Relativní důležitost jednotlivých features pro klasifikační model.

## 9.4 Porovnání Sektorů

![Sector Comparison](data/30_tickers/figures/sector_comparison.png)

**Popis:** Porovnání accuracy, precision, recall a F1 mezi sektory.

---

# 10. OMEZENÍ A BUDOUCÍ PRÁCE

## 10.1 Datová Omezení

### 10.1.1 Survivorship Bias

**Problém:** Dataset obsahuje pouze akcie aktuálně v S&P 500. Firmy, které zbankrotovaly nebo byly vyřazeny, chybí.

**Důsledek:** Potenciální nadhodnocení výkonnosti modelu.

**Mitigace:**
- Použití historických konstituentů indexu (vyžaduje placená data)
- Explicitní disclaimer v interpretaci

### 10.1.2 Look-Ahead Bias

**Problém:** Fundamentální metriky jsou publikovány se zpožděním (quarterly reports 1-2 měsíce po konci kvartálu).

**Mitigace:**
- Použití lagovaných dat
- Point-in-time databáze

### 10.1.3 Kvalita Imputovaných Dat

**Problém:** ~80% fundamentálních dat je predikováno modelem, nikoli skutečných.

**Důsledek:** Chyby imputace se propagují do klasifikátoru.

**Mitigace:**
- Confidence intervals pro imputované hodnoty
- Sensitivity analýza
- Sloupec `data_source` pro transparentnost

## 10.2 Modelová Omezení

### 10.2.1 Stacionarita

**Předpoklad:** Vztahy mezi features a targetem jsou stabilní v čase.

**Realita:** Tržní dynamika se mění (COVID-19, úrokové sazby, geopolitika).

**Mitigace:**
- Rolling window training
- Periodic retraining
- Regime detection

### 10.2.2 Transakční Náklady

**Problém:** Model nezahrnuje bid-ask spread, poplatky, market impact, daně.

**Důsledek:** Skutečná výkonnost bude nižší než backtest.

## 10.3 Budoucí Rozšíření

| Rozšíření | Popis | Priorita |
|-----------|-------|----------|
| **Více tickerů** | 100-150 tickerů, více sektorů | ⭐⭐⭐⭐⭐ |
| **Alternative data** | Sentiment z news/social media | ⭐⭐⭐⭐ |
| **Deep Learning** | LSTM/Transformer pro časové řady | ⭐⭐⭐ |
| **Ensemble** | Kombinace více modelů | ⭐⭐⭐ |
| **Real-time** | Automatizovaný trading systém | ⭐⭐ |

---

# 11. ZÁVĚR

## 11.1 Shrnutí Dosažených Výsledků

### Co funguje dobře ✅

1. **RF Regressor pro imputaci** - R² 0.76-0.97 je excelentní
2. **Hybridní přístup** - Umožňuje využít fundamenty i pro historii
3. **Technické indikátory** - Returns a volatility jsou nejdůležitější
4. **Industrials sektor** - Model zde funguje nejlépe (35.9%)

### Limitace ⚠️

1. **Accuracy ~32%** - Blízko random baseline
2. **HOLD třída** - Nejhůře rozpoznávaná (20% recall)
3. **Finanční trhy** - Inherentně těžko predikovatelné (EMH)

## 11.2 Vědecký Přínos

1. **Metodologický:** Demonstrace hybridního přístupu k řešení chybějících dat
2. **Praktický:** Funkční end-to-end ML pipeline pro finanční predikce
3. **Analytický:** Feature importance analýza technických vs. fundamentálních faktorů

## 11.3 Doporučení

Pro zlepšení výsledků doporučuji:

1. **Více dat** - 100+ tickerů, delší historie
2. **Feature engineering** - Sentiment, makroekonomické indikátory
3. **Jiné modely** - XGBoost, LSTM
4. **Binární klasifikace** - UP vs NOT UP (snazší problém)
5. **Confidence thresholds** - Obchodovat pouze při vysoké jistotě

---

# 12. REFERENCE

## Akademické Zdroje

1. Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383-417.

2. Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. *The Journal of Finance*, 47(2), 427-465.

3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

4. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223-2273.

5. Graham, B., & Dodd, D. (1934). *Security Analysis*. McGraw-Hill.

## Technické Reference

6. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

7. McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*.

## Online Zdroje

8. Yahoo Finance API: https://pypi.org/project/yfinance/
9. Scikit-learn Documentation: https://scikit-learn.org/stable/

---

# 13. PŘÍLOHY

## Příloha A: Kompletní Seznam Features

### A.1 OHLCV Features (5)
```
open, high, low, close, volume
```

### A.2 Technical Indicators (13)
```
volatility, returns,
rsi_14, macd, macd_signal, macd_hist,
sma_3, sma_6, sma_12,
ema_3, ema_6, ema_12,
volume_change
```

### A.3 Fundamental Metrics (11)
```
trailingPE, forwardPE, priceToBook,
returnOnEquity, returnOnAssets,
profitMargins, operatingMargins, grossMargins,
debtToEquity, currentRatio,
beta
```

## Příloha B: Struktura Projektu

```
CleanSolution/
│
├── 📄 DIPLOMOVA_PRACE_DOKUMENTACE.md    # Tento dokument
├── 📄 README.md                          # Přehled projektu
├── 📄 requirements.txt                   # Python závislosti
│
├── 📂 data/
│   └── 📂 30_tickers/
│       ├── 📂 ohlcv/                     # Surová OHLCV data
│       ├── 📂 fundamentals/              # Fundamentální data
│       ├── 📂 complete/                  # Kompletní dataset
│       └── 📂 figures/                   # Vizualizace
│
├── 📂 models/
│   └── 📂 30_tickers/
│       ├── 📂 classifiers/               # RF Classifier modely
│       ├── 📂 regressors/                # RF Regressor modely
│       ├── 📂 scalers/                   # StandardScaler objekty
│       └── 📂 metadata/                  # JSON/CSV výsledky
│
├── 📂 docs/
│   ├── METHODOLOGY.md
│   ├── MATHEMATICAL_FOUNDATIONS.md
│   ├── ALGORITHM_SELECTION.md
│   ├── WORKFLOW.md
│   └── SUMMARY.md
│
└── 📄 Skripty:
    ├── download_30_tickers.py
    ├── download_fundamentals.py
    ├── train_rf_regressor.py
    ├── train_rf_classifier.py
    ├── hyperparameter_tuning.py
    └── final_evaluation.py
```

## Příloha C: Instalace a Spuštění

### Požadavky

```txt
# requirements.txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
yfinance>=0.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
```

### Spuštění Pipeline

```bash
# Aktivace prostředí
cd CleanSolution
python -m venv venv
.\venv\Scripts\activate  # Windows

# Instalace závislostí
pip install -r requirements.txt

# Spuštění celé pipeline (v pořadí)
python download_30_tickers.py
python download_fundamentals.py
python train_rf_regressor.py
python train_rf_classifier.py
python hyperparameter_tuning.py
python final_evaluation.py
```

## Příloha D: Výstupní Soubory

| Soubor | Popis |
|--------|-------|
| `data/30_tickers/ohlcv/all_sectors_ohlcv_10y.csv` | Surová OHLCV data |
| `data/30_tickers/fundamentals/all_sectors_fundamentals.csv` | Fundamentální metriky |
| `data/30_tickers/complete/all_sectors_complete_10y.csv` | Kompletní dataset |
| `models/30_tickers/regressors/fundamental_predictor.pkl` | RF Regressor model |
| `models/30_tickers/classifiers/rf_classifier_tuned.pkl` | RF Classifier model |
| `models/30_tickers/metadata/final_evaluation_results.json` | Výsledky evaluace |
| `data/30_tickers/figures/confusion_matrix.png` | Confusion matrix |
| `data/30_tickers/figures/roc_curves.png` | ROC křivky |
| `data/30_tickers/figures/feature_importance.png` | Feature importance |

---

# KONEC DOKUMENTACE

**Celkový rozsah:** ~30 stran  
**Poslední aktualizace:** Prosinec 2025  
**Autor:** Bc. Jan Dub

---

*Tento dokument byl vytvořen jako kompletní dokumentace diplomové práce a obsahuje veškeré teoretické, metodologické a implementační aspekty projektu klasifikace cenových pohybů akcií pomocí strojového učení.*
