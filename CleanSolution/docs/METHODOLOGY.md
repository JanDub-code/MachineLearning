# Metodologie Predikce Akciových Trhů pomocí Strojového Učení

## Teoreticko-Metodologický Rámec Diplomové Práce

**Autor:** Bc. Jan Dub  
**Studijní program:** Informatika (Ing.)  
**Datum:** Prosinec 2025  
**Verze dokumentu:** 2.0

---

## Obsah

1. [Úvod a Motivace](#1-úvod-a-motivace)
2. [Teoretický Základ](#2-teoretický-základ)
3. [Problém Neúplnosti Fundamentálních Dat](#3-problém-neúplnosti-fundamentálních-dat)
4. [Volba Algoritmů a Jejich Zdůvodnění](#4-volba-algoritmů-a-jejich-zdůvodnění)
5. [Architektura Řešení](#5-architektura-řešení)
6. [Fáze 1: Sběr a Předzpracování Dat](#6-fáze-1-sběr-a-předzpracování-dat)
7. [Fáze 2: Imputace Chybějících Dat pomocí Random Forest](#7-fáze-2-imputace-chybějících-dat-pomocí-random-forest)
8. [Fáze 3: Klasifikace Cenových Pohybů](#8-fáze-3-klasifikace-cenových-pohybů)
9. [Evaluační Metriky](#9-evaluační-metriky)
10. [Omezení a Budoucí Práce](#10-omezení-a-budoucí-práce)
11. [Reference](#11-reference)

---

## 1. Úvod a Motivace

### 1.1 Kontext Problému

Predikce pohybů na akciových trzích představuje jeden z nejnáročnějších problémů v oblasti kvantitativních financí a strojového učení. Efektivní tržní hypotéza (Fama, 1970) postuluje, že ceny akcií plně reflektují všechny dostupné informace, což implikuje nemožnost systematicky dosahovat nadprůměrných výnosů. Nicméně empirické studie (Lo & MacKinlay, 1999; Jegadeesh & Titman, 1993) dokumentují existenci anomálií a predikovatelných vzorců.

### 1.2 Cíle Práce

Tato práce si klade následující cíle:

1. **Primární cíl:** Vyvinout klasifikační model pro predikci směru cenových pohybů akcií na základě kombinace technických a fundamentálních faktorů.

2. **Sekundární cíl:** Navrhnout a implementovat metodu pro imputaci chybějících fundamentálních dat v historických časových řadách pomocí strojového učení.

3. **Metodologický cíl:** Poskytnout rigorózní srovnání různých přístupů k řešení problému s důrazem na interpretabilitu a praktickou aplikovatelnost.

### 1.3 Vědecký Přínos

Práce přináší následující přínosy:

- **Hybridní přístup:** Kombinace ensemble metod (Random Forest) pro imputaci dat a klasifikačních modelů pro predikci
- **Multifaktorový model:** Integrace 14 fundamentálních a 5 technických indikátorů
- **Řešení problému neúplnosti dat:** Inovativní využití ML pro zpětnou rekonstrukci fundamentálních metrik
- **Sektorová analýza:** Diferenciace modelů pro Technology, Consumer a Industrials sektory

---

## 2. Teoretický Základ

### 2.1 Fundamentální vs. Technická Analýza

#### 2.1.1 Fundamentální Analýza

Fundamentální analýza vychází z předpokladu, že tržní cena akcie se dlouhodobě konverguje k její vnitřní hodnotě (intrinsic value). Klíčové metriky zahrnují:

**Valuační poměry:**
- **P/E (Price-to-Earnings):** Poměr ceny k zisku na akcii. Vysoké P/E může indikovat nadhodnocení nebo očekávání růstu.
- **P/B (Price-to-Book):** Poměr tržní ceny k účetní hodnotě vlastního kapitálu.
- **P/S (Price-to-Sales):** Poměr ceny k tržbám, užitečný pro ztrátové společnosti.
- **EV/EBITDA:** Poměr hodnoty podniku k provoznímu zisku před odpisy.

**Ukazatele profitability:**
- **ROE (Return on Equity):** Návratnost vlastního kapitálu, klíčový indikátor efektivity.
- **ROA (Return on Assets):** Návratnost celkových aktiv.
- **Profit Margin:** Čistá zisková marže.
- **Operating Margin:** Provozní zisková marže.
- **Gross Margin:** Hrubá zisková marže.

**Ukazatele finančního zdraví:**
- **Debt-to-Equity:** Poměr dluhu k vlastnímu kapitálu (finanční páka).
- **Current Ratio:** Běžná likvidita (oběžná aktiva / krátkodobé závazky).
- **Quick Ratio:** Pohotová likvidita (bez zásob).

**Růstové metriky:**
- **Revenue Growth YoY:** Meziroční růst tržeb.
- **Earnings Growth YoY:** Meziroční růst zisku.

#### 2.1.2 Technická Analýza

Technická analýza předpokládá, že historické cenové vzorce se opakují a obsahují prediktivní informace:

**Momentum indikátory:**
- **RSI (Relative Strength Index):** Oscilující indikátor v rozsahu 0-100, hodnoty >70 indikují překoupenost, <30 přeprodanost.
- **MACD (Moving Average Convergence Divergence):** Rozdíl mezi krátkodobým a dlouhodobým exponenciálním klouzavým průměrem.

**Trendové indikátory:**
- **SMA (Simple Moving Average):** Jednoduchý klouzavý průměr za periody 3, 6, 12 měsíců.
- **EMA (Exponential Moving Average):** Exponenciální klouzavý průměr s vyšší vahou recentních dat.

**Volatilita:**
- **Historická volatilita:** Směrodatná odchylka logaritmických výnosů.

### 2.2 Hypotéza Efektivních Trhů a Její Kritika

Eugene Fama (1970) definoval tři formy tržní efektivity:

| Forma | Definice | Implikace |
|-------|----------|-----------|
| **Slabá** | Ceny reflektují všechny historické informace | Technická analýza je neúčinná |
| **Střední** | Ceny reflektují všechny veřejné informace | Fundamentální analýza je neúčinná |
| **Silná** | Ceny reflektují všechny informace (včetně insider) | Žádná analýza není účinná |

**Kritika a anomálie:**
- **Momentum efekt** (Jegadeesh & Titman, 1993): Akcie s vysokými výnosy v posledních 3-12 měsících pokračují v nadprůměrných výnosech.
- **Value efekt** (Fama & French, 1992): Akcie s nízkým P/B dosahují vyšších výnosů.
- **Size efekt:** Malé firmy mají vyšší rizikově očištěné výnosy.

Tyto anomálie poskytují teoretické zdůvodnění pro využití prediktivních modelů.

### 2.3 Strojové Učení v Kvantitativních Financích

#### 2.3.1 Supervised Learning pro Predikci

Supervised learning vyžaduje párovaná data (X, y), kde:
- **X** = feature matrix (technické + fundamentální indikátory)
- **y** = target variable (budoucí cenový pohyb)

**Regrese vs. Klasifikace:**

| Aspekt | Regrese | Klasifikace |
|--------|---------|-------------|
| **Output** | Kontinuální (přesná cena) | Diskrétní (třída pohybu) |
| **Příklad** | "Cena bude $152.30" | "Cena vzroste o >5%" |
| **Interpretace** | Složitější | Přímočará (BUY/SELL) |
| **Evaluace** | MAE, RMSE, R² | Accuracy, F1, AUC-ROC |
| **Praktické využití** | Portfolio optimization | Trading signály |

**Zdůvodnění volby klasifikace v této práci:**

1. **Redukce šumu:** Přesná predikce ceny je prakticky nemožná kvůli stochastické povaze trhů. Klasifikace redukuje problém na predikovatelný směr.

2. **Praktická aplikovatelnost:** Investoři činí binární rozhodnutí (koupit/nekoupit), nikoli rozhodnutí o přesné ceně.

3. **Robustnost vůči outlierům:** Extrémní pohyby (např. +200% NVIDIA v 2023) nezkreslují model.

4. **Interpretabilita:** Confusion matrix poskytuje jasné metriky úspěšnosti.

---

## 3. Problém Neúplnosti Fundamentálních Dat

### 3.1 Popis Problému

Fundamentální data z API poskytovatelů (Yahoo Finance, Bloomberg, Refinitiv) jsou typicky dostupná pouze za omezené historické období:

| Zdroj dat | Typická dostupnost | Poznámka |
|-----------|-------------------|----------|
| Yahoo Finance (yfinance) | 1.5-2 roky | Quarterly/TTM |
| Bloomberg Terminal | 5-10 let | Placený |
| SEC EDGAR | 10+ let | Pouze US, raw formát |
| Refinitiv | 20+ let | Placený |

**Důsledky pro tuto práci:**
- OHLCV data: dostupná za 10 let (2015-2025)
- Fundamentální data: dostupná za ~1.5 roku (2024-2025)
- **GAP: 8.5 let chybějících fundamentálních dat**

### 3.2 Přístupy k Řešení Neúplnosti Dat

#### 3.2.1 Tradiční Statistické Metody

| Metoda | Popis | Nevýhody |
|--------|-------|----------|
| **Mean imputation** | Nahrazení průměrem | Snižuje varianci, nezachycuje vztahy |
| **Forward/Backward fill** | Propagace poslední známé hodnoty | Nerealistické pro dynamické metriky |
| **Interpolace** | Lineární/spline interpolace | Předpokládá kontinuitu, která neexistuje |
| **Multiple imputation** | Generování více imputovaných datasetů | Výpočetně náročné, komplexní inference |

#### 3.2.2 Machine Learning Přístup (Zvolený)

**Koncept:** Natrénovat model na období, kde jsou data kompletní, a použít ho pro imputaci chybějících hodnot.

**Předpoklad:** Existuje systematický vztah mezi OHLCV daty (které máme za celé období) a fundamentálními metrikami.

**Matematická formulace:**

$$
\hat{F}_t = f(OHLCV_t, TechnicalIndicators_t) + \epsilon
$$

Kde:
- $\hat{F}_t$ = predikované fundamentální metriky v čase t
- $OHLCV_t$ = cenová data (Open, High, Low, Close, Volume)
- $TechnicalIndicators_t$ = odvozené technické indikátory
- $f(\cdot)$ = naučená funkce (Random Forest)
- $\epsilon$ = reziduální chyba

### 3.3 Teoretické Zdůvodnění Vztahu OHLCV → Fundamenty

Existuje implicitní vztah mezi cenovými daty a fundamentálními metrikami:

1. **P/E a cena:** $P/E = \frac{Price}{EPS}$, kde EPS se mění pomalu (quarterly), ale Price je v OHLCV.

2. **Volatilita a riziko:** Vyšší volatilita implikuje vyšší požadovanou návratnost → ovlivňuje valuační poměry.

3. **Volume a likvidita:** Vysoký objem koreluje s institucionálním zájmem → často vyšší valuace.

4. **Momentum a růst:** Rostoucí cena (zachycená v MACD, RSI) často předchází zlepšení fundamentů.

---

## 4. Volba Algoritmů a Jejich Zdůvodnění

### 4.1 Random Forest pro Imputaci Fundamentálních Dat

#### 4.1.1 Popis Algoritmu

Random Forest (Breiman, 2001) je ensemble metoda kombinující predikce mnoha rozhodovacích stromů:

$$
\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
$$

Kde:
- $B$ = počet stromů (n_estimators)
- $T_b(x)$ = predikce b-tého stromu pro vstup x

**Klíčové mechanismy:**
- **Bootstrap aggregating (Bagging):** Každý strom je trénován na náhodném vzorku dat s opakováním.
- **Feature randomness:** V každém uzlu se vybírá z náhodné podmnožiny features.

#### 4.1.2 Zdůvodnění Volby pro Imputaci

| Kritérium | Random Forest | Alternativy |
|-----------|---------------|-------------|
| **Nelineární vztahy** | ✅ Zachycuje komplexní interakce | Linear regression: pouze lineární |
| **Robustnost vůči outlierům** | ✅ Rozhodovací stromy jsou inherentně robustní | Neural Networks: citlivé na outliers |
| **Multi-output regrese** | ✅ Přirozená podpora (14 výstupů) | Většina modelů vyžaduje wrapper |
| **Feature importance** | ✅ Integrovaná interpretabilita | Deep learning: black-box |
| **Potřeba hyperparameter tuningu** | ✅ Funguje dobře out-of-box | Gradient Boosting: citlivý na tuning |
| **Výpočetní náročnost** | ✅ Paralelizovatelný, rychlý | Neural Networks: vyžaduje GPU |
| **Riziko overfittingu** | ✅ Nízké díky ensemble efektu | Single decision tree: vysoké |

#### 4.1.3 Hyperparametry a Jejich Volba

```python
RF_PARAMS = {
    'n_estimators': 100,      # Počet stromů
    'max_depth': 20,          # Maximální hloubka (prevence overfittingu)
    'min_samples_split': 5,   # Minimum vzorků pro split
    'min_samples_leaf': 2,    # Minimum vzorků v listu
    'random_state': 42,       # Reprodukovatelnost
    'n_jobs': -1              # Paralelizace
}
```

**Zdůvodnění:**
- **n_estimators=100:** Empiricky ověřeno, že marginální přínos dalších stromů klesá po ~100.
- **max_depth=20:** Omezuje komplexitu stromů, prevence overfittingu na menším datasetu.
- **min_samples_split=5, min_samples_leaf=2:** Regularizace stromů.

### 4.2 Klasifikace Cenových Pohybů

#### 4.2.1 Definice Klasifikačního Problému

Místo regrese (predikce přesné ceny) definujeme klasifikační problém:

**Ternární klasifikace:**

$$
y_t = \begin{cases}
0 \text{ (DOWN)} & \text{if } r_{t+1} < -\theta \\
1 \text{ (HOLD)} & \text{if } -\theta \leq r_{t+1} \leq +\theta \\
2 \text{ (UP)} & \text{if } r_{t+1} > +\theta
\end{cases}
$$

Kde:
- $r_{t+1} = \frac{P_{t+1} - P_t}{P_t}$ = měsíční výnos
- $\theta$ = threshold (typicky 3-5%)

**Volba thresholdu θ = 3%:**
- Transakční náklady: ~0.1-0.5%
- Bid-ask spread: ~0.1-1%
- Opportunity cost: ~1-2%
- **Celkem: ~3% je minimální pohyb pro profitabilní obchod**

#### 4.2.2 Volba Klasifikačního Algoritmu

Pro klasifikaci cenových pohybů zvažujeme:

| Algoritmus | Výhody | Nevýhody | Vhodnost |
|------------|--------|----------|----------|
| **Logistic Regression** | Interpretabilní, rychlý | Pouze lineární hranice | ⭐⭐⭐ |
| **Random Forest Classifier** | Nelineární, robustní | Méně interpretabilní | ⭐⭐⭐⭐⭐ |
| **Gradient Boosting (XGBoost)** | Vysoká přesnost | Náchylný k overfittingu | ⭐⭐⭐⭐ |
| **SVM** | Dobrý pro high-dim | Pomalý, citlivý na škálu | ⭐⭐ |
| **Neural Networks** | Flexibilní | Vyžaduje mnoho dat, GPU | ⭐⭐ |

**Zvolená strategie: Random Forest Classifier**

Důvody:
1. Konzistence s imputační fází (stejná rodina algoritmů)
2. Nativní podpora multi-class klasifikace
3. Feature importance pro interpretaci
4. Robustnost vůči nevyváženým třídám (class_weight='balanced')

#### 4.2.3 Proč ne Ridge Regression?

V původním návrhu byl použit Ridge Regression pro predikci log-ceny. Přechod na klasifikaci je motivován:

1. **Redukce komplexity problému:** Predikce směru je fundamentálně snazší než predikce přesné hodnoty.

2. **Praktická interpretace:** 
   - Regrese: "MAE = $12" → Co to znamená pro investora?
   - Klasifikace: "Precision = 72%" → Když model říká BUY, v 72% případů cena skutečně roste.

3. **Evaluační metriky:**
   - Regrese: R² může být vysoké, ale model stále nesprávně predikuje směr.
   - Klasifikace: Přímo měří úspěšnost predikce směru.

4. **Risk management:**
   - Klasifikace umožňuje nastavit různé thresholdy pro různé risk appetite.
   - Probability outputs poskytují confidence score.

---

## 5. Architektura Řešení

### 5.1 High-Level Pipeline

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
│  Input Features (18):              Output Targets (14):                      │
│  ├── OHLCV (5)                     ├── Valuation (4)                        │
│  ├── Technical (8)                 ├── Profitability (5)                    │
│  └── Derived (5)                   ├── Health (3)                           │
│                                    └── Growth (2)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMPLETE DATASET (2015-2025)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  2015-2024: OHLCV + Predicted Fundamentals (data_source='predicted')        │
│  2024-2025: OHLCV + Real Fundamentals (data_source='real')                  │
│                                                                              │
│  Total: ~18,000 records × 150 tickers × 3 sectors                           │
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
│  Features: Fundamentals (14) + Technical (5) = 19 features                  │
│  Training: Chronological split (80% train / 20% test)                       │
│  Sector-specific models: Technology, Consumer, Industrials                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT & EVALUATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Metrics:                          Outputs:                                  │
│  ├── Accuracy                      ├── Trained models (.pkl)                │
│  ├── Precision, Recall, F1         ├── Predictions vs Actual                │
│  ├── Confusion Matrix              ├── Feature Importance                   │
│  └── AUC-ROC (per class)           └── Visualizations                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Datový Tok

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Script 0   │────▶│   Script 1   │────▶│   Script 2   │────▶│   Script 3   │
│  Download    │     │  Download    │     │    Train     │     │   Complete   │
│   Prices     │     │ Fundamentals │     │   RF Model   │     │  Historical  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
   data_10y/           data/              models/               data/
   ├── all_           fundamentals/      ├── fundamental_     complete/
   │   sectors_       └── all_              predictor.pkl     └── all_sectors_
   │   full_10y.         sectors_        └── feature_            complete_
   │   csv               fundamentals.      scaler.pkl           10y.csv
   └── {Sector}_         csv
       full_10y.
       csv
                                                                     │
                                                                     ▼
                                                          ┌──────────────┐
                                                          │   Script 4   │
                                                          │    Train     │
                                                          │ Classifier   │
                                                          └──────────────┘
                                                                 │
                                                                 ▼
                                                            models/
                                                            ├── {Sector}_
                                                            │   price_
                                                            │   classifier.pkl
                                                            └── predictions/
```

---

## 6. Fáze 1: Sběr a Předzpracování Dat

### 6.1 Zdroje Dat

#### 6.1.1 OHLCV Data

**Zdroj:** Yahoo Finance API (via yfinance library)

**Parametry:**
- **Období:** 2015-01-01 až 2025-12-31 (10+ let)
- **Frekvence:** Měsíční (Monthly)
- **Universe:** 150 akcií z S&P 500
- **Sektory:** Technology (50), Consumer (50), Industrials (50)

**Struktura:**
```
date       | ticker | sector     | open    | high    | low     | close   | volume
2015-01-31 | AAPL   | Technology | 26.25   | 29.53   | 24.17   | 29.06   | 2.1B
2015-01-31 | MSFT   | Technology | 46.66   | 48.63   | 40.12   | 40.45   | 1.8B
...
```

#### 6.1.2 Odvozené Technické Indikátory

Následující indikátory jsou vypočteny z OHLCV dat:

| Indikátor | Formule | Interpretace |
|-----------|---------|--------------|
| **Returns** | $r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ | Měsíční výnos |
| **Volatility** | $\sigma = std(r_{t-n:t})$ | Historická volatilita (12M) |
| **RSI_14** | $100 - \frac{100}{1 + RS}$ | Relative Strength Index |
| **MACD** | $EMA_{12} - EMA_{26}$ | Momentum |
| **MACD_Signal** | $EMA_9(MACD)$ | Signal line |
| **MACD_Hist** | $MACD - Signal$ | Histogram |
| **SMA_3/6/12** | $\frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}$ | Simple Moving Average |
| **EMA_3/6/12** | $\alpha P_t + (1-\alpha)EMA_{t-1}$ | Exponential MA |
| **Volume_Change** | $\frac{V_t - V_{t-1}}{V_{t-1}}$ | Volume momentum |

### 6.2 Výběr Akcií (Stock Universe)

#### 6.2.1 Kritéria Výběru

1. **Členství v S&P 500:** Zajišťuje likviditu a datovou dostupnost
2. **Minimální historie:** 10 let obchodní historie
3. **Sektorová diverzifikace:** Rovnoměrné zastoupení sektorů
4. **Absence survivorship bias:** Diskutováno v omezeních (Sekce 10)

#### 6.2.2 Sektorová Struktura

| Sektor | Počet akcií | Příklady |
|--------|-------------|----------|
| **Technology** | 50 | AAPL, MSFT, GOOGL, NVDA, META |
| **Consumer** | 50 | AMZN, TSLA, NKE, SBUX, MCD |
| **Industrials** | 50 | CAT, BA, UPS, HON, GE |

### 6.3 Předzpracování Dat

#### 6.3.1 Čištění Dat

```python
# 1. Odstranění chybějících hodnot
df = df.dropna(subset=['close', 'volume'])

# 2. Odstranění nekonečných hodnot
df = df.replace([np.inf, -np.inf], np.nan)

# 3. Odstranění outlierů (optional)
# Používáme IQR metodu pro technické indikátory
Q1 = df['returns'].quantile(0.01)
Q99 = df['returns'].quantile(0.99)
df = df[(df['returns'] >= Q1) & (df['returns'] <= Q99)]
```

#### 6.3.2 Feature Engineering

**Lagované features:**
Pro zachycení časové dynamiky jsou vytvořeny lagované verze klíčových features:

```python
for lag in [1, 3, 6]:
    df[f'returns_lag_{lag}'] = df.groupby('ticker')['returns'].shift(lag)
    df[f'volume_change_lag_{lag}'] = df.groupby('ticker')['volume_change'].shift(lag)
```

**Interakční features:**
```python
df['momentum_volume'] = df['returns'] * df['volume_change']
df['rsi_macd_interaction'] = df['rsi_14'] * df['macd']
```

---

## 7. Fáze 2: Imputace Chybějících Dat pomocí Random Forest

### 7.1 Formulace Problému

**Vstup:** OHLCV data + technické indikátory (18 features)
**Výstup:** 14 fundamentálních metrik

**Trénovací data:** Období 2024-01-01 až 2025-10-31, kde máme kompletní data

**Inference:** Období 2015-01-01 až 2023-12-31, kde chybí fundamenty

### 7.2 Architektura Modelu

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Definice modelu
base_estimator = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model = MultiOutputRegressor(base_estimator)
```

**MultiOutputRegressor:** Wrapper, který trénuje separátní Random Forest pro každý target. Alternativou je nativní multi-output podpora RF, ale wrapper umožňuje flexibilnější konfiguraci.

### 7.3 Validace Imputačního Modelu

#### 7.3.1 Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
```

**Poznámka k shuffle:** Pro imputační model používáme shuffle=True, protože cílem je naučit obecný vztah OHLCV→Fundamenty, nikoli predikovat budoucnost.

#### 7.3.2 Metriky Evaluace

Pro každou z 14 fundamentálních metrik:

| Metrika | Formule | Interpretace |
|---------|---------|--------------|
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Průměrná absolutní chyba |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Penalizuje velké chyby |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Vysvětlená variance |
| **MAE%** | $\frac{MAE}{\bar{y}} \times 100$ | Relativní chyba |

**Cílové hodnoty:**
- MAE% < 15% pro všechny metriky
- R² > 0.70 pro klíčové metriky (P/E, ROE)

### 7.4 Feature Importance Analýza

Random Forest poskytuje inherentní měření důležitosti features:

**Mean Decrease in Impurity (MDI):**
$$
Importance(X_j) = \sum_{t \in T} \frac{n_t}{n} \Delta impurity(t, X_j)
$$

Kde:
- $T$ = množina všech uzlů, kde se splituje na feature $X_j$
- $n_t$ = počet vzorků v uzlu $t$
- $\Delta impurity$ = pokles impurity (variance pro regresi)

**Očekávané výsledky:**
- `close` → nejvyšší důležitost (přímo vstupuje do valuačních poměrů)
- `rsi_14`, `macd` → zachycují sentiment/momentum
- `volume` → likvidita, institucionální zájem

---

## 8. Fáze 3: Klasifikace Cenových Pohybů

### 8.1 Definice Target Variable

```python
def create_classification_target(df, threshold=0.03):
    """
    Vytvoří ternární klasifikační target.
    
    Args:
        df: DataFrame s cenami
        threshold: Hranice pro DOWN/UP (default 3%)
    
    Returns:
        Series s třídami 0 (DOWN), 1 (HOLD), 2 (UP)
    """
    # Výpočet měsíčního výnosu
    df['return_next_month'] = df.groupby('ticker')['close'].pct_change(-1) * -1
    
    # Klasifikace
    conditions = [
        df['return_next_month'] < -threshold,
        df['return_next_month'] > threshold
    ]
    choices = [0, 2]  # DOWN, UP
    
    df['target'] = np.select(conditions, choices, default=1)  # HOLD
    
    return df['target']
```

### 8.2 Distribuce Tříd

Očekávaná distribuce (na základě historických dat S&P 500):

| Třída | Popis | Očekávaný podíl | Interpretace |
|-------|-------|-----------------|--------------|
| 0 (DOWN) | return < -3% | ~25-30% | Signifikantní pokles |
| 1 (HOLD) | -3% ≤ return ≤ +3% | ~35-40% | Stagnace, transakční náklady by "sežraly" profit |
| 2 (UP) | return > +3% | ~30-35% | Signifikantní růst |

**Class Imbalance:** Mírná nevyváženost je adresována pomocí `class_weight='balanced'`.

### 8.3 Model Architecture

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

**Zdůvodnění hyperparametrů:**

| Parametr | Hodnota | Zdůvodnění |
|----------|---------|------------|
| `n_estimators` | 200 | Více stromů pro stabilnější pravděpodobnostní odhady |
| `max_depth` | 15 | Nižší než pro imputaci → prevence overfittingu na klasifikaci |
| `min_samples_split` | 10 | Regularizace |
| `min_samples_leaf` | 5 | Zajišťuje robustní listy |
| `class_weight` | 'balanced' | Kompenzace mírné nevyváženosti tříd |

### 8.4 Chronologický Train/Test Split

**KRITICKÉ:** Pro predikci budoucích pohybů je nutné použít chronologický split, ne náhodný!

```python
# Seřazení dat chronologicky
df = df.sort_values('date')

# Split: 80% train (starší data), 20% test (novější data)
split_date = df['date'].quantile(0.8)
train_df = df[df['date'] < split_date]
test_df = df[df['date'] >= split_date]
```

**Důvod:** Náhodný split by způsobil "data leakage" - model by se učil z budoucích dat.

### 8.5 Probability Calibration

Random Forest může produkovat nekalibrované pravděpodobnosti. Pro praktické použití je vhodná kalibrace:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    model, 
    method='sigmoid',  # Platt scaling
    cv=5
)
calibrated_model.fit(X_train, y_train)

# Kalibrované pravděpodobnosti
probas = calibrated_model.predict_proba(X_test)
```

---

## 9. Evaluační Metriky

### 9.1 Klasifikační Metriky

#### 9.1.1 Confusion Matrix

```
                    Predicted
                 DOWN  HOLD  UP
Actual   DOWN     TP₀   E₀₁   E₀₂
         HOLD     E₁₀   TP₁   E₁₂
         UP       E₂₀   E₂₁   TP₂
```

**Interpretace pro trading:**
- **False Positive (UP):** Model říká BUY, ale cena klesá → ZTRÁTA
- **False Negative (UP):** Model říká HOLD/SELL, ale cena roste → Propásnutá příležitost
- **True Positive (UP):** Model správně identifikuje růst → PROFIT

#### 9.1.2 Per-Class Metriky

| Metrika | Formule | Význam |
|---------|---------|--------|
| **Precision** | $\frac{TP}{TP + FP}$ | Když řeknu UP, jak často mám pravdu? |
| **Recall** | $\frac{TP}{TP + FN}$ | Kolik UP situací jsem zachytil? |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Harmonický průměr P a R |

**Praktická interpretace:**

- **Vysoká Precision (UP):** Konzervativní strategie - méně obchodů, ale spolehlivějších
- **Vysoký Recall (UP):** Agresivní strategie - chytáme většinu příležitostí, ale s vyšším rizikem

#### 9.1.3 Agregované Metriky

| Metrika | Použití |
|---------|---------|
| **Macro F1** | Průměr F1 přes všechny třídy (váha 1:1:1) |
| **Weighted F1** | Vážený průměr podle velikosti tříd |
| **Accuracy** | Celková správnost (může být zavádějící při imbalanci) |

### 9.2 Finanční Metriky

Kromě ML metrik hodnotíme i finanční výkonnost:

#### 9.2.1 Backtesting Strategy

```python
def backtest_strategy(predictions, actual_returns):
    """
    Jednoduchá long-only strategie:
    - Kup akcie, kde model predikuje UP (třída 2)
    - Drž portfolio jeden měsíc
    - Rebalance
    """
    # Signály: 1 = long, 0 = nedrží
    signals = (predictions == 2).astype(int)
    
    # Výnosy strategie
    strategy_returns = signals * actual_returns
    
    # Metriky
    cumulative_return = (1 + strategy_returns).prod() - 1
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(12)
    max_drawdown = calculate_max_drawdown(strategy_returns)
    
    return {
        'cumulative_return': cumulative_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }
```

#### 9.2.2 Benchmark Comparison

**Baseline strategie:**
1. **Buy & Hold:** Kup všechny akcie, drž celé období
2. **Random:** Náhodně vyber 50% akcií každý měsíc
3. **Momentum:** Kup top 20% akcií podle minulého měsíce

**Cíl:** Model by měl překonat alespoň Buy & Hold po započtení transakčních nákladů.

### 9.3 Statistická Významnost

#### 9.3.1 Bootstrap Confidence Intervals

```python
from sklearn.utils import resample

n_iterations = 1000
accuracies = []

for i in range(n_iterations):
    # Resample test set
    X_boot, y_boot = resample(X_test, y_test)
    y_pred = model.predict(X_boot)
    acc = accuracy_score(y_boot, y_pred)
    accuracies.append(acc)

# 95% confidence interval
ci_lower = np.percentile(accuracies, 2.5)
ci_upper = np.percentile(accuracies, 97.5)
```

#### 9.3.2 McNemar's Test

Pro srovnání dvou modelů:

$$
\chi^2 = \frac{(b - c)^2}{b + c}
$$

Kde b a c jsou počty případů, kde modely nesouhlasí.

---

## 10. Omezení a Budoucí Práce

### 10.1 Datová Omezení

#### 10.1.1 Survivorship Bias

**Problém:** Dataset obsahuje pouze akcie aktuálně v S&P 500. Firmy, které zbankrotovaly nebo byly vyřazeny z indexu, chybí.

**Důsledek:** Model může nadhodnocovat výkonnost, protože "nevidí" neúspěšné firmy.

**Mitigace:**
- Použití historických konstituentů indexu (vyžaduje placená data)
- Explicitní disclaimer v interpretaci výsledků

#### 10.1.2 Look-Ahead Bias

**Problém:** Některé fundamentální metriky jsou publikovány se zpožděním (quarterly reports vychází 1-2 měsíce po konci kvartálu).

**Mitigace:**
- Použití lagovaných fundamentálních dat (shift o 1-2 měsíce)
- Point-in-time databáze (vyžaduje specializované zdroje)

#### 10.1.3 Kvalita Imputovaných Dat

**Problém:** 85% fundamentálních dat (2015-2024) je predikováno AI modelem, nikoli skutečná.

**Důsledek:** Chyby v imputaci se propagují do klasifikačního modelu.

**Mitigace:**
- Confidence intervals pro imputované hodnoty
- Sensitivity analýza: jaký vliv má chyba imputace na finální predikce?
- Sloupec `data_source` pro transparentnost

### 10.2 Modelová Omezení

#### 10.2.1 Stacionarita

**Předpoklad:** Vztahy mezi features a targetem jsou stabilní v čase.

**Realita:** Tržní dynamika se mění (např. COVID-19, úrokové sazby, geopolitika).

**Mitigace:**
- Rolling window training
- Periodic retraining
- Regime detection modely

#### 10.2.2 Transakční Náklady

**Problém:** Model nezahrnuje:
- Bid-ask spread
- Poplatky brokera
- Market impact (pro velké pozice)
- Daně

**Důsledek:** Skutečná výkonnost bude nižší než backtest.

### 10.3 Budoucí Rozšíření

1. **Alternative Data:** Sentiment z news/social media, satelitní data
2. **Deep Learning:** LSTM/Transformer pro časové řady
3. **Reinforcement Learning:** Optimalizace portfolio allocation
4. **Ensemble Methods:** Kombinace více modelů pro robustnější predikce
5. **Real-time Pipeline:** Automatizovaný trading systém

---

## 11. Reference

### Akademické Zdroje

1. Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383-417.

2. Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. *The Journal of Finance*, 47(2), 427-465.

3. Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *The Journal of Finance*, 48(1), 65-91.

4. Lo, A. W., & MacKinlay, A. C. (1999). *A Non-Random Walk Down Wall Street*. Princeton University Press.

5. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

6. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223-2273.

### Technické Reference

7. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

8. McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56-61.

### Online Zdroje

9. Yahoo Finance API Documentation: https://pypi.org/project/yfinance/

10. Scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html

---

## Přílohy

### A. Kompletní Seznam Features

#### A.1 OHLCV Features (5)
```
open, high, low, close, volume
```

#### A.2 Technical Indicators (13)
```
volatility, returns,
rsi_14, macd, macd_signal, macd_hist,
sma_3, sma_6, sma_12,
ema_3, ema_6, ema_12,
volume_change
```

#### A.3 Fundamental Metrics (14)
```
PE, PB, PS, EV_EBITDA,                              # Valuation
ROE, ROA, Profit_Margin, Operating_Margin, Gross_Margin,  # Profitability
Debt_to_Equity, Current_Ratio, Quick_Ratio,         # Financial Health
Revenue_Growth_YoY, Earnings_Growth_YoY             # Growth
```

### B. Struktura Výstupních Souborů

```
CleanSolution/
├── data/
│   ├── fundamentals/
│   │   └── all_sectors_fundamentals.csv
│   ├── complete/
│   │   └── all_sectors_complete_10y.csv
│   ├── analysis/
│   │   ├── fundamental_predictor_metrics.csv
│   │   ├── classification_metrics.csv
│   │   ├── confusion_matrix_{sector}.png
│   │   └── feature_importance_{sector}.csv
│   └── predictions/
│       └── {sector}_predictions.csv
├── models/
│   ├── fundamental_predictor.pkl
│   ├── feature_scaler.pkl
│   └── {sector}_classifier.pkl
└── docs/
    ├── METHODOLOGY.md (tento dokument)
    ├── WORKFLOW.md
    └── SUMMARY.md
```

---

**Dokument připraven pro účely diplomové práce.**

**Poslední aktualizace:** Prosinec 2025
