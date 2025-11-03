# 🔄 WORKFLOW - Krok za Krokem Průvodce

## 📖 Úvod

Tento dokument poskytuje **detailní průvodce** celým procesem predikce cen akcií pomocí AI a lineární regrese. Projdeme všech 5 fází projektu s praktickými příklady.

---

## 🎯 Přehled Fází

```
FÁZE 1: Sběr OHLCV Dat (10 let)           ✅ HOTOVO (nadřazený projekt)
          ↓
FÁZE 2: Stažení Fundamentů (1.5 roku)     📥 download_fundamentals.py
          ↓
FÁZE 3: AI Model (OHLCV → Fundamenty)     🤖 train_fundamental_predictor.py
          ↓
FÁZE 4: Doplnění Historie (2015-2024)     🔮 complete_historical_data.py
          ↓
FÁZE 5: Predikce Ceny (Fundamenty → $)    💰 train_price_predictor.py
```

---

## ✅ FÁZE 1: Sběr OHLCV Dat (Již hotovo)

### Co máme připravené:

📂 `../data_10y/` obsahuje:
- `all_sectors_full_10y.csv` - kompletní dataset
- `Technology_full_10y.csv`, `Consumer_full_10y.csv`, `Industrials_full_10y.csv`
- Tickery pro každý sektor

### Struktura dat:

```csv
date, ticker, sector, open, high, low, close, volume,
volatility, returns, rsi_14, macd, macd_signal, macd_hist,
sma_3, sma_6, sma_12, ema_3, ema_6, ema_12,
dividends, split_occurred, volume_change
```

### Ověření:

```python
import pandas as pd

df = pd.read_csv('../data_10y/all_sectors_full_10y.csv')
print(f"Záznamů: {len(df)}")
print(f"Období: {df['date'].min()} → {df['date'].max()}")
print(f"Tickery: {df['ticker'].nunique()}")
```

**Očekávaný výstup:**
```
Záznamů: ~18,000
Období: 2015-01-31 → 2025-10-31
Tickery: 150
```

---

## 📥 FÁZE 2: Stažení Fundamentálních Dat

### Cíl:
Stáhnout fundamentální metriky pro období 2024-2025 (cca 1.5 roku)

### Způsob A: Python Skript (lokálně)

```bash
cd CleanSolution/scripts
python 1_download_fundamentals.py
```

**Co skript dělá:**
1. Načte seznam tickerů z `../data_10y/`
2. Pro každý ticker stáhne quarterly financials z yfinance
3. Vypočítá 14 fundamentálních metrik
4. Uloží do `../data/fundamentals/`

**Výstup:**
```
data/fundamentals/
├── all_sectors_fundamentals.csv
├── Technology_fundamentals.csv
├── Consumer_fundamentals.csv
└── Industrials_fundamentals.csv
```

### Způsob B: Google Colab Notebook

1. Otevřete `notebooks/Part1_DataPreparation_AI.ipynb`
2. Nahrajte OHLCV data na Google Drive
3. Připojte Drive a spusťte notebook
4. Sekce 4 stahuje fundamenty automaticky

### Očekávané metriky:

| Kategorie | Metriky |
|-----------|---------|
| **Valuační** | P/E, P/B, P/S, EV/EBITDA |
| **Profitabilita** | ROE, ROA, Profit Margin, Operating Margin, Gross Margin |
| **Finanční zdraví** | Debt-to-Equity, Current Ratio, Quick Ratio |
| **Růst** | Revenue Growth YoY, Earnings Growth YoY |

### Ověření:

```python
import pandas as pd

df = pd.read_csv('data/fundamentals/all_sectors_fundamentals.csv')
print(f"Záznamů: {len(df)}")
print(f"Tickery: {df['ticker'].nunique()}")
print(f"Columns: {df.columns.tolist()}")
```

**Očekávaný výstup:**
```
Záznamů: ~600-900 (závisí na dostupnosti dat)
Tickery: 100-150
Columns: ['date', 'ticker', 'sector', 'PE', 'PB', 'PS', ...]
```

### ⚠️ Možné problémy:

**Problém:** yfinance vrací prázdná data pro některé tickery
- **Řešení:** Normální, ne všechny firmy mají kompletní quarterly data
- Skript automaticky přeskočí problematické tickery

**Problém:** Rate limiting (příliš mnoho requestů)
- **Řešení:** Skript má built-in `time.sleep(0.5)` mezi requesty
- Pro větší bezpečnost zvyšte na `time.sleep(1.0)`

---

## 🤖 FÁZE 3: Trénování AI Modelu

### Cíl:
Natrénovat Random Forest model, který predikuje fundamenty z OHLCV dat

### Způsob A: Python Skript

```bash
python scripts/2_train_fundamental_predictor.py
```

**Co skript dělá:**
1. Načte OHLCV data (2015-2025) a fundamenty (2024-2025)
2. Spojí data pomocí forward-fill
3. Připraví features (OHLCV + technické indikátory)
4. Trénuje Multi-output Random Forest (100 trees, max_depth=20)
5. Evaluuje na test setu (80/20 split)
6. Analyzuje feature importance
7. Uloží model a výsledky

**Výstup:**
```
models/
├── fundamental_predictor.pkl      # Natrénovaný model
└── feature_scaler.pkl             # StandardScaler pro features

data/analysis/
├── fundamental_predictor_metrics.csv        # MAE, RMSE, R² pro každou metriku
├── feature_importance_fundamentals.csv      # Důležitost features
└── fundamental_predictions_vs_actual.csv    # Predikce vs. skutečnost
```

### Způsob B: Google Colab

Spusťte sekce 6-8 v `Part1_DataPreparation_AI.ipynb`

### Očekávané výsledky:

**Cílové metriky:**
- **MAE < 15%** (relativní chyba)
- **R² > 0.70** (vysvětleno 70% variance)

**Příklad výstupu:**
```
📊 PRŮMĚR:
   MAE: 3.245
   MAE%: 14.2%
   RMSE: 5.123
   R²: 0.743
```

### Interpretace výsledků:

| MAE% | Hodnocení | Akce |
|------|-----------|------|
| < 15% | ✨ Výborně! | Pokračujte na FÁZI 4 |
| 15-20% | 👍 Dobře | Použitelné, pokračujte |
| > 20% | ⚠️ Vyšší chyba | Zvažte tuning nebo více dat |

### Feature Importance analýza:

**Očekávané top features:**
- `close` - současná cena (silná korelace s valuačními ratios)
- `rsi_14` - RSI indikátor (sentiment)
- `volume` - objem obchodování
- `volatility` - volatilita (souvisí s rizikem)
- `macd` - momentum

---

## 🔮 FÁZE 4: Doplnění Historických Dat

### Cíl:
Použít AI model k predikci fundamentů pro období 2015-2024

### Spuštění:

```bash
python scripts/3_complete_historical_data.py
```

**Co skript dělá:**
1. Načte natrénovaný AI model
2. Načte OHLCV data (2015-2025)
3. **Predikuje fundamenty pro 2015-2024** pomocí AI modelu
4. Spojí predikované (2015-2024) + reálné (2024-2025) fundamenty
5. Vytvoří kompletní 10letý dataset
6. Validuje predikce (srovnání průměrů)
7. Uloží kompletní data

**Výstup:**
```
data/complete/
├── all_sectors_complete_10y.csv
├── Technology_complete_10y.csv
├── Consumer_complete_10y.csv
└── Industrials_complete_10y.csv
```

### Struktura výstupního datasetu:

```csv
date, ticker, sector,
open, high, low, close, volume, volatility, returns, rsi_14, ...  # OHLCV + technické
PE, PB, PS, EV_EBITDA, ROE, ROA, ...                              # Fundamenty
data_source                                                         # 'predicted' nebo 'real'
```

**Sloupec `data_source`:**
- `predicted` = fundamenty predikované AI modelem (2015-2024)
- `real` = reálné fundamenty z yfinance (2024-2025)

### Validace:

Skript automaticky srovná průměry predikovaných vs. reálných hodnot:

```
📊 Srovnání predikovaných vs. reálných hodnot:
Metrika                   Predikované (mean)   Reálné (mean)        Rozdíl %
-----------------------------------------------------------------------------------
PE                        24.3215              26.1820              7.2%
ROE                       0.1823               0.1965               7.8%
Revenue_Growth_YoY        0.0842               0.0915               8.7%
```

**Dobrá validace:** Rozdíly < 20%  
**Pozor:** Rozdíly > 30% mohou indikovat problém s modelem

---

## 💰 FÁZE 5: Trénování Modelu pro Predikci Ceny

### Cíl:
Natrénovat Ridge Regression model, který predikuje cenu z fundamentů

### Spuštění:

```bash
python scripts/4_train_price_predictor.py
```

**Co skript dělá:**
1. Načte kompletní dataset (10 let OHLCV + fundamenty)
2. Vytvoří target: `log_price_next_month`
3. Připraví features: fundamenty + technické indikátory
4. Trénuje **samostatný Ridge model pro každý sektor**
5. Evaluuje na test setu (chronologický split 80/20)
6. Analyzuje koeficienty (feature importance)
7. Vytváří vizualizace
8. Uloží modely

**Výstup:**
```
models/
├── Technology_price_model.pkl
├── Technology_price_scaler.pkl
├── Consumer_price_model.pkl
├── Consumer_price_scaler.pkl
├── Industrials_price_model.pkl
└── Industrials_price_scaler.pkl

data/analysis/
├── price_prediction_metrics_summary.csv
├── Technology_price_predictions.csv
├── Technology_price_coefficients.csv
├── sector_mae_comparison.png
└── sector_r2_comparison.png
```

### Očekávané výsledky:

**Cílové metriky:**
- **MAE < $15** (průměrná absolutní chyba v dolarech)
- **R² > 0.75** (vysvětleno 75% variance)

**Příklad výstupu:**
```
📊 SOUHRNNÉ VÝSLEDKY

Technology:
  Test MAE:  $14.23
  Test RMSE: $19.87
  Test R²:   0.781

Consumer:
  Test MAE:  $10.54
  Test RMSE: $14.21
  Test R²:   0.823

Industrials:
  Test MAE:  $11.89
  Test RMSE: $15.44
  Test R²:   0.798

📈 PRŮMĚR VŠECH SEKTORŮ:
  • MAE:  $12.22
  • R²:   0.801
```

### Feature Coefficients analýza:

**TOP 10 FEATURES pro Technology:**
```
+ PE                      :   0.3421  (vyšší P/E → vyšší cena)
+ Revenue_Growth_YoY      :   0.2873  (růst tržeb zvyšuje cenu)
+ ROE                     :   0.2156  (profitabilita)
+ PB                      :   0.1987
+ Profit_Margin           :   0.1562
- Debt_to_Equity          :  -0.1343  (dluh snižuje cenu)
- volatility              :  -0.0894  (volatilita je riziková)
+ close                   :   0.0832
+ Operating_Margin        :   0.0765
+ rsi_14                  :   0.0621
```

**Interpretace:**
- **Pozitivní koeficient** = zvýšení této metriky zvyšuje cenu
- **Negativní koeficient** = zvýšení této metriky snižuje cenu
- **Velikost koeficientu** = síla vlivu

---

## 🎯 Použití Natrénovaných Modelů

### Predikce ceny pro novou firmu:

```python
import pandas as pd
import numpy as np
from joblib import load

# 1. Načtení modelu a scaleru
model = load('models/Technology_price_model.pkl')
scaler = load('models/Technology_price_scaler.pkl')

# 2. Příprava vstupních dat
input_data = pd.DataFrame({
    # Fundamenty
    'PE': [28.5],
    'PB': [40.2],
    'PS': [7.8],
    'EV_EBITDA': [22.1],
    'ROE': [0.45],
    'ROA': [0.18],
    'Profit_Margin': [0.25],
    'Operating_Margin': [0.30],
    'Gross_Margin': [0.42],
    'Debt_to_Equity': [1.5],
    'Current_Ratio': [1.8],
    'Quick_Ratio': [1.5],
    'Revenue_Growth_YoY': [0.12],
    'Earnings_Growth_YoY': [0.15],
    
    # Technické
    'volatility': [0.015],
    'returns': [0.02],
    'rsi_14': [62.0],
    'macd': [1.2],
    'volume_change': [0.05]
})

# 3. Standardizace
X_scaled = scaler.transform(input_data)

# 4. Predikce
log_price_pred = model.predict(X_scaled)[0]
predicted_price = np.exp(log_price_pred)

print(f"Predikovaná cena za měsíc: ${predicted_price:.2f}")
```

### Analýza důležitosti faktorů:

```python
# Načtení koeficientů
coef_df = pd.read_csv('data/analysis/Technology_price_coefficients.csv')
coef_df = coef_df.sort_values('abs_coefficient', ascending=False)

print("TOP 10 FAKTORŮ OVLIVŇUJÍCÍCH CENU:")
print(coef_df.head(10))
```

---

## 📊 Analýza a Vizualizace

### Srovnání predikcí s reálnými cenami:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Načtení predikcí
pred = pd.read_csv('data/analysis/Technology_price_predictions.csv')
pred['date'] = pd.to_datetime(pred['date'])

# Vizualizace pro jeden ticker
ticker = 'AAPL'
ticker_pred = pred[pred['ticker'] == ticker]

plt.figure(figsize=(14, 6))
plt.plot(ticker_pred['date'], ticker_pred['price_true'], label='Skutečná cena', linewidth=2)
plt.plot(ticker_pred['date'], ticker_pred['price_pred'], label='Predikovaná cena', linestyle='--', linewidth=2)
plt.xlabel('Datum')
plt.ylabel('Cena ($)')
plt.title(f'{ticker} - Predikce vs. Skutečnost')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### Error analýza:

```python
# MAE distribution
errors = abs(pred['price_pred'] - pred['price_true'])

plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, label=f'Průměr: ${errors.mean():.2f}')
plt.xlabel('Absolutní chyba ($)')
plt.ylabel('Počet predikcí')
plt.title('Distribuce Chyb Predikce')
plt.legend()
plt.show()
```

---

## ⚠️ Troubleshooting

### Problém 1: Chybějící data

**Chyba:**
```
FileNotFoundError: ../data_10y/all_sectors_full_10y.csv
```

**Řešení:**
- Ujistěte se, že jste spustili `prepare_10y_data_full.py` z nadřazeného projektu
- Zkontrolujte relativní cesty v konfiguračních konstantách
- Případně vytvořte symlink: `ln -s ../../data_10y data/ohlcv_10y`

### Problém 2: Nízká přesnost AI modelu (MAE > 20%)

**Možné příčiny:**
- Málo trénovacích dat (< 500 vzorků)
- Chybějící fundamenty pro mnoho tickerů
- Outliers v datech

**Řešení:**
1. Zvýšit počet tickerů (stáhnout více fundamentálních dat)
2. Hyperparameter tuning:
   ```python
   RF_PARAMS = {
       'n_estimators': 200,  # zvýšit
       'max_depth': 30,      # zvýšit
       'min_samples_split': 3
   }
   ```
3. Feature selection (odstranit málo důležité features)

### Problém 3: Nízký R² score pro predikci ceny (< 0.60)

**Možné příčiny:**
- Predikované fundamenty mají vysokou chybu
- Linearita není vhodná pro data
- Chybějící důležité faktory

**Řešení:**
1. Zlepšit AI model z FÁZE 3
2. Zkusit jiný model (ElasticNet, Gradient Boosting)
3. Přidat více features
4. Ensemble modely

### Problém 4: Memory Error při trénování

**Řešení:**
```python
# Redukovat velikost datasetu
df = df.sample(frac=0.5, random_state=42)  # Použít 50% dat

# Nebo trénovat po sektorech
for sector in ['Technology', 'Consumer', 'Industrials']:
    sector_df = df[df['sector'] == sector]
    # trénování...
```

---

## 📈 Best Practices

### 1. Pravidelná Re-trénování

Modely by měly být re-trénovány každých **3-6 měsíců** s novými daty.

### 2. Cross-Validation

Pro robustnější evaluaci použijte K-fold cross-validation:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
print(f"CV MAE: {-scores.mean():.2f} ± {scores.std():.2f}")
```

### 3. Confidence Intervals

Pro odhad nejistoty použijte bootstrap:

```python
from sklearn.utils import resample

predictions = []
for _ in range(100):
    X_boot, y_boot = resample(X_test, y_test)
    pred = model.predict(X_boot)
    predictions.append(pred)

predictions = np.array(predictions)
lower = np.percentile(predictions, 2.5, axis=0)
upper = np.percentile(predictions, 97.5, axis=0)
```

### 4. Monitoring

Sledujte průběžně:
- MAE trend v čase
- Distribution shifty (změny v distribuci dat)
- Feature drift (změny v importanci features)

---

## 🎓 Další Zdroje

### Doporučená Literatura:
- **Scikit-learn Documentation:** https://scikit-learn.org/
- **yfinance GitHub:** https://github.com/ranaroussi/yfinance
- **Financial ML:** "Advances in Financial Machine Learning" - Marcos López de Prado

### Užitečné Tutoriály:
- Time Series Cross-Validation
- Feature Engineering for Financial Data
- Ensemble Methods in ML

---

**Autor:** Bc. Jan Dub  
**Poslední aktualizace:** Říjen 2025  
**Verze:** 1.0.0
