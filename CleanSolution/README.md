# 🎯 CleanSolution - Predikce Cen Akcií pomocí AI & Lineární Regrese

## 📖 O Projektu

Tento projekt implementuje **inovativní přístup k predikci cen akcií** kombinací:
1. **AI modelu (Random Forest)** - pro doplnění historických fundamentálních dat
2. **Lineární regrese (Ridge)** - pro interpretovatelnou predikci cen z fundamentů

### 🔑 Klíčová Myšlenka

## Metodologie

### 1. Sběr Dat
- **Cenová data**: 10 let historie (OHLCV) + Technické indikátory (RSI, MACD, atd.)
- **Fundamentální data**: Finanční metriky (P/E, ROE, atd.)
- **Doplnění historie**: Použití AI modelu pro dopočítání chybějících fundamentálních dat v historii.

### 2. Validace a Tuning Modelů (CRITICAL)
Abychom zajistili robustnost a kvalitu modelů, používáme pokročilé validační techniky:
- **Cross Validation**: Pro ověření stability modelu na různých podmnožinách dat.
- **Grid Search**: Pro systematické hledání optimálních hyperparametrů.
- **Cíl**: Matematicky podložený výběr nejlepšího modelu, nikoliv "náhodný tip".

### 3. Predikce
- Predikce budoucí ceny na základě kombinace technických a fundamentálních faktorů.

**Problém:**
- Máme 10 let historických cen (OHLCV data)
- Ale pouze 1.5 roku fundamentálních dat (P/E, ROE, atd.)

**Řešení:**
1. Sbíráme fundamentální data za dostupné období (1.5 roku)
2. Trénujeme AI model, který se naučí predikovat fundamenty z OHLCV dat
3. Používáme AI model k doplnění chybějících 8.5 let fundamentů
4. Trénujeme lineární regresi na kompletním 10letém datasetu
5. Predikujeme budoucí ceny na základě fundamentálních metrik

---

## 📂 Struktura Projektu

```
CleanSolution/
│
├── 📄 README.md                              # Tento soubor
├── 📄 WORKFLOW.md                            # Detailní průvodce workflow
├── 📄 requirements.txt                       # Python závislosti
│
├── 📂 data/                                  # Datové soubory
│   ├── ohlcv_10y/                           # OHLCV data z nadřazeného projektu (symlink)
│   ├── fundamentals/                        # Fundamentální data (1.5 roku)
│   ├── complete/                            # Kompletní dataset (10 let)
│   └── predictions/                         # Výsledky predikcí
│
├── 📂 scripts/                               # Python skripty
│   ├── 1_download_fundamentals.py           # FÁZE 2: Stažení fundamentů
│   ├── 2_train_fundamental_predictor.py     # FÁZE 3: AI model
│   ├── 3_complete_historical_data.py        # FÁZE 4: Doplnění dat
│   └── 4_train_price_predictor.py           # FÁZE 5: Lineární regrese
│
├── 📂 notebooks/                             # Jupyter Notebooky pro Google Colab
│   ├── Part1_DataPreparation_AI.ipynb       # FÁZE 2-3: Data + AI model
│   └── Part2_PricePrediction.ipynb          # FÁZE 4-5: Predikce cen
│
├── 📂 models/                                # Uložené modely
│   ├── fundamental_predictor.pkl            # Random Forest model
│   ├── feature_scaler.pkl                   # StandardScaler pro features
│   ├── Technology_price_model.pkl           # Ridge modely po sektorech
│   ├── Consumer_price_model.pkl
│   └── Industrials_price_model.pkl
│
└── 📂 docs/                                  # Dokumentace
    ├── PHASE_OVERVIEW.md                    # Přehled všech fází
    ├── RESULTS_ANALYSIS.md                  # Analýza výsledků
    └── API_REFERENCE.md                     # Dokumentace funkcí
```

---

## 🚀 Rychlý Start

### Předpoklady

- Python 3.8+
- Přístup k internetu (pro stahování dat z yfinance)
- OHLCV data z nadřazeného projektu (složka `../data_10y/`)

### Instalace

```bash
# 1. Přejděte do složky CleanSolution
cd CleanSolution

# 2. Nainstalujte závislosti
pip install -r requirements.txt

# 3. (Volitelné) Vytvořte symlink na OHLCV data
# Windows (PowerShell jako admin):
New-Item -ItemType SymbolicLink -Path "data\ohlcv_10y" -Target "..\data_10y"

# Linux/Mac:
ln -s ../data_10y data/ohlcv_10y
```

### Spuštění Pipeline

#### **Varianta A: Python Skripty (lokálně)**

```bash
# FÁZE 2: Stáhnout fundamentální data (1.5 roku)
python scripts/1_download_fundamentals.py

# FÁZE 3: Natrénovat AI model (OHLCV → Fundamenty)
python scripts/2_train_fundamental_predictor.py

# FÁZE 4: Doplnit historická data (2015-2024)
python scripts/3_complete_historical_data.py

# FÁZE 5: Natrénovat model pro predikci cen
python scripts/4_train_price_predictor.py
```

#### **Varianta B: Google Colab Notebooky**

1. **Nahrajte OHLCV data** do Google Drive
2. Otevřete `notebooks/Part1_DataPreparation_AI.ipynb` v Google Colabu
3. Spusťte všechny buňky (FÁZE 2-3)
4. Otevřete `notebooks/Part2_PricePrediction.ipynb` 
5. Spusťte všechny buňky (FÁZE 4-5)

---

## 📊 Přehled Fází

### ✅ **FÁZE 1: Sběr OHLCV Dat** (Hotovo v nadřazeném projektu)
- 10 let měsíčních OHLCV dat (2015-2025)
- 150 firem z 3 sektorů (Technology, Consumer, Industrials)
- Technické indikátory: RSI, MACD, SMA, EMA, volatilita, returns

### 📥 **FÁZE 2: Stažení Fundamentálních Dat** (1.5 roku)
**Skript:** `scripts/1_download_fundamentals.py`

**Co stahujeme:**
- P/E ratio, P/B ratio, P/S ratio, EV/EBITDA, PEG ratio
- ROE, ROA, Profit Margin, Operating Margin, Gross Margin
- Debt-to-Equity, Current Ratio, Quick Ratio
- Revenue Growth YoY, Earnings Growth YoY

**Období:** 2024-01-01 až 2025-10-01 (~18 měsíců)

**Výstup:** `data/fundamentals/all_sectors_fundamentals.csv`

### 🤖 **FÁZE 3: AI Model pro Predikci Fundamentů**
**Skript:** `scripts/2_train_fundamental_predictor.py`

**Model:** Multi-output Random Forest Regressor

**Input Features:**
- OHLCV: open, high, low, close, volume
- Technické: volatility, returns, RSI, MACD, SMA, EMA
- Další: dividends, volume_change

**Output (15 targets):**
- Všechny fundamentální metriky z FÁZE 2

**Metrika úspěchu:** MAE < 15% (relativní chyba)

**Výstup:** `models/fundamental_predictor.pkl`

### 🔮 **FÁZE 4: Doplnění Historických Dat**
**Skript:** `scripts/3_complete_historical_data.py`

**Proces:**
1. Načte OHLCV data (2015-2025)
2. Aplikuje AI model na období 2015-2024 (predikce fundamentů)
3. Spojí s reálnými fundamenty z 2024-2025
4. Vytvoří kompletní 10letý dataset

**Výstup:** `data/complete/all_sectors_complete_10y.csv`

### 💰 **FÁZE 5: Lineární Regrese - Predikce Ceny**
**Skript:** `scripts/4_train_price_predictor.py`

**Model:** Ridge Regression (po sektorech)

**Input Features:**
- Všechny fundamentální metriky
- Vybrané technické indikátory

**Output:** `log_price_next_month` (logaritmická cena za měsíc)

**Metrika úspěchu:** MAE < $15 (absolutní chyba v USD)

**Výstupy:**
- `models/Technology_price_model.pkl`
- `models/Consumer_price_model.pkl`
- `models/Industrials_price_model.pkl`

---

## 📈 Očekávané Výsledky

### AI Model (Predikce Fundamentů)
```
✅ P/E ratio: MAE < 3.0 bodů
✅ ROE: MAE < 5%
✅ Revenue Growth: MAE < 10%
✅ Celkový průměr: MAE < 15%
```

### Lineární Regrese (Predikce Ceny)
```
✅ Technology: MAE ~$15
✅ Consumer: MAE ~$11
✅ Industrials: MAE ~$11
✅ R² score: >0.75 (vysvětleno 75% variance)
```

### Srovnání s Baseline
```
Baseline (průměr sektoru): MAE ~$45
Náš model: MAE ~$12-15
→ Zlepšení o 67-73%! 🎉
```

---

## 🔬 Použití Modelů

### Predikce Ceny z Fundamentů

```python
import pandas as pd
import numpy as np
from joblib import load

# 1. Načtení modelu
model = load('models/Technology_price_model.pkl')
scaler = load('models/feature_scaler.pkl')

# 2. Příprava vstupních dat
input_data = pd.DataFrame({
    'P/E': [28.5],
    'P/B': [40.2],
    'P/S': [7.8],
    'ROE': [0.45],
    'Revenue_Growth_YoY': [0.12],
    'Debt_to_Equity': [1.5],
    # ... další features
})

# 3. Predikce
X_scaled = scaler.transform(input_data)
log_price_pred = model.predict(X_scaled)[0]
predicted_price = np.exp(log_price_pred)

print(f"Predikovaná cena: ${predicted_price:.2f}")
```

---

## 📚 Dokumentace

- **[WORKFLOW.md](docs/WORKFLOW.md)** - Detailní průvodce krok za krokem
- **[PHASE_OVERVIEW.md](docs/PHASE_OVERVIEW.md)** - Přehled všech fází
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Dokumentace funkcí a tříd

---

## ⚠️ Důležité Poznámky

### Datová Omezení
- **Fundamenty jen 1.5 roku** → AI predikce pro starší data mají vyšší nejistotu
- **Survivorship bias** → S&P 500 neobsahuje firmy, které vypadly z indexu
- **Look-ahead bias** → Pozor na použití budoucích dat při trénování

### Modelová Omezení
- **AI predikce fundamentů** → Není 100% přesná (~15% chyba)
- **Linearita** → Vztah fundamenty→cena nemusí být lineární
- **Externí šoky** → COVID, války, recese nejsou predikované z fundamentů

### Doporučení
- ✅ Používejte confidence intervals (bootstrap)
- ✅ Validujte na různých časových obdobích
- ✅ Srovnávejte s baseline modely
- ✅ Nepředpokládejte kauzalitu (pouze korelace)

---

## 🤝 Přispívání

Tento projekt je vyvíjen jako diplomová/bakalářská práce. Feedback a návrhy na vylepšení jsou vítány!

---

## 📝 Licence

Tento projekt je určen pro **vzdělávací účely**. Používání pro reálné investiční rozhodnutí je na vlastní riziko.

---

## 📧 Kontakt

- **Autor:** Bc. Jan Dub
- **Datum:** Říjen 2025
- **Projekt:** Predikce Cen Akcií pomocí ML

---

**Vytvořeno:** 31. října 2025  
**Verze:** 1.0.0  
**Status:** 🚧 V implementaci
