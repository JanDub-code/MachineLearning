# 🚀 QUICKSTART - Google Colab Workflow

## Rychlý Průvodce pro Diplomovou Práci

---

## ✅ Předpoklady

1. **Google účet** s přístupem ke Google Drive a Google Colab
2. **OHLCV data** - 10 let historie (již připravena v `data_10y/`)
3. ~30 minut volného času pro kompletní průběh

---

## 📋 Krok za Krokem

### Krok 1: Příprava Google Drive

1. Nahrajte složku `CleanSolution` do Google Drive:
   ```
   Google Drive/
   └── MachineLearning/
       ├── data_10y/
       │   ├── Technology_full_10y.csv
       │   ├── Consumer_full_10y.csv
       │   └── Industrials_full_10y.csv
       ├── notebooks/
       └── models/
   ```

2. Případně upravte cestu `DRIVE_PATH` v noteboocích

### Krok 2: Spusťte Notebooky (v pořadí)

| # | Notebook | Co dělá | Výstup |
|---|----------|---------|--------|
| **01** | Data Collection | Stahuje a připravuje data | `data/ohlcv/*.csv` |
| **02** | Train Fundamental Predictor | Trénuje RF Regressor | `models/fundamental_predictor.pkl` |
| **03** | Complete Historical Data | Imputuje chybějící data | `data/complete/*.csv` |
| **04** | Train Price Classifier | Trénuje RF Classifier | `models/rf_classifier*.pkl` |
| **05** | Hyperparameter Tuning | Optimalizuje parametry | `models/optimal_hyperparameters.json` |
| **06** | Final Evaluation | Generuje výsledky | `figures/*.png`, `final_results.json` |

### Krok 3: Stáhněte Výsledky

Po dokončení Notebooku 06 stáhněte:
- 📈 Grafy z `figures/` pro diplomovou práci
- 📄 `final_results.json` s metrikami

---

## 🎯 Rychlá Verze (pouze esenciální)

Pokud chcete pouze výsledky bez hyperparameter tuning:

1. Spusťte **Notebook 01** → Data
2. Spusťte **Notebook 02** → Model pro imputaci
3. Spusťte **Notebook 03** → Kompletní dataset
4. Spusťte **Notebook 04** → Klasifikátor
5. Spusťte **Notebook 06** → Výsledky

(Notebook 05 - Hyperparameter Tuning je volitelný)

---

## 🔧 Řešení Problémů

### "Drive not mounted"
```python
from google.colab import drive
drive.mount('/content/drive')
```

### "File not found"
Zkontrolujte, že `DRIVE_PATH` odpovídá vaší struktuře složek.

### "Out of memory"
Použijte Colab Pro nebo snižte počet tickerů v konfiguraci.

---

## 📊 Očekávané Výsledky

| Metrika | Očekávaná hodnota |
|---------|-------------------|
| Accuracy | 55-60% |
| F1-Score | 0.55-0.60 |
| Win Rate | 55-60% |
| AUC (UP class) | 0.60-0.70 |

---

## 📁 Vytvořené Soubory

Po úspěšném dokončení budete mít:

```
MachineLearning/
├── data/
│   ├── ohlcv/              # Stažená OHLCV data
│   ├── fundamentals/       # Stažené fundamenty
│   └── complete/           # Kompletní dataset
├── models/
│   ├── fundamental_predictor.pkl
│   ├── price_classifier_tuned.pkl
│   └── optimal_hyperparameters.json
└── figures/
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── sector_comparison.png
    ├── feature_importance.png
    └── backtest_equity.png
```

---

## 🖥️ Lokální Spuštění (Alternativa)

Pokud preferujete lokální prostředí:

```bash
cd CleanSolution
pip install -r requirements.txt

# Stáhnout data
python scripts/0_download_prices.py
python scripts/1_download_fundamentals.py

# Spustit Jupyter
jupyter lab
```

**Nebo vše najednou (Windows PowerShell):**
```powershell
python 1_download_fundamentals.py; python 2_train_fundamental_predictor.py; python 3_complete_historical_data.py; python 4_train_price_predictor.py
```

**Linux/Mac:**
```bash
python 1_download_fundamentals.py && python 2_train_fundamental_predictor.py && python 3_complete_historical_data.py && python 4_train_price_predictor.py
```

---

## 📊 KROK 3: Kontrola Výsledků (2 minuty)

### Zkontrolujte vytvořené soubory:

```bash
# Modely
ls ../models/
# Měli byste vidět:
# - fundamental_predictor.pkl
# - feature_scaler.pkl
# - Technology_price_model.pkl
# - Consumer_price_model.pkl
# - Industrials_price_model.pkl

# Data
ls ../data/complete/
# Měli byste vidět:
# - all_sectors_complete_10y.csv
# - Technology_complete_10y.csv
# - Consumer_complete_10y.csv
# - Industrials_complete_10y.csv

# Analýzy
ls ../data/analysis/
# Měli byste vidět:
# - fundamental_predictor_metrics.csv
# - feature_importance_fundamentals.csv
# - price_prediction_metrics_summary.csv
# - *.png (vizualizace)
```

---

## 🎯 KROK 4: První Predikce (2 minuty)

### Vyzkoušejte model na novém vstupu:

```python
import pandas as pd
import numpy as np
from joblib import load

# Načtení modelu
model = load('../models/Technology_price_model.pkl')
scaler = load('../models/Technology_price_scaler.pkl')

# Vstupní data (AAPL příklad)
input_data = pd.DataFrame({
    'PE': [28.5], 'PB': [40.2], 'PS': [7.8], 'EV_EBITDA': [22.1],
    'ROE': [0.45], 'ROA': [0.18], 'Profit_Margin': [0.25],
    'Operating_Margin': [0.30], 'Gross_Margin': [0.42],
    'Debt_to_Equity': [1.5], 'Current_Ratio': [1.8], 'Quick_Ratio': [1.5],
    'Revenue_Growth_YoY': [0.12], 'Earnings_Growth_YoY': [0.15],
    'volatility': [0.015], 'returns': [0.02], 'rsi_14': [62.0],
    'macd': [1.2], 'volume_change': [0.05]
})

# Predikce
X_scaled = scaler.transform(input_data)
log_price = model.predict(X_scaled)[0]
price = np.exp(log_price)

print(f"Predikovaná cena za měsíc: ${price:.2f}")
```

---

## 📈 KROK 5: Zobrazení Výsledků (1 minuta)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Načtení metrik
metrics = pd.read_csv('../data/analysis/price_prediction_metrics_summary.csv')
print("\n📊 VÝSLEDKY PO SEKTORECH:\n")
print(metrics[['sector', 'test_mae', 'test_r2']])

# Vizualizace
img = plt.imread('../data/analysis/sector_mae_comparison.png')
plt.figure(figsize=(12, 6))
plt.imshow(img)
plt.axis('off')
plt.show()
```

---

## 🎉 Hotovo!

Pokud všechno proběhlo v pořádku, měli byste vidět:

```
✅ FÁZE 2: Staženo ~600-900 fundamentálních záznamů
✅ FÁZE 3: AI model s MAE ~14% a R² ~0.74
✅ FÁZE 4: Kompletní 10letý dataset (~18,000 záznamů)
✅ FÁZE 5: Predikční modely s MAE ~$12 a R² ~0.80
```

---

## ⚠️ Problémy?

### "FileNotFoundError: ../data_10y/..."
**Řešení:** Ujistěte se, že máte OHLCV data v nadřazené složce  
```bash
ls ../data_10y/all_sectors_full_10y.csv
```

### "ModuleNotFoundError: No module named..."
**Řešení:** Znovu nainstalujte závislosti  
```bash
pip install -r requirements.txt
```

### "yfinance returns empty data"
**Řešení:** Normální, ne všechny tickery mají kompletní data  
Skript automaticky přeskakuje problematické tickery

### Stahování trvá příliš dlouho
**Řešení:** Omezení rate limitingu yfinance  
Upravte v `1_download_fundamentals.py`:
```python
time.sleep(1.0)  # místo 0.5
```

---

## 📚 Další Kroky

1. **Prozkoumejte výsledky:** `data/analysis/`
2. **Experimentujte s hyperparametry:** upravte RF_PARAMS, RIDGE_ALPHA
3. **Vyzkoušejte Google Colab:** `notebooks/Part1_DataPreparation_AI.ipynb`
4. **Přečtěte si WORKFLOW.md:** detailní průvodce

---

## 🔗 Užitečné Odkazy

- **README.md** - Přehled projektu
- **WORKFLOW.md** - Detailní návod
- **SUMMARY.md** - Kompletní shrnutí
- **docs/** - Další dokumentace

---

**Vytvořeno:** 31. října 2025  
**Verze:** 1.0.0  

**🚀 Hodně štěstí!**
