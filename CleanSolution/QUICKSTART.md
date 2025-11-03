# 🚀 QUICK START GUIDE

## Rychlý návod pro spuštění projektu (5 minut)

---

## ✅ Předpoklady

- ✅ Python 3.8 nebo vyšší nainstalován
- ✅ OHLCV data z nadřazeného projektu (`../data_10y/all_sectors_full_10y.csv`)
- ✅ Přístup k internetu (pro stahování fundamentálních dat)

---

## 📦 KROK 1: Instalace (1 minuta)

```bash
# Přejděte do složky CleanSolution
cd CleanSolution

# Nainstalujte závislosti
pip install -r requirements.txt
```

**Nebo rychle:**
```bash
pip install pandas numpy scikit-learn yfinance matplotlib seaborn joblib
```

---

## 🎯 KROK 2: Spuštění Pipeline (30-60 minut)

### Automatické spuštění všech fází:

```bash
cd scripts

# FÁZE 2: Stažení fundamentálních dat (~30-45 min)
python 1_download_fundamentals.py

# FÁZE 3: Trénování AI modelu (~5 min)
python 2_train_fundamental_predictor.py

# FÁZE 4: Doplnění historických dat (~5 min)
python 3_complete_historical_data.py

# FÁZE 5: Trénování predikčního modelu (~5 min)
python 4_train_price_predictor.py
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
