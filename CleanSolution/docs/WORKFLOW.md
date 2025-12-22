# 🔄 WORKFLOW - Google Colab Průvodce

## 📖 Úvod

Tento dokument poskytuje **detailní průvodce** celým procesem klasifikace cenových pohybů akcií pomocí ML. Workflow je optimalizován pro **Google Colab**.

---

## 🎯 Přehled Notebooků

```
📓 01_Data_Collection.ipynb
   └── Teoretický úvod, stažení OHLCV, technické indikátory
          ↓
📓 02_Train_Fundamental_Predictor.ipynb
   └── Random Forest Regressor (OHLCV → Fundamenty)
          ↓
📓 03_Complete_Historical_Data.ipynb
   └── Imputace chybějících fundamentů (2015-2024)
          ↓
📓 04_Train_Price_Classifier.ipynb
   └── Random Forest Classifier (DOWN/HOLD/UP)
          ↓
📓 05_Hyperparameter_Tuning.ipynb
   └── Grid Search s TimeSeriesSplit (volitelný)
          ↓
📓 06_Final_Evaluation.ipynb
   └── Kompletní evaluace + grafy pro diplomovou práci
```

---

## 📓 Notebook 01: Data Collection

### Cíl
Připravit kompletní dataset OHLCV + technické indikátory

### Obsah
1. **Teoretický úvod**
   - Efektivní hypotéza trhů (EMH)
   - Omezení predikce cen
   - Klasifikace vs regrese

2. **Stažení OHLCV dat**
   - yfinance API
   - 10 let měsíční historie
   - 150 S&P 500 akcií

3. **Technické indikátory**
   - RSI (14 period)
   - MACD (12, 26, 9)
   - SMA/EMA (3, 6, 12 měsíců)
   - Volatilita, momentum

### Výstup
```
data/ohlcv/
├── all_sectors_ohlcv_10y.csv
├── Technology_ohlcv_10y.csv
├── Consumer_ohlcv_10y.csv
└── Industrials_ohlcv_10y.csv
```

---

## 📓 Notebook 02: Train Fundamental Predictor

### Cíl
Natrénovat RF Regressor pro predikci fundamentálních metrik z OHLCV

### Problém
- Fundamentální data dostupná pouze za 1.5 roku (2024-2025)
- OHLCV data za 10 let (2015-2025)
- Pro klasifikaci potřebujeme kompletní dataset

### Řešení
Multi-output Random Forest Regressor:
- **Input:** 18 OHLCV + technických features
- **Output:** 11 fundamentálních metrik

### Obsah
1. Načtení dat
2. Feature engineering
3. Trénink RF Regressor
4. Evaluace (MAE, RMSE, R²)
5. Feature importance analýza

### Výstup
```
models/
├── fundamental_predictor.pkl
└── feature_scaler.pkl
```

---

## 📓 Notebook 03: Complete Historical Data

### Cíl
Použít natrénovaný model k doplnění chybějících fundamentů

### Proces
1. Načíst OHLCV data (2015-2024)
2. Aplikovat feature scaler
3. Predikovat fundamentální metriky
4. Validovat výsledky (sanity checks)
5. Spojit s reálnými daty (2024-2025)

### Sanity Checks
- P/E ratio: 0 < P/E < 100
- ROE: -50% < ROE < 100%
- Debt/Equity: 0 < D/E < 10

### Výstup
```
data/complete/
└── all_sectors_complete_10y.csv
```

---

## 📓 Notebook 04: Train Price Classifier

### Cíl
Natrénovat ternární klasifikátor pro predikci cenových pohybů

### Definice Tříd (±3% threshold)
| Třída | Label | Definice |
|-------|-------|----------|
| DOWN | 0 | Return < -3% |
| HOLD | 1 | -3% ≤ Return ≤ +3% |
| UP | 2 | Return > +3% |

### Proč 3%?
Pokrývá transakční náklady:
- Bid-ask spread: ~0.5%
- Broker fees: ~0.5%
- Slippage: ~1%
- Reserve: ~1%

### Obsah
1. Vytvoření target variable
2. Feature selection
3. Chronologický train/test split
4. Trénink RF Classifier
5. Evaluace per sektor

### Výstup
```
models/
└── rf_classifier_all_sectors.pkl
```

---

## 📓 Notebook 05: Hyperparameter Tuning

### Cíl
Najít optimální hyperparametry pomocí Grid Search

### TimeSeriesSplit
Speciální cross-validation pro časové řady:
```
Fold 1: [Train: ████████] [Test: ██]
Fold 2: [Train: ██████████] [Test: ██]
Fold 3: [Train: ████████████] [Test: ██]
```

### Parametrový prostor (RF)
| Parametr | Hodnoty |
|----------|---------|
| n_estimators | [100, 200, 300] |
| max_depth | [10, 15, 20, None] |
| min_samples_split | [2, 5, 10] |
| min_samples_leaf | [1, 2, 4] |

### Obsah
1. Grid Search pro RF Regressor
2. Grid Search pro RF Classifier
3. Porovnání s Gradient Boosting
4. Vizualizace výsledků

### Výstup
```
models/
├── fundamental_predictor_tuned.pkl
├── price_classifier_tuned.pkl
└── optimal_hyperparameters.json
```

---

## 📓 Notebook 06: Final Evaluation

### Cíl
Kompletní evaluace + vizualizace pro diplomovou práci

### Obsah
1. **Klasifikační metriky**
   - Accuracy, Precision, Recall, F1
   - Classification Report

2. **Vizualizace**
   - Confusion Matrix
   - ROC křivky (per class)
   - Feature Importance

3. **Sektorová analýza**
   - Porovnání Technology vs Consumer vs Industrials

4. **Backtesting**
   - Simulace obchodní strategie
   - Equity curve, Drawdown
   - Sharpe Ratio

### Výstup
```
figures/
├── confusion_matrix.png
├── roc_curves.png
├── sector_comparison.png
├── feature_importance.png
└── backtest_equity.png
```

---

## 🔧 Praktické Tipy

### Google Colab Setup
```python
from google.colab import drive
drive.mount('/content/drive')

DRIVE_PATH = '/content/drive/MyDrive/MachineLearning'
```

### Ukládání modelů
```python
import joblib

# Uložit
joblib.dump(model, f'{MODEL_PATH}/model.pkl')

# Načíst
model = joblib.load(f'{MODEL_PATH}/model.pkl')
```

### Ukládání grafů
```python
plt.savefig(f'{FIGURES_PATH}/graph.png', dpi=300, bbox_inches='tight')
```

---

## 📊 Očekávané Výsledky

| Notebook | Klíčová metrika | Očekávaná hodnota |
|----------|-----------------|-------------------|
| 02 | RF Regressor R² | > 0.60 |
| 04 | Classifier Accuracy | 55-60% |
| 04 | F1-Score (weighted) | 0.55-0.60 |
| 06 | Win Rate (backtest) | 55-60% |
| 06 | Sharpe Ratio | > 0.5 |

---

## ❓ FAQ

### Proč Google Colab místo lokálního Jupyter?
1. Bezplatné GPU/TPU
2. Jednotné prostředí
3. Snadné sdílení
4. Integrace s Google Drive

### Proč klasifikace místo regrese?
1. Praktičtější output (BUY/HOLD/SELL)
2. Robustní vůči outliers
3. Lépe interpretovatelné výsledky

### Proč Random Forest místo Neural Network?
1. Menší dataset (tisíce, ne miliony záznamů)
2. Interpretabilita (feature importance)
3. Nepotřebuje GPU
4. Rychlý trénink

---

## ✅ Checklist

- [ ] Nahrát data do Google Drive
- [ ] Spustit Notebook 01 (Data Collection)
- [ ] Spustit Notebook 02 (Fundamental Predictor)
- [ ] Spustit Notebook 03 (Complete Data)
- [ ] Spustit Notebook 04 (Price Classifier)
- [ ] Spustit Notebook 05 (Hyperparameter Tuning) - volitelné
- [ ] Spustit Notebook 06 (Final Evaluation)
- [ ] Stáhnout grafy pro diplomovou práci
