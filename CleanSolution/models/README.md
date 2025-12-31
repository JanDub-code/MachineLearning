# 📂 Models - Uložené ML modely

Tato složka obsahuje natrénované modely pro ML pipeline klasifikace cenových pohybů.

**Každý experiment má vlastní podsložku** (např. `30_tickers/`, `50_tickers/`, `100_tickers/`).

---

## 📁 Struktura

```
models/
├── 30_tickers/          # Experiment: 30 tickerů (10 per sektor)
│   ├── classifiers/     # Klasifikační modely (DOWN/HOLD/UP)
│   ├── regressors/      # Regresní modely (pro imputaci fundamentů)
│   ├── scalers/         # StandardScaler objekty
│   └── metadata/        # Metadata a výsledky experimentů
│
├── 50_tickers/          # (budoucí experiment)
├── 100_tickers/         # (budoucí experiment)
└── README.md
```

---

## 📂 30_tickers/classifiers/

Modely pro ternární klasifikaci cenových pohybů.

| Soubor | Popis | Accuracy | F1 |
|--------|-------|----------|-----|
| `rf_classifier_all_sectors.pkl` | Baseline RF Classifier | 33.4% | 32.6% |
| `rf_classifier_tuned.pkl` | Po hyperparameter tuningu | 32.1% | 31.0% |

**Použití:**
```python
import joblib
model = joblib.load('models/30_tickers/classifiers/rf_classifier_tuned.pkl')
predictions = model.predict(X_scaled)
```

---

## 📂 30_tickers/regressors/

Modely pro predikci fundamentálních metrik z OHLCV dat.

| Soubor | Popis | Průměrné R² |
|--------|-------|-------------|
| `fundamental_predictor.pkl` | Multi-output RF Regressor | 0.91 |

**Targets:**
- trailingPE, forwardPE, priceToBook
- returnOnEquity, returnOnAssets
- profitMargins, operatingMargins, grossMargins
- debtToEquity, currentRatio, beta

**Použití:**
```python
import joblib
model = joblib.load('models/30_tickers/regressors/fundamental_predictor.pkl')
fundamentals = model.predict(X_ohlcv_scaled)
```

---

## 📂 30_tickers/scalers/

StandardScaler objekty pro normalizaci dat.

| Soubor | Použití |
|--------|--------|
| `feature_scaler.pkl` | Pro RF Regressor (OHLCV → fundamenty) |
| `classifier_scaler.pkl` | Pro baseline RF Classifier |
| `classifier_scaler_tuned.pkl` | Pro tuned RF Classifier |

**Použití:**
```python
import joblib
scaler = joblib.load('models/30_tickers/scalers/classifier_scaler_tuned.pkl')
X_scaled = scaler.transform(X)
```

---

## 📂 30_tickers/metadata/

Metadata a výsledky experimentů.

| Soubor | Obsah |
|--------|-------|
| `optimal_hyperparameters.json` | Nejlepší parametry z Grid Search |
| `final_evaluation_results.json` | Finální metriky a confusion matrix |
| `classifier_metadata.json` | Info o classifier modelu |
| `grid_search_results.csv` | Všechny kombinace z tuningu |
| `feature_importance.csv` | Důležitost features (regressor) |
| `classifier_feature_importance.csv` | Důležitost features (classifier) |

---

## 🔧 Nejlepší hyperparametry

```json
{
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced"
}
```

---

*Vytvořeno: 31. prosince 2025*
