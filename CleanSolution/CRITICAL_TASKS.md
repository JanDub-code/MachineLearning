# CRITICAL TASKS - CleanSolution

## ✅ Status Přehled

| Úkol | Status | Notebook |
|------|--------|----------|
| Cross Validation (TimeSeriesSplit) | ✅ Implementováno | 05_Hyperparameter_Tuning.ipynb |
| Grid Search | ✅ Implementováno | 05_Hyperparameter_Tuning.ipynb |
| Data Pipeline | ✅ Kompletní | 01-03 notebooky |
| RF Classifier | ✅ Implementováno | 04_Train_Price_Classifier.ipynb |
| Evaluace | ✅ Implementováno | 06_Final_Evaluation.ipynb |

---

## 1. Model Validation & Tuning ✅

- **Cross Validation**: TimeSeriesSplit implementován v `05_Hyperparameter_Tuning.ipynb`
- **Grid Search**: Hyperparameter tuning pro RF Regressor i RF Classifier
- **Výstup**: `models/optimal_hyperparameters.json`

## 2. Data Pipeline ✅

- **Notebook 01**: Sběr OHLCV dat + technické indikátory
- **Notebook 02**: RF Regressor pro imputaci fundamentů
- **Notebook 03**: Doplnění chybějících historických dat

## 3. Klasifikační Model ✅

- **Notebook 04**: RF Classifier pro ternární klasifikaci (DOWN/HOLD/UP)
- **Threshold**: ±3% (pokrývá transakční náklady)
- **Validace**: Chronologický train/test split

## 4. Evaluace ✅

- **Notebook 06**: Confusion Matrix, ROC křivky, per-sector analýza
- **Metriky**: Accuracy, Precision, Recall, F1-Score

---

## 🚀 Jak Spustit

1. Nahrajte data do Google Drive
2. Spusťte notebooky 01-06 v pořadí
3. Výsledky v `models/` a `data/`

---

## 📝 Poznámky

- Modely jsou prázdné dokud nespustíte pipeline
- Doporučeno spustit v Google Colab (bezplatné GPU)
