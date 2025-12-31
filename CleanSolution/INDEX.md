# 🎯 CleanSolution - Index Dokumentace

Vítejte v **CleanSolution** - kompletním řešení pro klasifikaci cenových pohybů akcií pomocí Random Forest!

---

## 📚 Dokumentace

### 🚀 Začínáme

| Dokument | Popis | Pro koho |
|----------|-------|----------|
| **[QUICKSTART.md](QUICKSTART.md)** | 5minutový rychlý start | ✅ Začátečníci |
| **[README.md](README.md)** | Přehled projektu, instalace | ✅ Všichni |
| **[docs/WORKFLOW.md](docs/WORKFLOW.md)** | Detailní průvodce krok za krokem | 📖 Pokročilí |

### 📊 Reference

| Dokument | Popis |
|----------|-------|
| **[docs/SUMMARY.md](docs/SUMMARY.md)** | Kompletní shrnutí projektu |
| **requirements.txt** | Python závislosti |

---

## 🗂️ Struktura Projektu

```
CleanSolution/
│
├── 📄 README.md                    ← ZAČNĚTE TADY
├── 📄 QUICKSTART.md                ← 5min rychlý start
├── 📄 INDEX.md                     ← Tento soubor
├── 📄 requirements.txt             ← Python závislosti
│
├── 📂 notebooks/                   ← 🎯 HLAVNÍ - Jupyter Notebooky pro Google Colab
│   ├── 01_Data_Collection.ipynb             # Sběr dat
│   ├── 02_Train_Fundamental_Predictor.ipynb # RF Regressor
│   ├── 03_Complete_Historical_Data.ipynb    # Imputace dat
│   ├── 04_Train_Price_Classifier.ipynb      # RF Classifier
│   ├── 05_Hyperparameter_Tuning.ipynb       # Grid Search
│   └── 06_Final_Evaluation.ipynb            # Evaluace
│
├── 📂 scripts/                     ← Pomocné Python skripty (API)
│   ├── 0_download_prices.py                 # Stažení OHLCV
│   └── 1_download_fundamentals.py           # Stažení fundamentů
│
├── 📂 data/                        ← Datové soubory (generované)
│   ├── ohlcv/
│   ├── fundamentals/
│   ├── complete/
│   └── figures/
│
├── 📂 data_10y/                    ← Vstupní data (10 let)
│   ├── Technology_full_10y.csv
│   ├── Consumer_full_10y.csv
│   └── Industrials_full_10y.csv
│
├── 📂 models/                      ← ML modely (generované)
│   ├── fundamental_predictor.pkl
│   ├── rf_classifier_all_sectors.pkl
│   └── optimal_hyperparameters.json
│
├── 📂 docs/                        ← Dokumentace
│   ├── METHODOLOGY.md
│   ├── MATHEMATICAL_FOUNDATIONS.md
│   ├── ALGORITHM_SELECTION.md
│   ├── WORKFLOW.md
│   └── SUMMARY.md
│
└── 📂 archive/                     ← Archivované staré skripty
```

---

## 🎓 Doporučený Postup Čtení

### Pro Úplné Začátečníky:

1. ✅ **[QUICKSTART.md](QUICKSTART.md)** - Rychlé spuštění za 5 minut
2. ✅ **[README.md](README.md)** - Pochopení projektu
3. ✅ Spusťte notebooky 01-06 v Google Colab
4. ✅ **[docs/WORKFLOW.md](docs/WORKFLOW.md)** - Detailní pochopení

### Pro Pokročilé:

1. ✅ **[README.md](README.md)** - Přehled
2. ✅ **[docs/WORKFLOW.md](docs/WORKFLOW.md)** - Detailní workflow
3. ✅ Prozkoumejte notebooky v `notebooks/`
4. ✅ **[docs/SUMMARY.md](docs/SUMMARY.md)** - Kompletní reference

### Pro Google Colab:

1. ✅ **[README.md](README.md)** - Sekce "Rychlý Start"
2. ✅ Otevřete `notebooks/01_Data_Collection.ipynb`
3. ✅ Spusťte všechny notebooky v pořadí 01-06

---

## 🚀 Rychlý Start (TL;DR)

**Doporučený postup - Google Colab:**

1. Nahrajte data do Google Drive (`MachineLearning/data_10y/`)
2. Otevřete notebooky v Colab (v pořadí):

| # | Notebook | Popis | Čas |
|---|----------|-------|-----|
| 1 | `01_Data_Collection.ipynb` | Sběr dat | ~10 min |
| 2 | `02_Train_Fundamental_Predictor.ipynb` | RF Regressor | ~5 min |
| 3 | `03_Complete_Historical_Data.ipynb` | Imputace dat | ~2 min |
| 4 | `04_Train_Price_Classifier.ipynb` | RF Classifier | ~5 min |
| 5 | `05_Hyperparameter_Tuning.ipynb` | Grid Search | ~15 min |
| 6 | `06_Final_Evaluation.ipynb` | Evaluace | ~5 min |

**Výsledky:**
- `models/` - natrénované modely
- `data/complete/` - kompletní dataset
- `data/figures/` - vizualizace

**Očekávaný čas:** ~45 minut

---

## 📊 Co Projekt Dělá?

### Problém:
- Máme 10 let historických OHLCV dat
- Ale pouze 1.5 roku fundamentálních dat (P/E, ROE, atd.)

### Řešení:

```
📓 01: Sběr OHLCV dat + tech. indikátory      ✅ Notebook 01
          ↓
📓 02: RF Regressor (OHLCV → Fundamenty)      🤖 Notebook 02
          ↓
📓 03: Imputace chybějících fundamentů        🔮 Notebook 03
          ↓
📓 04: RF Classifier (DOWN/HOLD/UP)           📊 Notebook 04
          ↓
📓 05: Hyperparameter Tuning                  🎛️ Notebook 05
          ↓
📓 06: Finální evaluace + vizualizace         📈 Notebook 06
```

### Výsledek:

- ✅ RF Regressor pro imputaci fundamentálních dat
- ✅ RF Classifier pro ternární klasifikaci (DOWN/HOLD/UP)
- ✅ Kompletní 10letý dataset připravený k analýze
- ✅ Accuracy ~55-60%, F1-Score ~0.55-0.60

---

## 🛠️ Dostupné Nástroje

### Jupyter Notebooky (Google Colab) - HLAVNÍ WORKFLOW

| Notebook | Popis | Čas |
|----------|-------|-----|
| `01_Data_Collection.ipynb` | Sběr dat + technické indikátory | ~10 min |
| `02_Train_Fundamental_Predictor.ipynb` | RF Regressor pro imputaci | ~5 min |
| `03_Complete_Historical_Data.ipynb` | Doplnění chybějících dat | ~2 min |
| `04_Train_Price_Classifier.ipynb` | RF Classifier (DOWN/HOLD/UP) | ~5 min |
| `05_Hyperparameter_Tuning.ipynb` | Grid Search optimalizace | ~15 min |
| `06_Final_Evaluation.ipynb` | Evaluace + grafy pro DP | ~5 min |

### Pomocné Python Skripty (API)

| Skript | Popis |
|--------|-------|
| `0_download_prices.py` | Stažení OHLCV dat z yfinance |
| `1_download_fundamentals.py` | Stažení fundamentálních dat |

---

## 📈 Očekávané Výsledky

### RF Regressor (Imputace fundamentů):
```
✅ Predikuje 11 fundamentálních metrik z OHLCV
✅ MAE: ~14-18% (závislé na metrice)
✅ R²: ~0.70-0.85
```

### RF Classifier (Klasifikace pohybů):
```
✅ Ternární klasifikace: DOWN/HOLD/UP
✅ Threshold: ±3% (pokrývá transakční náklady)
✅ Accuracy: ~55-60%
✅ F1-Score (weighted): ~0.55-0.60
```

**Definice tříd:**
- DOWN (0): Měsíční výnos < -3%
- HOLD (1): Výnos mezi -3% a +3%
- UP (2): Měsíční výnos > +3%

---

## ❓ FAQ

### Q: Potřebuji GPU?
**A:** Ne, všechny modely běží na CPU (skripty i notebooky).

### Q: Jak dlouho trvá celý pipeline?
**A:** ~45-90 minut (většinu času trvá stahování fundamentálních dat).

### Q: Mohu použít jiné tickery?
**A:** Ano! Stačí upravit OHLCV data v `../data_10y/` a spustit pipeline znovu.

### Q: Funguje to na Windows?
**A:** Ano! Všechny skripty jsou cross-platform (Windows, Linux, Mac).

### Q: Potřebuji yfinance API klíč?
**A:** Ne, yfinance je free a nevyžaduje API klíč.

### Q: Co když mám málo RAM?
**A:** Redukujte počet tickerů nebo použijte Google Colab (zdarma 12GB RAM).

---

## 🔗 Externí Odkazy

### Knihovny:
- [scikit-learn](https://scikit-learn.org/) - Machine Learning
- [yfinance](https://github.com/ranaroussi/yfinance) - Financial Data
- [pandas](https://pandas.pydata.org/) - Data Manipulation

### Tutoriály:
- [Time Series ML](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [Feature Engineering](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 📞 Podpora

### Máte problém?

1. **Zkontrolujte [docs/WORKFLOW.md](docs/WORKFLOW.md) sekci "Troubleshooting"**
2. Zkontrolujte instalaci: `pip list | grep -E "pandas|numpy|scikit-learn|yfinance"`
3. Ověřte OHLCV data: `ls ../data_10y/all_sectors_full_10y.csv`

### Našli jste bug?

- Popište problém (chybová hláška, kroky k reprodukci)
- Zkontrolujte log výstup skriptů
- Kontaktujte autora (viz README.md)

---

## 🎯 Další Kroky

Po dokončení základního pipeline:

1. ✅ **Experimentujte** s hyperparametry
2. ✅ **Analyzujte** feature importance
3. ✅ **Vizualizujte** predikce vs. skutečnost
4. ✅ **Rozšiřte** o další sektory nebo metriky
5. ✅ **Sdílejte** své výsledky!

---

## 📜 Licence

Tento projekt je určen pro **vzdělávací účely**.  
Používání pro reálné investiční rozhodnutí je **na vlastní riziko**.

---

## 🙏 Poděkování

**Vytvořeno pomocí:**
- GitHub Copilot
- scikit-learn Community
- yfinance Contributors

---

## 📅 Verze

**Verze:** 2.0.0  
**Datum:** 31. prosince 2025  
**Status:** Notebook Workflow ✅

---

**🚀 Hodně štěstí s vaším projektem!**

*Vytvořeno s ❤️ pro predikci akcií pomocí ML*
