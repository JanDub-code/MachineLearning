# 🎯 CleanSolution - Klasifikace Cenových Pohybů Akcií pomocí ML

## Diplomová Práce - Ing. Informatika

**Autor:** Bc. Jan Dub  
**Datum:** Prosinec 2025

---

## 📖 O Projektu

Tento projekt implementuje **hybridní přístup k predikci směru cenových pohybů akcií** kombinací:

1. **Random Forest Regressor** - pro imputaci chybějících historických fundamentálních dat
2. **Random Forest Classifier** - pro klasifikaci budoucích cenových pohybů (DOWN/HOLD/UP)

### 🔑 Klíčová Inovace

Projekt řeší fundamentální problém v kvantitativních financích: **neúplnost historických fundamentálních dat**. Zatímco cenová data (OHLCV) jsou dostupná za 10+ let, fundamentální metriky (P/E, ROE, atd.) jsou typicky dostupné pouze za 1-2 roky.

**Řešení:**
1. Natrénovat ML model na období, kde máme kompletní data (OHLCV + Fundamenty)
2. Použít tento model k rekonstrukci chybějících fundamentálních dat
3. Klasifikovat budoucí cenové pohyby na základě kompletního datasetu

### 🎯 Klasifikační Přístup

| Aspekt | Klasifikace |
|--------|-------------|
| **Output** | Třída pohybu (DOWN/HOLD/UP) |
| **Interpretace** | "Cena vzroste/klesne o >3%" |
| **Praktické využití** | Přímé trading signály |
| **Robustnost** | Robustní vůči outliers |
| **Evaluace** | Accuracy, Precision, Recall, F1 |

**Definice tříd (±3% threshold):**
- **DOWN (0):** Měsíční výnos < -3%
- **HOLD (1):** Měsíční výnos mezi -3% a +3%
- **UP (2):** Měsíční výnos > +3%

Threshold 3% odpovídá minimálnímu profitabilnímu pohybu po započtení transakčních nákladů.

---

## 📂 Struktura Projektu

```
CleanSolution/
│
├── 📄 README.md                              # Tento soubor
├── 📄 QUICKSTART.md                          # Rychlý start pro Colab
├── 📄 requirements.txt                       # Python závislosti
│
├── 📂 notebooks/                             # 🎯 HLAVNÍ - Jupyter Notebooky pro Google Colab
│   ├── 01_Data_Collection.ipynb             # Sběr dat (teoretický úvod + stahování)
│   ├── 02_Train_Fundamental_Predictor.ipynb # RF Regressor pro imputaci
│   ├── 03_Complete_Historical_Data.ipynb    # Doplnění chybějících dat
│   ├── 04_Train_Price_Classifier.ipynb      # RF Classifier pro klasifikaci
│   ├── 05_Hyperparameter_Tuning.ipynb       # Grid Search optimalizace
│   └── 06_Final_Evaluation.ipynb            # Kompletní evaluace + vizualizace
│
├── 📂 scripts/                               # Pomocné Python skripty (pouze API)
│   ├── 0_download_prices.py                 # Stažení OHLCV dat z yfinance
│   └── 1_download_fundamentals.py           # Stažení fundamentálních dat
│
├── 📂 data/                                  # Datové soubory
│   ├── ohlcv_10y/                           # OHLCV data (10 let)
│   ├── fundamentals/                        # Fundamentální data (1.5 roku)
│   ├── complete/                            # Kompletní dataset
│   └── figures/                             # Generované grafy
│
├── 📂 data_10y/                              # Vstupní data (10 let historie)
│   ├── Technology_full_10y.csv
│   ├── Consumer_full_10y.csv
│   └── Industrials_full_10y.csv
│
├── 📂 models/                                # Uložené modely
│   ├── fundamental_predictor.pkl            # RF Regressor
│   ├── fundamental_predictor_tuned.pkl      # Optimalizovaný RF Regressor
│   ├── rf_classifier_all_sectors.pkl        # RF Classifier
│   ├── price_classifier_tuned.pkl           # Optimalizovaný RF Classifier
│   └── optimal_hyperparameters.json         # Nejlepší parametry
│
├── 📂 docs/                                  # Dokumentace
│   ├── METHODOLOGY.md                       # Detailní metodologie
│   ├── MATHEMATICAL_FOUNDATIONS.md          # Matematické základy
│   ├── ALGORITHM_SELECTION.md               # Výběr algoritmů
│   ├── WORKFLOW.md                          # Průvodce workflow
│   └── SUMMARY.md                           # Shrnutí projektu
│
└── 📂 archive/                               # Archivované staré skripty
    ├── 2_train_fundamental_predictor.py
    ├── 3_complete_historical_data.py
    └── 4_train_price_predictor.py
```

---

## 🚀 Rychlý Start (Google Colab)

### Doporučený Workflow

Všechny ML operace jsou implementovány v **Jupyter Noteboocích** optimalizovaných pro Google Colab.

**Postup:**

1. **Nahrajte data do Google Drive:**
   ```
   Google Drive/
   └── MachineLearning/
       └── data_10y/
           ├── Technology_full_10y.csv
           ├── Consumer_full_10y.csv
           └── Industrials_full_10y.csv
   ```

2. **Otevřete notebooky v Google Colab (v pořadí):**

   | # | Notebook | Popis | Doba |
   |---|----------|-------|------|
   | 1 | `01_Data_Collection.ipynb` | Teoretický úvod, stahování dat | ~10 min |
   | 2 | `02_Train_Fundamental_Predictor.ipynb` | Trénink RF Regressor | ~5 min |
   | 3 | `03_Complete_Historical_Data.ipynb` | Imputace chybějících dat | ~2 min |
   | 4 | `04_Train_Price_Classifier.ipynb` | Trénink RF Classifier | ~5 min |
   | 5 | `05_Hyperparameter_Tuning.ipynb` | Optimalizace hyperparametrů | ~15 min |
   | 6 | `06_Final_Evaluation.ipynb` | Evaluace + grafy pro DP | ~5 min |

3. **Každý notebook obsahuje:**
   - 📚 Teoretický úvod s akademickými vysvětleními
   - 📊 Matematické vzorce (LaTeX)
   - 💻 Spustitelný Python kód
   - 📈 Vizualizace výsledků
   - 💾 Automatické ukládání do Google Drive

---

## 📊 Metodologie

### Fáze 1: Sběr Dat
- **OHLCV data:** 10 let měsíční historie (2015-2025) pro 150 S&P 500 akcií
- **Technické indikátory:** RSI, MACD, SMA, EMA, volatilita
- **Fundamentální data:** 11 metrik (P/E, ROE, Debt/Equity, atd.)
- **Sektory:** Technology, Consumer Discretionary, Industrials

### Fáze 2: Imputace Dat (Random Forest Regressor)
- **Problém:** Fundamentální data dostupná pouze za 1.5 roku
- **Řešení:** Multi-output RF natrénovaný na vztahu OHLCV → Fundamenty
- **Výstup:** Kompletní dataset 2015-2025

### Fáze 3: Klasifikace (Random Forest Classifier)
- **Input:** OHLCV + Technické + Fundamentální features
- **Output:** Ternární klasifikace (DOWN/HOLD/UP)
- **Validace:** Chronologický split + TimeSeriesSplit

### Fáze 4: Evaluace
- Confusion Matrix, ROC křivky
- Per-sector analýza
- Backtesting obchodní strategie
- Feature Importance

---

## 📈 Výsledky

### Klasifikace

| Metrika | Hodnota |
|---------|---------|
| Accuracy | ~55-60% |
| F1-Score (weighted) | ~0.55-0.60 |
| Win Rate (backtest) | ~55-60% |

### Klíčová Zjištění

- ✅ Random Forest poskytuje robustní klasifikaci
- ✅ 3% threshold efektivně pokrývá transakční náklady
- ✅ Fundamentální data zlepšují predikci
- ✅ TimeSeriesSplit je kritický pro validní evaluaci
- ✅ Balanced class weights zlepšují recall minoritních tříd

---

## 📚 Dokumentace

| Dokument | Obsah |
|----------|-------|
| [METHODOLOGY.md](docs/METHODOLOGY.md) | Kompletní metodologie projektu |
| [MATHEMATICAL_FOUNDATIONS.md](docs/MATHEMATICAL_FOUNDATIONS.md) | Matematické základy algoritmů |
| [ALGORITHM_SELECTION.md](docs/ALGORITHM_SELECTION.md) | Zdůvodnění výběru algoritmů |
| [WORKFLOW.md](docs/WORKFLOW.md) | Detailní průvodce workflow |
| [QUICKSTART.md](QUICKSTART.md) | Rychlý start |

---

## 🛠️ Lokální Spuštění (Volitelné)

Pokud preferujete lokální prostředí místo Google Colab:

```bash
# 1. Klonujte repozitář
git clone https://github.com/user/MachineLearning.git
cd MachineLearning/CleanSolution

# 2. Vytvořte virtuální prostředí
python -m venv venv
source venv/bin/activate  # Linux/Mac
# nebo: .\venv\Scripts\activate  # Windows

# 3. Nainstalujte závislosti
pip install -r requirements.txt

# 4. (Volitelné) Stáhněte data
python scripts/0_download_prices.py
python scripts/1_download_fundamentals.py

# 5. Spusťte Jupyter
jupyter lab
```

---

## 📜 Licence

MIT License - viz [LICENSE](../LICENSE)

---

## 👤 Autor

**Bc. Jan Dub**  
Diplomová práce - Ing. Informatika  
Prosinec 2025
