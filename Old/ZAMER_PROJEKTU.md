# 🎯 ZÁMĚR PROJEKTU - Predikce Cen Akcií s ML

**Datum:** 22. října 2025  
**Cíl:** Využít AI/ML pro doplnění historických fundamentálních dat a následně predikovat ceny akcií pomocí multifaktorové lineární regrese

---

## 🔍 HLAVNÍ MYŠLENKA

### Problém:
- ✅ Máme **10 let OHLCV dat** (2015-2025) - kompletní
- ⚠️ Máme **pouze 1.5 roku fundamentálních dat** (2024-2025) - neúplné

### Řešení:
1. **FÁZE 1-2:** Sesbírat fundamentální data za 1.5 roku (P/E, P/B, ROE, atd.)
2. **FÁZE 3:** Natrénovat **AI model** který dokáže predikovat fundamentální hodnoty z OHLCV + technických indikátorů
3. **FÁZE 4:** Použít natrénovaný AI model pro **doplnění chybějících 8.5 let fundamentů** (2015-2024)
4. **FÁZE 5:** S kompletním datasetem (10 let OHLCV + fundamenty) natrénovat **multifaktorovou lineární regresi** pro predikci budoucích cen

### Výsledek:
📊 **Kompletní dataset (10 let)** → 🤖 **Lineární regrese model** → 💰 **Predikce budoucí ceny ze zadaných fundamentů**

---

## 📋 FÁZE PROJEKTU

### **FÁZE 1: Sběr OHLCV Dat** ✅ (v průběhu)

#### 1.1 Historická Cenová Data (10 let: 2015-2025)
- ✅ **OHLCV data** (Open, High, Low, Close, Volume) - měsíční agregace denních dat
- ✅ **Corporate Actions** (Dividendy, Stock Splits)
- ✅ **Technické indikátory** (RSI, MACD, SMA, EMA, Volatilita, Returns)

**Výstup:** `data_10y/Technology_full_10y.csv` (120 měsíců × 50 firem × features)

---

### **FÁZE 2: Sběr Fundamentálních Dat** ⏳

#### 2.1 Fundamentální Metriky (1.5 roku: 2024-2025)
**Zdroj:** yfinance quarterly/annual data, financial APIs

**Features k získání:**
- **Valuační ratios:**
  - P/E ratio (Price-to-Earnings)
  - P/B ratio (Price-to-Book)
  - P/S ratio (Price-to-Sales)
  - EV/EBITDA (Enterprise Value to EBITDA)
  - PEG ratio (P/E to Growth)

- **Profitabilita:**
  - ROE (Return on Equity)
  - ROA (Return on Assets)
  - Profit Margin (Čistá zisková marže)
  - Operating Margin
  - Gross Margin

- **Finanční zdraví:**
  - Debt-to-Equity (Zadluženost)
  - Current Ratio (Likvidita)
  - Quick Ratio
  - Cash Ratio

- **Růst:**
  - Revenue Growth YoY (% meziroční růst tržeb)
  - Earnings Growth YoY
  - Book Value Growth

**Výstup:** `data_fundamentals/Technology_fundamentals_1.5y.csv` (~18 měsíců × 50 firem × 15 fundamentů)

---

### **FÁZE 3: AI Model pro Predikci Fundamentů** 🤖 ⏳

#### 3.1 Cíl
**Natrénovat AI model který dokáže predikovat fundamentální hodnoty z OHLCV a technických indikátorů**

#### 3.2 Training Dataset (1.5 roku: 2024-2025)
```
Input Features (X):
- open, high, low, close, volume
- volatility, returns
- RSI_14, MACD, MACD_signal, MACD_hist
- SMA_3, SMA_6, SMA_12
- EMA_3, EMA_6, EMA_12
- volume_change
- dividends, split_occurred
- sector (category)

Target Variables (y) - každý fundament je samostatný target:
- P/E ratio
- P/B ratio  
- P/S ratio
- EV/EBITDA
- ROE, ROA
- Profit_Margin
- Debt_to_Equity
- Revenue_Growth_YoY
... (15 fundamentů)
```

#### 3.3 Model Architecture
**Multi-output Regression** - jeden model pro všechny fundamenty současně

**Možné algoritmy:**
1. **Random Forest Regressor** (doporučeno)
   - Zvládá non-linearity
   - Odolný vůči outliers
   - Feature importance
   
2. **Gradient Boosting (XGBoost/LightGBM)**
   - Vysoká přesnost
   - Zvládá missing values
   
3. **Neural Network (Multi-output)**
   - Komplexní vzorce
   - Potřebuje více dat

**Výběr:** Random Forest (robustní, interpretovatelný)

#### 3.4 Training Process
```python
# 1. Příprava dat
X_train = ohlcv_technical_data[2024-2025]  # ~18 měsíců × 150 firem
y_train = fundamental_data[2024-2025]       # 15 fundamentů

# 2. Train/validation split
X_train, X_val = train_test_split(80/20)

# 3. Trénování multi-output modelu
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )
)

model.fit(X_train, y_train)

# 4. Validace
predictions = model.predict(X_val)
mae_per_fundamental = mean_absolute_error(y_val, predictions, multioutput='raw_values')
```

#### 3.5 Očekávané Výsledky
```
Metriky úspěchu (na validačních datech):
- P/E ratio: MAE < 3.0 (±3 bodů)
- P/B ratio: MAE < 0.5
- ROE: MAE < 5% 
- Revenue Growth: MAE < 10%

Celkový průměrný MAE: < 15% relativní chyba
```

#### 3.6 Feature Importance
Po natrénování zjistíme:
- Které technické indikátory nejlépe predikují které fundamenty?
- Je RSI dobrý prediktor pro ROE?
- Predikuje Volume změnu Revenue Growth?

**Výstup FÁZE 3:** 
- ✅ Natrénovaný AI model `fundamental_predictor.pkl`
- ✅ Feature importance analysis
- ✅ Validation metrics (MAE, RMSE, R²)

---

### **FÁZE 4: Doplnění Historických Fundamentů (2015-2024)** 🔮 ⏳

#### 4.1 Cíl
**Použít natrénovaný AI model k predikci chybějících 8.5 let fundamentálních dat**

#### 4.2 Process
```python
# 1. Načíst kompletní OHLCV data (2015-2025)
full_ohlcv = pd.read_csv('data_10y/all_sectors_full_10y.csv')

# 2. Filtrovat pouze období bez fundamentů (2015-2024)
historical_data = full_ohlcv[full_ohlcv['date'] < '2024-01-01']

# 3. Připravit features (stejné jako při trénování)
X_historical = historical_data[feature_columns]

# 4. Predikovat fundamenty
predicted_fundamentals = model.predict(X_historical)

# 5. Vytvořit kompletní dataset
historical_data['P/E'] = predicted_fundamentals[:, 0]
historical_data['P/B'] = predicted_fundamentals[:, 1]
historical_data['ROE'] = predicted_fundamentals[:, 2]
# ... všech 15 fundamentů

# 6. Spojit s reálnými fundamenty (2024-2025)
complete_dataset = pd.concat([
    historical_data,  # 2015-2024 s predikovanými fundamenty
    real_data_2024_2025  # 2024-2025 s reálnými fundamenty
])
```

#### 4.3 Validace Predikovaných Fundamentů
**Cross-check s reálnými hodnotami kde jsou dostupné:**
```python
# Test: Predikuj 2024 data a porovnej s reálnými
X_2024 = ohlcv_data[2024]
y_pred_2024 = model.predict(X_2024)
y_real_2024 = real_fundamentals[2024]

mae = mean_absolute_error(y_real_2024, y_pred_2024)
# Očekáváme podobný MAE jako na validaci (~15%)
```

#### 4.4 Výstup
```
data_10y_complete/
├── Technology_complete_10y.csv  (120 měsíců × 50 firem × (OHLCV + Tech + 15 Fundamentů))
├── Consumer_complete_10y.csv
├── Industrials_complete_10y.csv
└── all_sectors_complete_10y.csv

Struktura řádku:
date | ticker | sector | open | high | low | close | volume | 
volatility | returns | rsi_14 | macd | ... | 
P/E | P/B | P/S | ROE | ROA | ... | source (real/predicted)
```

**Sloupec `source`:**
- `real` - fundamenty z 2024-2025 (skutečná data)
- `predicted` - fundamenty z 2015-2024 (AI predikce)

**Výstup FÁZE 4:**
- ✅ Kompletní 10letý dataset s OHLCV + Technické + **Fundamenty (real + predicted)**
- ✅ ~18,000 řádků (150 firem × 120 měsíců)
- ✅ Připraven pro multifaktorovou lineární regresi

---

### **FÁZE 5: Multifaktorová Lineární Regrese - Predikce Ceny** 💰 ⏳

#### 5.1 Cíl
**Natrénovat lineární regresi která predikuje budoucí cenu akcie ze zadaných fundamentálních hodnot**

#### 5.2 Dataset pro Training
**Použijeme kompletní 10letý dataset z FÁZE 4**

```python
Input Features (X):
# Fundamentální faktory (hlavní prediktory)
- P/E ratio
- P/B ratio
- P/S ratio
- EV/EBITDA
- ROE, ROA
- Profit_Margin
- Debt_to_Equity
- Revenue_Growth_YoY
- Operating_Margin
- Current_Ratio

# Technické faktory (podpůrné)
- volatility
- RSI_14
- MACD
- volume_change

# Sektorová příslušnost
- sector (one-hot encoded)

Target (y):
- log_price_next_month (log transformovaná cena za 1 měsíc)
```

#### 5.3 Model Architecture
**Multifaktorová Lineární Regrese**

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# Standardizace features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ridge Regression (L2 regularizace pro stabilitu)
model = Ridge(alpha=1.0)
model.fit(X_scaled, y)

# Koeficienty ukazují důležitost každého faktoru
coefficients = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_,
    'importance': abs(model.coef_)
}).sort_values('importance', ascending=False)
```

#### 5.4 Training Strategy
**Po sektorech + Celkový model**

```python
# 1. Model pro každý sektor samostatně
for sector in ['Technology', 'Consumer', 'Industrials']:
    sector_data = complete_data[complete_data['sector'] == sector]
    
    # Train/test split (80/20)
    X_train, X_test = train_test_split(sector_data, test_size=0.2)
    
    # Trénování
    model_sector = Ridge(alpha=1.0)
    model_sector.fit(X_train, y_train)
    
    # Evaluace
    predictions = model_sector.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    # Uložení
    joblib.dump(model_sector, f'models/{sector}_price_predictor.pkl')

# 2. Globální model (všechny sektory)
model_global = Ridge(alpha=1.0)
model_global.fit(X_all, y_all)
```

#### 5.5 Použití Modelu - Predikce Budoucí Ceny

**Scénář:** Chci predikovat cenu AAPL za 1 měsíc

```python
# 1. Zadám fundamentální hodnoty (aktuální nebo očekávané)
input_data = {
    'P/E': 28.5,
    'P/B': 40.2,
    'P/S': 7.8,
    'ROE': 0.45,
    'Revenue_Growth_YoY': 0.12,
    'Debt_to_Equity': 1.5,
    'volatility': 0.015,
    'RSI_14': 62.0,
    'sector': 'Technology'
}

# 2. Předzpracování (stejné jako při trénování)
X_input = pd.DataFrame([input_data])
X_input = scaler.transform(X_input)

# 3. Predikce
log_price_pred = model.predict(X_input)[0]
predicted_price = np.exp(log_price_pred)

print(f"Predikovaná cena AAPL: ${predicted_price:.2f}")
```

#### 5.6 Feature Importance - Co Ovlivňuje Cenu?

```python
# Analýza koeficientů
top_features = coefficients.head(10)

Příklad výsledku:
┌────────────────────────┬─────────────┬──────────────┐
│ Feature                │ Coefficient │ Importance   │
├────────────────────────┼─────────────┼──────────────┤
│ P/E ratio              │   0.342     │   0.342      │  ← Nejvíce ovlivňuje
│ Revenue_Growth_YoY     │   0.287     │   0.287      │
│ ROE                    │   0.215     │   0.215      │
│ P/B ratio              │   0.198     │   0.198      │
│ Profit_Margin          │   0.156     │   0.156      │
│ RSI_14                 │  -0.089     │   0.089      │  ← Negativní korelace
│ Debt_to_Equity         │  -0.134     │   0.134      │
│ volatility             │  -0.045     │   0.045      │
└────────────────────────┴─────────────┴──────────────┘
```

**Interpretace:**
- ✅ **P/E ratio má největší vliv** - vyšší P/E → vyšší cena (growth premium)
- ✅ **Revenue Growth** - růst tržeb zvyšuje cenu
- ⚠️ **Vysoká volatilita snižuje cenu** - investoři se vyhýbají risku
- ⚠️ **Vysoký debt snižuje cenu** - zadlužené firmy jsou rizikovější

#### 5.7 Evaluace Modelu

**Metriky:**
```python
# Celkový dataset (10 let)
MAE:  $12.50  (průměrná absolutní chyba)
RMSE: $18.30  (root mean squared error)
R²:   0.78    (78% variance vysvětleno)

# Po sektorech
Technology MAE:  $15.20
Consumer MAE:    $10.80
Industrials MAE: $11.30
```

**Srovnání s baseline:**
```
Baseline (průměrná cena): MAE ~$45
Náš model: MAE ~$12.50
→ Zlepšení o 72%!
```

#### 5.8 Výstup FÁZE 5

**Natrénované modely:**
```
models/
├── Technology_price_predictor.pkl
├── Consumer_price_predictor.pkl
├── Industrials_price_predictor.pkl
├── Global_price_predictor.pkl
└── feature_scaler.pkl
```

**Analýzy:**
```
analysis/
├── feature_importance_by_sector.csv
├── model_coefficients.csv
├── predictions_vs_actual.csv
└── sector_comparison.png
```

**Produkční API:**
```python
def predict_stock_price(ticker, fundamentals):
    """
    Predikuje budoucí cenu akcie ze zadaných fundamentů.
    
    Args:
        ticker: str - symbol akcie (např. 'AAPL')
        fundamentals: dict - fundamentální metriky
            {
                'P/E': float,
                'P/B': float,
                'ROE': float,
                ...
            }
    
    Returns:
        {
            'predicted_price': float,
            'confidence_interval_95': (lower, upper),
            'key_drivers': [(feature, impact), ...],
            'sector_comparison': str  # 'overvalued'/'undervalued'/'fair'
        }
    """
```

---

## 🎯 KONEČNÝ CÍL

**Vytvořit komplexní ML systém který:**

### 1️⃣ **Využívá AI pro Doplnění Historických Dat**
- ✅ Sbírá fundamentální data za 1.5 roku (reálná data)
- ✅ Trénuje Random Forest model na predikci fundamentů z OHLCV
- ✅ Doplňuje chybějících 8.5 let fundamentálních hodnot AI predikcí
- ✅ Vytváří kompletní 10letý dataset

### 2️⃣ **Multifaktorová Lineární Regrese**
- ✅ Využívá 10 let kompletních dat (OHLCV + Technical + Fundamentals)
- ✅ Trénuje lineární regresi pro predikci ceny z fundamentálních faktorů
- ✅ Identifikuje klíčové faktory ovlivňující cenu (P/E, ROE, Growth)
- ✅ Funguje napříč sektory (Technology, Consumer, Industrials)

### 3️⃣ **Produkční Použití**
- ✅ **Input:** Zadám fundamentální hodnoty firmy (P/E, P/B, ROE, ...)
- ✅ **Output:** Predikovaná budoucí cena + confidence interval
- ✅ **Analýza:** Které faktory mají největší vliv na cenu?
- ✅ **Benchmarking:** Je firma nadhodnocená/podhodnocená vs. sektor?

---

## 🔬 INOVATIVNÍ PŘÍSTUP

### Proč je to unikátní?

**Kombinace AI + Klasické Lineární Regrese:**

1. **AI (Random Forest)** → Doplní historická data  
   ├─ Učí se vzorce mezi OHLCV a fundamenty  
   └─ Vytvoří kompletní dataset (10 let)

2. **Lineární Regrese** → Predikuje cenu z fundamentů  
   ├─ Interpretovatelné koeficienty  
   ├─ Jasné vztahy (P/E ↑ → Cena ↑)  
   └─ Použitelné pro investiční rozhodování

**Výhody oproti tradičním přístupům:**
- ❌ **Tradiční:** Pouze 1.5 roku dat → málo vzorků, přetrénování
- ✅ **Náš přístup:** 10 let dat → robustní model, více vzorců

---

## 📊 OČEKÁVANÉ VÝSLEDKY

### Metriky Úspěchu

#### FÁZE 3 (AI Predikce Fundamentů):
```
Baseline: MAE ~30% (náhodný odhad)
Cíl: MAE <15% (relativní chyba fundamentů)

Příklad:
- Reálné P/E: 28.5
- Predikované P/E: 26.2
- Chyba: 8% ✅
```

#### FÁZE 5 (Lineární Regrese - Cena):
```
Baseline: MAE ~$45 (průměrná cena sektoru)
Cíl: MAE <$15 (predikce z fundamentů)
R²: >0.75 (vysvětleno 75% variance)

Příklad:
- Reálná cena AAPL: $185.20
- Predikovaná cena: $178.50
- Chyba: $6.70 (3.6%) ✅
```

### Srovnání Přístupů

| Přístup | Data | Features | Očekávaný MAE | Interpretabilita |
|---------|------|----------|---------------|------------------|
| Baseline (průměr sektoru) | N/A | N/A | ~$45 | ❌ |
| Jen technické indikátory | 10 let | OHLCV + RSI + MACD | ~$25 | ⚠️ |
| **NÁŠ: AI + Fundamenty** | 10 let | OHLCV + Tech + 15 Fundamentů | ~$12-15 | ✅ |
| Neural Network (black box) | 10 let | Vše | ~$10 | ❌❌ |

**Závěr:** Náš přístup má nejlepší poměr **přesnost/interpretabilita**

---

## 🚀 IMPLEMENTAČNÍ PLÁN

### ✅ Týden 1: Příprava OHLCV Dat (HOTOVO)
- [x] Stáhnout denní OHLCV data (10 let)
- [x] Agregovat na měsíční
- [x] Vypočítat technické indikátory (RSI, MACD, SMA, EMA)
- [x] Uložit do `data_10y/`

### ⏳ Týden 2: Sběr Fundamentálních Dat
- [ ] Stáhnout quarterly fundamentals (2024-2025)
- [ ] Extrahovat P/E, P/B, P/S, ROE, ROA, atd.
- [ ] Čištění a validace
- [ ] Mergovat s OHLCV daty
- [ ] Uložit do `data_fundamentals/`

### ⏳ Týden 3: AI Model - Predikce Fundamentů
- [ ] Příprava train/validation split
- [ ] Trénování Random Forest multi-output regressoru
- [ ] Hyperparameter tuning (grid search)
- [ ] Validace (MAE, RMSE, R²)
- [ ] Feature importance analýza
- [ ] Uložit model: `models/fundamental_predictor.pkl`

### ⏳ Týden 4: Doplnění Historických Dat
- [ ] Aplikovat AI model na 2015-2024 data
- [ ] Predikovat 15 fundamentálních metrik
- [ ] Spojit s reálnými daty (2024-2025)
- [ ] Validace (cross-check kde máme reálná data)
- [ ] Uložit kompletní dataset: `data_10y_complete/`

### ⏳ Týden 5: Lineární Regrese - Predikce Ceny
- [ ] Feature engineering (standardizace, one-hot encoding)
- [ ] Train/test split (80/20)
- [ ] Trénování Ridge Regression (po sektorech)
- [ ] Evaluace (MAE, RMSE, R²)
- [ ] Analýza koeficientů (které fundamenty ovlivňují cenu?)
- [ ] Uložit modely: `models/Technology_price_predictor.pkl`

### ⏳ Týden 6: Evaluace & Produkční API
- [ ] Backtesting na historických datech
- [ ] Vytvoření predikčního API
- [ ] Vizualizace (predictions vs actual, feature importance)
- [ ] Dokumentace
- [ ] Srovnání s baseline modely

---

## ⚠️ RIZIKA A OMEZENÍ

### Datová Omezení
- ❌ **Fundamenty jen 1.5 roku** → AI model může mít nižší přesnost pro starší data
- ❌ **Look-ahead bias** → Musíme zajistit že nepoužíváme budoucí data při trénování
- ❌ **Survivorship bias** → S&P 500 obsahuje jen úspěšné firmy (vypadlé firmy chybí)

### Modelová Omezení
- ⚠️ **AI predikce fundamentů** → Není 100% přesná (očekáváme ~15% chybu)
- ⚠️ **Linearita** → Vztah fundamenty → cena nemusí být lineární
- ⚠️ **External shocks** → COVID, recese, války → těžko predikovatelné z fundamentů

### Řešení
1. **Ensemble AI modelů** → Random Forest + XGBoost průměr
2. **Regularizace** → Ridge/Lasso prevence overfittingu
3. **Rolling validation** → Testovat na různých časových úsecích
4. **Confidence intervals** → Bootstrap pro odhad nejistoty
5. **Sektorová segmentace** → Každý sektor má vlastní model

---

## 🔍 KLÍČOVÉ OTÁZKY, NA KTERÉ ODPOVÍME

### Po FÁZI 3 (AI Predikce Fundamentů):
1. ✅ **Lze predikovat fundamenty z OHLCV?** → MAE, R²
2. ✅ **Které technické indikátory nejlépe korelují s fundamenty?** → Feature importance
3. ✅ **Je RSI dobrý prediktor pro ROE?** → Correlation matrix
4. ✅ **Funguje to stejně pro všechny sektory?** → Per-sector MAE

### Po FÁZI 4 (Kompletní Dataset):
1. ✅ **Jsou predikované fundamenty realistické?** → Porovnání s průměry sektoru
2. ✅ **Mění se fundamenty v čase logicky?** → Trend analýza
3. ✅ **Korelace predikovaných vs. reálných?** → Scatter plots

### Po FÁZI 5 (Lineární Regrese):
1. ✅ **Které fundamenty nejvíce ovlivňují cenu?** → Koeficienty
2. ✅ **Je P/E důležitější než ROE?** → Coefficient magnitude
3. ✅ **Funguje model napříč sektory?** → Per-sector MAE comparison
4. ✅ **Lze identifikovat under/overvalued firmy?** → Residual analysis
5. ✅ **Produkční použití?** → Live predictions on new data

---

## 📈 BUSINESS VALUE

### Pro Investory:
- 💰 **Odhad fair value** → Je firma pod/nadhodnocená?
- 📊 **Benchmarking vs. sektor** → Jak firma stojí oproti konkurenci?
- 🔮 **Predikce ceny** → Co se stane když se změní fundamenty?

### Pro Analytiky:
- 🔬 **Feature importance** → Které faktory jsou klíčové pro valuaci?
- 📈 **Historická analýza** → Jak se měnily fundamenty v čase?
- 🎯 **Sektorové rozdíly** → Technology vs. Consumer vs. Industrials

### Pro Data Scientists:
- 🤖 **Hybrid AI + Classical ML** → Kombinace modelů
- 📚 **Metodologie** → Doplnění chybějících historických dat pomocí AI
- 🏆 **Benchmarking** → Srovnání různých přístupů (RF vs. LR vs. NN)

---

**Vytvořeno:** 22. října 2025  
**Status:** FÁZE 1 - Sběr OHLCV dat (80% hotovo)  
**Další krok:** FÁZE 2 - Stáhnout fundamentální data (1.5 roku)  
**Konečný cíl:** Multifaktorová lineární regrese s 10 lety kompletních dat

---

## 📂 STRUKTURA PROJEKTU

```
Strojové učení/
│
├── 📄 ZAMER_PROJEKTU.md                      # Tento dokument
├── 📄 summary.md                             # Průběžné poznámky
│
├── 📂 data_10y/                              # FÁZE 1: OHLCV data (10 let)
│   ├── all_sectors_full_10y.csv             # Všechny sektory
│   ├── Technology_full_10y.csv              # 120 měsíců × 50 firem
│   ├── Consumer_full_10y.csv
│   ├── Industrials_full_10y.csv
│   ├── Technology_tickers.txt
│   ├── Consumer_tickers.txt
│   └── Industrials_tickers.txt
│
├── 📂 data_fundamentals/                     # FÁZE 2: Fundamentální data (1.5 roku)
│   ├── Technology_fundamentals_1.5y.csv     # 18 měsíců × 50 firem × 15 fundamentů
│   ├── Consumer_fundamentals_1.5y.csv
│   ├── Industrials_fundamentals_1.5y.csv
│   └── all_fundamentals_1.5y.csv
│
├── 📂 data_10y_complete/                     # FÁZE 4: Kompletní data (OHLCV + Fundamentals)
│   ├── Technology_complete_10y.csv          # 120 měsíců × 50 firem × všechny features
│   ├── Consumer_complete_10y.csv
│   ├── Industrials_complete_10y.csv
│   └── all_sectors_complete_10y.csv
│
├── 📂 models/                                # Natrénované modely
│   ├── fundamental_predictor.pkl            # FÁZE 3: Random Forest (AI)
│   ├── Technology_price_predictor.pkl       # FÁZE 5: Linear Regression
│   ├── Consumer_price_predictor.pkl
│   ├── Industrials_price_predictor.pkl
│   ├── Global_price_predictor.pkl
│   └── feature_scaler.pkl
│
├── 📂 analysis/                              # Analýzy a grafy
│   ├── feature_importance_fundamentals.csv  # Které tech indikátory predikují fundamenty?
│   ├── feature_importance_price.csv         # Které fundamenty ovlivňují cenu?
│   ├── model_coefficients.csv               # Koeficienty lineární regrese
│   ├── predictions_vs_actual.csv            # Validace predikcí
│   └── sector_comparison.png
│
├── � scripts/                               # Python skripty
│   ├── prepare_10y_data_full.py             # FÁZE 1: Stažení OHLCV (150 firem)
│   ├── prepare_10y_data_test.py             # FÁZE 1: Test (30 firem)
│   ├── download_fundamentals.py             # FÁZE 2: Stažení fundamentů
│   ├── train_fundamental_predictor.py       # FÁZE 3: AI model
│   ├── complete_historical_data.py          # FÁZE 4: Doplnění 8.5 let
│   ├── train_price_predictor.py             # FÁZE 5: Lineární regrese
│   └── sector_linear_pipeline.py            # Starý pipeline (deprecated)
│
└── 📂 out/                                   # Výstupy (starý systém)
    ├── metrics_summary.csv
    └── models/
```
