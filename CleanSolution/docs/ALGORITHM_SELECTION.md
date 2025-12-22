# Volba Algoritmů - Zdůvodnění

## Executive Summary pro Diplomovou Práci

Tento dokument poskytuje stručné, ale rigorózní zdůvodnění volby algoritmů použitých v projektu.

---

## 1. Random Forest pro Imputaci Fundamentálních Dat

### Proč Random Forest?

| Kritérium | Random Forest | Alternativy | Výhoda RF |
|-----------|---------------|-------------|-----------|
| **Nelineární vztahy** | ✅ Zachycuje | Linear Regression: Ne | RF modeluje komplexní interakce OHLCV → Fundamenty |
| **Multi-output** | ✅ Přirozená podpora | Většina: Wrapper nutný | Jeden model pro 14 výstupů |
| **Robustnost** | ✅ Ensemble stabilita | Single tree: Vysoká variance | Agregace snižuje chybu |
| **Outliers** | ✅ Inherentně robustní | NN: Citlivé | Finanční data mají outliers |
| **Interpretabilita** | ✅ Feature importance | Deep Learning: Black-box | Důležité pro akademickou práci |
| **Data requirements** | ✅ Funguje s menším datasetem | NN: Vyžaduje více dat | Máme pouze 1.5 roku fundamentů |
| **Tuning complexity** | ✅ Out-of-box dobré výsledky | XGBoost: Citlivý na tuning | Menší riziko špatného nastavení |

### Teoretické Zdůvodnění

Random Forest kombinuje **bagging** (bootstrap aggregating) a **feature randomness**:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

**Variance reduction:**
$$Var(\bar{T}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2 \xrightarrow{B \to \infty} \rho \sigma^2$$

Feature randomness snižuje korelaci $\rho$ mezi stromy → nižší celková variance.

### Proč NE Neuronové Sítě?

1. **Nedostatek dat:** ~1,800 vzorků (18 měsíců × 100 tickerů) je nedostatečné pro NN
2. **Overfitting risk:** NN s mnoha parametry přetrénuje na malém datasetu
3. **Interpretabilita:** NN jsou black-box, RF poskytuje feature importance
4. **Výpočetní náročnost:** RF trénuje v minutách, NN vyžaduje GPU

---

## 2. Random Forest Classifier pro Predikci Cenových Pohybů

### Proč Klasifikace místo Regrese?

| Aspekt | Regrese | Klasifikace |
|--------|---------|-------------|
| **Praktické využití** | "Cena bude $152.30" → Co s tím? | "BUY/HOLD/SELL" → Přímý signál |
| **Robustnost** | NVIDIA +200% zkresluje MAE | Extrémní pohyby jsou stále "UP" |
| **Evaluace** | R² = 0.8 může stále predikovat špatný směr | Precision = 70% → jasná interpretace |
| **Trading aplikace** | Vyžaduje další zpracování | Přímo mapuje na akce |
| **Akademická hodnota** | "MAE = $12" je abstraktní | "Hit rate 68%" je srozumitelné |

### Proč Ternární (3 třídy) místo Binární?

**Binární (UP/DOWN):**
- Problém: +0.5% a +15% jsou obě "UP", ale velmi odlišné
- Problém: Transakční náklady ~1-2% → malé pohyby nejsou profitabilní

**Ternární (DOWN/HOLD/UP):**
```
DOWN: return < -3%   → SELL signál
HOLD: -3% ≤ return ≤ +3%  → Žádná akce (transakční náklady by převýšily zisk)
UP:   return > +3%   → BUY signál
```

Threshold 3% odpovídá:
- Bid-ask spread: ~0.1-1%
- Broker fees: ~0.1-0.5%
- Slippage: ~0.5-1%
- **Risk premium: ~1-2%**

### Proč Random Forest místo jiných klasifikátorů?

| Model | Výhoda | Nevýhoda | Vhodnost |
|-------|--------|----------|----------|
| **Logistic Regression** | Interpretabilní | Pouze lineární hranice | ⭐⭐⭐ |
| **Random Forest** | Nelineární, robustní, feature importance | - | ⭐⭐⭐⭐⭐ |
| **XGBoost** | Vysoká přesnost | Overfitting risk, složitý tuning | ⭐⭐⭐⭐ |
| **SVM** | Dobré pro high-dim | Pomalý, citlivý na škálu | ⭐⭐ |
| **Neural Network** | Flexibilní | Vyžaduje více dat, black-box | ⭐⭐ |

**Zvolený model: Random Forest Classifier**

Důvody:
1. **Konzistence** s imputační fází (stejná rodina algoritmů)
2. **Probability outputs** pro confidence-based filtering
3. **Class balancing** pomocí `class_weight='balanced'`
4. **Robustnost** vůči třídní nevyváženosti

---

## 3. Proč NE Ridge Regression (původní přístup)?

### Původní Design

```
Fundamenty → Ridge Regression → log(Price_next_month)
```

### Problémy

1. **Nepraktická interpretace:**
   - "MAE = $12" → Je to dobré nebo špatné?
   - Model může mít vysoké R², ale stále predikovat špatný směr

2. **Citlivost na outliers:**
   - NVIDIA: cena se zčtyřnásobila za 2 roky
   - Tyto extrémní pohyby dominují MAE/RMSE

3. **Linearní předpoklad:**
   - Ridge předpokládá lineární vztah Fundamenty → Cena
   - Realita je komplexnější (momentum, sentiment, atd.)

4. **Žádný přímý trading signál:**
   - Predikce ceny vyžaduje další rozhodovací pravidla
   - Klasifikace přímo říká "kup/drž/prodej"

### Nový Design

```
Fundamenty + Technické → Random Forest Classifier → {DOWN, HOLD, UP}
```

**Výhody:**
- Přímé trading signály
- Robustnost vůči outlierům
- Jasné evaluační metriky (Precision, Recall, F1)
- Probability outputs pro risk management

---

## 4. Kombinace Modelů - Architektura Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    FÁZE 1-2: DATA COLLECTION                │
│  OHLCV (10 let) + Fundamenty (1.5 roku)                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            FÁZE 3: RANDOM FOREST REGRESSOR                  │
│  Úkol: Imputace chybějících fundamentálních dat             │
│  Proč RF: Nelineární, multi-output, robustní                │
│  Input: OHLCV + Technical (18 features)                     │
│  Output: Fundamenty (14 metrik)                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               FÁZE 4: DATA COMPLETION                       │
│  Spojení predikovaných (2015-2024) + reálných (2024-2025)   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           FÁZE 5: RANDOM FOREST CLASSIFIER                  │
│  Úkol: Klasifikace cenových pohybů                          │
│  Proč RF: Konzistence, robustnost, interpretabilita         │
│  Input: Fundamenty + Technical (19 features)                │
│  Output: {DOWN, HOLD, UP}                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRADING SIGNÁLY                          │
│  DOWN (prob > 0.6) → SELL/SHORT                             │
│  HOLD → Žádná akce                                          │
│  UP (prob > 0.6) → BUY/LONG                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Srovnání s Literaturou

### Akademické Reference

| Studie | Metoda | Dataset | Výsledek |
|--------|--------|---------|----------|
| Gu et al. (2020) | Neural Networks, Random Forest | US equities | RF competitive s NN |
| Krauss et al. (2017) | Random Forest, Deep NN | S&P 500 | RF: Sharpe ~0.5 |
| Leung et al. (2000) | Classification vs Regression | Forex | Klasifikace stabilnější |

### Klíčové Zjištění

> "Tree-based methods such as random forests achieve competitive performance compared to neural networks while offering better interpretability and requiring less computational resources."
> — Gu, Kelly & Xiu (2020), Review of Financial Studies

---

## 6. Shrnutí Voleb

| Fáze | Algoritmus | Hlavní Důvod |
|------|------------|--------------|
| **Imputace** | Random Forest Regressor | Nelineární multi-output, robustnost, interpretabilita |
| **Klasifikace** | Random Forest Classifier | Konzistence, praktické signály, probability outputs |
| **NE Ridge Regression** | - | Nepraktická interpretace, outlier sensitivity |
| **NE Neural Networks** | - | Nedostatek dat, black-box, výpočetní náročnost |

---

## Reference

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
2. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies*, 33(5), 2223-2273.
3. Krauss, C., Do, X. A., & Huck, N. (2017). Deep Neural Networks, Gradient-Boosted Trees, Random Forests: Statistical Arbitrage on the S&P 500. *European Journal of Operational Research*, 259(2), 689-702.
