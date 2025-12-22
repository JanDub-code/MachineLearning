# Matematické a Statistické Základy

## Formální Definice a Důkazy pro Diplomovou Práci

---

## 1. Formalizace Problému Predikce

### 1.1 Notace

| Symbol | Význam |
|--------|--------|
| $\mathcal{D}$ | Dataset $\{(x_i, y_i)\}_{i=1}^{N}$ |
| $x_i \in \mathbb{R}^d$ | Feature vector (d = 19 features) |
| $y_i \in \{0, 1, 2\}$ | Target class (DOWN, HOLD, UP) |
| $f: \mathbb{R}^d \rightarrow \{0, 1, 2\}$ | Klasifikační funkce |
| $\hat{y}_i = f(x_i)$ | Predikovaná třída |
| $P(Y = k | X = x)$ | Podmíněná pravděpodobnost třídy k |
| $r_t$ | Výnos v čase t |
| $\theta$ | Threshold pro klasifikaci |

### 1.2 Definice Klasifikačního Problému

**Definice 1 (Ternární Klasifikace Výnosů):**
Nechť $r_{t+1}$ je měsíční výnos akcie v čase $t+1$. Definujeme klasifikační funkci $g: \mathbb{R} \rightarrow \{0, 1, 2\}$:

$$
g(r_{t+1}; \theta) = \begin{cases}
0 & \text{pokud } r_{t+1} < -\theta \\
1 & \text{pokud } -\theta \leq r_{t+1} \leq \theta \\
2 & \text{pokud } r_{t+1} > \theta
\end{cases}
$$

**Definice 2 (Měsíční Výnos):**
$$
r_{t+1} = \frac{P_{t+1} - P_t}{P_t} = \frac{P_{t+1}}{P_t} - 1
$$

kde $P_t$ je uzavírací cena v čase $t$.

**Definice 3 (Logaritmický Výnos):**
$$
r^{log}_{t+1} = \ln\left(\frac{P_{t+1}}{P_t}\right) = \ln(P_{t+1}) - \ln(P_t)
$$

**Poznámka:** Pro malé výnosy platí aproximace $r^{log} \approx r$, protože $\ln(1+x) \approx x$ pro $|x| \ll 1$.

---

## 2. Teorie Random Forest

### 2.1 Rozhodovací Stromy

**Definice 4 (Rozhodovací Strom):**
Rozhodovací strom $T$ je binární strom, kde:
- Každý vnitřní uzel obsahuje test $x_j \leq t$ pro nějakou feature $j$ a threshold $t$
- Každý list obsahuje predikci $\hat{y}$

**Definice 5 (Impurity - Gini Index):**
Pro uzel $m$ s distribucí tříd $p_{mk}$ (podíl třídy $k$ v uzlu $m$):

$$
Gini(m) = \sum_{k=1}^{K} p_{mk}(1 - p_{mk}) = 1 - \sum_{k=1}^{K} p_{mk}^2
$$

**Definice 6 (Information Gain):**
Při splitu uzlu $m$ na potomky $m_L$ a $m_R$:

$$
\Delta Impurity = Impurity(m) - \frac{n_{m_L}}{n_m} Impurity(m_L) - \frac{n_{m_R}}{n_m} Impurity(m_R)
$$

### 2.2 Random Forest Ensemble

**Definice 7 (Random Forest):**
Random Forest $\mathcal{F} = \{T_1, T_2, ..., T_B\}$ je kolekce $B$ rozhodovacích stromů, kde:

1. Každý strom $T_b$ je trénován na bootstrap vzorku $\mathcal{D}_b$ velikosti $n$ s opakováním
2. V každém uzlu je náhodně vybráno $m$ features (typicky $m = \sqrt{d}$ pro klasifikaci)
3. Strom je pěstován do maximální hloubky nebo do splnění stopping kritéria

**Věta 1 (Predikce Random Forest - Klasifikace):**
Pro vstup $x$, finální predikce je většinové hlasování:

$$
\hat{y} = \arg\max_k \sum_{b=1}^{B} \mathbb{1}[T_b(x) = k]
$$

**Věta 2 (Pravděpodobnostní Odhad):**
Pravděpodobnost třídy $k$ pro vstup $x$:

$$
\hat{P}(Y = k | X = x) = \frac{1}{B} \sum_{b=1}^{B} \mathbb{1}[T_b(x) = k]
$$

### 2.3 Teoretické Vlastnosti

**Věta 3 (Konzistence Random Forest, Breiman 2001):**
Za určitých podmínek (dostatečný počet stromů, správná regularizace) je Random Forest konzistentní estimátor:

$$
\lim_{n \rightarrow \infty} \mathbb{E}[L(\hat{f}_n, f^*)] = 0
$$

kde $f^*$ je optimální Bayesovský klasifikátor.

**Věta 4 (Variance Reduction):**
Pro $B$ nezávislých stromů s variancí $\sigma^2$ a korelací $\rho$:

$$
Var(\bar{T}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2
$$

Při $B \rightarrow \infty$: $Var(\bar{T}) \rightarrow \rho \sigma^2$

**Důsledek:** Zvyšování počtu stromů snižuje varianci, ale pouze do limitu daného korelací $\rho$. Feature randomness snižuje $\rho$.

---

## 3. Feature Importance

### 3.1 Mean Decrease in Impurity (MDI)

**Definice 8 (MDI Feature Importance):**
Pro feature $X_j$ v stromu $T$:

$$
Importance_{MDI}(X_j, T) = \sum_{m: v(m) = j} \frac{n_m}{n} \Delta Impurity(m)
$$

kde suma je přes všechny uzly $m$, kde se splituje na feature $j$.

Pro Random Forest:
$$
Importance_{MDI}(X_j) = \frac{1}{B} \sum_{b=1}^{B} Importance_{MDI}(X_j, T_b)
$$

### 3.2 Permutation Importance

**Definice 9 (Permutation Importance):**
Pro feature $X_j$:

$$
Importance_{Perm}(X_j) = L(\hat{f}, \mathcal{D}_{perm_j}) - L(\hat{f}, \mathcal{D})
$$

kde $\mathcal{D}_{perm_j}$ je dataset s permutovanými hodnotami feature $j$ a $L$ je loss funkce.

**Výhoda:** Permutation importance je model-agnostická a měří skutečný dopad feature na predikci.

---

## 4. Evaluační Metriky - Formální Definice

### 4.1 Confusion Matrix

**Definice 10 (Confusion Matrix):**
Pro $K$-třídní klasifikaci je confusion matrix $C \in \mathbb{N}^{K \times K}$, kde:

$$
C_{ij} = |\{x : \hat{y}(x) = i \land y(x) = j\}|
$$

tj. počet vzorků s predikovanou třídou $i$ a skutečnou třídou $j$.

### 4.2 Precision, Recall, F1

**Definice 11 (Precision pro třídu k):**
$$
Precision_k = \frac{TP_k}{TP_k + FP_k} = \frac{C_{kk}}{\sum_{j} C_{kj}}
$$

**Definice 12 (Recall pro třídu k):**
$$
Recall_k = \frac{TP_k}{TP_k + FN_k} = \frac{C_{kk}}{\sum_{i} C_{ik}}
$$

**Definice 13 (F1-Score pro třídu k):**
$$
F1_k = 2 \cdot \frac{Precision_k \cdot Recall_k}{Precision_k + Recall_k} = \frac{2 \cdot TP_k}{2 \cdot TP_k + FP_k + FN_k}
$$

### 4.3 Agregované Metriky

**Definice 14 (Macro-averaged F1):**
$$
F1_{macro} = \frac{1}{K} \sum_{k=1}^{K} F1_k
$$

**Definice 15 (Weighted F1):**
$$
F1_{weighted} = \sum_{k=1}^{K} \frac{n_k}{n} \cdot F1_k
$$

kde $n_k$ je počet vzorků třídy $k$.

### 4.4 ROC a AUC

**Definice 16 (ROC Curve):**
Pro binární klasifikaci s threshold $\tau$:
- True Positive Rate: $TPR(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}$
- False Positive Rate: $FPR(\tau) = \frac{FP(\tau)}{FP(\tau) + TN(\tau)}$

ROC křivka je graf $TPR$ vs $FPR$ pro všechna $\tau \in [0, 1]$.

**Definice 17 (AUC - Area Under Curve):**
$$
AUC = \int_0^1 TPR(FPR^{-1}(x)) dx
$$

**Interpretace:** AUC = pravděpodobnost, že náhodně vybraný pozitivní vzorek má vyšší score než náhodně vybraný negativní vzorek.

### 4.5 Multi-class AUC

**Definice 18 (One-vs-Rest AUC):**
Pro multi-class problém počítáme AUC pro každou třídu vs. ostatní:

$$
AUC_{OvR} = \frac{1}{K} \sum_{k=1}^{K} AUC_k
$$

---

## 5. Statistické Testy

### 5.1 McNemar's Test

**Použití:** Srovnání dvou klasifikátorů na stejném test setu.

**Definice 19 (McNemar's Test):**
Nechť:
- $b$ = počet vzorků, kde model 1 je správný a model 2 nesprávný
- $c$ = počet vzorků, kde model 1 je nesprávný a model 2 správný

Test statistika:
$$
\chi^2 = \frac{(|b - c| - 1)^2}{b + c}
$$

Pod $H_0$ (modely mají stejnou chybovost): $\chi^2 \sim \chi^2_1$

### 5.2 Bootstrap Confidence Intervals

**Definice 20 (Percentile Bootstrap CI):**
Pro statistiku $\theta$ (např. accuracy):

1. Generuj $B$ bootstrap vzorků z test setu
2. Pro každý vzorek vypočti $\hat{\theta}_b$
3. CI na úrovni $1-\alpha$:
$$
CI = [\hat{\theta}_{(\alpha/2)}, \hat{\theta}_{(1-\alpha/2)}]
$$

kde $\hat{\theta}_{(q)}$ je $q$-kvantil bootstrap distribuce.

### 5.3 Cross-Validation

**Definice 21 (K-Fold Cross-Validation):**
1. Rozděl data do $K$ fold
2. Pro $k = 1, ..., K$:
   - Trénuj na foldech $\{1, ..., K\} \setminus \{k\}$
   - Evaluuj na foldu $k$
3. Průměrná metrika: $\bar{M} = \frac{1}{K} \sum_{k=1}^{K} M_k$

**Časová Cross-Validation (pro časové řady):**
```
Fold 1: Train [1, ..., n₁]     Test [n₁+1, ..., n₂]
Fold 2: Train [1, ..., n₂]     Test [n₂+1, ..., n₃]
Fold 3: Train [1, ..., n₃]     Test [n₃+1, ..., n₄]
...
```

---

## 6. Teorie Imputace Dat

### 6.1 Missing Data Mechanismy

**Definice 22 (Missing Completely at Random - MCAR):**
$$
P(M | X_{obs}, X_{mis}) = P(M)
$$

Chybějící data nezávisí na žádných hodnotách.

**Definice 23 (Missing at Random - MAR):**
$$
P(M | X_{obs}, X_{mis}) = P(M | X_{obs})
$$

Chybějící data závisí pouze na pozorovaných hodnotách.

**Definice 24 (Missing Not at Random - MNAR):**
Chybějící data závisí na nepozorovaných hodnotách.

**V našem případě:** Fundamentální data chybí kvůli omezení API (MCAR) - mechanismus chybění nesouvisí s hodnotami samotných fundamentů.

### 6.2 Regresní Imputace

**Definice 25 (Regresní Imputace):**
Pro chybějící hodnotu $X_{j,mis}$:

$$
\hat{X}_{j,mis} = f(X_{obs})
$$

kde $f$ je regresní model natrénovaný na kompletních datech.

**V našem případě:**
$$
\hat{F}_t = RF(OHLCV_t, TechIndicators_t)
$$

kde $RF$ je Random Forest regressor.

### 6.3 Nejistota Imputace

**Důležité:** Imputované hodnoty mají vyšší nejistotu než pozorované hodnoty.

**Definice 26 (Prediction Interval pro RF):**
Pro Random Forest lze aproximovat prediction interval pomocí variance mezi stromy:

$$
\hat{\sigma}^2(x) = \frac{1}{B-1} \sum_{b=1}^{B} (T_b(x) - \bar{T}(x))^2
$$

Aproximativní 95% PI: $\hat{y}(x) \pm 1.96 \cdot \hat{\sigma}(x)$

---

## 7. Finanční Metriky

### 7.1 Výkonnostní Metriky

**Definice 27 (Kumulativní Výnos):**
$$
R_{cum} = \prod_{t=1}^{T} (1 + r_t) - 1
$$

**Definice 28 (Annualizovaný Výnos):**
$$
R_{ann} = (1 + R_{cum})^{12/T} - 1
$$

kde $T$ je počet měsíců.

**Definice 29 (Sharpe Ratio):**
$$
SR = \frac{\bar{r} - r_f}{\sigma_r}
$$

kde:
- $\bar{r}$ = průměrný výnos strategie
- $r_f$ = bezriziková sazba
- $\sigma_r$ = směrodatná odchylka výnosů

**Annualizovaný Sharpe Ratio:**
$$
SR_{ann} = SR \cdot \sqrt{12}
$$

### 7.2 Rizikové Metriky

**Definice 30 (Maximum Drawdown):**
$$
MDD = \max_{t} \left( \max_{s \leq t} P_s - P_t \right) / \max_{s \leq t} P_s
$$

Největší procentuální pokles od vrcholu.

**Definice 31 (Value at Risk - VaR):**
$$
VaR_\alpha = -\inf\{x : P(r \leq x) \geq \alpha\}
$$

Ztráta, která nebude překročena s pravděpodobností $1-\alpha$.

**Definice 32 (Expected Shortfall / CVaR):**
$$
ES_\alpha = -\mathbb{E}[r | r \leq -VaR_\alpha]
$$

Očekávaná ztráta podmíněná překročením VaR.

---

## 8. Hypotézy a Jejich Testování

### 8.1 Hlavní Hypotézy

**H1 (Prediktivní Schopnost):**
$H_0$: Model nemá lepší predikční schopnost než náhodný klasifikátor
$H_1$: Model má statisticky významně lepší accuracy než $1/K$ (pro $K$ tříd)

Test: Binomický test s $p_0 = 1/K$

**H2 (Fundamentální Faktory):**
$H_0$: Fundamentální metriky nepřispívají k predikci
$H_1$: Fundamentální metriky statisticky významně zlepšují predikci

Test: Srovnání modelu s fundamenty vs. bez fundamentů pomocí McNemar's test

**H3 (Sektorová Specificita):**
$H_0$: Jeden model pro všechny sektory je stejně dobrý jako sektorově specifické modely
$H_1$: Sektorově specifické modely mají lepší výkon

Test: Cross-validation s jednotným vs. sektorovými modely

### 8.2 Statistická Síla

Pro binární výsledek (správná/nesprávná predikce) s accuracy $p$:

**Sample Size pro detekci zlepšení o $\Delta p$:**
$$
n \geq \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \cdot 2p(1-p)}{\Delta p^2}
$$

Kde:
- $z_{1-\alpha/2}$ = z-score pro hladinu významnosti
- $z_{1-\beta}$ = z-score pro statistickou sílu
- $p$ = očekávaná accuracy
- $\Delta p$ = minimální detekovatelné zlepšení

---

## Závěr

Tento dokument poskytuje formální matematický základ pro metody použité v diplomové práci. Všechny definice a věty jsou uvedeny v kontextu specifického problému predikce akciových pohybů a jsou základem pro rigorózní interpretaci výsledků.

---

**Reference:**

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
2. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
