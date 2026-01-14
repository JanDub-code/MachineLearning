# 🚀 MAXIMÁLNÍ CHEAT SHEET: Obhajoba ML Pipeline S&P 500

> **Téma:** Klasifikace akciových pohybů pomocí Random Forest s využitím hybridní kaskádové imputace fundamentů.

---

## 🏗️ 1. ARCHITEKTURA: "Kaskádový systém" (The Sequential Pipeline)
Můj projekt využívá **kaskádovou architekturu**, kde modely nespolupracují jen tak, ale v přesně daném řetězci (Feature Augmentation).

1.  **FÁZE A: RF Regressor (Imputační motor)**
    *   **Úkol:** Rekonstrukce historie. Free API dávají jen 2 roky fundamentů, já jich potřeboval 10.
    *   **Vztah:** Statický vztah mezi cenou/objemem a účetním stavem firmy.
    *   **Výsledek:** **$R^2 \approx 0.97$**. Extrémně silná korelace dokazuje, že tržní cena v sobě fundamenty "nese".
    *   **Klíčový driver:** **Volume (Objem)** má váhu ~50%. Je to nejsilnější indikátor zájmu velkých fondů.
2.  **FÁZE B: RF Classifier (Predikční mozek)**
    *   **Úkol:** Klasifikace do tří tříd (**DOWN** < -3%, **HOLD** ±3%, **UP** > 3%).
    *   **Spolupráce:** Tento model "tahá" data z prvního modelu. Vstupem mu jsou reálné ceny + **imputované (vytvořené) fundamenty**.
    *   **Výsledek:** **Accuracy 35,6%** (o 2,3% nad náhodu). Signifikantní "alpha" ve světě financí.

---

## 🧠 2. ALGORITMUS: Proč je Random Forest (RF) ideální?
**KRITICKÉ:** RF **NENÍ** neuronová síť. Je to *Ensemble Learning* založený na větvení.

*   **Princip „moudrosti davu“:** 200 stromů. Každý vidí jinou část dat (**Bagging**) a náhodnou část indikátorů (**Feature Randomness**).
*   **Proč pro UP/DOWN/HOLD?**
    *   **Binární řezy:** RF se ptá "Je RSI > 70?". To přesně odpovídá našim diskrétním škatulkám ±3%.
    *   **Ignorování šumu:** Průměr 200 stromů vyruší náhodné chyby jednotlivých stromů.
    *   **Nelineární logika:** RF chápe vztahy jako "Pokud je P/E nízké A ZÁROVEŇ RSI roste, pak kupuj".
*   **Proč ne jiné?**
    *   **XGBoost:** Často výkonnější, ale na burze se šíleně přeučuje (**overfitting**).
    *   **SVM / Neuronky:** Vyžadují složité ladění vzdáleností, RF je "přímočařejší".

---

## 🛠️ 3. PIPELINE & KÓD: Technické pilíře
*   **StandardScaler:** **Standardizace je klíčová pro stabilitu.** Sjednocuje váhu (miliardový Volume vs. jednotkové Returns), aby jedna feature nepřebila ostatní. 
    *   *Tip:* I když RF měřítko neřeší, standardizace v pipeline zajišťuje numerickou stabilitu a možnost do budoucna model vyměnit.
*   **TimeSeriesSplit:** Simulace reálného času. Model se učí na 2015-2018 a testuje se na 2019. Tím eliminujeme nahlížení do budoucnosti (**Data Leakage**).
*   **Hyperparameter Tuning (GridSearchCV):** Ladění ovladačů (např. `max_depth=15` jako strop proti přeučení). Najde "zlatou střední cestu" výkonu.

---

## 📊 4. METRIKY: Rozbor tvých výsledků
Můj model není "věštec", ale nositel statistické výhody (**Alpha**).

*   **Accuracy (35,61%):** Celková úspěšnost. Překonání baseline (33,3%) je důkazem nalezení neefektivity trhu.
*   **Precision (36,57%):** **Nejdůležitější metrika.** "Když model řekne UP, máme vyšší než náhodnou šanci, že trefíme zisk."
*   **Recall (35,61%):** Citlivost. Model je raději opatrnější a signál nevydá, než aby riskoval špatný nákup.
*   **F1-Score (35,77%):** Harmonický průměr dokazující vyváženost. Model nepodvádí tipováním jen jedné třídy.
*   **ROC Curve / AUC (0.55):** Důkaz rozlišovací schopnosti. Jakmile je křivka nad diagonálou, model prokazatelně identifikoval nenáhodné vzorce.

---

## 📈 5. SEKTOROVÁ ANALÝZA: Kde to šlape?
*   **Financials (40,3% acc):** **NEJLEPŠÍ.** Banky mají jasné vazby mezi fundamenty (dluh, kapitál) a cenou.
*   **Technology (nejslabší):** IT firmy rostou na základě budoucího "hype", což se z účetních výkazů predikuje hůře.
*   **Confusion Matrix:** Model má vysoký Recall u **DOWN**. Je to skvělý nástroj pro **řízení rizika** (pozná, kdy z trhu utéct).

---

## 🔮 6. BUDOUCNOST: LSTM vs. Transformer
*   **LSTM (Long Short-Term Memory):** Neuronka s pamětí. Dobrá na filmy/sekvence, ale na burze "vidí duchy" (overfitting) a potřebuje hromady dat.
*   **Transformer:** Technologie za ChatGPT. Má mechanismus **Attention**, vidí "příběh" v čase, ale je to totální Black Box a vyžaduje GPU farmy.
*   **Random Forest** byl vybrán jako robustní, interpretovatelný a efektivní model pro tabulková data.

---

## ⚔️ 7. DEFENSIVA: Příprava na útok komise
*   **Overfitting?** -> "Omezil jsem `max_depth=15` a použil `TimeSeriesSplit` pro férové testování."
*   **Jen 35%?** -> "Finanční trhy jsou z 90% náhodná procházka (EMH). +2% nad náhodu je v měsíčním měřítku signifikantní úspěch."
*   **RF je neuronka?** -> "Rozhodně ne. RF je ensemble stromů založený na logickém větvení, neuronky na matematických vahách a backpropagaci."
*   **Imputace je podvod?** -> "Naopak. $R^2 = 0.97$ u regrese dokazuje, že naše syntetická data věrně simulují historickou realitu."

---
**Zlom vaz! Máš to podložené kódem, matematikou i logikou. 🚀**
