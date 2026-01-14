# 🚀 Plán rozšíření projektu: Od prototypu k produkčnímu systému

Tento dokument slouží jako roadmapa pro transformaci současné pipeline na robustní kvantitativní platformu pro výzkum a automatizované obchodování.

---

## 📊 1. Datová infrastruktura a API
Aktuální systém využívá `yfinance`, které je skvělé pro statický výzkum, ale pro live nasazení má omezení.

### Navrhované zdroje dat:
| Úroveň | Platforma | Využití |
| :--- | :--- | :--- |
| **Prototyping** | Finnhub, Alpha Vantage | Rychlý přístup k cenám a fundamentům přes REST. |
| **High-Precision** | Polygon.io, Massive API | Realtime tick data a WebSockety pro intradenní signály. |
| **Institucionální** | CME Group, dxFeed | Data přímo z burzy, futures a opce. |
| **Exekuce** | Alpaca, IBKR API | Kombinace tržních dat s možností přímého zadávání příkazů. |

> **💡 Produkční upgrade:** V reálném nasazení by byla současná **RF Imputace** (dopočítávání historie) nahrazena nákupem tzv. **Point-in-Time databází**. Tím by se eliminovala jakákoliv chyba predikce v historii a zajistilo se, že model vidí přesně ty informace, které byly na trhu dostupné v čase $T$, bez rizika pohledu do budoucnosti (Look-Ahead Bias).

---

## 🏗️ 2. Architektura robustní pipeline
Navrhuji přechod na modulární systém, který oddělí sběr dat, výpočty a exekuci.

### A) Data Ingestion & Storage
*   **WebSockets:** Implementace feedu pro sledování cen v reálném čase.
*   **Time-series Databáze:** Nasazení **InfluxDB** nebo **TimescaleDB** pro efektivní ukládání tickových dat (místo ukládání do CSV).

### B) Feature Engineering 2.0
*   **Alternativní data:** Integrace NLP modelů pro analýzu sentimentu ze zpráv a sociálních sítí (News Sentiment Overlay).
*   **Orderflow signály:** Využití dat úrovně 2 (L2) pro sledování nerovnováhy v knize objednávek (Orderbook Imbalance).

---

## 🧠 3. Modelování a validace
Současný Random Forest je stabilní základ, pro další posun navrhuji:

*   **Walk-forward Backtesting:** Místo statického rozdělení použít klouzavé okno, které lépe simuluje měnící se tržní režimy.
*   **Hyperparameter Optimization:** Přechod z GridSearch na **Optuna** (Bayesovská optimalizace) pro rychlejší a efektivnější ladění modelu.
*   **Deep Learning:** Experiment s architekturou **LSTM** (pro zachycení časové sekvence) nebo **Transformer** (pro pozornost na klíčové tržní události).

---

## 🛡️ 4. Live Execution & Risk Management
Při přechodu na live trading je klíčové přidat vrstvu ochrany kapitálu:

1.  **Signal Filtering:** Obchodovat pouze tehdy, pokud jistota modelu (prediction probability) přesáhne definovaný práh (např. 60 %).
2.  **Vol-based Sizing:** Velikost pozice se dynamicky mění podle aktuální volatility trhu (ATR).
3.  **Trailing Drawdown Circuit Breaker (Pojistka z maxima):** 
    *   **Princip:** Implementace dynamického stop-spínače. Model běží neomezeně, dokud je ziskový.
    *   **Logika:** Pokud kapitál poklesne o **2–5 % od dosaženého denního maxima (High-Water Mark)**, model okamžitě zastaví veškerou obchodní činnost. 
    *   **Příklad ochrany:** Pokud model během dne vygeneruje zisk +60 % (stav 160 % baseline) a trh se náhle otočí, pojistka se aktivuje při poklesu na 155 % baseline. Tím je ochráněna naprostá většina denního zisku a zabráněno jeho úplnému odevzdání při náhlé změně tržních pravidel.
    *   **Cíl:** Zabránit "vymazání" úspěšného dne při nečekané volatilitě. Restart systému vyžaduje lidskou intervenci (revizi tržního kontextu).
4.  **Automatic Stop-Loss/Take-Profit:** Implementace přímo v exekuční pipeline.

---

##  5. Strategický přesun: Forex jako ideální ML hřiště
Přechod z akcií na měnové páry (Forex) nabízí pro náš systém několik zásadních výhod, které mohou významně zvýšit ziskovost a stabilitu.

### Proč je Forex pro algoritmy "lehčí"?
*   **Dominance algoritmů (80-90 %):** Trh ovládají stroje, nikoliv lidské emoce. To vytváří matematicky čitelnější vzorce. Náš model se tak neučí predikovat "trh", ale předvídat chování ostatních algoritmů, což je mnohem stabilnější cíl.
*   **Likvidita a Nonstop trading:** Forex běží 24/5. To eliminuje "gapy" (skoky v ceně mezi dny), které u akcií často vedou k výpadkům v datech a nečekaným ztrátám. Obrovská likvidita zajišťuje okamžitou exekuci s minimálními náklady.
*   **Technické nástroje a nízké bariéry:**
    *   **Kvalitní data zdarma:** Na rozdíl od akcií, kde jsou Point-in-Time data drahá, Forex brokeři (OANDA, Pepperstone, IC Markets) poskytují historická tick data a real-time feedy s vysokou granulositou často zcela zdarma v rámci demo účtů.
    *   **Nižší poplatky:** Spready na hlavních párech (EUR/USD) jsou často zlomkem poplatků za akcie.
    *   **Prototypování zdarma:** Platformy jako MetaTrader (MT5) nebo OANDA API umožňují neomezené testování na demo účtech s reálnými daty bez nutnosti vkládat kapitál.
*   **Čistota dat:** Na Forexu odpadá potřeba imputace (dopočítávání) fundamentů. Makroekonomické ukazatele (úrokové sazby, inflace) jsou veřejně dostupné a jasně definované, čímž vzniká "čistší" datový signál.

### Adaptivní učení a modelování dynamiky
Na trhu, který se mění ze týdne na týden, je statický model odsouzen k zániku. Navrhujeme tyto přístupy:

1.  **Rolling Window Adaptive Learning:** 
    *   **Ignorování starých dat:** U Forexu je klíčové starší data (např. starší než 6 měsíců) postupně **ignorovat nebo jim dát nižší váhu**. Tržní režimy se neustále mění a data z roku 2020 modelu v roce 2025 spíše "otráví" úsudek šumem, který již neplatí.
    *   **Týdenní retraining:** Model se automaticky přetrénovává každou neděli na oknech posledních týdnů/měsíců, aby zachytil aktuální "strojitost" algoritmů.
2.  **Mixture of Experts (Soustava specialistů):** 
    *   Nasazení více menších modelů specializovaných na konkrétní stavy (Trendový expert, Volatilní expert, Range expert).
    *   **Gating Network:** Řídicí vrstva vyhodnocuje aktuální režim trhu a předává slovo nejvhodnějšímu specialistovi.
3.  **LightGBM / CatBoost:** Přechod na Gradient Boosting algoritmy, které jsou řádově rychlejší při retrainingu a lépe zvládají čerstvá, vysoce granulární data.

### Praktický experiment: Optimalizace okna (Multi-Horizon Analysis)
Jako klíčový test pro Forex navrhuji implementaci **paralelní testovací pipeline**, která odpoví na otázku: *"Jak stará data jsou ještě užitečná?"*

*   **Metodika:** Souběžný trénink a testování **12 nezávislých modelů** ($M_1$ až $M_{12}$).
*   **Struktura:** Model $M_k$ je trénován výhradně na datech za posledních $k$ týdnů.
*   **Analýza výkonu:** Všechny modely predikují stejná data (aktuální týden). Sledování metrik v reálném čase odhalí:
    *   **Informační útlum:** V momentě, kdy 12týdenní model začne výrazně zaostávat za 2týdenním, víme, že na trhu došlo k zásadní změně režimu.
    *   **Vážený Ensemble:** Finální signál nemusí pocházet z jednoho modelu, ale z váženého průměru všech 12, kde nejúspěšnější horizonty v daném měsíci mají nejsilnější hlas.
*   **Výsledek:** Tato "bitva modelů" v reálném čase nám umožní dynamicky přepínat mezi krátkodobou agresivní predikcí a konzervativnějším delším pohledem.

---

## 🛠️ 6. Praktické kroky k implementaci
1.  **Zvolit API:** Doporučuji začít s **Alpaca API** (zdarma pro paper trading i data).
2.  **Modularizace kódu:** Rozdělit `run_150_pipeline.py` na samostatné skripty pro `data_fetcher`, `trainer` a `executor`.
3.  **Paper Trading:** Spustit systém na demo účtu po dobu alespoň jednoho měsíce pro ověření reálného skluzu (slippage) a latence.

---
*Tato nástavba posouvá projekt z akademické sféry do světa profesionálního kvantitativního tradingu.*
