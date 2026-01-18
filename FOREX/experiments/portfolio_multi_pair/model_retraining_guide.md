# Retrénování, udržení edge a zlepšení modelu (praktický plán)

Tento dokument shrnuje **jak často retrénovat**, jak **udržet edge**, a jaké jsou **varianty vylepšení** (data, časové úseky, modelové změny) bez zbytečného overfittingu. Je psaný prakticky pro tento projekt (LogReg‑Momentum, high‑vol režim).

---

## 1) Jak často retrénovat

**Krátká odpověď:**
- **1× za 1–3 měsíce** jako default.
- **Měsíčně** je nejčastěji rozumné, pokud sleduješ metriky a market regime.

**Proč ne častěji?**
- Edge je malý a režimový → časté retrénování vede k přeučení na šum.
- V krátkých oknech se mění mikrostruktura (spready, volatility), model „běhá“ za trhem.

**Kdy dřív než 1 měsíc?**
- Prudká změna volatility režimu (např. dlouhé low‑vol období a pak risk‑off šok).
- Systém začne **systematicky ztrácet** (rolling expectancy pod 0).

**Kdy později než 3 měsíce?**
- Pokud edge drží stabilně, frekvence signálů je v normě a drawdown kontrolovatelný.

---

## 2) Jak monitorovat, že edge „žije“

Používej **rolling okno 30–50 obchodů** pro každý pár i pro portfolio:
- **Expectancy/Trade** (klíčové)
- **Win rate**
- **Profit factor**
- **Trade frequency**

**Trigger k retréninku:**
- Expectancy < 0 po 30–50 obchodech
- Výrazný pokles frekvence signálů bez změny volatility
- DD překročí historický limit (např. 2× typický max DD)

---

## 3) Varianty vylepšení modelu (bez rozbití edge)

### A) Stabilní, bezpečné změny (preferované)
1) **Update vah** (re‑fit) bez změny feature setu
2) **Update thresholdů** (probability threshold, min gap)
3) **Re‑kalibrace ATR filtrů** (min/max ATR pips) per pair
4) **Volatility‑weighted sizing** místo fixního risku

### B) Středně rizikové změny
1) **Krátké rozšíření feature setu** (např. volatility slope, session flags)
2) **Session filtry** (např. jen London/NY overlap)
3) **Jiný horizon** (např. 3–10 min místo 1–5)

### C) Rizikové změny (jen pokud A/B nepomůže)
1) Změna modelu (např. GBM, XGBoost)
2) Re‑definice targetu (např. min move filter)
3) Silná feature engineering (může přeučit)

---

## 4) Jak zlepšit aktuální model (konkrétní plán)

### Krok 1: Stabilizace portfolia
- Drop / downweight páry s negativní edge (aktuálně **USDCAD**).
- Udržet core: EURUSD + USDJPY + AUDJPY + EURGBP + AUDUSD.

### Krok 2: Kalibrace prahů
- Prob threshold + min gap optimalizovat per pair.
- ATR bounds per pair (JPY vs non‑JPY).

### Krok 3: Režimové filtry
- High‑vol filter zachovat.
- Testovat „vol regime = high“ pouze.

### Krok 4: Sizing
- Volatility‑weighted risk (nižší váha v extrémní volatilitě).
- Max portfolio risk cap (např. 2–3 %).

---

## 5) Jak přidat více dat (delší historie)

### Doporučení
- Přidej **6–24 měsíců** historie pro robustnost.
- Pozor na **režimové změny** – delší historie může přidat jiný profil volatility.

### Praktická strategie
- Hlavní model trénuj na **posledních 3–6 měsících** (rolling).
- Starší data používej jen na **robustnost** a sanity check.

---

## 6) Doporučený „professional“ retrénink režim

**Měsíční cyklus**
1) Re‑fit modelu na posledních 3 měsících
2) Retune thresholdů per pair
3) Re‑run full‑period walk‑forward
4) Uprav portfolio váhy / drop pary s negativním edge

**Monitoring**
- Týdně kontrolovat rolling expectancy + signal frequency

---

## 7) Závěr

- Retrénování **měsíčně** je bezpečný default.
- Edge je malý → **stabilita > agresivní změny**.
- Nejrychlejší zlepšení: **správné páry + kalibrace prahů + sizing**.
- Dlouhá historie pomůže, ale core model drž v „recent“ režimu.

---

## Další kroky (prakticky)

1) Připravím variantu **bez USDCAD** + přepočet portfolio výnosu.
2) Přidám **basket_8** test.
3) Přidám **volatility‑weighted sizing** a přepočet edge.
