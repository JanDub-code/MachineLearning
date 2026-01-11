# 🎯 Detailní plán vylepšení projektu: Přechod na 150 Pipeline

Tento dokument obsahuje konkrétní kroky pro "upgrade" práce z verze 30 tickerů na robustní verzi 150 tickerů a zlepšení vizuální i odborné stránky dokumentace.

## 📈 1. Upgrade Pipeline (run_150_pipeline.py)
Cílem je získat maximální množství reprezentativních dat a obrázků pro dokumentaci.
- [ ] **Nová funkce `generate_premium_visuals()`**:
    - [ ] **Equity Curve (Backtest)**: Klíčový graf srovnávající kumulativní výnos ML strategie vs. index S&P 500 (Buy & Hold).
    - [ ] **R² Imputation Heatmap**: Matice ukazující přesnost doplňování fundamentů pro každou metriku (0.0 až 1.0).
    - [ ] **Normalized Confusion Matrix**: Procentuální vyjádření úspěšnosti (kolik % UP pohybů jsme skutečně trefili).
    - [ ] **Sector Alpha Plot**: Srovnání stability predikce napříč 5 sektory.
    - [ ] **High-Resolution Export**: Všechny grafy ukládat v 300 DPI s jednotným vizuálním stylem.

## ✍️ 2. Restrukturalizace LaTeX (DIPLOMOVA_PRACE_LATEX.md)
Práce teď popisuje malý experiment, musíme ji přepsat na "velkou hru".
- [ ] **Kapitola 1-3 (Teorie)**: 
    - [ ] Doplnit vysvětlení pojmů (EMH - Efficient Market Hypothesis, Overfitting, Stationarity).
    - [ ] Jasnější rozdělení mezi technickou a fundamentální analýzou (tabulka rozdílů).
- [ ] **Kapitola 5 (Data)**:
    - [ ] Přepsat rozsah z 30 na **150 tickerů (5 sektorů po 30 firmách)**.
    - [ ] Zdůraznit objem dat: **10 let historie = ~18 000 záznamů**.
- [ ] **Kapitola 8 (Experiment)**: 
    - [ ] Přejmenovat na: **"Robustní verifikace modelu na datech indexu S&P 500"**.
    - [ ] Přidat popis hybridního přístupu (Imputace -> Klasifikace).
- [ ] **Kapitola 9 (Výsledky)**: 
    - [ ] **Nahradit všechny tabulky a grafy z 30 tickerů verzí pro 150 tickerů.**
    - [ ] Přidat sekci **"Backtesting a reálná aplikovatelnost"** (zde bude Equity Curve).
    - [ ] Rozepsat interpretaci výsledků: Proč je accuracy ~33 % u akcií ve skutečnosti dobrý/upřímný výsledek.

## �️ 3. Vizuální Storytelling (Více fotek a popisek)
Méně "moře textu", více informací v obrázcích.
- [ ] **Schéma Pipeline**: Vytvořit (nebo popsat) blokové schéma celého procesu (stahování -> čištění -> imputace -> tuning -> predikce).
- [ ] **Bohaté popisky**: Každý graf musí mít popisek na 3-4 řádky, který vysvětluje:
    - *"Co graf ukazuje?"* (osa X, osa Y)
    - *"Co z toho vyplývá?"* (např. interpretace AUC skóre).
- [ ] **Sektorová galerie**: Přidat srovnávací grafy výkonnosti mezi sektory (např. Tech vs. Industrials).

## 🏁 4. Závěry a interpretace
- [ ] **Reframing výsledků**: Model neprezentovat jako "nepřesný", ale jako "stabilitu udržující v šumu finančních trhů".
- [ ] **Zobecnitelnost**: Přidat závěr o tom, že 150 tickerů dokazuje schopnost modelu generalizovat na různé typy byznysů.
- [ ] **Srovnání**: Krátká zmínka o tom, že u 30 tickerů byla vyšší náhoda, zatímco 150 tickerů dává stabilnější (byť zdánlivě nižší) metriky.
