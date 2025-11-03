# 🤖 AGENT CONTEXT - CleanSolution

> **Účel:** Kritický kontext pro AI agenty. Tento soubor je STRUČNÝ a odkazuje na detailní dokumentaci.  
> **Poslední update:** 1. listopadu 2025  
> **Status:** Production Ready ✅

---

## 🎯 Co Je Tento Projekt?

**Hybrid AI/ML pipeline pro predikci cen akcií:**
- Random Forest AI → predikuje fundamentals z OHLCV (řeší problém chybějících 8.5 let dat)
- Ridge Regression → predikuje ceny z fundamentals

**Kontext:** Parent projekt má 10 let OHLCV dat, ale fundamentals existují jen 1.5 roku. AI nám doplní historii.

---

## 📚 HLAVNÍ PRAVIDLO: Použij Existující Dokumentaci!

### Primární Reference (čti VŽDY před začátkem práce):

| Dokument | Kdy Použít |
|----------|-----------|
| **[README.md](README.md)** | První orientace v projektu |
| **[QUICKSTART.md](QUICKSTART.md)** | Jak rychle spustit pipeline |
| **[docs/WORKFLOW.md](docs/WORKFLOW.md)** | Detailní kroky každé fáze + troubleshooting |
| **[docs/SUMMARY.md](docs/SUMMARY.md)** | Technické detaily, architektury, výsledky |
| **[INDEX.md](INDEX.md)** | Mapa všech dokumentů |
| **[PROJECT_COMPLETE.md](PROJECT_COMPLETE.md)** | Status deliverables, statistiky |

### ⚠️ NEOPAKUJ obsah těchto dokumentů - ODKAZUJ na ně!

---

## 🏗️ Pipeline Overview (5 Fází)

```
FÁZE 1: OHLCV Data → [Parent projekt: data_10y/]
FÁZE 2: Download Fundamentals → scripts/1_download_fundamentals.py
FÁZE 3: Train AI Model → scripts/2_train_fundamental_predictor.py
FÁZE 4: Complete History → scripts/3_complete_historical_data.py
FÁZE 5: Train Price Predictor → scripts/4_train_price_predictor.py
```

**Detaily každé fáze:** Viz [docs/WORKFLOW.md](docs/WORKFLOW.md)

---

## 📁 Struktura (Kritická Místa)

```
CleanSolution/
├── scripts/               # 4 Python skripty (FÁZE 2-5)
├── notebooks/             # Google Colab ready (Part1 = FÁZE 2-3)
├── data/                  # Outputs z pipeline
│   ├── fundamentals/      # FÁZE 2 output
│   ├── complete/          # FÁZE 4 output
│   └── analysis/          # FÁZE 5 visualizations
├── models/                # Pkl files (RF + Ridge + Scaler)
└── docs/                  # Detailní dokumentace
```

**Kompletní popis:** Viz [README.md](README.md) sekce "Project Structure"

---

## 🚨 Kritická Upozornění

### ⛔ NIKDY:
- Nesmaž `../data_10y/` (parent projekt data)
- Necommituj `data/` nebo `models/*.pkl` (velké soubory)
- Neměň scripts bez testování celého pipeline
- Neignoruj rate limiting v Script 1 (→ 429 error)

### ✅ VŽDY:
- Spusť scripts v pořadí 1→2→3→4
- Checkni dependencies před spuštěním (`pip install -r requirements.txt`)
- Zálohuj modely před retrainingem (`cp models/ models_backup/`)
- Update dokumentaci při změnách

---

## 🔧 Běžné Úkoly

### Spustit Pipeline

```bash
# Automaticky (doporučeno):
run_pipeline.bat  # Windows
./run_pipeline.sh  # Linux/Mac

# Manuálně viz: QUICKSTART.md
```

### Debugovat Problém

1. **Najdi error v console** (má timestamp)
2. **Otevři [docs/WORKFLOW.md](docs/WORKFLOW.md)** → sekce "Troubleshooting"
3. **Zkontroluj Debug Checklist** níže
4. **Zeptej se uživatele** pokud není v dokumentaci

### Přidat Feature

1. **Edituj relevantní script** (např. `scripts/1_download_fundamentals.py`)
2. **Testuj změnu** (spusť ten script samostatně)
3. **Update dokumentaci:**
   - `README.md` (Features section)
   - `docs/WORKFLOW.md` (Detailed steps)
   - Tento soubor (pokud kritické)

---

## 🔍 Debug Checklist (Quick)

```bash
# 1. Data existují?
ls ../data_10y/all_sectors_full_10y.csv  # Musí existovat

# 2. Dependencies OK?
pip show scikit-learn yfinance pandas

# 3. Správné pořadí?
# FÁZE 2 → 3 → 4 → 5 (nelze přeskočit)

# 4. Outputs jsou vytvořeny?
ls data/fundamentals/  # Po Script 1
ls models/*.pkl        # Po Script 2
ls data/complete/      # Po Script 3
ls data/analysis/      # Po Script 4
```

**Detailní debugging:** Viz [docs/WORKFLOW.md](docs/WORKFLOW.md) → "Common Issues"

---

## 📊 Očekávané Metriky

**FÁZE 3 (AI Model):** MAE ~14.2%, R² ~0.743  
**FÁZE 5 (Price Predictor):** MAE ~$12.22, R² ~0.801  

**Threshold alarmy:**
- Pokud MAE > $15 nebo R² < 0.70 → investigate
- Možné příčiny: data drift, outliers, špatné hyperparametry

**Detailní výsledky:** Viz [docs/SUMMARY.md](docs/SUMMARY.md) → "Results"

---

## 🎯 Roadmap (Prioritizovaný)

### HIGH Priority:
1. **Part2 Notebook** (FÁZE 4-5 pro Google Colab)
2. **Cross-validation** (time-series CV)
3. **Real-time API** (live predictions)

### MEDIUM/LOW Priority:
Viz [docs/SUMMARY.md](docs/SUMMARY.md) → "Future Enhancements"

---

## 📞 Pro AI Agenty: Workflow

```
1. User request → Přečti relevantní dokumentaci (README/WORKFLOW/SUMMARY)
2. Determine task → Modifikace? Debugging? Nový feature?
3. Find relevant script → scripts/1-4, nebo notebook
4. Make change → Testuj lokálně
5. Update docs → README + WORKFLOW (+ tento soubor pokud kritické)
6. Report back → Stručně, co bylo uděláno + reference na docs
```

### Standardní Odpověď:

```
"Upravil jsem [soubor]. Detaily: [link na doc].
Pro spuštění viz [QUICKSTART.md](QUICKSTART.md)."
```

**❌ NE:** Dlouhé vysvětlování co už je v docs  
**✅ ANO:** Stručně + reference na dokumentaci

---

## 🔐 Metadata

**Autor:** Bc. Jan Dub  
**Datum:** 31. října 2025 (poslední update: 1. listopadu 2025)  
**Verze:** 1.0.0  
**Tech Stack:** Python 3.8+, scikit-learn, yfinance, pandas  

**Parent projekt:** `../` (obsahuje data_10y/, původní skripty)  
**CleanSolution:** Nová, čistá implementace s kompletní dokumentací

---

## 🚀 TL;DR

```
• Projekt: Hybrid AI predikce cen akcií (RF → Ridge)
• Pipeline: 5 fází (1=parent, 2-5=CleanSolution scripts)
• Dokumentace: README → rychlý start, WORKFLOW → detaily
• Spuštění: run_pipeline.bat/sh
• Debugging: WORKFLOW.md → Troubleshooting
• Změny: Edit script → test → update docs
• Tento soubor: Stručný guide + odkazy na detailní docs
```

**🤖 Pro agenty: Přečti dokumentaci místo hádání. Odkazuj místo opakování. Buď stručný.**

---

*Tento soubor: Minimální kontext. Detaily: Odkazy výše.*  
*Update: Při kritických změnách projektu.*
