# 🧠 Jak to celé funguje (Simple README)

Tento projekt slouží k **predikci cen akcií** pomocí umělé inteligence. Protože nemáme kompletní data pro všech 10 let zpětně, musíme si je "dopočítat".

Celý proces má 3 hlavní kroky:

## 1. Sběr Dat (Data Ingestion)
*   **Co děláme:** Stahujeme data, která jsou dostupná.
*   **Máme:** Ceny akcií (OHLCV) za 10 let.
*   **Chybí:** Fundamentální data (zisky, tržby, P/E ratio) pro starší roky (máme jen posledních 1.5 roku).

## 2. Doplnění Historie pomocí AI (Imputace)
*   **Problém:** Abychom mohli trénovat hlavní model, potřebujeme kompletní historii fundamentů, kterou nemáme.
*   **Řešení:** Natrénujeme "pomocnou AI" (**Random Forest**), která se na datech z posledního 1.5 roku naučí, jak cena akcie souvisí s jejími fundamenty.
*   **Výsledek:** Tato AI se podívá na ceny před 5 nebo 10 lety a s vysokou přesností "odhadne" (dopočítá), jaké tehdy musely být fundamenty. Tím získáme **kompletní 10letou historii**.

## 3. Predikce Budoucnosti (Forecasting)
*   **Co děláme:** Vezmeme kompletní 10letá data (část reálná, část dopočítaná AI) a vložíme je do hlavního modelu (**Ridge Regression**).
*   **Cíl:** Tento model hledá vzory v celé této historii a na jejich základě předpovídá, kam se cena pohne v příštím měsíci.

---
### 🚀 Shrnutí v jedné větě
**"Používáme AI, abychom zrekonstruovali minulost, a díky tomu mohli lépe předpovídat budoucnost."**

---

## ❓ Časté otázky (FAQ)

**Q: Můžu si vybrat sektor a predikovat cenu na 10 let dopředu?**
**A: Ne, model predikuje cenu na 1 měsíc dopředu.**

*   **Co model umí:** Pro **konkrétní firmu v konkrétním sektoru** předpovědět cenu na **následující měsíc**.
*   **Proč ne 10 let:** U dlouhodobých předpovědí dochází k tzv. **násobící se chybě** (compounding error). Malá nepřesnost v prvním měsíci by se ve druhém měsíci zvětšila, ve třetím ještě více, až by byla předpověď na 10 let naprosto nepoužitelná (čisté hádání).
*   **K čemu to tedy je:** Model využívá 10letou historii k tomu, aby co nejpřesněji odhadl ten *nejbližší* krok. Je to jako navigace v autě – vidí celou mapu (historii), ale říká vám přesně, kam zahnout na příští křižovatce (příští měsíc).
