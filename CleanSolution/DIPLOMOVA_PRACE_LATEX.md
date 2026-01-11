\documentclass[twoside, 12pt]{article}
\usepackage{amsmath}
\usepackage{graphicx, xdipp, url, fancyvrb, tcolorbox}
\usepackage[hidelinks]{hyperref}
\cestina
\usepackage{minted}
\usepackage{caption}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{float}

\tcbuselibrary{listingsutf8}
\tcbuselibrary{skins}

\begin{document}

\makeatletter
\def\typprace{Semestrální}
\def\c@prace{projekt}
\def\a@prace{project}
\makeatother
\titul{Klasifikace cenových pohybů akcií pomocí strojového učení}
{\begin{tabular}[b]{@{}r} Bc. Jan Dub \\ Bc. David Krčmář \end{tabular}}{doc. Ing. František Dařena, Ph.D.}{Brno 2025}


% ============================================
% ABSTRAKTY
% ============================================
\abstract{Dub, J. Classification of Stock Price Movements Using Machine Learning. Semester project. Mendel University in Brno, 2025.}
{This semester project focuses on the design and implementation of a machine learning system for classifying monthly stock price movements from the S\&P 500 index. The theoretical part discusses the Efficient Market Hypothesis, fundamental and technical analysis approaches, and the mathematical foundations of the Random Forest algorithm. The practical part presents a hybrid approach that addresses the problem of missing historical fundamental data through RF-based imputation. The classification model achieves accuracy comparable to random baseline, reflecting the inherent difficulty of financial market prediction. The project concludes with an evaluation of feature importance and recommendations for future improvements.}

\abstrakt{Dub, J. Klasifikace cenových pohybů akcií pomocí strojového učení. Semestrální projekt. Mendelova univerzita v Brně, 2025.}
{Tento semestrální projekt se zaměřuje na návrh a implementaci systému strojového učení pro klasifikaci měsíčních cenových pohybů akcií z indexu S\&P 500. Teoretická část pojednává o hypotéze efektivních trhů, přístupech fundamentální a technické analýzy a matematických základech algoritmu Random Forest. Praktická část představuje hybridní přístup řešící problém chybějících historických fundamentálních dat pomocí RF-based imputace. Klasifikační model dosahuje accuracy srovnatelné s náhodným baseline, což odráží inherentní obtížnost predikce finančních trhů. Práce je zakončena vyhodnocením důležitosti features a doporučeními pro budoucí vylepšení.}

% ============================================
% KLÍČOVÁ SLOVA
% ============================================
\keywords{machine learning, Random Forest, stock market, classification, S\&P 500, fundamental analysis, technical analysis}
\klslova{strojové učení, Random Forest, akciový trh, klasifikace, S\&P 500, fundamentální analýza, technická analýza}

% ============================================
% OBSAH
% ============================================
\obsah

% ============================================
% SEZNAM POJMŮ
% ============================================
\section*{\LARGE\textbf{Seznam použitých pojmů}}

\begin{itemize}
    \item \textbf{\textit{EMH (Efficient Market Hypothesis)}}: Hypotéza efektivních trhů tvrdící, že ceny akcií odrážejí všechny dostupné informace.
    
    \item \textbf{\textit{OHLCV}}: Cenová data obsahující Open (otevírací), High (nejvyšší), Low (nejnižší), Close (zavírací) cenu a Volume (objem).
    
    \item \textbf{\textit{Random Forest}}: Ensemble metoda strojového učení kombinující více rozhodovacích stromů.
    
    \item \textbf{\textit{Feature Importance}}: Měřítko relativní důležitosti jednotlivých vstupních proměnných pro predikci modelu.
    
    \item \textbf{\textit{Imputace}}: Proces doplnění chybějících hodnot v datasetu pomocí statistických nebo ML metod.
    
    \item \textbf{\textit{P/E (Price-to-Earnings)}}: Poměr ceny akcie k zisku na akcii, základní valuační metrika.
    
    \item \textbf{\textit{ROE (Return on Equity)}}: Rentabilita vlastního kapitálu, měřítko profitability.
    
    \item \textbf{\textit{RSI (Relative Strength Index)}}: Technický indikátor měřící překoupenost/přeprodanost.
    
    \item \textbf{\textit{MACD (Moving Average Convergence Divergence)}}: Technický indikátor pro identifikaci změn trendu.
    
    \item \textbf{\textit{AUC (Area Under Curve)}}: Metrika kvality klasifikátoru měřící plochu pod ROC křivkou.
    
    \item \textbf{\textit{Cross-Validation}}: Technika pro validaci modelu rozdělením dat na více částí.
    
    \item \textbf{\textit{TimeSeriesSplit}}: Speciální cross-validace pro časové řady respektující časovou posloupnost.
\end{itemize}
% =========================================================================
% ČÁST 2: Kapitoly 1-3 (Úvod, Teorie, Matematika)
% =========================================================================

% ============================================
% KAPITOLA 1: ÚVOD A MOTIVACE
% ============================================
\kapitola{Úvod a motivace}

Predikce pohybů akciových trhů představuje jeden z nejnáročnějších problémů kvantitativních financí. Tato práce se zaměřuje na vývoj a evaluaci ML systému pro klasifikaci měsíčních cenových pohybů akcií z indexu S\&P 500.

\sekce{Kontext problému}

\podsekce{Hypotéza efektivních trhů (EMH)}

Podle Eugene Famy (1970) existují tři formy tržní efektivity:

\begin{table}[H]
\centering
\caption{Formy tržní efektivity podle EMH}
\begin{tabular}{|p{2.5cm}|p{4cm}|p{5cm}|}
\hline
\textbf{Forma} & \textbf{Dostupné informace} & \textbf{Implikace} \\ \hline
Slabá & Historické ceny & Technická analýza nefunguje \\ \hline
Polo-silná & Veřejné informace & Fundamentální analýza nefunguje \\ \hline
Silná & Veškeré informace & Žádná strategie nepřekoná trh \\ \hline
\end{tabular}
\end{table}

\textbf{Naše pozice:} Pokud existují tržní neefektivity, ML modely mohou tyto neefektivity identifikovat a využít. Práce testuje hypotézu, že kombinace fundamentálních a technických faktorů může poskytnout prediktivní signál.

\sekce{Cíle práce}

\begin{itemize}
    \item \textbf{Primární cíl:} Vyvinout ML model pro klasifikaci měsíčních cenových pohybů
    \item \textbf{Sekundární cíl:} Řešit problém chybějících historických fundamentálních dat
    \item \textbf{Terciární cíl:} Analyzovat prediktivní sílu různých typů features
\end{itemize}

\sekce{Klíčová inovace}

Projekt řeší fundamentální problém v kvantitativních financích: \textbf{neúplnost historických fundamentálních dat}. Zatímco cenová data (OHLCV) jsou dostupná za 10+ let, fundamentální metriky (P/E, ROE, atd.) jsou typicky dostupné pouze za 1-2 roky.

\textbf{Navrhované řešení - Hybridní přístup:}

\begin{enumerate}
    \item \textbf{Random Forest Regressor (Imputace):} Input: OHLCV + Technické indikátory $\rightarrow$ Output: Fundamentální metriky
    \item \textbf{Random Forest Classifier (Predikce):} Input: OHLCV + Technické + Fundamenty $\rightarrow$ Output: Třída pohybu (DOWN / HOLD / UP)
\end{enumerate}

% ============================================
% KAPITOLA 2: TEORETICKÝ RÁMEC
% ============================================
\kapitola{Teoretický rámec}

\sekce{Fundamentální vs. technická analýza}

\podsekce{Fundamentální analýza}

Fundamentální analýza se zaměřuje na vnitřní hodnotu aktiva na základě finančních výkazů, ekonomických podmínek a konkurenčního postavení firmy.

\textbf{Klíčové metriky používané v této práci:}

\begin{table}[H]
\centering
\caption{Fundamentální metriky}
\begin{tabular}{|p{3cm}|p{5cm}|p{4cm}|}
\hline
\textbf{Kategorie} & \textbf{Metriky} & \textbf{Interpretace} \\ \hline
Valuační & P/E, P/B, P/S, EV/EBITDA & Nadhodnocení/podhodnocení \\ \hline
Profitabilita & ROE, ROA, Marže & Efektivita generování zisku \\ \hline
Finanční zdraví & Debt/Equity, Current Ratio & Schopnost splácet závazky \\ \hline
Růst & Revenue Growth, Earnings Growth & Dynamika růstu \\ \hline
\end{tabular}
\end{table}

Benjamin Graham a David Dodd ve své práci ``Security Analysis'' (1934) argumentují, že dlouhodobě cena akcie konverguje k její vnitřní hodnotě.

\podsekce{Technická analýza}

Technická analýza předpokládá, že veškeré informace jsou zahrnuty v ceně a objemu obchodování.

\textbf{Používané indikátory:}

\begin{table}[H]
\centering
\caption{Technické indikátory}
\begin{tabular}{|p{2.5cm}|p{5cm}|p{4cm}|}
\hline
\textbf{Indikátor} & \textbf{Formule} & \textbf{Interpretace} \\ \hline
RSI (14) & $100 - \frac{100}{1 + RS}$ & Překoupenost/přeprodanost \\ \hline
MACD & $EMA_{12} - EMA_{26}$ & Momentum, změna trendu \\ \hline
SMA/EMA & Klouzavé průměry & Trend, support/resistance \\ \hline
Volatilita & $\sigma = \frac{High - Low}{Close}$ & Míra rizika \\ \hline
\end{tabular}
\end{table}

\podsekce{Proč kombinace obou přístupů?}

\begin{table}[H]
\centering
\caption{Srovnání přístupů}
\begin{tabular}{|p{2.5cm}|p{3cm}|p{3cm}|p{3cm}|}
\hline
\textbf{Aspekt} & \textbf{Fundamentální} & \textbf{Technická} & \textbf{Kombinace} \\ \hline
Horizont & Dlouhodobý & Krátkodobý & Střední \\ \hline
Data & Kvartální & Denní/měsíční & Oba zdroje \\ \hline
Lag & Vysoký & Nízký & Vyvážený \\ \hline
Noise & Nízký & Vysoký & Střední \\ \hline
\end{tabular}
\end{table}

\sekce{Klasifikace vs. regrese}

\podsekce{Proč klasifikace?}

V původním návrhu byl použit regresní přístup pro predikci přesné hodnoty ceny. Přechod na klasifikaci je motivován:

\begin{table}[H]
\centering
\caption{Srovnání regrese a klasifikace}
\begin{tabular}{|p{3cm}|p{4.5cm}|p{4.5cm}|}
\hline
\textbf{Aspekt} & \textbf{Regrese} & \textbf{Klasifikace} \\ \hline
Output & Přesná cena/výnos & Třída pohybu \\ \hline
Interpretace & ``Cena bude \$152.34'' & ``Cena vzroste o >3\%'' \\ \hline
Praktické využití & Obtížné & Přímé trading signály \\ \hline
Robustnost & Citlivá na outliers & Robustní \\ \hline
\end{tabular}
\end{table}

\podsekce{Definice tříd}

\begin{itemize}
    \item \textbf{Třída 0 (DOWN):} return $< -3\%$ $\rightarrow$ Signifikantní pokles
    \item \textbf{Třída 1 (HOLD):} $-3\% \leq$ return $\leq +3\%$ $\rightarrow$ Stagnace
    \item \textbf{Třída 2 (UP):} return $> +3\%$ $\rightarrow$ Signifikantní růst
\end{itemize}

\textbf{Zdůvodnění prahu $\pm 3\%$:}
\begin{itemize}
    \item Typické transakční náklady: 0.1-0.5\%
    \item Minimální pohyb pro profitabilní obchod: $\sim 1\%$
    \item 3\% poskytuje dostatečnou ``bezpečnostní rezervu''
    \item Historicky $\sim 30\%$ měsíců má pohyb $> \pm 3\%$
\end{itemize}

\sekce{Problém chybějících dat}

\podsekce{Klasifikace Missing Data Mechanismů}

Podle Rubina (1976) existují tři mechanismy:

\begin{table}[H]
\centering
\caption{Missing Data Mechanismy}
\begin{tabular}{|p{2cm}|p{5cm}|p{4cm}|}
\hline
\textbf{Mechanismus} & \textbf{Definice} & \textbf{V našem případě} \\ \hline
MCAR & Chybění nezávisí na žádných hodnotách & API limit, neexistující data \\ \hline
MAR & Chybění závisí na pozorovaných hodnotách & - \\ \hline
MNAR & Chybění závisí na nepozorovaných hodnotách & - \\ \hline
\end{tabular}
\end{table}

\podsekce{Přístup k imputaci}

Regresní imputace pomocí Random Forest:

$$\hat{F}_t = RF(OHLCV_t, TechIndicators_t)$$

Kde:
\begin{itemize}
    \item $\hat{F}_t$ = predikované fundamentální metriky v čase $t$
    \item $RF$ = Random Forest regressor
    \item $OHLCV_t$ = cenová data v čase $t$
    \item $TechIndicators_t$ = technické indikátory v čase $t$
\end{itemize}

% ============================================
% KAPITOLA 3: MATEMATICKÉ ZÁKLADY
% ============================================
\kapitola{Matematické základy}

\sekce{Random Forest}

\podsekce{Definice}

\textbf{Random Forest} je ensemble metoda kombinující více rozhodovacích stromů:

$$\hat{f}_{RF}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

Kde:
\begin{itemize}
    \item $B$ = počet stromů (n\_estimators)
    \item $T_b$ = b-tý rozhodovací strom
    \item $x$ = vstupní vektor features
\end{itemize}

\podsekce{Konstrukce stromu}

Pro každý uzel $t$ s daty $D_t$:

\begin{enumerate}
    \item Náhodně vyber $m$ features z celkových $p$ (typicky $m = \sqrt{p}$)
    \item Najdi nejlepší split $(j^*, s^*)$:
\end{enumerate}

$$(j^*, s^*) = \arg\min_{j \in M} \arg\min_{s} [L(D_{left}) + L(D_{right})]$$

Kde $L$ je loss funkce (Gini impurity pro klasifikaci, MSE pro regresi).

\podsekce{Gini Impurity}

$$Gini(t) = 1 - \sum_{k=1}^{K} p_{tk}^2$$

Kde $p_{tk}$ je proporce třídy $k$ v uzlu $t$.

\textbf{Interpretace:}
\begin{itemize}
    \item $Gini = 0$: Čistý uzel (všechny vzorky jedné třídy)
    \item $Gini = 0.5$: Maximální impurity pro binární klasifikaci
\end{itemize}

\podsekce{Feature Importance}

Mean Decrease in Impurity (MDI):

$$Importance(X_j) = \sum_{t \in T} \frac{n_t}{n} \cdot \Delta impurity(t, X_j)$$

\sekce{Evaluační metriky}

\podsekce{Klasifikační metriky}

Per-class metriky:

$$Precision_k = \frac{TP_k}{TP_k + FP_k}$$

$$Recall_k = \frac{TP_k}{TP_k + FN_k}$$

$$F1_k = 2 \cdot \frac{Precision_k \cdot Recall_k}{Precision_k + Recall_k}$$

\podsekce{Agregované metriky}

$$Accuracy = \frac{\sum_k TP_k}{N}$$

$$Macro\ F1 = \frac{1}{K} \sum_{k=1}^{K} F1_k$$

$$Weighted\ F1 = \sum_{k=1}^{K} \frac{n_k}{N} \cdot F1_k$$

\podsekce{ROC a AUC}

\textbf{ROC Curve:}
\begin{itemize}
    \item True Positive Rate: $TPR = \frac{TP}{TP + FN}$
    \item False Positive Rate: $FPR = \frac{FP}{FP + TN}$
\end{itemize}

\textbf{AUC (Area Under Curve):}

$$AUC = \int_0^1 TPR(FPR^{-1}(x)) dx$$

\textbf{Interpretace:}
\begin{itemize}
    \item AUC = 0.5: Náhodný klasifikátor
    \item AUC = 1.0: Perfektní klasifikátor
\end{itemize}

\podsekce{Regresní metriky (pro imputaci)}

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

\sekce{Cross-Validation pro časové řady}

\podsekce{TimeSeriesSplit}

Pro časové řady nelze použít náhodnou cross-validaci (data leakage). TimeSeriesSplit zajišťuje, že trénovací data jsou vždy před testovacími:

\begin{verbatim}
Fold 1: Train [1, ..., n₁]     Test [n₁+1, ..., n₂]
Fold 2: Train [1, ..., n₂]     Test [n₂+1, ..., n₃]
Fold 3: Train [1, ..., n₃]     Test [n₃+1, ..., n₄]
\end{verbatim}
% =========================================================================
% ČÁST 3: Kapitoly 4-6 (Výběr algoritmů, Architektura, Implementace)
% =========================================================================

% ============================================
% KAPITOLA 4: VÝBĚR ALGORITMŮ
% ============================================
\kapitola{Výběr algoritmů}

\sekce{Proč Random Forest?}

\podsekce{Srovnání s alternativami}

\begin{table}[H]
\centering
\caption{Srovnání ML algoritmů}
\begin{tabular}{|p{2.5cm}|p{4cm}|p{4cm}|p{1.5cm}|}
\hline
\textbf{Algoritmus} & \textbf{Výhody} & \textbf{Nevýhody} & \textbf{Vhodnost} \\ \hline
Random Forest & Interpretovatelný, robustní, nativní feature importance & Pomalejší než boosting & $\star\star\star\star\star$ \\ \hline
XGBoost/LightGBM & Rychlý, vysoká přesnost & Méně interpretovatelný, náchylný k overfittingu & $\star\star\star\star$ \\ \hline
Neural Networks & Zachycuje komplexní vzory & Black-box, potřebuje hodně dat & $\star\star\star$ \\ \hline
SVM & Dobrý pro malé datasety & Pomalý trénink, obtížná interpretace & $\star\star$ \\ \hline
Logistic Regression & Velmi interpretovatelný & Lineární, omezená kapacita & $\star\star$ \\ \hline
\end{tabular}
\end{table}

\podsekce{Zdůvodnění volby RF}

\begin{enumerate}
    \item \textbf{Konzistence:} Stejný algoritmus pro imputaci i klasifikaci
    \item \textbf{Interpretovatelnost:} Feature importance pro analýzu
    \item \textbf{Robustnost:} Ensemble metoda odolná vůči šumu
    \item \textbf{Flexibilita:} Nativní podpora multi-class klasifikace
    \item \textbf{Class balancing:} \texttt{class\_weight='balanced'}
\end{enumerate}

\sekce{Hyperparametry RF}

\begin{table}[H]
\centering
\caption{Hyperparametry Random Forest}
\begin{tabular}{|p{3.5cm}|p{2.5cm}|p{6cm}|}
\hline
\textbf{Parametr} & \textbf{Hodnota} & \textbf{Zdůvodnění} \\ \hline
\texttt{n\_estimators} & 100-200 & Více stromů = stabilnější predikce \\ \hline
\texttt{max\_depth} & 10-15 & Prevence overfittingu \\ \hline
\texttt{min\_samples\_split} & 5-10 & Regularizace \\ \hline
\texttt{min\_samples\_leaf} & 2-4 & Zajištění robustních listů \\ \hline
\texttt{class\_weight} & 'balanced' & Kompenzace nevyvážených tříd \\ \hline
\texttt{random\_state} & 42 & Reprodukovatelnost \\ \hline
\end{tabular}
\end{table}

% ============================================
% KAPITOLA 5: ARCHITEKTURA ŘEŠENÍ
% ============================================
\kapitola{Architektura řešení}

\sekce{High-Level Pipeline}

Celková architektura řešení se skládá z následujících fází:

\textbf{Fáze 1: Data Collection}
\begin{itemize}
    \item OHLCV Data (2015-2025): Open, High, Low, Close, Volume + Technické indikátory (RSI, MACD, SMA, EMA)
    \item Fundamental Data (2024-2025): P/E, P/B, P/S, EV/EBITDA, ROE, ROA, Margins, Debt ratios
\end{itemize}

\textbf{Fáze 2: Imputation Model (Random Forest Regressor)}
\begin{itemize}
    \item Training: OHLCV (2024-2025) $\rightarrow$ Fundamentals (2024-2025)
    \item Inference: OHLCV (2015-2024) $\rightarrow$ Predicted Fundamentals (2015-2024)
    \item Input Features (18): OHLCV (5) + Technical (8) + Derived (5)
    \item Output Targets (11): Valuation (3) + Profitability (5) + Health (3)
\end{itemize}

\textbf{Fáze 3: Complete Dataset (2015-2025)}
\begin{itemize}
    \item 2015-2024: OHLCV + Predicted Fundamentals (\texttt{data\_source='predicted'})
    \item 2024-2025: OHLCV + Real Fundamentals (\texttt{data\_source='real'})
    \item Total: $\sim$3,380 records $\times$ 30 tickerů $\times$ 3 sektory
\end{itemize}

\textbf{Fáze 4: Classification Model (Random Forest Classifier)}
\begin{itemize}
    \item Target Definition: Class 0 (DOWN), Class 1 (HOLD), Class 2 (UP)
    \item Features: OHLCV (5) + Technical (13) + Fundamental (11) = 29 features
    \item Training: Chronological split (80\% train / 20\% test)
\end{itemize}

\textbf{Fáze 5: Output \& Evaluation}
\begin{itemize}
    \item Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, AUC-ROC
    \item Outputs: Trained models (.pkl), Predictions, Feature Importance, Visualizations
\end{itemize}

\sekce{Datový tok}

Pipeline se skládá z následujících kroků:

\begin{enumerate}
    \item \textbf{Krok 1:} Download OHLCV $\rightarrow$ \texttt{data/ohlcv/all\_sectors\_ohlcv.csv}
    \item \textbf{Krok 2:} Download Fundamentals $\rightarrow$ \texttt{data/fundamentals/all\_sectors\_fundamentals.csv}
    \item \textbf{Krok 3:} Train RF Regressor $\rightarrow$ \texttt{models/fundamental\_predictor.pkl}
    \item \textbf{Krok 4:} Complete Historical $\rightarrow$ \texttt{data/complete/all\_sectors\_complete\_10y.csv}
    \item \textbf{Krok 5:} Train RF Classifier
    \item \textbf{Krok 6-7:} Tuning + Evaluation
\end{enumerate}

% ============================================
% KAPITOLA 6: IMPLEMENTACE PIPELINE
% ============================================
\kapitola{Implementace Pipeline}

\sekce{Krok 1: Stažení OHLCV Dat}

\podsekce{Skript: download\_30\_tickers.py}

\begin{minted}[frame=single,fontsize=\small,linenos]{python}
#!/usr/bin/env python3
"""Stažení 30 tickerů (10 per sektor) pro pipeline."""

import yfinance as yf
import pandas as pd

# Konfigurace
TICKERS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", 
                   "AVGO", "ORCL", "CSCO", "ADBE", "CRM"],
    "Consumer": ["AMZN", "TSLA", "HD", "MCD", "NKE",
                 "SBUX", "TGT", "LOW", "PG", "KO"],
    "Industrials": ["CAT", "HON", "UPS", "BA", "GE",
                    "RTX", "DE", "LMT", "MMM", "UNP"]
}

def calculate_rsi(series, period=14):
    """RSI indikátor"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    """MACD indikátor"""
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal
\end{minted}

\textbf{Výstup:} Soubor \texttt{data/ohlcv/all\_sectors\_ohlcv\_10y.csv} obsahující 3,870 řádků, 30 tickerů, 10.7 let historie.

\sekce{Krok 2: Stažení Fundamentálních Dat}

\podsekce{Skript: download\_fundamentals.py}

\begin{minted}[frame=single,fontsize=\small,linenos]{python}
def get_fundamentals(ticker):
    """Stáhne fundamentální metriky pro ticker"""
    info = yf.Ticker(ticker).info
    
    return {
        # Valuační
        'trailingPE': info.get('trailingPE'),
        'forwardPE': info.get('forwardPE'),
        'priceToBook': info.get('priceToBook'),
        
        # Profitabilita
        'returnOnEquity': info.get('returnOnEquity'),
        'returnOnAssets': info.get('returnOnAssets'),
        'profitMargins': info.get('profitMargins'),
        
        # Finanční zdraví
        'debtToEquity': info.get('debtToEquity'),
        'currentRatio': info.get('currentRatio'),
        'beta': info.get('beta')
    }
\end{minted}

\textbf{Stažené metriky (25 sloupců):}

\begin{table}[H]
\centering
\caption{Kategorie fundamentálních metrik}
\begin{tabular}{|p{3cm}|p{8cm}|}
\hline
\textbf{Kategorie} & \textbf{Metriky} \\ \hline
Valuační & trailingPE, forwardPE, priceToBook, enterpriseToEbitda \\ \hline
Profitabilita & returnOnEquity, returnOnAssets, profitMargins, operatingMargins \\ \hline
Zadluženost & debtToEquity, currentRatio, quickRatio \\ \hline
Riziko & beta \\ \hline
\end{tabular}
\end{table}

\sekce{Krok 3: Trénink RF Regressoru}

\podsekce{Skript: train\_rf\_regressor.py}

\begin{minted}[frame=single,fontsize=\small,linenos]{python}
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Features pro predikci fundamentů
OHLCV_FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'volatility', 'returns', 'rsi_14', 
    'macd', 'macd_signal', 'macd_hist',
    'sma_3', 'sma_6', 'sma_12',
    'ema_3', 'ema_6', 'ema_12',
    'volume_change'
]

# Model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Trénink
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
\end{minted}

\textbf{Výsledky imputace:}

\begin{table}[H]
\centering
\caption{Výsledky RF Regressoru pro imputaci}
\begin{tabular}{|p{3cm}|p{2cm}|p{2cm}|p{3cm}|}
\hline
\textbf{Target} & \textbf{MAE} & \textbf{R² Score} & \textbf{Kvalita} \\ \hline
trailingPE & 4.419 & 0.957 & $\star\star\star\star\star$ \\ \hline
forwardPE & 2.595 & 0.964 & $\star\star\star\star\star$ \\ \hline
returnOnAssets & 0.015 & 0.970 & $\star\star\star\star\star$ \\ \hline
returnOnEquity & 0.045 & 0.935 & $\star\star\star\star$ \\ \hline
priceToBook & 1.854 & 0.891 & $\star\star\star\star$ \\ \hline
profitMargins & 0.031 & 0.886 & $\star\star\star\star$ \\ \hline
debtToEquity & 38.513 & 0.765 & $\star\star\star$ \\ \hline
\end{tabular}
\end{table}

\textbf{Průměrné $R^2$: 0.91} - Excelentní kvalita imputace.

\sekce{Krok 4: Kompletace Historických Dat}

\begin{minted}[frame=single,fontsize=\small,linenos]{python}
# Rozdělení dat
cutoff_date = df['date'].max() - pd.DateOffset(months=24)
df_recent = df[df['date'] >= cutoff_date]      # Reálné fundamenty
df_historical = df[df['date'] < cutoff_date]   # K imputaci

# Imputace
X_hist = df_historical[OHLCV_FEATURES]
X_hist_scaled = scaler.transform(X_hist)
predicted_funds = model.predict(X_hist_scaled)

# Označení zdroje dat
df_recent['data_source'] = 'real'
df_historical['data_source'] = 'predicted'

# Spojení
df_complete = pd.concat([df_historical, df_recent])
\end{minted}

\textbf{Statistiky:}

\begin{table}[H]
\centering
\caption{Rozdělení dat podle zdroje}
\begin{tabular}{|p{5cm}|p{3cm}|}
\hline
\textbf{Část} & \textbf{Počet řádků} \\ \hline
Recent (reálné fundamenty) & 650 \\ \hline
Historical (predikované) & 2,730 \\ \hline
\textbf{Celkem} & \textbf{3,380} \\ \hline
\end{tabular}
\end{table}

\sekce{Krok 5: Trénink RF Classifieru}

\podsekce{Skript: train\_rf\_classifier.py}

\begin{minted}[frame=single,fontsize=\small,linenos]{python}
from sklearn.ensemble import RandomForestClassifier

# Definice target variable
THRESHOLD = 0.03  # ±3%

def create_target(df):
    """Vytvoří klasifikační target"""
    df['future_close'] = df.groupby('ticker')['close'].shift(-1)
    df['future_return'] = (df['future_close'] - df['close']) / df['close']
    
    def classify(ret):
        if ret < -THRESHOLD:
            return 0  # DOWN
        elif ret > THRESHOLD:
            return 2  # UP
        else:
            return 1  # HOLD
    
    df['target'] = df['future_return'].apply(classify)
    return df

# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
\end{minted}

\textbf{Distribuce tříd:}

\begin{table}[H]
\centering
\caption{Distribuce cílových tříd}
\begin{tabular}{|p{2cm}|p{2cm}|p{2cm}|}
\hline
\textbf{Třída} & \textbf{Počet} & \textbf{Procento} \\ \hline
DOWN & 871 & 26.0\% \\ \hline
HOLD & 1,111 & 33.2\% \\ \hline
UP & 1,368 & 40.8\% \\ \hline
\end{tabular}
\end{table}

\sekce{Krok 6: Hyperparameter Tuning}

\begin{minted}[frame=single,fontsize=\small,linenos]{python}
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Grid search prostor
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'class_weight': ['balanced']
}

# TimeSeriesSplit pro časovou konzistenci
tscv = TimeSeriesSplit(n_splits=5)

# Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=tscv,
    scoring='f1_weighted',
    n_jobs=-1
)

grid_search.fit(X_scaled, y)
\end{minted}

\textbf{Nejlepší parametry:}

\begin{verbatim}
{
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced"
}
\end{verbatim}
% =========================================================================
% ČÁST 4: Kapitoly 7-9 (Experiment, Výsledky, Vizualizace)
% =========================================================================

% ============================================
% KAPITOLA 7: EXPERIMENT 30 TICKERŮ
% ============================================
\kapitola{Experiment: 30 tickerů}

\sekce{Konfigurace experimentu}

\begin{table}[H]
\centering
\caption{Konfigurace experimentu}
\begin{tabular}{|p{4cm}|p{6cm}|}
\hline
\textbf{Parametr} & \textbf{Hodnota} \\ \hline
Počet tickerů & 30 \\ \hline
Počet sektorů & 3 \\ \hline
Tickerů per sektor & 10 \\ \hline
Období & 2014-01-01 až 2024-12-31 \\ \hline
Frekvence & Měsíční \\ \hline
Target threshold & $\pm 3\%$ \\ \hline
\end{tabular}
\end{table}

\sekce{Vybrané tickery}

\begin{table}[H]
\centering
\caption{Rozdělení tickerů podle sektorů}
\begin{tabular}{|p{2.5cm}|p{10cm}|}
\hline
\textbf{Sektor} & \textbf{Tickery} \\ \hline
Technology & AAPL, MSFT, NVDA, GOOGL, META, AVGO, ORCL, CSCO, ADBE, CRM \\ \hline
Consumer & AMZN, TSLA, HD, MCD, NKE, SBUX, TGT, LOW, PG, KO \\ \hline
Industrials & CAT, HON, UPS, BA, GE, RTX, DE, LMT, MMM, UNP \\ \hline
\end{tabular}
\end{table}

\sekce{Statistiky datasetu}

\begin{table}[H]
\centering
\caption{Statistiky finálního datasetu}
\begin{tabular}{|p{5cm}|p{4cm}|}
\hline
\textbf{Metrika} & \textbf{Hodnota} \\ \hline
Celkem řádků & 3,870 \\ \hline
Po čištění & 3,380 \\ \hline
Časové období & 10.7 let \\ \hline
OHLCV features & 5 \\ \hline
Technické indikátory & 13 \\ \hline
Fundamentální metriky & 11 \\ \hline
\textbf{Celkem features} & \textbf{29} \\ \hline
\end{tabular}
\end{table}

% ============================================
% KAPITOLA 8: VÝSLEDKY A ANALÝZA
% ============================================
\kapitola{Výsledky a analýza}

\sekce{RF Regressor (Imputace)}

\podsekce{Výsledky per-target}

\begin{table}[H]
\centering
\caption{Výsledky RF Regressoru - detailní}
\begin{tabular}{|p{3cm}|p{2cm}|p{2cm}|p{3cm}|}
\hline
\textbf{Target} & \textbf{MAE} & \textbf{R² Score} & \textbf{Kvalita} \\ \hline
trailingPE & 4.419 & 0.957 & $\star\star\star\star\star$ \\ \hline
forwardPE & 2.595 & 0.964 & $\star\star\star\star\star$ \\ \hline
returnOnAssets & 0.015 & 0.970 & $\star\star\star\star\star$ \\ \hline
returnOnEquity & 0.045 & 0.935 & $\star\star\star\star$ \\ \hline
priceToBook & 1.854 & 0.891 & $\star\star\star\star$ \\ \hline
profitMargins & 0.031 & 0.886 & $\star\star\star\star$ \\ \hline
debtToEquity & 38.513 & 0.765 & $\star\star\star$ \\ \hline
\end{tabular}
\end{table}

\textbf{Průměrné $R^2$: 0.91}

\podsekce{Feature Importance (Regressor)}

\begin{table}[H]
\centering
\caption{Top 5 nejdůležitějších features pro imputaci}
\begin{tabular}{|p{1cm}|p{3cm}|p{2.5cm}|}
\hline
\textbf{Rank} & \textbf{Feature} & \textbf{Importance} \\ \hline
1 & \textbf{volume} & 0.4995 \\ \hline
2 & sma\_12 & 0.0734 \\ \hline
3 & ema\_12 & 0.0730 \\ \hline
4 & sma\_6 & 0.0586 \\ \hline
5 & ema\_6 & 0.0583 \\ \hline
\end{tabular}
\end{table}

\textbf{Poznatek:} Volume je dominantní prediktor fundamentálních metrik (korelace s tržní kapitalizací a likviditou).

\sekce{RF Classifier (Klasifikace)}

\podsekce{Celkové výsledky}

\begin{table}[H]
\centering
\caption{Celkové výsledky klasifikátoru}
\begin{tabular}{|p{4cm}|p{3cm}|}
\hline
\textbf{Metrika} & \textbf{Hodnota} \\ \hline
Accuracy & 32.09\% \\ \hline
Precision & 32.87\% \\ \hline
Recall & 32.09\% \\ \hline
F1-Score & 31.00\% \\ \hline
Random baseline & 33.33\% \\ \hline
Test samples & 670 \\ \hline
\end{tabular}
\end{table}

\podsekce{Classification Report}

\begin{verbatim}
              precision    recall  f1-score   support

        DOWN       0.30      0.51      0.38       193
        HOLD       0.33      0.20      0.25       216
          UP       0.35      0.28      0.31       261

    accuracy                           0.32       670
   macro avg       0.33      0.33      0.31       670
weighted avg       0.33      0.32      0.31       670
\end{verbatim}

\podsekce{Per-Sector Analýza}

\begin{table}[H]
\centering
\caption{Výsledky podle sektorů}
\begin{tabular}{|p{3cm}|p{2cm}|p{2cm}|p{2cm}|}
\hline
\textbf{Sektor} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{Samples} \\ \hline
\textbf{Industrials} & 35.9\% & 34.6\% & 231 \\ \hline
Consumer & 30.4\% & 29.8\% & 181 \\ \hline
Technology & 29.8\% & 27.6\% & 258 \\ \hline
\end{tabular}
\end{table}

\textbf{Poznatek:} Industrials sektor je nejlépe predikovatelný. Technology má nejvyšší volatilitu a je nejtěžší k predikci.

\podsekce{Feature Importance (Classifier)}

\begin{table}[H]
\centering
\caption{Top 10 nejdůležitějších features pro klasifikaci}
\begin{tabular}{|p{1cm}|p{3.5cm}|p{2.5cm}|p{3cm}|}
\hline
\textbf{Rank} & \textbf{Feature} & \textbf{Importance} & \textbf{Typ} \\ \hline
1 & returns & 0.0577 & Technický \\ \hline
2 & volatility & 0.0560 & Technický \\ \hline
3 & macd\_hist & 0.0489 & Technický \\ \hline
4 & macd\_signal & 0.0481 & Technický \\ \hline
5 & volume\_change & 0.0449 & Technický \\ \hline
6 & rsi\_14 & 0.0430 & Technický \\ \hline
7 & macd & 0.0392 & Technický \\ \hline
8 & returnOnEquity & 0.0380 & Fundamentální \\ \hline
9 & returnOnAssets & 0.0373 & Fundamentální \\ \hline
10 & currentRatio & 0.0359 & Fundamentální \\ \hline
\end{tabular}
\end{table}

\textbf{Poznatky:}
\begin{itemize}
    \item Technické indikátory dominují (7 z top 10)
    \item Momentum features (returns, MACD) jsou nejdůležitější
    \item Fundamenty (ROE, ROA) jsou stále významné (top 10)
\end{itemize}

\sekce{Interpretace výsledků}

\podsekce{Accuracy vs. Random Baseline}

\begin{itemize}
    \item \textbf{Model accuracy:} 32.1\%
    \item \textbf{Random baseline (3 třídy):} 33.3\%
    \item \textbf{Rozdíl:} -1.2\%
\end{itemize}

\textbf{Interpretace:} Model dosahuje accuracy blízké náhodnému klasifikátoru. Toto je typické pro finanční predikce a odráží vysokou efektivitu trhů.

\podsekce{Analýza Confusion Matrix}

\begin{verbatim}
              DOWN  HOLD    UP
   DOWN       98    39      56    (51% recall)
   HOLD       72    44      100   (20% recall)
   UP         84    85      92    (35% recall)
\end{verbatim}

\textbf{Poznatky:}
\begin{enumerate}
    \item Model má tendenci predikovat DOWN častěji
    \item HOLD je nejhůře rozpoznávaná třída (pouze 20\% recall)
    \item Nejvíce záměn mezi UP a HOLD
\end{enumerate}

\podsekce{AUC Skóre}

\begin{table}[H]
\centering
\caption{AUC skóre pro jednotlivé třídy}
\begin{tabular}{|p{2cm}|p{2cm}|}
\hline
\textbf{Třída} & \textbf{AUC} \\ \hline
DOWN & $\sim$0.55 \\ \hline
HOLD & $\sim$0.52 \\ \hline
UP & $\sim$0.54 \\ \hline
\end{tabular}
\end{table}

Hodnoty AUC blízko 0.5 indikují slabou separabilitu tříd.

% ============================================
% KAPITOLA 9: VIZUALIZACE
% ============================================
\kapitola{Vizualizace}

\sekce{Confusion Matrix}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{data/30_tickers/figures/confusion_matrix.png}
    \caption{Confusion Matrix - matice záměn ukazující distribuci skutečných vs. predikovaných tříd. Diagonála reprezentuje správné predikce.}
    \label{fig:confusion_matrix}
\end{figure}

\sekce{ROC Křivky}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{data/30_tickers/figures/roc_curves.png}
    \caption{ROC křivky pro každou třídu. Čím blíže křivka k levému hornímu rohu, tím lepší separabilita.}
    \label{fig:roc_curves}
\end{figure}

\sekce{Feature Importance}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\linewidth]{data/30_tickers/figures/feature_importance.png}
    \caption{Relativní důležitost jednotlivých features pro klasifikační model.}
    \label{fig:feature_importance}
\end{figure}

\sekce{Porovnání Sektorů}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\linewidth]{data/30_tickers/figures/sector_comparison.png}
    \caption{Porovnání accuracy, precision, recall a F1 mezi sektory.}
    \label{fig:sector_comparison}
\end{figure}

% =========================================================================
% ČÁST 5: Kapitoly 10-13 (Omezení, Závěr, Reference, Přílohy)
% =========================================================================

% ============================================
% KAPITOLA 10: OMEZENÍ A BUDOUCÍ PRÁCE
% ============================================
\kapitola{Omezení a budoucí práce}

\sekce{Datová omezení}

\podsekce{Survivorship Bias}

\textbf{Problém:} Dataset obsahuje pouze akcie aktuálně v S\&P 500. Firmy, které zbankrotovaly nebo byly vyřazeny, chybí.

\textbf{Důsledek:} Potenciální nadhodnocení výkonnosti modelu.

\textbf{Mitigace:}
\begin{itemize}
    \item Použití historických konstituentů indexu (vyžaduje placená data)
    \item Explicitní disclaimer v interpretaci
\end{itemize}

\podsekce{Look-Ahead Bias}

\textbf{Problém:} Fundamentální metriky jsou publikovány se zpožděním (quarterly reports 1-2 měsíce po konci kvartálu).

\textbf{Mitigace:}
\begin{itemize}
    \item Použití lagovaných dat
    \item Point-in-time databáze
\end{itemize}

\podsekce{Kvalita imputovaných dat}

\textbf{Problém:} $\sim$80\% fundamentálních dat je predikováno modelem, nikoli skutečných.

\textbf{Důsledek:} Chyby imputace se propagují do klasifikátoru.

\textbf{Mitigace:}
\begin{itemize}
    \item Confidence intervals pro imputované hodnoty
    \item Sensitivity analýza
    \item Sloupec \texttt{data\_source} pro transparentnost
\end{itemize}

\sekce{Modelová omezení}

\podsekce{Stacionarita}

\textbf{Předpoklad:} Vztahy mezi features a targetem jsou stabilní v čase.

\textbf{Realita:} Tržní dynamika se mění (COVID-19, úrokové sazby, geopolitika).

\textbf{Mitigace:}
\begin{itemize}
    \item Rolling window training
    \item Periodic retraining
    \item Regime detection
\end{itemize}

\podsekce{Transakční náklady}

\textbf{Problém:} Model nezahrnuje bid-ask spread, poplatky, market impact, daně.

\textbf{Důsledek:} Skutečná výkonnost bude nižší než backtest.

\sekce{Budoucí rozšíření}

\begin{table}[H]
\centering
\caption{Navrhovaná budoucí rozšíření}
\begin{tabular}{|p{3cm}|p{6cm}|p{2cm}|}
\hline
\textbf{Rozšíření} & \textbf{Popis} & \textbf{Priorita} \\ \hline
Více tickerů & 100-150 tickerů, více sektorů & $\star\star\star\star\star$ \\ \hline
Alternative data & Sentiment z news/social media & $\star\star\star\star$ \\ \hline
Deep Learning & LSTM/Transformer pro časové řady & $\star\star\star$ \\ \hline
Ensemble & Kombinace více modelů & $\star\star\star$ \\ \hline
Real-time & Automatizovaný trading systém & $\star\star$ \\ \hline
\end{tabular}
\end{table}

% ============================================
% KAPITOLA 11: ZÁVĚR
% ============================================
\kapitola{Závěr}

\sekce{Shrnutí dosažených výsledků}

\podsekce{Co funguje dobře}

\begin{enumerate}
    \item \textbf{RF Regressor pro imputaci} - $R^2$ 0.76-0.97 je excelentní
    \item \textbf{Hybridní přístup} - Umožňuje využít fundamenty i pro historii
    \item \textbf{Technické indikátory} - Returns a volatility jsou nejdůležitější
    \item \textbf{Industrials sektor} - Model zde funguje nejlépe (35.9\%)
\end{enumerate}

\podsekce{Limitace}

\begin{enumerate}
    \item \textbf{Accuracy $\sim$32\%} - Blízko random baseline
    \item \textbf{HOLD třída} - Nejhůře rozpoznávaná (20\% recall)
    \item \textbf{Finanční trhy} - Inherentně těžko predikovatelné (EMH)
\end{enumerate}

\sekce{Vědecký přínos}

\begin{enumerate}
    \item \textbf{Metodologický:} Demonstrace hybridního přístupu k řešení chybějících dat
    \item \textbf{Praktický:} Funkční end-to-end ML pipeline pro finanční predikce
    \item \textbf{Analytický:} Feature importance analýza technických vs. fundamentálních faktorů
\end{enumerate}

\sekce{Doporučení}

Pro zlepšení výsledků doporučuji:

\begin{enumerate}
    \item \textbf{Více dat} - 100+ tickerů, delší historie
    \item \textbf{Feature engineering} - Sentiment, makroekonomické indikátory
    \item \textbf{Jiné modely} - XGBoost, LSTM
    \item \textbf{Binární klasifikace} - UP vs NOT UP (snazší problém)
    \item \textbf{Confidence thresholds} - Obchodovat pouze při vysoké jistotě
\end{enumerate}

% ============================================
% KAPITOLA 12: REFERENCE
% ============================================
\kapitola{Reference}

\begin{literatura}

\citace{fama1970}{Fama, 1970}{\autor{Fama, E. F.} \nazev{Efficient capital markets: A review of theory and empirical work.} The Journal of Finance, 25(2), 383-417. 1970.}

\citace{fama1992}{Fama a French, 1992}{\autor{Fama, E. F., \& French, K. R.} \nazev{The cross-section of expected stock returns.} The Journal of Finance, 47(2), 427-465. 1992.}

\citace{breiman2001}{Breiman, 2001}{\autor{Breiman, L.} \nazev{Random forests.} Machine Learning, 45(1), 5-32. 2001.}

\citace{gu2020}{Gu, Kelly a Xiu, 2020}{\autor{Gu, S., Kelly, B., \& Xiu, D.} \nazev{Empirical asset pricing via machine learning.} The Review of Financial Studies, 33(5), 2223-2273. 2020.}

\citace{graham1934}{Graham a Dodd, 1934}{\autor{Graham, B., \& Dodd, D.} \nazev{Security Analysis.} McGraw-Hill. 1934.}

\citace{pedregosa2011}{Pedregosa et al., 2011}{\autor{Pedregosa, F., et al.} \nazev{Scikit-learn: Machine learning in Python.} Journal of Machine Learning Research, 12, 2825-2830. 2011.}

\citace{mckinney2010}{McKinney, 2010}{\autor{McKinney, W.} \nazev{Data structures for statistical computing in Python.} Proceedings of the 9th Python in Science Conference. 2010.}

\citace{yfinance2024}{Yahoo Finance API, 2024}{\autor{Yahoo Finance.} \nazev{yfinance Python package.} [online]. 2024. Dostupné z: \url{https://pypi.org/project/yfinance/}.}

\citace{sklearn2024}{Scikit-learn Documentation, 2024}{\autor{Scikit-learn developers.} \nazev{Scikit-learn Documentation.} [online]. 2024. Dostupné z: \url{https://scikit-learn.org/stable/}.}

\end{literatura}

% ============================================
% KAPITOLA 13: PŘÍLOHY
% ============================================
\kapitola{Přílohy}

\sekce{Příloha A: Kompletní seznam Features}

\podsekce{A.1 OHLCV Features (5)}

\begin{verbatim}
open, high, low, close, volume
\end{verbatim}

\podsekce{A.2 Technical Indicators (13)}

\begin{verbatim}
volatility, returns,
rsi_14, macd, macd_signal, macd_hist,
sma_3, sma_6, sma_12,
ema_3, ema_6, ema_12,
volume_change
\end{verbatim}

\podsekce{A.3 Fundamental Metrics (11)}

\begin{verbatim}
trailingPE, forwardPE, priceToBook,
returnOnEquity, returnOnAssets,
profitMargins, operatingMargins, grossMargins,
debtToEquity, currentRatio,
beta
\end{verbatim}

\sekce{Příloha B: Struktura projektu}

\begin{verbatim}
CleanSolution/
│
├── DIPLOMOVA_PRACE_DOKUMENTACE.md    # Dokumentace
├── README.md                          # Přehled projektu
├── requirements.txt                   # Python závislosti
│
├── data/
│   └── 30_tickers/
│       ├── ohlcv/                     # Surová OHLCV data
│       ├── fundamentals/              # Fundamentální data
│       ├── complete/                  # Kompletní dataset
│       └── figures/                   # Vizualizace
│
├── models/
│   └── 30_tickers/
│       ├── classifiers/               # RF Classifier modely
│       ├── regressors/                # RF Regressor modely
│       ├── scalers/                   # StandardScaler objekty
│       └── metadata/                  # JSON/CSV výsledky
│
└── Skripty:
    ├── download_30_tickers.py
    ├── download_fundamentals.py
    ├── train_rf_regressor.py
    ├── train_rf_classifier.py
    ├── hyperparameter_tuning.py
    └── final_evaluation.py
\end{verbatim}

\sekce{Příloha C: Instalace a spuštění}

\podsekce{Požadavky}

\begin{verbatim}
# requirements.txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
yfinance>=0.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
\end{verbatim}

\podsekce{Spuštění Pipeline}

\begin{verbatim}
# Aktivace prostředí
cd CleanSolution
python -m venv venv
.\venv\Scripts\activate  # Windows

# Instalace závislostí
pip install -r requirements.txt

# Spuštění celé pipeline (v pořadí)
python download_30_tickers.py
python download_fundamentals.py
python train_rf_regressor.py
python train_rf_classifier.py
python hyperparameter_tuning.py
python final_evaluation.py
\end{verbatim}

\sekce{Příloha D: Výstupní soubory}

\begin{table}[H]
\centering
\caption{Hlavní výstupní soubory}
\begin{tabular}{|p{8.5cm}|p{4cm}|}
\hline
\textbf{Soubor} & \textbf{Popis} \\ \hline
\nolinkurl{data/ohlcv/all_sectors_ohlcv_10y.csv} & Surová OHLCV data \\ \hline
\nolinkurl{data/fundamentals/all_sectors_fundamentals.csv} & Fundamentální metriky \\ \hline
\nolinkurl{data/complete/all_sectors_complete_10y.csv} & Kompletní dataset \\ \hline
\nolinkurl{models/fundamental_predictor.pkl} & RF Regressor model \\ \hline
\nolinkurl{models/rf_classifier_tuned.pkl} & RF Classifier model \\ \hline
\nolinkurl{models/final_evaluation_results.json} & Výsledky evaluace \\ \hline
\nolinkurl{data/figures/confusion_matrix.png} & Confusion matrix \\ \hline
\nolinkurl{data/figures/roc_curves.png} & ROC křivky \\ \hline
\nolinkurl{data/figures/feature_importance.png} & Feature importance \\ \hline
\end{tabular}
\end{table}

% ============================================
% KONEC DOKUMENTU
% ============================================

\end{document}
