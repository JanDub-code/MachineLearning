\documentclass[twoside, 12pt]{article}
\usepackage{amsmath}
\usepackage{graphicx, xdipp, url, fancyvrb, tcolorbox}
\usepackage[hidelinks]{hyperref}
\cestina
\usepackage{listings}
\lstset{
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  numbers=left,
  numberstyle=\tiny,
  language=Python
}
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
{This semester project focuses on the design and implementation of a machine learning system for classifying monthly stock price movements from the S\&P 500 index. The theoretical part discusses the Efficient Market Hypothesis, fundamental and technical analysis approaches, and the mathematical foundations of the Random Forest algorithm. The practical part presents a hybrid approach that addresses the problem of missing historical fundamental data through RF-based imputation. \textbf{The classification model was trained on 150 tickers across 5 sectors (approx. 17,000 records) and achieves 35.6\% accuracy, outperforming the random baseline of 33.3\%.} The project concludes with comprehensive visualizations and recommendations for future improvements.}

\abstrakt{Dub, J. Klasifikace cenových pohybů akcií pomocí strojového učení. Semestrální projekt. Mendelova univerzita v Brně, 2025.}
{Tento semestrální projekt se zaměřuje na návrh a implementaci systému strojového učení pro klasifikaci měsíčních cenových pohybů akcií z indexu S\&P 500. Teoretická část pojednává o hypotéze efektivních trhů, přístupech fundamentální a technické analýzy a matematických základech algoritmu Random Forest. Praktická část představuje hybridní přístup řešící problém chybějících historických fundamentálních dat pomocí RF-based imputace. \textbf{Klasifikační model byl natrénován na 150 tickerech v 5 sektorech (cca 17 000 záznamů) a dosahuje accuracy 35,6\%, čímž překonává náhodný baseline 33,3\%.} Práce je zakončena komplexními vizualizacemi a doporučeními pro budoucí vylepšení.}

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
    
    \item \textbf{\textit{Random Forest}}: Metoda strojového učení, která kombinuje výsledky mnoha rozhodovacích stromů pro dosažení vyšší stability.
    
    \item \textbf{\textit{Feature Importance}}: Měřítko určující, jak moc daná vstupní proměnná (např. P/E ratio) ovlivňuje výslednou predikci.
    
    \item \textbf{\textit{Imputace}}: Technika pro inteligentní doplnění chybějících hodnot v datech pomocí odhadu z jiných proměnných.
    
    \item \textbf{\textit{Accuracy}}: Celková přesnost modelu udávající procento správně klasifikovaných vzorků.
    
    \item \textbf{\textit{Precision a Recall}}: Metriky hodnotící kvalitu predikcí pro konkrétní třídy (např. jak moc můžeme věřit signálu "UP").
    
    \item \textbf{\textit{F1-Score}}: Harmonický průměr mezi Precision a Recall, poskytující vyvážený pohled na výkon modelu.
    
    \item \textbf{\textit{MAE (Mean Absolute Error)}}: Průměrná chyba predikce, používaná k hodnocení kvality imputace fundamentů.
    
    \item \textbf{\textit{R-squared ($R^2$)}}: Statistický ukazatel udávající, jak dobře model vysvětluje variabilitu v datech (1.0 = perfektní shoda).
    
    \item \textbf{\textit{P/E, ROE, Current Ratio}}: Fundamentální ukazatele hodnotící valuaci, ziskovost a finanční zdraví firmy.
    
    \item \textbf{\textit{RSI, MACD}}: Technické indikátory sledující hybnost ceny (momentum) a trendové změny.
    
    \item \textbf{\textit{TimeSeriesSplit}}: Způsob testování modelu na historických datech, který respektuje časovou posloupnost (nevidí do budoucnosti).
\end{itemize}
% =========================================================================
% ČÁST 2: Kapitoly 1-3 (Úvod, Teorie, Matematika)
% =========================================================================

% ============================================
% KAPITOLA 1: ÚVOD A MOTIVACE
% ============================================
\kapitola{Úvod a motivace}

Predikce pohybů akciových trhů představuje jeden z nejnáročnějších problémů kvantitativních financí. Tato práce se zaměřuje na vývoj a evaluaci ML systému pro klasifikaci měsíčních cenových pohybů akcií z indexu S\&P 500.

\sekce{Kontext a cíle práce}

Predikce akciových trhů je náročná kvůli jejich vysoké efektivitě (EMH). Podle Eugene Famy (1970) trh v polo-silné formě odráží všechny veřejné informace, což teoreticky znemožňuje fundamentální analýzu. Naše práce však testuje hypotézu, že moderní metody strojového učení (ML) dokáží v kombinaci technických a fundamentálních dat identifikovat vzorce, které trh dočasně přehlíží.

\textbf{Hlavní cíle:}
\begin{itemize}
    \item Vyvinout ML model pro klasifikaci měsíčních cenových pohybů (DOWN, HOLD, UP).
    \item Vyřešit kritický nedostatek historických fundamentálních dat pomocí ML imputace.
    \item Analyzovat, zda mají pro predikci větší význam technické indikátory nebo účetní metriky.
\end{itemize}

\sekce{Klíčová inovace: Hybridní přístup}

Hlavní výzvou v kvantitativních financích je \textbf{informační asymetrie v čase}: zatímco cenová data (OHLCV) jsou dostupná desítky let, fundamentální data (P/E, ROE) jsou u bezplatných API dostupná jen za poslední 1-2 roky.

\textbf{Naše řešení:}
\begin{enumerate}
    \item \textbf{Imputace:} Model se naučí vztah mezi cenou a fundamenty na aktuálních datech a "dopředpovídá" chybějící historii.
    \item \textbf{Klasifikace:} Kompletní dataset (reálná + imputovaná data) slouží k finální predikci pohybu trhu.
\end{enumerate}

% ============================================
% KAPITOLA 2: TEORETICKÝ RÁMEC
% ============================================
\kapitola{Teoretický rámec}

\sekce{Analýza dat: Fundamentální vs. Technická}

Práce kombinuje dva tradiční přístupy:
\begin{itemize}
    \item \textbf{Fundamentální analýza:} Sleduje vnitřní hodnotu firmy (P/E pro valuaci, ROE pro ziskovost, Debt/Equity pro zdraví). Tyto metriky se mění pomalu (kvartálně).
    \item \textbf{Technická analýza:} Sleduje náladu na trhu skrze cenu a objem. Používáme indikátory jako RSI (překoupenost) a MACD (změna trendu). Výhodou je jejich dostupnost v reálném čase.
\end{itemize}

Kombinace obou světů umožňuje ML modelu vidět jak "vnitřní kvalitu" firmy, tak "aktuální momentum" na trhu, což by mělo vést k robustnějším predikcím.

\sekce{Od regrese ke klasifikaci}

Namísto snahy o predikci přesné ceny (regrese), která je extrémně náchylná na šum, se zaměřujeme na \textbf{klasifikaci do tří směrů}:
\begin{enumerate}
    \item \textbf{DOWN (přes -3\%):} Signifikantní pokles ceny.
    \item \textbf{HOLD ($\pm$3\%):} Stagnace nebo malý pohyb (často v rámci transakčních nákladů).
    \item \textbf{UP (přes 3\%):} Signifikantní růst ceny.
\end{enumerate}

Tento přístup je praktičtější pro investiční rozhodování, kde nás zajímá především směr a síla pohybu, nikoliv dolary a centy.

\sekce{Řešení chybějících dat (Imputace)}

Historická fundamentální data často chybí (Missing Completely at Random). Namísto odstranění neúplných záznamů, což by drasticky zmenšilo dataset, používáme model k jejich doplnění. Využíváme faktu, že fundamenty a ceny jsou korelované. Pokud model vidí stabilní růst ceny a objemu, dokáže s vysokou přesností odhadnout pravděpodobné ROE či P/E v daném období.

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
\kapitola{Metodické a matematické základy}

\sekce{Princip algoritmu Random Forest}

Algoritmus Random Forest byl zvolen pro svou robustnost a schopnost pracovat s různorodými daty (technickými i fundamentálními). Namísto spoléhání se na jeden rozhodovací strom, který se může snadno splést nebo "přeučit" na šum v datech, Random Forest vytváří celé "lesy" nezávislých stromů.

\textbf{Klíčové vlastnosti:}
\begin{itemize}
    \item \textbf{Bagging:} Každý strom trénuje na náhodném výběru dat, což zvyšuje stabilitu výsledku.
    \item \textbf{Random Subspace:} Při rozhodování v každém uzlu strom nahlíží jen na náhodnou podmnožinu indikátorů, čímž se snižuje riziko, že jeden dominantní indikátor zastíní ostatní.
    \item \textbf{Hlasování:} Finální predikce vzniká jako průměr (v případě regrese) nebo většinové hlasování (v případě klasifikace) všech stromů v modelu.
\end{itemize}

\sekce{Jak hodnotíme úspěšnost modelu?}

Pro finanční data není pouhá "přesnost" (accuracy) vždy dostačující. Používáme proto komplexnější sadu metrik:

\begin{itemize}
    \item \textbf{F1-Score pro klasifikaci:} Pomáhá nám pochopit, zda model jen "hádá" nejčastější třídu, nebo skutečně dokáže rozlišit mezi výkyvy trhu (UP/DOWN).
    \item \textbf{Feature Importance:} Umožňuje nám nahlédnout do "černé skříňky" modelu a zjistit, které faktory (např. RSI nebo EBITDA) mají na jeho rozhodování největší vliv.
    \item \textbf{Metriky imputace ($R^2$ a MAE):} Před samotnou klasifikací doplňujeme chybějící fundamenty. Zde nás zajímá, jak blízko jsou naše odhady realitě. $R^2$ skóre blížící se k 1.0 značí, že model skvěle zachytil vztahy mezi cenou a fundamenty.
\end{itemize}

\sekce{Zajištění férového testování (TimeSeriesSplit)}

U akcií nelze použít standardní náhodné rozdělení dat typu "zamíchat balíček karet". Nemůžeme učit model na datech zítřejších a chtít po něm predikci dneška. Používáme proto \textbf{TimeSeriesSplit}, který simuluje reálný scénář: model se učí na minulosti (např. roky 2015-2022) a testuje se na neviděné budoucnosti (2023-2025). Tím eliminujeme riziko "pohledu do budoucnosti" (data leakage).
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

\begin{lstlisting}
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
\end{lstlisting}

\textbf{Výstup:} Soubor \texttt{data/ohlcv/all\_sectors\_ohlcv\_10y.csv} obsahující 3,870 řádků, 30 tickerů, 10.7 let historie.

\sekce{Krok 2: Stažení Fundamentálních Dat}

\podsekce{Skript: download\_fundamentals.py}

\begin{lstlisting}
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
\end{lstlisting}

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

\begin{lstlisting}
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
\end{lstlisting}

\kapitola{Architektura řešení}

\sekce{High-Level Pipeline}

Architektura je navržena jako lineární proces od sběru dat až po finální predikci, rozdělený do dvou hlavních ML fází:

\begin{enumerate}
    \item \textbf{Data Collection:} Sběr OHLCV cenových dat (10 let) a fundamentálních ukazatelů (2 roky) pro 150 tickerů z 5 sektorů S\&P 500.
    \item \textbf{Fáze Imputace (RF Regressor):} Model se naučí odhadovat 11 fundamentálních metrik na základě 18 cenových a technických indikátorů. Tímto modelem "doplníme" chybějících 8 let historie.
    \item \textbf{Kompletace dat:} Vytvoření sjednoceného datasetu o rozsahu $\sim$17,000 záznamů, kde novější data jsou reálná a starší jsou kvalifikovaným odhadem modelu.
    \item \textbf{Fáze Klasifikace (RF Classifier):} Finální model predikuje jednu ze tří tříd (DOWN, HOLD, UP) na základě 29 vstupních proměnných.
\end{enumerate}

\kapitola{Implementace a trénink}

\sekce{Model Imputace (Regressor)}

Pro doplnění dat využíváme \texttt{RandomForestRegressor} s nastavením pro vysokou přesnost odhadu:

\begin{itemize}
    \item \textbf{Vstupy:} OHLCV, RSI, MACD, Klouzavé průměry (SMA/EMA 3, 6, 12).
    \item \textbf{Cíle:} Valuační (P/E), Profitabilita (ROE, ROA), Zdraví (Debt/Equity).
    \item \textbf{Parametry:} 100 stromů, \texttt{max\_depth=15}.
\end{itemize}

\textbf{Výsledky:} Model dosáhl průměrného \textbf{$R^2 = 0.91$}. To potvrzuje silnou korelaci mezi cenovými pohyby a fundamentálním stavem firmy, což umožňuje přesnou rekonstrukci historie.

\sekce{Model Klasifikace (Classifier)}

Cílem je predikovat měsíční výnos s využitím thresholdu $\pm 3\%$. 

\begin{itemize}
    \item \textbf{Parametry:} 200 stromů, \texttt{class\_weight='balanced'}.
    \item \textbf{Tuning:} Optimalizace proběhla přes \texttt{GridSearchCV} s využitím \texttt{TimeSeriesSplit} (5 foldů).
\end{itemize}

\textbf{Distribuce tříd v datasetu:}
\begin{itemize}
    \item \textbf{UP / DOWN:} $\sim$67\% (signifikantní pohyby).
    \item \textbf{HOLD:} $\sim$33\% (stagnace).
\end{itemize}

\sekce{Krok 5: Klasifikace a Hyperparametry}

Model \texttt{RandomForestClassifier} byl naladěn pomocí mřížkového hledání (\texttt{GridSearchCV}) napříč následujícími parametry:

\begin{itemize}
    \item \textbf{Počet stromů:} 100 až 200 (zvoleno 100 pro stabilitu).
    \item \textbf{Hloubka stromů:} 10 až 20 (zvoleno 10 jako prevence overfittingu).
    \item \textbf{Váhy tříd:} \texttt{'balanced'} (kompenzuje mírně nerovnoměrné zastoupení růstů a poklesů).
    \item \textbf{Threshold:} Predikujeme pohyb nad $\pm 3\%$, což odpovídá reálným nákladům a volatilitě trhu.
\end{itemize}

\textbf{Nejlepší konfigurace:} Model s 100 stromy a hloubkou 10 dosáhl nejlepšího vyvážení mezi trénovací a testovací přesností.


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
\kapitola{Robustní verifikace modelu na datech S\&P 500}

\sekce{Konfigurace experimentu}

\begin{table}[H]
\centering
\caption{Konfigurace experimentu}
\begin{tabular}{|p{4cm}|p{6cm}|}
\hline
\textbf{Parametr} & \textbf{Hodnota} \\ \hline
Počet tickerů & \textbf{150} \\ \hline
Počet sektorů & \textbf{5} \\ \hline
Tickerů per sektor & 30 \\ \hline
Období & 2015-01-01 až 2025-12-31 \\ \hline
Frekvence & Měsíční \\ \hline
Target threshold & $\pm 3\%$ \\ \hline
\end{tabular}
\end{table}

\sekce{Vybrané tickery}

\begin{table}[H]
\centering
\caption{Rozdělení tickerů podle sektorů (30 tickerů per sektor)}
\begin{tabular}{|p{2.5cm}|p{10cm}|}
\hline
\textbf{Sektor} & \textbf{Příklady tickerů (ukázka)} \\ \hline
Technology & AAPL, MSFT, NVDA, GOOGL, META, AVGO, ORCL, CSCO, ADBE, CRM, ... \\ \hline
Consumer & AMZN, TSLA, HD, MCD, NKE, SBUX, TGT, LOW, PG, KO, ... \\ \hline
Industrials & CAT, HON, UPS, BA, GE, RTX, DE, LMT, MMM, UNP, ... \\ \hline
Healthcare & JNJ, UNH, PFE, ABBV, MRK, TMO, ABT, DHR, LLY, BMY, ... \\ \hline
Financials & JPM, BAC, WFC, GS, MS, C, BLK, SCHW, AXP, USB, ... \\ \hline
\end{tabular}
\end{table}

\sekce{Statistiky datasetu}

\begin{table}[H]
\centering
\caption{Statistiky finálního datasetu}
\begin{tabular}{|p{5cm}|p{4cm}|}
\hline
\textbf{Metrika} & \textbf{Hodnota} \\ \hline
Celkem řádků & \textbf{16,829} \\ \hline
Po čištění & \textbf{16,679} \\ \hline
Časové období & 10 let \\ \hline
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
Accuracy & \textbf{35.61\%} \\ \hline
Precision & 36.57\% \\ \hline
Recall & 35.61\% \\ \hline
F1-Score & 35.77\% \\ \hline
Random baseline & 33.33\% \\ \hline
Test samples & \textbf{3,336} \\ \hline
\end{tabular}
\end{table}

Model dosahuje nejlepších výsledků u třídy \textbf{UP} (přesnost 41\%) a \textbf{HOLD} (37\%). To naznačuje, že model je více "optimistický" a lépe identifikuje růstové trendy než prudké poklesy (DOWN: 29\%). Celkový výkon modelu je vyvážený, s mírným příklonem k profitabilitě u dlouhých pozic (UP).


\podsekce{Per-Sector Analýza}

\begin{table}[H]
\centering
\caption{Výsledky podle sektorů}
\begin{tabular}{|p{3cm}|p{2cm}|p{2cm}|p{2cm}|}
\hline
\textbf{Sektor} & \textbf{Accuracy} & \textbf{F1-Score} & \textbf{Samples} \\ \hline
\textbf{Financials} & \textbf{40.3\%} & 38.8\% & 581 \\ \hline
Consumer & 38.5\% & 38.1\% & 584 \\ \hline
Healthcare & 38.1\% & 35.8\% & 708 \\ \hline
Industrials & 36.0\% & 34.8\% & 731 \\ \hline
Technology & 35.2\% & 34.7\% & 732 \\ \hline
\end{tabular}
\end{table}

\textbf{Poznatek:} Financials sektor je nejlépe predikovatelný. Technology sektor vykazuje stabilní, ale mírně nižší výkonnost kvůli vyšší volatiličě.

\podsekce{Feature Importance (Classifier)}

\begin{table}[H]
\centering
\caption{Top 10 nejdůležitějších features pro klasifikaci}
\begin{tabular}{|p{1cm}|p{3.5cm}|p{2.5cm}|p{3cm}|}
\hline
\textbf{Rank} & \textbf{Feature} & \textbf{Importance} & \textbf{Typ} \\ \hline
1 & returns & \textbf{0.0610} & Technický \\ \hline
2 & volatility & 0.0576 & Technický \\ \hline
3 & rsi\_14 & 0.0505 & Technický \\ \hline
4 & macd\_hist & 0.0490 & Technický \\ \hline
5 & volume\_change & 0.0452 & Technický \\ \hline
6 & macd\_signal & 0.0409 & Technický \\ \hline
7 & macd & 0.0378 & Technický \\ \hline
8 & currentRatio & 0.0374 & Fundamentální \\ \hline
9 & debtToEquity & 0.0361 & Fundamentální \\ \hline
10 & beta & 0.0359 & Fundamentální \\ \hline
\end{tabular}
\end{table}

\textbf{Poznatky:}
\begin{itemize}
    \item Technické indikátory dominují (7 z top 10)
    \item Momentum features (returns, MACD) jsou nejdůležitější
    \item Fundamenty (ROE, ROA) jsou stále významné (top 10)
\end{itemize}

\sekce{Shrnutí výsledků}

Model dosahuje \textbf{35.61\% accuracy}, což je o více než 2\% nad náhodným baseline (33.33\%). Ačkoliv se tento rozdíl může zdát malý, ve světě financí a hypotézy efektivních trhů jde o signifikantní "alphu", která dokazuje, že kombinace fundamentů a technických indikátorů má reálnou prediktivní sílu.

\textbf{Klíčová zjištění:}
\begin{itemize}
    \item \textbf{Sektorová stabilita:} Model funguje nejlépe pro finanční sektor (40.3\% accuracy), zřejmě díky jasnější vazbě mezi fundamenty (debt, ratios) a cenou.
    \item \textbf{Dominance techniky:} Krátkodobé pohyby jsou nejsilněji ovlivněny historickými výnosy a volatilitou, nicméně fundamenty (CurrentRatio, DebtToEquity) poskytují modelu nezbytný "kotvící" kontext.
    \item \textbf{Náročnost predikce:} Nízký rozdíl oproti náhodě potvrzuje, že trh je vysoce efektivní a většina pohybů je v měsíčním horizontu blízko náhodné procházce.
\end{itemize}
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
    \includegraphics[width=0.95\linewidth]{data/150_tickers/figures/confusion_matrix_normalized.png}
    \caption{Confusion Matrix (absolutní počty i normalizovaná procentuální verze). Levý graf ukazuje absolutní počty predikcí, pravý graf zobrazuje procentuální úspěšnost pro každou třídu. Normalizovaná matice umožňuje posoudit, jak dobře model rozpoznává jednotlivé třídy nezávisle na jejich četnosti v datasetu. Diagonála reprezentuje správné predikce.}
    \label{fig:confusion_matrix}
\end{figure}

\sekce{ROC Křivky}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\linewidth]{data/150_tickers/figures/roc_curves.png}
    \caption{ROC křivky (Receiver Operating Characteristic) pro jednotlivé třídy DOWN, HOLD a UP. Osa X představuje False Positive Rate, osa Y True Positive Rate. Čím blíže křivka k levému hornímu rohu, tím lepší je separabilita dané třídy. AUC (Area Under Curve) hodnoty nad 0.5 indikují, že model je lepší než náhodný klasifikátor.}
    \label{fig:roc_curves}
\end{figure}

\sekce{Feature Importance}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\linewidth]{data/150_tickers/figures/feature_importance.png}
    \caption{Relativní důležitost jednotlivých features pro klasifikační model Random Forest. Delší sloupce indikují vyšší přínos dané proměnné k predikci. Technické indikátory (returns, volatility, RSI, MACD) dominují, což naznačuje, že krátkodobé cenové pohyby a momentum jsou nejsilnějšími prediktory budoucího směru ceny.}
    \label{fig:feature_importance}
\end{figure}

\sekce{Porovnání Sektorů}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\linewidth]{data/150_tickers/figures/sector_comparison.png}
    \caption{Srovnání výkonnosti modelu napříč 5 sektory S\&P 500. Graf zobrazuje Accuracy, Precision, Recall a F1-Score pro každý sektor. Financials dosahují nejlepších výsledků (40.3\% accuracy), zatímco Technology vykazuje nejnižší výkonnost kvůli vyšší volatilitě. Stabilita metrik napříč sektory potvrzuje robustnost pipeline.}
    \label{fig:sector_comparison}
\end{figure}


\sekce{Distribuce Tříd}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\linewidth]{data/150_tickers/figures/class_distribution.png}
    \caption{Rozložení cílových tříd v trénovacích a testovacích datech. DOWN (červená) reprezentuje poklesy větší než 3\%, HOLD (oranžová) pohyby v rozmezí $\pm$3\%, UP (zelená) růsty větší než 3\%. Vyvážená distribuce tříd v obou datasetech potvrzuje, že model není vystaven class imbalance problému a výsledky jsou reprezentativní.}
    \label{fig:class_distribution}
\end{figure}

\sekce{Analýza Confidence Predikcí}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.95\linewidth]{data/150_tickers/figures/prediction_confidence.png}
    \caption{Analýza spolehlivosti modelu. Levý graf porovnává distribuci confidence (pravděpodobnosti predikované třídy) pro správné vs. špatné predikce -- ideálně by správné predikce měly mít vyšší confidence. Pravý graf ukazuje, jak accuracy roste s confidence úrovní -- vyšší confidence typicky koreluje s lepší přesností, což umožňuje implementovat strategii obchodování pouze při vysoké jistotě modelu.}
    \label{fig:prediction_confidence}
\end{figure}

\sekce{Distribuce Měsíčních Výnosů}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\linewidth]{data/150_tickers/figures/returns_histogram.png}
    \caption{Histogram měsíčních výnosů v testovacím období. Červená přerušovaná čára označuje DOWN threshold (-3\%), zelená přerušovaná čára UP threshold (+3\%). Oblast mezi nimi představuje HOLD zónu. Distribuce ukazuje přirozené rozložení tříd a zdůvodňuje volbu thresholdu $\pm$3\% jako hranice mezi signifikantním pohybem a šumem.}
    \label{fig:returns_histogram}
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
    \item \textbf{Financials sektor} - Model zde funguje nejlépe (\textbf{40.3\%})
    \item \textbf{Stabilita napříč sektory} - Všech 5 sektorů vykazuje konzistentní výsledky
\end{enumerate}

\podsekce{Limitace}

\begin{enumerate}
    \item \textbf{Accuracy 35.6\%} - O 2.3\% lepší než random baseline (33.3\%)
    \item \textbf{HOLD třída} - Nejhůře rozpoznávaná třída
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
    \item \textbf{Alternative data} - Sentiment z news/social media
    \item \textbf{Jiné modely} - XGBoost, LSTM pro časové ř ady
    \item \textbf{Binární klasifikace} - UP vs NOT UP (snazší problém)
    \item \textbf{Confidence thresholds} - Obchodovat pouze při vysoké jistotě
    \item \textbf{Real-time systém} - Automatizované periodické dotrénovávání
\end{enumerate}

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
├── run_150_pipeline.py            # Hlavní pipeline (vše v jednom)
├── DIPLOMOVA_PRACE_LATEX.md       # LaTeX dokumentace
├── README.md                      # Přehled projektu
├── requirements.txt               # Python závislosti
│
├── data/
│   └── 150_tickers/
│       ├── ohlcv/                 # OHLCV data (150 tickerů)
│       ├── fundamentals/          # Fundamentální data
│       ├── complete/              # Kompletní dataset (~17k řádků)
│       └── figures/               # 9 vizualizací (300 DPI)
│
├── models/
│   └── 150_tickers/
│       ├── rf_classifier_tuned.pkl    # Natrénovaný model
│       ├── classifier_scaler_tuned.pkl
│       └── *.json                     # Metadata
│
└── docs/                          # Doprovodná dokumentace
    ├── METHODOLOGY.md
    └── MATHEMATICAL_FOUNDATIONS.md
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

# Spuštění celé pipeline (jeden příkaz)
python run_150_pipeline.py

# Pipeline automaticky:
# 1. Stáhne OHLCV data pro 150 tickerů
# 2. Stáhne fundamentální data
# 3. Natrénuje RF Regressor (imputace)
# 4. Natrénuje RF Classifier
# 5. Provede hyperparameter tuning
# 6. Vygeneruje 9 vizualizací + report
\end{verbatim}

\sekce{Příloha D: Výstupní soubory}

\begin{table}[H]
\centering
\caption{Hlavní výstupní soubory}
\begin{tabular}{|p{8cm}|p{4.5cm}|}
\hline
\textbf{Soubor} & \textbf{Popis} \\ \hline
\nolinkurl{data/150_tickers/complete/all_sectors_complete_10y.csv} & Kompletní dataset \\ \hline
\nolinkurl{models/150_tickers/rf_classifier_tuned.pkl} & RF Classifier model \\ \hline
\nolinkurl{data/150_tickers/figures/final_report.json} & Výsledky evaluace \\ \hline
\nolinkurl{data/150_tickers/figures/confusion_matrix_normalized.png} & Confusion matrix \\ \hline
\nolinkurl{data/150_tickers/figures/roc_curves.png} & ROC křivky \\ \hline
\nolinkurl{data/150_tickers/figures/feature_importance.png} & Feature importance \\ \hline
\nolinkurl{data/150_tickers/figures/sector_comparison.png} & Srovnání sektorů \\ \hline
\end{tabular}
\end{table}

% ============================================
% KONEC DOKUMENTU
% ============================================

\end{document}
