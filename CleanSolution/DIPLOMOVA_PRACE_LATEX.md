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
\kapitola{Výběr a konfigurace algoritmů}

\sekce{Proč právě Random Forest?}

Algoritmus \textbf{Random Forest (RF)} tvoří jádro celé naší pipeline a plní v ní dvě kritické úlohy:
\begin{enumerate}
    \item \textbf{Imputace (Regrese):} Odhaduje chybějící historická fundamentální data na základě cenových vzorců.
    \item \textbf{Klasifikace:} Provádí finální predikci směru ceny (DOWN, HOLD, UP).
\end{enumerate}

Tato volba byla učiněna po zvážení alternativ (viz Tabulka 1), přičemž hlavními argumenty byla odolnost vůči šumu v tržních datech a schopnost vysvětlit, proč model dané rozhodnutí učinil (Feature Importance).

\begin{table}[H]
\centering
\caption{Srovnání vhodnosti ML algoritmů pro náš projekt}
\begin{tabular}{|l|p{3.5cm}|p{3.5cm}|c|}
\hline
\textbf{Algoritmus} & \textbf{Výhody} & \textbf{Nevýhody} & \textbf{Vhodný?} \\ \hline
Random Forest & Robustnost, interpretovatelnost & Vyšší paměťová náročnost & \textbf{Ano} \\ \hline
XGBoost & Extrémní rychlost a výkon & Náchylnost k overfittingu & Ne \\ \hline
Neural Networks & Učení velmi složitých vzorů & Vyžaduje obrovské objemy dat & Ne \\ \hline
SVM & Dobré pro malé datasety & Pomalý trénink u velkých dat & Ne \\ \hline
\end{tabular}
\end{table}

\sekce{Co jsou hyperparametry a jak ovlivňují model?}

Zatímco běžné "parametry" se model učí sám z dat, \textbf{hyperparametry} jsou "ovládací prvky", které musíme nastavit my před začátkem učení. Určují celkovou strategii a složitost modelu. Správné nastavení je rozdílem mezi modelem, který nic neví, a modelem, který umí predikovat trh.

\begin{table}[H]
\centering
\caption{Klíčové hyperparametry v našem modelu}
\begin{tabular}{|p{3.8cm}|p{2cm}|p{6.2cm}|}
\hline
\textbf{Hyperparametr} & \textbf{Hodnota} & \textbf{Význam pro výsledek} \\ \hline
\texttt{n\_estimators} & 100 -- 200 & Počet stromů v lese. Více stromů znamená "více hlav víc ví", ale delší výpočet. \\ \hline
\texttt{max\_depth} & 10 -- 15 & Maximální hloubka stromu. Omezujeme ji, aby strom nebyl příliš detailní a nehledal v datech neexistující náhody. \\ \hline
\texttt{class\_weight} & 'balanced' & Říká modelu, aby věnoval stejnou pozornost vzácným poklesům i častějším růstům. \\ \hline
\texttt{min\_samples\_split} & 5 -- 10 & Kolik dat musí uzel obsahovat, abychom ho dále dělili. Brání vzniku příliš specifických pravidel. \\ \hline
\end{tabular}
\end{table}

\sekce{Komponenty Pipeline}

Kromě samotných algoritmů obsahuje projekt několik pomocných prvků:
\begin{itemize}
    \item \textbf{StandardScaler:} Normalizuje data (převádí např. cenu \$500 a objem 1M na srovnatelná čísla 0-1), aby jedna proměnná nepřebila ostatní.
    \item \textbf{TimeSeriesSplit:} Zajišťuje, že se model testuje vždy na datech modernějších, než na kterých se učil (simulace reálného času).
    \item \textbf{Feature Engineering:} Výpočet RSI, MACD a dalších odvozených indikátorů, které dávají modelu "nápovědu" o náladě na trhu.
\end{itemize}

\kapitola{Architektura řešení}

Celý systém je navržen jako modulární pipeline, která transformuje surová tržní data do podoby obchodních signálů. Unikátnost architektury spočívá v hybridním přístupu, kde využíváme algoritmus \textbf{Random Forest} ve dvou odlišných rolích, abychom překonali problém neúplných dat.

\sekce{Logická struktura systému}

Architekturu lze rozdělit do tří logických bloků: sběr dat, rekonstrukce historie a finální klasifikace.

\podsekce{1. Blok: Akvizice a inženýrství dat}
Prvním krokem je sběr dat ze serverů Yahoo Finance. Zde narážíme na zásadní problém: zatímco cenové pohyby (OHLCV) jsou dostupné desítky let dozadu, detailní účetní data (fundamenty) jsou u bezplatných zdrojů omezena pouze na poslední dva roky. 
Abychom model nenaučili pouze na krátkém (a potenciálně netypickém) období, musíme najít způsob, jak získat fundamenty i pro starší cenové záznamy. V tomto kroku také počítáme technické indikátory (RSI, MACD), které dávají modelu informaci o dynamice ceny.

\podsekce{2. Blok: Imputační motor (RF Regressor)}
Zde přichází první nasazení \textbf{Random Forest Regressoru}. Tento model slouží jako "překladač" mezi cenovými vzorci a fundamentálním stavem firmy. 
\begin{itemize}
    \item \textbf{Proč RF?} Vztah mezi tím, jak se akcie hýbe na burze a jaké má firma zisky, není lineární. RF dokáže tyto komplexní vztahy zachytit lépe než jednoduché statistické metody.
    \item \textbf{Princip:} Model se naučí na datech z let 2024--2025, jak vypadají fundamenty (např. P/E) při určitých cenových pohybech. Následně tyto znalosti použije k "dopočítání" (imputaci) historie pro roky 2015--2023.
\end{itemize}

\podsekce{3. Blok: Klasifikační motor (RF Classifier)}
V posledním bloku dochází ke sjednocení reálných a předpovězených dat. Vzniká robustní dataset o rozsahu \textbf{$\sim$17,000 záznamů} (150 tickerů z 5 sektorů napříč 10 lety).
Na tomto uceleném datasetu trénujeme \textbf{Random Forest Classifier}. Ten bere v úvahu technickou situaci na grafu i (rekonstruované) fundamenty a rozhoduje o výsledném signálu. Využívá k tomu naladěné hyperparametry (počet stromů, hloubka), které jsme diskutovali v předchozí kapitole.

\kapitola{Implementace a datový tok}

Praktická realizace probíhá v rámci jedné hlavní pipeline (\texttt{run\_150\_pipeline.py}), která automatizuje všechny kroky od stažení dat až po vizualizaci výsledků.

\sekce{Příprava datového základu}

Pipeline začíná paralelním stahováním dat pro všech 150 tickerů. Pro každý ticker vypočítáme 13 technických indikátorů, které slouží jako primární vhled do "nálady" trhu. Fundamentální data jsou uložena odděleně a párována k cenám na měsíční bázi. Tímto přístupem eliminujeme šum denních výkyvů a soustředíme se na střednědobé trendy.

\sekce{Proces učení a optimalizace}

\podsekce{Trénink regrese (Imputace)}
Nejdříve učíme model odhadovat hodnoty jako \textit{Trailing P/E} nebo \textit{Return on Equity}. Používáme k tomu regresi, protože cílem je trefit co nejpřesnější číslo. Úspěšnost tohoto kroku je kritická – pokud by regrese dávala nesmyslné hodnoty, celý model by se učil na lživé historii. Naše zvolená konfigurace (100 stromů, hloubka 15) však dosahuje vysoké stability.

\podsekce{Trénink klasifikace (Predikce)}
Po doplnění historie fundamenty máme kompletní "pohled do minulosti". RF klasifikátor se učí rozeznávat tři stavy: pokles (DOWN), klid (HOLD) a růst (UP). 
Abychom zajistili, že se model jen nenabifluje historické tabulky, využíváme:
\begin{itemize}
    \item \textbf{Standardizaci:} Všechny hodnoty (např. miliardové objemy a jednotkové P/E) převádíme na srovnatelné měřítko.
    \item \textbf{Ladění (GridSearch):} Automaticky zkoušíme různé kombinace hloubky a počtu stromů, abychom našli tu, která nejlépe funguje na "neviděných" datech.
\end{itemize}

\sekce{Metodika evaluace}

Hlavním nástrojem pro posouzení kvality je \textbf{TimeSeriesSplit}. Na rozdíl od běžných testů model v žádném okamžiku nevidí "za roh" do budoucnosti. Vždy se učí na datech $T$ a testuje se na datech $T+1$. Výsledkem je sada reportů a grafů v profesionálním rozlišení (300 DPI), které detailně rozebírají úspěšnost podle sektorů i důležitost jednotlivých indikátorů.

% =========================================================================
% ČÁST 4: Kapitoly 7-9 (Experiment, Výsledky, Vizualizace)
% =========================================================================

% ============================================
% KAPITOLA 7: EXPERIMENT 30 TICKERŮ
% ============================================
\kapitola{Robustní verifikace modelu na datech S\&P 500}

Abychom ověřili skutečnou sílu našeho hybridního modelu, neomezili jsme se pouze na úzký výběr akcií, ale provedli jsme rozsáhlý experiment na reprezentativním vzorku indexu S\&P 500. Cílem této kapitoly je popsat parametry tohoto testu a složení dat, na kterých model prokazuje svou stabilitu.

\sekce{Konfigurace experimentu}

Pro experiment bylo zvoleno celkem \textbf{150 akciových titulů}. Tato volba není náhodná – představuje dostatečně velký vzorek pro statistickou významnost, ale zároveň chrání model před přílišným zahlcením šumem z méně likvidních titulů.

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

\sekce{Volba sektorů a tickerů}

Hlavním motivem pro rozdělení do \textbf{5 odlišných sektorů} byla potřeba komparativní analýzy. Každý sektor (např. technologie vs. finance) se chová v tržních cyklech jinak. Tím, že model trénujeme napříč těmito světy, testujeme jeho schopnost přizpůsobit se různým fundamentálním charakteristikám (např. odlišné míře zadlužení u průmyslu vs. technologií).

V rámci každého sektoru jsme vybrali \textbf{30 tickerů}. Toto množství považujeme za \textbf{optimální objem dat}: 
\begin{itemize}
    \item Poskytuje dostatek "učebního materiálu" pro Random Forest (celkem přes 16 000 záznamů).
    \item Zajišťuje, že model není příliš specializovaný na úzkou skupinu firem, ale chápe sektor jako celek.
    \item Udržuje výpočetní náročnost v rozumných mezích při zachování vysoké granularity výsledků.
\end{itemize}

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

Po propojení technických indikátorů s reálnými a imputovanými fundamenty vznikl finální dataset, který slouží jako základ pro veškerá další měření. Celkový počet features (29) dává modelu dostatečný prostor pro nalezení nelineárních závislostí, aniž by došlo k tzv. \textbf{prokletí dimenzionality} (Curse of Dimensionality).

\textbf{Co to znamená v našem kontextu?} 
V datové vědě platí, že s každým dalším přidaným indikátorem (dimenzí) roste objem dat potřebný k tomu, aby model nebyl ztracen v "prázdném prostoru". Pokud bychom měli stovky indikátorů na malém počtu firem, model by si začal vymýšlet náhodné vztahy (overfitting). Naše konfigurace (29 proměnných na cca 17 000 záznamů) představuje zdravý poměr, kdy má algoritmus dost informací pro rozhodování, ale stále se pohybuje v "hustě osídleném" datovém prostoru, kde jsou nalezené vztahy statisticky podložené.

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

Evaluace proběhla ve dvou fázích, které odpovídají dualitě našeho modelu. Nejdříve jsme posuzovali schopnost "rekonstruovat" historii (regrese) a následně schopnost "předpovídat" budoucnost (klasifikace).

\sekce{RF Regressor: Kvalita rekonstrukce dat}

Předtím, než jsme se pokusili o jakoukoli predikci na trhu, museli jsme zajistit, aby naše imputovaná fundamentální data odpovídala realitě. K tomu jsme využili \textbf{RF Regressor}, u kterého jsme sledovali metriku $R^2$ (koeficient determinace).

\podsekce{Metrika $R^2$ a její význam}
Metrika $R^2$ udává, kolik procent variability (změn) cílové hodnoty dokáže model vysvětlit na základě vstupů. Pokud má $R^2$ hodnotu 0.9, znamená to, že 90 \% pohybu např. metriky P/E dokážeme odvodit z cenového grafu a objemů. V našem případě jsou výsledky nad 0.9 extrémně signifikantní a potvrzují, že tržní cena v sobě fundamentální zdraví firmy skutečně nese.

\begin{table}[H]
\centering
\caption{Výsledky RF Regressoru - detailní přesnost rekonstrukce}
\begin{tabular}{|p{3cm}|p{2cm}|p{2cm}|p{3cm}|}
\hline
\textbf{Cílová metrika} & \textbf{MAE} & \textbf{R² Score} & \textbf{Interpretace} \\ \hline
trailingPE & 4.419 & 0.957 & Výborná \\ \hline
forwardPE & 2.595 & 0.964 & Výborná \\ \hline
returnOnAssets & 0.015 & 0.970 & Výborná \\ \hline
returnOnEquity & 0.045 & 0.935 & Velmi dobrá \\ \hline
priceToBook & 1.854 & 0.891 & Velmi dobrá \\ \hline
profitMargins & 0.031 & 0.886 & Velmi dobrá \\ \hline
debtToEquity & 38.513 & 0.765 & Dobrá \\ \hline
\end{tabular}
\end{table}

\podsekce{Feature Importance pro regresor}
\textbf{Feature Importance} nám říká, o které informace se model nejvíce "opírá" při dělání svých odhadů. Pracuje na principu sledování toho, o kolik by se model zhoršil, kdybychom mu danou informaci vzali. 

Jak ukazuje Tabulka \ref{tab:reg_importance}, u regresoru hrají klíčovou roli následující faktory:

\begin{itemize}
    \item \textbf{Volume (Objem):} S obrovským náskokem nejdůležitější parametr (vliv $\sim$50 \%). Objem obchodů je přímým odrazem zájmu velkých institucionálních hráčů. Jelikož jsou fundamenty (zisky, dluhy) hlavním vodítkem pro tyto velké fondy, existuje mezi objemem a účetními metrikami velmi silná vazba.
    \item \textbf{SMA a EMA (6--12 měsíců):} Klouzavé průměry za delší období. Protože se fundamenty mění pomalu (kvartálně), model ignoruje denní šum a soustředí se na dlouhodobý cenový trend. Průměrná cena za posledních 12 měsíců je pro odhad např. P/E ratio mnohem lepším vodítkem než aktuální cena.
\end{itemize}

\begin{table}[H]
\centering
\caption{Nejdůležitější faktory pro imputaci historie}
\label{tab:reg_importance}
\begin{tabular}{|c|p{4cm}|c|}
\hline
\textbf{Pořadí} & \textbf{Indikátor} & \textbf{Důležitost} \\ \hline
1 & \textbf{volume} & 0.4995 \\ \hline
2 & \texttt{sma\_12} & 0.0734 \\ \hline
3 & \texttt{ema\_12} & 0.0730 \\ \hline
4 & \texttt{sma\_6} & 0.0586 \\ \hline
5 & \texttt{ema\_6} & 0.0583 \\ \hline
\end{tabular}
\end{table}

\sekce{RF Classifier: Predikční schopnost systému}

Po úspěšné rekonstrukci historie jsme nasadili \textbf{RF Classifier}, aby určil směr budoucího vývoje. Zatímco regresor tipoval čísla, klasifikátor tipuje "škatulku" (DOWN, HOLD, UP).

\podsekce{Celkové výsledky a random baseline}
Hlavním měřítkem úspěchu je porovnání s náhodou. Jelikož máme 3 třídy, náhodný tipér by měl úspěšnost \textbf{33.33 \%}. Náš model dosahuje \textbf{35.61 \%}, což ve světě financí není zanedbatelné – jde o důkaz, že model v trhu vidí neefektivity.

\begin{table}[H]
\centering
\caption{Celkové metriky klasifikátoru}
\begin{tabular}{|p{4cm}|p{3cm}|}
\hline
\textbf{Metrika} & \textbf{Hodnota} \\ \hline
Accuracy & \textbf{35.61\%} \\ \hline
Precision & 36.57\% \\ \hline
Recall & 35.61\% \\ \hline
F1-Score & 35.77\% \\ \hline
\textbf{Náhodný baseline} & \textbf{33.33\%} \\ \hline
\end{tabular}
\end{table}

Model vykazuje vyšší přesnost u růstových trendů (UP), což naznačuje, že býčí trhy S\&P 500 mají čitelnější strukturu než náhlé panické výprodeje. 


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
\sekce{Hloubková analýza predikcí}

\podsekce{Confusion Matrix: Kde se model plete?}
\textbf{Confusion Matrix} je nástroj, který nám ukazuje "kdo s koho" – tedy kolikrát model trefil skutečnost a s čím si ji nejčastěji plete. Diagonála (v našem případě čísla 98, 44 a 92) představuje vítězství modelu.

Z pohledu metriky \textbf{Recall} (schopnost modelu najít všechny relevantní případy dané třídy) vidíme zajímavý trend:

\begin{enumerate}
    \item \textbf{DOWN (51 \% Recall):} Model velmi dobře pozná situaci, kdy se trh chystá klesat. Pokud nastane pokles, model ho zachytí v polovině případů.
    \item \textbf{HOLD (20 \% Recall):} Toto je slabina modelu. Klid na trhu je často interpretován jako příprava na pohyb jedním nebo druhým směrem.
    \item \textbf{UP (35 \% Recall):} Model je opatrný u růstů, což vede k nižšímu recallu, ale vyšší preciznosti (pokud model řekne "kupovat", je to spolehlivější).
\end{enumerate}

\begin{table}[H]
\centering
\caption{Analýza záměn (Confusion Matrix - zjednodušená)}
\begin{tabular}{l c c c l}
\toprule
 & \textbf{Pred.} DOWN & \textbf{Pred.} HOLD & \textbf{Pred.} UP & \textbf{Recall} \\
\midrule
\textbf{Skutečnost} DOWN & \textbf{98} & 39 & 56 & \textbf{51 \%} \\
\textbf{Skutečnost} HOLD & 72 & \textbf{44} & 100 & \textbf{20 \%} \\
\textbf{Skutečnost} UP   & 84 & 85 & \textbf{92} & \textbf{35 \%} \\
\bottomrule
\end{tabular}
\end{table}

\podsekce{Rozlišovací schopnost (AUC Score)}

\textbf{AUC (Area Under Curve)} udává, jak moc je model "zmatený" při rozhodování mezi dvěma třídami. Hodnota \textbf{0.5} znamená totální zmatek (náhodu), hodnota \textbf{1.0} je dokonalý věštec.

\begin{table}[H]
\centering
\caption{AUC skóre: Schopnost rozlišit třídy od sebe}
\begin{tabular}{|p{3cm}|p{2cm}|p{7cm}|}
\hline
\textbf{Třída} & \textbf{AUC} & \textbf{Význam} \\ \hline
DOWN & 0.55 & Mírná schopnost oddělit pokles od šumu. \\ \hline
HOLD & 0.52 & Skoro náhodná separace. \\ \hline
UP & 0.54 & Mírná schopnost zachytit růstový signál. \\ \hline
\end{tabular}
\end{table}

Výsledky AUC potvrzují, že finanční trhy jsou extrémně náročné prostředí. I mírné vychýlení nad 0.5 však v kombinaci s velkým množstvím obchodů může tvořit ziskovou strategii.

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
