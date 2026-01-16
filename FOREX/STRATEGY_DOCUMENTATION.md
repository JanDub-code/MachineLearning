# 📚 Detailní Dokumentace Trading Strategií

## Přehled

Systém obsahuje **2 vítězné strategie** po rigorózním testování 8 variant.
Obě strategie používají stejný ML model (Random Forest), ale liší se v:
- Pravidech vstupu (probability threshold)
- Pravidlech výstupu (SL/TP úrovně)
- Risk/Reward ratio

---

# 🥇 V5.3 Tight R:R (CHAMPION)

## Filosofie: "Prohrávej malé, vyhrávej velké"

```
Win Rate: 38% (prohraje většinu obchodů)
Ale když vyhraje: získá 3× více než ztratí
Matematická edge: 0.38 × 3.0 - 0.62 × 1.0 = +0.52 na trade
```

## Parametry

| Parametr | Hodnota | Význam |
|----------|---------|--------|
| `probability_threshold` | **0.58** | Model musí predikovat ≥58% šanci na UP pro LONG |
| `min_probability_gap` | **0.08** | Pravděpodobnost musí být ≥8% od 50% |
| `sl_atr_multiplier` | **1.0** | Stop Loss = 1.0 × ATR (velmi těsný) |
| `tp_atr_multiplier` | **3.0** | Take Profit = 3.0 × ATR (vysoký cíl) |
| `max_holding_bars` | **60** | Max 60 minut v obchodu (na 1m datech) |
| `min_atr_pips` | **0.2** | NeObchoduj při ATR < 0.2 pips |
| `max_atr_pips` | **10.0** | NeObchoduj při ATR > 10 pips |

## Příklad obchodu V5.3

```
EURUSD @ 1.0850
ATR = 5 pips (0.0005)

Stop Loss = 1.0 × 5 = 5 pips → Exit @ 1.0845 (LONG) nebo 1.0855 (SHORT)
Take Profit = 3.0 × 5 = 15 pips → Exit @ 1.0865 (LONG) nebo 1.0835 (SHORT)

R:R = 1:3 → Riskuji 5 pips, abych získal 15 pips
```

## Výkonnost

| Metrika | Hodnota |
|---------|---------|
| **PnL (5 dnů)** | +71.6 pips |
| **Profit Factor** | 1.16 |
| **Win Rate** | 38.1% |
| **Profitable Days** | 4/5 (80%) |
| **TP Hits** | 285 (38%) |
| **SL Hits** | 460 (61%) |

---

# 🥈 V5.6 Balanced (BACKUP)

## Filosofie: "Vyrovnaný přístup"

```
Win Rate: 50% (přibližně polovina úspěšná)
R:R = 1:1.7 → Vyrovnanější poměr riziko/zisk
Méně volativní equity křivka
```

## Parametry

| Parametr | Hodnota | Význam |
|----------|---------|--------|
| `probability_threshold` | **0.59** | Model musí predikovat ≥59% (přísnější) |
| `min_probability_gap` | **0.09** | Gap musí být ≥9% od 50% |
| `sl_atr_multiplier` | **1.3** | Stop Loss = 1.3 × ATR (středně těsný) |
| `tp_atr_multiplier` | **2.2** | Take Profit = 2.2 × ATR |
| `max_holding_bars` | **60** | Max 60 minut v obchodu |
| `min_atr_pips` | **0.2** | NeObchoduj při nízké volatilitě |
| `max_atr_pips` | **10.0** | NeObchoduj při extrémní volatilitě |

## Příklad obchodu V5.6

```
EURUSD @ 1.0850
ATR = 5 pips (0.0005)

Stop Loss = 1.3 × 5 = 6.5 pips → Exit @ 1.08435 (LONG)
Take Profit = 2.2 × 5 = 11 pips → Exit @ 1.0861 (LONG)

R:R = 1:1.7 → Riskuji 6.5 pips, abych získal 11 pips
```

## Výkonnost

| Metrika | Hodnota |
|---------|---------|
| **PnL (5 dnů)** | +44.3 pips |
| **Profit Factor** | 1.10 |
| **Win Rate** | 49.7% |
| **Profitable Days** | 2/5 (40%) |

---

# 📊 Srovnání strategií

| Aspekt | V5.3 Tight R:R | V5.6 Balanced |
|--------|----------------|---------------|
| **Filosofie** | Lose small, win big | Vyrovnaný |
| **Win Rate** | 38% | 50% |
| **R:R Ratio** | 1:3 | 1:1.7 |
| **Stop Loss** | 1.0 × ATR (těsný) | 1.3 × ATR (středně) |
| **Take Profit** | 3.0 × ATR (vysoký) | 2.2 × ATR (střední) |
| **Threshold** | 0.58 (volnější) | 0.59 (přísnější) |
| **PnL/5 dnů** | +71.6 pips | +44.3 pips |
| **Psychologie** | Těžké (hodně proher) | Snazší |

---

# 🔧 Krok za krokem: Jak strategie funguje

## KROK 1: Stažení dat (data_fetcher.py)

```python
# Stáhne posledních 7 dní 1-minutových OHLCV dat
fetch_yfinance_fx(pair="EURUSD", interval="1m", period="7d")

# Výstup: DataFrame s kolonkami
# time, open, high, low, close, volume
```

## KROK 2: Vytvoření features (features.py)

```python
def build_features(df):
    # 1. Základní returns
    df["return_1"] = df["close"].pct_change()  # 1-bar return
    
    # 2. Volatilita
    df["volatility"] = df["return_1"].rolling(60).std()  # 60-bar rolling std
    df["atr"] = atr(df, period=14)  # Average True Range
    
    # 3. Technické indikátory
    df["rsi_14"] = rsi(df["close"], period=14)  # RSI 0-100
    df["macd"], df["macd_signal"], df["macd_hist"] = macd(df["close"])
    
    # 4. Moving Averages (6 SMA + 6 EMA = 12 features)
    for w in [3, 6, 12, 24, 48, 96]:
        df[f"sma_{w}"] = df["close"].rolling(w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w).mean()
    
    # 5. Časové features
    df["minute"] = df["time"].dt.minute  # 0-59
    df["hour"] = df["time"].dt.hour      # 0-23
    df["dow"] = df["time"].dt.dayofweek  # 0-6 (Mon-Sun)
    
    # 6. Target: Půjde cena nahoru v příštím baru?
    df["return_h"] = df["close"].pct_change(periods=1).shift(-1)
    df["target"] = (df["return_h"] > 0).astype(int)  # 1=UP, 0=DOWN
    
    return df  # ~20 features celkem
```

**Finální features (20):**
- `return_1`, `volatility`, `atr`
- `rsi_14`, `macd`, `macd_signal`, `macd_hist`
- `sma_3`, `sma_6`, `sma_12`, `sma_24`, `sma_48`, `sma_96`
- `ema_3`, `ema_6`, `ema_12`, `ema_24`, `ema_48`, `ema_96`
- `minute`, `hour`, `dow`

## KROK 3: Trénování modelu

```python
# Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=300,      # 300 stromů v lese
    max_depth=10,          # Max hloubka stromu (anti-overfitting)
    min_samples_leaf=20,   # Min vzorků v listu (anti-overfitting)
    class_weight="balanced_subsample",  # Vyváží třídy
    random_state=42,       # Reprodukovatelnost
)

# Trénuj na 2 dnech dat (~2880 barů)
model.fit(X_train, y_train)
```

## KROK 4: Predikce

```python
# Pro každý bar v testovacích datech:
probabilities = model.predict_proba(X_test)[:, 1]  # P(UP)

# Příklad výstupu:
# Bar 1: P(UP) = 0.62 → LONG signál (nad 0.58 threshold)
# Bar 2: P(UP) = 0.53 → NO TRADE (gap < 0.08)
# Bar 3: P(UP) = 0.38 → SHORT signál (pod 0.42 = 1-0.58)
```

## KROK 5: Filtry před vstupem

```python
for i, prob in enumerate(probabilities):
    # FILTR 1: ATR volatilita
    atr_pips = df["atr"][i] / 0.0001
    if atr_pips < 0.2 or atr_pips > 10.0:
        continue  # Přeskoč - volatilita mimo rozsah
    
    # FILTR 2: Probability gap
    prob_gap = abs(prob - 0.5)
    if prob_gap < 0.08:  # Min gap pro V5.3
        continue  # Přeskoč - model není dost jistý
    
    # FILTR 3: Směr obchodu
    if prob >= 0.58:
        direction = "LONG"
    elif prob <= 0.42:  # 1 - 0.58
        direction = "SHORT"
    else:
        continue  # Přeskoč - v "no-trade" zóně
```

## KROK 6: Simulace obchodu

```python
def simulate_trade(entry_idx, direction, config):
    entry_price = df["close"][entry_idx]
    atr_pips = df["atr"][entry_idx] / 0.0001
    
    # Vypočítej SL a TP
    sl_distance = atr_pips * config.sl_atr_multiplier  # 1.0 pro V5.3
    tp_distance = atr_pips * config.tp_atr_multiplier  # 3.0 pro V5.3
    
    cost = 0.12 + 0.08  # spread + slippage = 0.20 pips
    
    # Projdi budoucí bary a kontroluj SL/TP
    for future_bar in range(1, 61):  # Max 60 barů
        high = df["high"][entry_idx + future_bar]
        low = df["low"][entry_idx + future_bar]
        
        if direction == "LONG":
            sl_price = entry_price - sl_distance * 0.0001
            tp_price = entry_price + tp_distance * 0.0001
            
            if low <= sl_price:
                return -sl_distance - cost  # SL HIT
            elif high >= tp_price:
                return +tp_distance - cost  # TP HIT
        
        else:  # SHORT
            sl_price = entry_price + sl_distance * 0.0001
            tp_price = entry_price - tp_distance * 0.0001
            
            if high >= sl_price:
                return -sl_distance - cost  # SL HIT
            elif low <= tp_price:
                return +tp_distance - cost  # TP HIT
    
    # Timeout - exit na close
    exit_price = df["close"][entry_idx + 60]
    if direction == "LONG":
        return (exit_price - entry_price) / 0.0001 - cost
    else:
        return (entry_price - exit_price) / 0.0001 - cost
```

## KROK 7: Daily Retrain Loop

```python
# Každý den:
for day in trading_days:
    # 1. Vezmi poslední 2 dny jako training data
    train_data = get_last_48_hours()
    
    # 2. Přetrénuj model na čerstvých datech
    model.fit(train_data)
    
    # 3. Obchoduj další den s novým modelem
    trade_next_day(model)
    
    # 4. Opakuj zítra s novými 48 hodinami
```

---

# 📈 Proč V5.3 vítězí nad V5.6?

## Matematika expectancy

### V5.3 Tight R:R
```
E[trade] = WinRate × AvgWin - LossRate × AvgLoss
E[trade] = 0.38 × 3.0 - 0.62 × 1.0
E[trade] = 1.14 - 0.62 = +0.52 pips per unit risk
```

### V5.6 Balanced
```
E[trade] = 0.50 × 2.2 - 0.50 × 1.3
E[trade] = 1.10 - 0.65 = +0.45 pips per unit risk
```

**V5.3 má vyšší expectancy per trade!**

---

# ⚠️ Rizika a omezení

1. **Testováno na 5 dnech** - malý vzorek
2. **V5.3 má nízkou win rate** - psychologicky náročné
3. **ATR se mění** - v nízké volatilitě méně obchodů
4. **Transakční náklady** - spread + slippage zahrnuty, ale reálné mohou být vyšší
5. **Gap risk** - přes víkend pozice nezabezpečené

---

*Dokumentace vygenerována: 2026-01-16*
