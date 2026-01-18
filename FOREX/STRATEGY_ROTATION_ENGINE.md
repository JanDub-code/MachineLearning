# Strategy Rotation Engine

## Koncept: 1000 modelů, hledání lokálního časového optima

Namísto hledání jedné "perfektní" strategie, která funguje navždy, budujeme systém,
který **kontinuálně generuje, testuje a rotuje tisíce strategií** pro nalezení
aktuálně optimálního edge v měnícím se trhu.

---

## 🎯 Filozofie

> *"If we have a disagreement, we'll just do both and measure."*
> — **John Carmack**

**Nikdo neví co funguje.** Ani člověk s IQ 500 nemá tušení, jestli je lepší:
- Lookback 14 nebo 21 dnů?
- RSI threshold 30 nebo 35?  
- TP 10 nebo 15 pips?
- London session nebo NY?

Každý "quant" jen hádá a pak racionalizuje proč zvolil právě ty parametry.
Post-hoc storytelling. Bullshit.

### Ekonomika

```
Tradiční přístup:
├── 10 quantů × $200k/rok = $2M ročně
├── Každý testuje 10-20 strategií měsíčně
├── = 1200-2400 strategií/rok
├── Lidské biasy, únava, ego
├── "Moje strategie je nejlepší" syndrom
└── Většina selhává

Brute-force přístup:
├── 2× RTX 4090 = $4k jednorázově
├── Elektřina ~$100/měsíc = $1.2k/rok
├── = 1000+ strategií DENNĚ
├── Žádné biasy, žádná únava
├── Pure statistical selection
└── Systém je nahraditelný, škálovatelný
```

**ROI:**
- 10 quantů: $2M/rok, testuje 2400 strategií
- GPU server: $5k/rok, testuje 365,000 strategií

**→ 150× více testů za 0.25% ceny**

### Princip

```
Člověk: "Myslím si, že RSI 14 s lookback 20 by mohl fungovat..."
Stroj:  "Otestoval jsem RSI 5-50 × lookback 5-100. Tady jsou výsledky."

Člověk: "Cítím, že London session je lepší..."
Stroj:  "London: Sharpe 0.8. NY: Sharpe 1.2. Overlap: Sharpe 1.5. Data."

Člověk: "Věřím v mean reversion strategie..."
Stroj:  "Mean reversion má PF 0.9 poslední měsíc. Momentum má PF 1.4. Přepínám."
```

### Mantry

1. **Měř všechno, nepředpokládej nic**
2. **Čísla > intuice**
3. **GPU hodiny jsou levnější než lidské hodiny**
4. **Model je komodita, infrastruktura je moat**
5. **Adapt, don't predict**

---

```
Tradiční workflow:
├── Quant má nápad
├── Testuje týden
├── Prezentuje před teamem
├── Debaty, politika
├── Deploy po měsících
└── Model přestane fungovat → blame game

Náš workflow:
├── Systém testuje 1000 nápadů denně
├── Statisticky vybere top 10
├── Automaticky deploy
├── Monitoruje performance
├── Model degraduje → už má náhradu ready
└── Žádní lidi, žádná politika
```

---

## 🏗️ Architektura

### Vrstvy systému

```
┌─────────────────────────────────────────────────────────────┐
│                    STRATEGY ROTATION ENGINE                  │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: EXECUTION                                          │
│  └── Trade ensemble top N strategií, risk management         │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: SELECTION                                          │
│  └── Rank strategie, vyber top performers, ensemble voting   │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: EVALUATION                                         │
│  └── Rolling backtest, Sharpe, PF, consistency metrics       │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: GENERATION                                         │
│  └── Kombinace modelů × parametrů × timeframes × features    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Layer 1: Strategy Generation

### Dimenze variability

Jeden základní model (např. LogReg) se rozloží do 100+ variant:

```python
STRATEGY_DIMENSIONS = {
    # Model type
    "model": [
        "logistic_regression",
        "random_forest", 
        "xgboost",
        "lightgbm",
        "neural_net_small",
        "ensemble_voting"
    ],
    
    # Training window (kolik dat pro trénink)
    "train_window_days": [7, 14, 30, 60, 90, 180],
    
    # Lookback pro features
    "feature_lookback_bars": [5, 10, 20, 50, 100],
    
    # Target definition
    "tp_pips": [5, 10, 15, 20, 30],
    "sl_pips": [5, 10, 15, 20],
    
    # Feature sets
    "feature_set": [
        "price_only",           # OHLC, returns
        "price_volume",         # + volume
        "technical_basic",      # + SMA, EMA, RSI
        "technical_advanced",   # + MACD, BB, ATR
        "microstructure",       # + spread, tick volume
        "multi_timeframe",      # + higher TF features
        "regime_aware"          # + regime detection
    ],
    
    # Entry filters
    "volatility_filter": [None, "low", "medium", "high"],
    "session_filter": [None, "london", "ny", "overlap"],
    "trend_filter": [None, "with_trend", "counter_trend"],
    
    # Probability threshold
    "entry_threshold": [0.52, 0.55, 0.58, 0.60, 0.65],
    
    # Timeframe
    "timeframe": ["1min", "5min", "15min", "1hour"],
    
    # Pair
    "pair": ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "EURGBP"]
}
```

### Počet kombinací

```python
# Příklad výpočtu
n_models = 6
n_train_windows = 6
n_lookbacks = 5
n_tp = 5
n_sl = 4
n_feature_sets = 7
n_vol_filters = 4
n_session_filters = 4
n_thresholds = 5
n_timeframes = 4
n_pairs = 6

total = (n_models * n_train_windows * n_lookbacks * n_tp * n_sl * 
         n_feature_sets * n_vol_filters * n_thresholds * n_pairs)
# = 6 * 6 * 5 * 5 * 4 * 7 * 4 * 5 * 6 = 3,024,000 kombinací

# Prakticky: sample 1000-10000 strategií náhodně nebo grid search podmnožiny
```

### Strategy Generator

```python
class StrategyGenerator:
    """Generuje strategie z prostoru parametrů."""
    
    def __init__(self, dimensions: dict):
        self.dimensions = dimensions
        
    def generate_random(self, n: int = 1000) -> list[StrategyConfig]:
        """Náhodně vygeneruj N strategií."""
        strategies = []
        for _ in range(n):
            config = {
                key: random.choice(values) 
                for key, values in self.dimensions.items()
            }
            strategies.append(StrategyConfig(**config))
        return strategies
    
    def generate_grid(self, subset_dims: list) -> list[StrategyConfig]:
        """Grid search přes subset dimenzí."""
        return list(itertools.product(*[
            self.dimensions[d] for d in subset_dims
        ]))
    
    def mutate(self, strategy: StrategyConfig, n_mutations: int = 3) -> StrategyConfig:
        """Mutuj existující strategii (pro evoluční přístup)."""
        new_config = strategy.copy()
        for key in random.sample(list(self.dimensions.keys()), n_mutations):
            new_config[key] = random.choice(self.dimensions[key])
        return StrategyConfig(**new_config)
```

---

## 📈 Layer 2: Strategy Evaluation

### Rolling Backtest Engine

```python
class RollingBacktester:
    """Evaluuje strategie na rolling window."""
    
    def __init__(self, 
                 eval_window_days: int = 30,
                 min_trades: int = 20):
        self.eval_window_days = eval_window_days
        self.min_trades = min_trades
    
    async def evaluate_strategy(self, 
                                 strategy: StrategyConfig,
                                 data: pd.DataFrame) -> StrategyResult:
        """Vyhodnoť strategii na posledních N dnech."""
        
        # 1. Train model na train window
        train_end = data.index[-1] - timedelta(days=self.eval_window_days)
        train_start = train_end - timedelta(days=strategy.train_window_days)
        train_data = data[train_start:train_end]
        
        model = train_model(strategy, train_data)
        
        # 2. Backtest na eval window
        eval_data = data[train_end:]
        trades = backtest(model, strategy, eval_data)
        
        # 3. Calculate metrics
        return StrategyResult(
            strategy_id=strategy.id,
            n_trades=len(trades),
            total_pnl=sum(t.pnl for t in trades),
            win_rate=sum(1 for t in trades if t.pnl > 0) / len(trades),
            profit_factor=calculate_pf(trades),
            sharpe=calculate_sharpe(trades),
            max_drawdown=calculate_max_dd(trades),
            expectancy=calculate_expectancy(trades),
            consistency=calculate_consistency(trades),  # pnl per week variance
            last_updated=datetime.now()
        )
    
    async def evaluate_all(self, 
                           strategies: list[StrategyConfig],
                           data: pd.DataFrame) -> list[StrategyResult]:
        """Paralelně evaluuj všechny strategie."""
        tasks = [
            self.evaluate_strategy(s, data) 
            for s in strategies
        ]
        return await asyncio.gather(*tasks)
```

### Metrics pro ranking

```python
@dataclass
class StrategyResult:
    strategy_id: str
    n_trades: int
    total_pnl: float
    win_rate: float
    profit_factor: float
    sharpe: float
    max_drawdown: float
    expectancy: float      # avg pips per trade
    consistency: float     # low variance = good
    last_updated: datetime
    
    @property
    def composite_score(self) -> float:
        """Kombinovaný score pro ranking."""
        if self.n_trades < 20:
            return -999  # Nedostatek dat
        
        # Vážený průměr normalizovaných metrik
        score = (
            0.25 * min(self.sharpe / 2.0, 1.0) +           # Sharpe, cap at 2
            0.25 * min(self.profit_factor / 2.0, 1.0) +    # PF, cap at 2
            0.20 * self.win_rate +                          # Win rate as-is
            0.15 * min(self.expectancy / 5.0, 1.0) +       # Expectancy, cap at 5 pips
            0.15 * (1 - min(self.max_drawdown / 100, 1.0)) # Lower DD = better
        )
        
        # Penalize inconsistency
        score *= (1 - 0.5 * min(self.consistency / 50, 1.0))
        
        return score
```

---

## 🏆 Layer 3: Strategy Selection

### Selection Engine

```python
class StrategySelector:
    """Vybírá top strategie pro trading."""
    
    def __init__(self,
                 top_n: int = 10,
                 min_score: float = 0.3,
                 max_correlation: float = 0.7):
        self.top_n = top_n
        self.min_score = min_score
        self.max_correlation = max_correlation
    
    def select(self, results: list[StrategyResult]) -> list[StrategyResult]:
        """Vyber top N diverzifikovaných strategií."""
        
        # 1. Filter by minimum quality
        qualified = [r for r in results if r.composite_score >= self.min_score]
        
        # 2. Sort by score
        sorted_results = sorted(qualified, key=lambda x: x.composite_score, reverse=True)
        
        # 3. Select with correlation filter (avoid similar strategies)
        selected = []
        for result in sorted_results:
            if len(selected) >= self.top_n:
                break
            
            # Check correlation with already selected
            if not self._is_correlated(result, selected):
                selected.append(result)
        
        return selected
    
    def _is_correlated(self, candidate: StrategyResult, 
                       selected: list[StrategyResult]) -> bool:
        """Check if candidate is too similar to already selected."""
        for s in selected:
            correlation = self._calculate_strategy_correlation(candidate, s)
            if correlation > self.max_correlation:
                return True
        return False
```

### Ensemble Voting

```python
class EnsembleVoter:
    """Kombinuje signály z více strategií."""
    
    def __init__(self, 
                 voting_method: str = "weighted",
                 min_agreement: float = 0.6):
        self.voting_method = voting_method
        self.min_agreement = min_agreement
    
    def vote(self, signals: list[Signal]) -> Signal | None:
        """Kombinuj signály do finálního rozhodnutí."""
        
        if not signals:
            return None
        
        if self.voting_method == "majority":
            return self._majority_vote(signals)
        elif self.voting_method == "weighted":
            return self._weighted_vote(signals)
        elif self.voting_method == "unanimous":
            return self._unanimous_vote(signals)
    
    def _weighted_vote(self, signals: list[Signal]) -> Signal | None:
        """Váhy podle Sharpe ratio strategie."""
        long_score = sum(
            s.strategy.sharpe * s.probability 
            for s in signals if s.direction == "long"
        )
        short_score = sum(
            s.strategy.sharpe * s.probability 
            for s in signals if s.direction == "short"
        )
        
        total_weight = sum(s.strategy.sharpe for s in signals)
        
        if long_score / total_weight > self.min_agreement:
            return Signal(direction="long", confidence=long_score/total_weight)
        elif short_score / total_weight > self.min_agreement:
            return Signal(direction="short", confidence=short_score/total_weight)
        
        return None  # No consensus
```

---

## ⚡ Layer 4: Execution

### Position Sizing

```python
class RiskManager:
    """Řídí velikost pozic a celkové riziko."""
    
    def __init__(self,
                 max_risk_per_trade: float = 0.01,  # 1% účtu
                 max_total_exposure: float = 0.05,   # 5% celkem
                 max_correlation_exposure: float = 0.03):  # 3% na korelované páry
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_exposure = max_total_exposure
        self.max_correlation_exposure = max_correlation_exposure
    
    def calculate_position_size(self,
                                signal: Signal,
                                account_balance: float,
                                current_positions: list) -> float:
        """Spočítej velikost pozice."""
        
        # Base size from risk
        risk_amount = account_balance * self.max_risk_per_trade
        sl_pips = signal.strategy.sl_pips
        pip_value = get_pip_value(signal.pair)
        
        base_lots = risk_amount / (sl_pips * pip_value)
        
        # Adjust for confidence
        adjusted_lots = base_lots * signal.confidence
        
        # Check total exposure
        current_exposure = sum(p.risk for p in current_positions)
        remaining_exposure = self.max_total_exposure * account_balance - current_exposure
        
        if adjusted_lots * sl_pips * pip_value > remaining_exposure:
            adjusted_lots = remaining_exposure / (sl_pips * pip_value)
        
        return max(0, adjusted_lots)
```

---

## 🔄 Main Loop

```python
class StrategyRotationEngine:
    """Hlavní engine pro strategy rotation."""
    
    def __init__(self, config: EngineConfig):
        self.generator = StrategyGenerator(config.dimensions)
        self.backtester = RollingBacktester(config.eval_window_days)
        self.selector = StrategySelector(config.top_n)
        self.voter = EnsembleVoter(config.voting_method)
        self.risk_manager = RiskManager(config.risk_params)
        
        self.strategy_pool: list[StrategyConfig] = []
        self.active_strategies: list[StrategyResult] = []
        self.performance_history: list = []
    
    async def initialize(self):
        """Inicializuj pool strategií."""
        self.strategy_pool = self.generator.generate_random(n=1000)
        await self._rebalance()
    
    async def run(self):
        """Hlavní loop."""
        while True:
            try:
                # 1. Get current market data
                data = await self.data_feed.get_latest()
                
                # 2. Generate signals from active strategies
                signals = []
                for strategy in self.active_strategies:
                    signal = await self._generate_signal(strategy, data)
                    if signal:
                        signals.append(signal)
                
                # 3. Ensemble voting
                final_signal = self.voter.vote(signals)
                
                # 4. Execute if signal
                if final_signal:
                    position_size = self.risk_manager.calculate_position_size(
                        final_signal, 
                        self.account.balance,
                        self.account.positions
                    )
                    if position_size > 0:
                        await self.executor.open_position(final_signal, position_size)
                
                # 5. Manage existing positions
                await self._manage_positions()
                
                # 6. Periodic rebalance (každou hodinu/den)
                if self._should_rebalance():
                    await self._rebalance()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)
    
    async def _rebalance(self):
        """Re-evaluuj a vyber nové aktivní strategie."""
        logger.info("Starting strategy rebalance...")
        
        # 1. Get fresh data
        data = await self.data_feed.get_historical(days=180)
        
        # 2. Evaluate all strategies
        results = await self.backtester.evaluate_all(self.strategy_pool, data)
        
        # 3. Select top performers
        self.active_strategies = self.selector.select(results)
        
        # 4. Evolve strategy pool (optional)
        if self.config.enable_evolution:
            self._evolve_pool(results)
        
        logger.info(f"Selected {len(self.active_strategies)} strategies")
        for s in self.active_strategies:
            logger.info(f"  {s.strategy_id}: score={s.composite_score:.3f}, "
                       f"sharpe={s.sharpe:.2f}, pf={s.profit_factor:.2f}")
    
    def _evolve_pool(self, results: list[StrategyResult]):
        """Evoluční vylepšení poolu."""
        # Kill bottom 10%
        sorted_results = sorted(results, key=lambda x: x.composite_score)
        kill_count = len(sorted_results) // 10
        
        # Replace with mutations of top performers
        top_performers = sorted_results[-kill_count*2:]
        new_strategies = [
            self.generator.mutate(random.choice(top_performers).strategy)
            for _ in range(kill_count)
        ]
        
        # Update pool
        dead_ids = {r.strategy_id for r in sorted_results[:kill_count]}
        self.strategy_pool = [
            s for s in self.strategy_pool if s.id not in dead_ids
        ] + new_strategies
```

---

## 📅 Scheduling

### Rebalance Frequency

```python
REBALANCE_SCHEDULES = {
    "aggressive": {
        "full_rebalance": "hourly",
        "quick_check": "5min",
        "evolution": "daily"
    },
    "moderate": {
        "full_rebalance": "daily",
        "quick_check": "hourly",
        "evolution": "weekly"
    },
    "conservative": {
        "full_rebalance": "weekly",
        "quick_check": "daily",
        "evolution": "monthly"
    }
}
```

---

## 🖥️ Infrastructure Requirements

### Pro 1000 strategií

```
Compute:
├── Backtesting: GPU cluster nebo multi-core CPU
├── ~1000 backtests × 30 days data × 1min bars = ~43M bars
├── Parallelizable: 16 cores → ~1 hodina pro full rebalance
└── GPU: 10-30 minut

Storage:
├── Historical data: ~10GB per pair per year (1min)
├── Strategy configs: ~10MB
├── Results history: ~100MB/month
└── Models: ~1GB (if caching trained models)

Memory:
├── Data in memory: 2-4GB per pair
├── Model training: depends on model type
└── Recommended: 32GB+ RAM

Network:
├── Real-time data feed
├── Broker API
└── Low latency execution (<100ms)
```

---

## 🎯 Key Success Factors

1. **Diversity over optimization**
   - Lepší mít 10 různých strategií se score 0.4 
   - Než 10 podobných se score 0.5

2. **Adapt, don't predict**
   - Nesnaž se predikovat regime change
   - Místo toho reaguj rychle když se změní

3. **Costs are king**
   - Každá strategie musí počítat s reálnými costs
   - Edge < costs = vyřadit

4. **Fail fast**
   - Strategie co nefunguje → rychle vyměnit
   - Nesentimentálně

5. **Infrastructure > Models**
   - Systém co rotuje modely má větší hodnotu
   - Než jakýkoliv jednotlivý model

---

## 🚀 Next Steps

1. [ ] Implementovat `StrategyGenerator` s full parameter space
2. [ ] Paralelní `RollingBacktester` s GPU support
3. [ ] `StrategySelector` s correlation filtering
4. [ ] `EnsembleVoter` s weighted voting
5. [ ] Live paper trading integration
6. [ ] Performance dashboard
7. [ ] Evolution/mutation engine
