from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Base paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Rolling windows in weeks
WINDOWS_WEEKS: List[int] = list(range(1, 13))

# Trading pairs
PAIRS = ["EUR_USD"]

@dataclass
class BrokerConfig:
    name: str = "oanda"
    practice: bool = True
    account_id: str = ""  # fill locally
    api_key: str = ""      # fill locally, do not commit
    rest_endpoint: str = "https://api-fxpractice.oanda.com/v3"
    stream_endpoint: str = "https://stream-fxpractice.oanda.com/v3"

@dataclass
class TrainingConfig:
    candle_granularity: str = "M1"  # 1-minute candles
    max_candles: int = 200_000       # safety cap per download
    holdout_hours: int = 24          # last day as validation per window
    target_horizon_minutes: int = 5  # predict next 5-minute direction
    probability_threshold: float = 0.6
    risk_per_trade: float = 0.0025   # 0.25%
    drawdown_cut: float = 0.03       # 3% breaker

@dataclass
class Paths:
    raw_dir: Path = RAW_DIR
    processed_dir: Path = PROCESSED_DIR
    models_dir: Path = MODELS_DIR
    reports_dir: Path = REPORTS_DIR

@dataclass
class Settings:
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: Paths = field(default_factory=Paths)
    pairs: List[str] = field(default_factory=lambda: PAIRS)
    windows_weeks: List[int] = field(default_factory=lambda: WINDOWS_WEEKS)


def load_settings() -> Settings:
    return Settings()
