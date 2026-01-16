"""
FOREX Live Paper Trading Bot

Connects to OANDA Practice account and executes trades based on
V5.3 Tight R:R strategy with daily model retraining.

REQUIREMENTS:
1. OANDA Practice account (free): https://www.oanda.com/register/
2. API key from OANDA Developer Portal
3. Set credentials in config/oanda_credentials.json

Usage:
    python run_live_paper.py
"""

import sys
import io
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging

# Fix Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

from src.features import build_features
from src.strategy_configs import get_primary_config, adapt_to_1m_regime

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_log.txt', encoding='utf-8')
    ]
)
log = logging.getLogger(__name__)


# =============================================================================
# OANDA API CLIENT
# =============================================================================

class OandaClient:
    """Simple OANDA v20 API client for paper trading."""
    
    def __init__(self, api_key: str, account_id: str, practice: bool = True):
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = "https://api-fxpractice.oanda.com" if practice else "https://api-fxtrade.oanda.com"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    
    def get_account(self) -> Dict:
        """Get account summary."""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/summary"
        resp = requests.get(url, headers=self.headers, timeout=10)
        resp.raise_for_status()
        return resp.json()["account"]
    
    def get_candles(self, instrument: str = "EUR_USD", granularity: str = "M1", 
                    count: int = 5000) -> pd.DataFrame:
        """Get historical candles."""
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "count": count,
            "price": "M",  # midpoint
        }
        resp = requests.get(url, headers=self.headers, params=params, timeout=30)
        resp.raise_for_status()
        
        candles = resp.json().get("candles", [])
        records = []
        for c in candles:
            if not c.get("complete", False):
                continue
            mid = c.get("mid", {})
            records.append({
                "time": pd.to_datetime(c["time"]),
                "open": float(mid["o"]),
                "high": float(mid["h"]),
                "low": float(mid["l"]),
                "close": float(mid["c"]),
                "volume": int(c.get("volume", 0)),
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("time").reset_index(drop=True)
            # Remove timezone for consistency
            if df["time"].dt.tz is not None:
                df["time"] = df["time"].dt.tz_localize(None)
        return df
    
    def get_current_price(self, instrument: str = "EUR_USD") -> Dict:
        """Get current bid/ask price."""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
        params = {"instruments": instrument}
        resp = requests.get(url, headers=self.headers, params=params, timeout=10)
        resp.raise_for_status()
        prices = resp.json().get("prices", [])
        if prices:
            p = prices[0]
            return {
                "bid": float(p["bids"][0]["price"]),
                "ask": float(p["asks"][0]["price"]),
                "spread": float(p["asks"][0]["price"]) - float(p["bids"][0]["price"]),
            }
        return {}
    
    def place_market_order(self, instrument: str, units: int, 
                           sl_price: float = None, tp_price: float = None) -> Dict:
        """Place a market order with optional SL/TP."""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
        
        order = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(units),  # Positive for LONG, negative for SHORT
            "timeInForce": "FOK",  # Fill or Kill
        }
        
        if sl_price:
            order["stopLossOnFill"] = {"price": f"{sl_price:.5f}"}
        if tp_price:
            order["takeProfitOnFill"] = {"price": f"{tp_price:.5f}"}
        
        payload = {"order": order}
        
        resp = requests.post(url, headers=self.headers, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    
    def get_open_trades(self) -> List[Dict]:
        """Get list of open trades."""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/openTrades"
        resp = requests.get(url, headers=self.headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("trades", [])
    
    def close_trade(self, trade_id: str) -> Dict:
        """Close a specific trade."""
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{trade_id}/close"
        resp = requests.put(url, headers=self.headers, timeout=10)
        resp.raise_for_status()
        return resp.json()


# =============================================================================
# TRADING BOT
# =============================================================================

class TradingBot:
    """Paper trading bot using V5.3 strategy."""
    
    def __init__(self, client: OandaClient, config=None):
        self.client = client
        self.config = config or adapt_to_1m_regime(get_primary_config())
        self.model = None
        self.last_train_time = None
        self.daily_trades = []
        self.equity_curve = []
        
        # Position sizing
        self.risk_per_trade = 0.01  # 1% of account per trade
        self.max_open_trades = 1    # Only 1 trade at a time
        
        log.info(f"Bot initialized with strategy: {self.config.name}")
        log.info(f"  Threshold: {self.config.probability_threshold}")
        log.info(f"  SL: {self.config.sl_atr_multiplier}x ATR")
        log.info(f"  TP: {self.config.tp_atr_multiplier}x ATR")
    
    def train_model(self, df: pd.DataFrame):
        """Train or retrain the model on latest data."""
        log.info("Training model...")
        
        df_features = build_features(df, horizon_minutes=1)
        feature_cols = [c for c in df_features.columns 
                        if c not in {"time", "target", "return_h", "date"}]
        
        X = df_features[feature_cols]
        y = df_features["target"]
        
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=42,
        )
        self.model.fit(X, y)
        self.last_train_time = datetime.now()
        self.feature_cols = feature_cols
        
        log.info(f"Model trained on {len(df_features)} samples")
    
    def should_retrain(self) -> bool:
        """Check if model needs retraining (daily)."""
        if self.model is None:
            return True
        if self.last_train_time is None:
            return True
        
        hours_since_train = (datetime.now() - self.last_train_time).total_seconds() / 3600
        return hours_since_train >= 24
    
    def get_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal from latest data."""
        if self.model is None:
            return None
        
        # Build features for latest bar
        df_features = build_features(df, horizon_minutes=1)
        if df_features.empty:
            return None
        
        latest = df_features.iloc[-1:]
        
        # Check ATR filter
        atr_pips = latest["atr"].values[0] / self.config.pip_value
        if atr_pips < self.config.min_atr_pips or atr_pips > self.config.max_atr_pips:
            return None
        
        # Get prediction
        X = latest[self.feature_cols]
        prob = self.model.predict_proba(X)[0, 1]
        
        # Check probability gap
        prob_gap = abs(prob - 0.5)
        if prob_gap < self.config.min_probability_gap:
            return None
        
        # Determine direction
        if prob >= self.config.probability_threshold:
            direction = "LONG"
        elif prob <= (1 - self.config.probability_threshold):
            direction = "SHORT"
        else:
            return None
        
        return {
            "direction": direction,
            "probability": prob,
            "atr_pips": atr_pips,
            "close": latest["close"].values[0],
            "time": latest["time"].values[0],
        }
    
    def calculate_position_size(self, account: Dict, atr_pips: float) -> int:
        """Calculate position size based on risk."""
        balance = float(account.get("balance", 10000))
        risk_amount = balance * self.risk_per_trade
        
        # For EUR/USD: 1 unit = $0.0001 per pip
        # 10,000 units = $1 per pip
        sl_pips = atr_pips * self.config.sl_atr_multiplier
        cost_pips = self.config.spread_pips + self.config.slippage_pips
        total_risk_pips = sl_pips + cost_pips
        
        # Units = risk_amount / (risk_pips * pip_value_per_unit)
        # For EUR/USD mini lot (10k units), 1 pip = $1
        units = int(risk_amount / total_risk_pips * 10000)
        
        # Cap at reasonable size
        units = min(units, 100000)  # Max 1 standard lot
        units = max(units, 1000)    # Min micro lot
        
        return units
    
    def execute_trade(self, signal: Dict) -> bool:
        """Execute a trade based on signal."""
        try:
            # Check if we already have open trades
            open_trades = self.client.get_open_trades()
            if len(open_trades) >= self.max_open_trades:
                log.info("Max open trades reached, skipping")
                return False
            
            # Get current price
            price_info = self.client.get_current_price("EUR_USD")
            if not price_info:
                log.error("Could not get current price")
                return False
            
            # Get account for position sizing
            account = self.client.get_account()
            units = self.calculate_position_size(account, signal["atr_pips"])
            
            # Calculate SL and TP
            atr = signal["atr_pips"] * self.config.pip_value
            
            if signal["direction"] == "LONG":
                entry_price = price_info["ask"]
                sl_price = entry_price - (self.config.sl_atr_multiplier * atr)
                tp_price = entry_price + (self.config.tp_atr_multiplier * atr)
            else:  # SHORT
                entry_price = price_info["bid"]
                sl_price = entry_price + (self.config.sl_atr_multiplier * atr)
                tp_price = entry_price - (self.config.tp_atr_multiplier * atr)
                units = -units  # Negative for short
            
            # Place order
            log.info(f"Placing {signal['direction']} order: {abs(units)} units")
            log.info(f"  Entry: {entry_price:.5f}, SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            log.info(f"  P(UP): {signal['probability']:.2%}")
            
            result = self.client.place_market_order(
                instrument="EUR_USD",
                units=units,
                sl_price=sl_price,
                tp_price=tp_price,
            )
            
            if "orderFillTransaction" in result:
                log.info(f"Order filled: {result['orderFillTransaction']['id']}")
                return True
            else:
                log.warning(f"Order not filled: {result}")
                return False
            
        except Exception as e:
            log.error(f"Trade execution failed: {e}")
            return False
    
    def run_once(self):
        """Run one iteration of the trading loop."""
        try:
            # Check if we need to retrain
            if self.should_retrain():
                log.info("Daily retrain triggered")
                df = self.client.get_candles("EUR_USD", "M1", count=5000)
                if len(df) > 1000:
                    self.train_model(df)
                else:
                    log.warning(f"Not enough data for training: {len(df)} candles")
                    return
            
            # Get latest data for signal
            df = self.client.get_candles("EUR_USD", "M1", count=200)
            if df.empty:
                log.warning("No candle data received")
                return
            
            # Generate signal
            signal = self.get_signal(df)
            
            if signal:
                log.info(f"Signal: {signal['direction']} @ {signal['probability']:.2%}")
                self.execute_trade(signal)
            else:
                log.debug("No signal generated")
            
        except Exception as e:
            log.error(f"Error in trading loop: {e}")
    
    def run(self, interval_seconds: int = 60):
        """Main trading loop."""
        log.info("=" * 60)
        log.info("FOREX ML Paper Trading Bot Started")
        log.info(f"Strategy: {self.config.name}")
        log.info(f"Interval: {interval_seconds} seconds")
        log.info("=" * 60)
        
        try:
            account = self.client.get_account()
            log.info(f"Account: {account.get('id')}")
            log.info(f"Balance: ${float(account.get('balance', 0)):,.2f}")
            log.info(f"Currency: {account.get('currency')}")
        except Exception as e:
            log.error(f"Could not connect to OANDA: {e}")
            return
        
        log.info("Starting trading loop... (Ctrl+C to stop)")
        
        while True:
            self.run_once()
            time.sleep(interval_seconds)


# =============================================================================
# MAIN
# =============================================================================

def load_credentials() -> Dict:
    """Load OANDA credentials from config file."""
    cred_path = Path(__file__).parent / "config" / "oanda_credentials.json"
    
    if not cred_path.exists():
        # Create template
        template = {
            "api_key": "YOUR_OANDA_API_KEY_HERE",
            "account_id": "YOUR_ACCOUNT_ID_HERE",
            "practice": True
        }
        cred_path.parent.mkdir(parents=True, exist_ok=True)
        cred_path.write_text(json.dumps(template, indent=2))
        
        print("=" * 60)
        print("SETUP REQUIRED")
        print("=" * 60)
        print()
        print(f"Created credential template at: {cred_path}")
        print()
        print("To get your credentials:")
        print("1. Create OANDA Practice account: https://www.oanda.com/register/")
        print("2. Go to: https://www.oanda.com/demo-account/tpa/personal_token")
        print("3. Generate API key")
        print("4. Edit the config file with your credentials")
        print()
        return None
    
    with open(cred_path) as f:
        creds = json.load(f)
    
    if creds.get("api_key") == "YOUR_OANDA_API_KEY_HERE":
        print("Please edit your credentials in:", cred_path)
        return None
    
    return creds


def main():
    creds = load_credentials()
    if not creds:
        return
    
    client = OandaClient(
        api_key=creds["api_key"],
        account_id=creds["account_id"],
        practice=creds.get("practice", True),
    )
    
    bot = TradingBot(client)
    
    try:
        bot.run(interval_seconds=60)  # Check every minute
    except KeyboardInterrupt:
        log.info("Bot stopped by user")


if __name__ == "__main__":
    main()
