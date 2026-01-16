from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from .configs import load_settings


def load_models(models_dir: Path = None) -> Dict[int, joblib]:
    settings = load_settings()
    base = Path(models_dir or settings.paths.models_dir)
    models = {}
    for w in settings.windows_weeks:
        path = base / f"w{w}" / "model.joblib"
        if path.exists():
            models[w] = joblib.load(path)
    if not models:
        raise FileNotFoundError(f"No models found under {base}")
    return models


def load_metrics(models_dir: Path = None) -> Dict[int, Dict]:
    settings = load_settings()
    base = Path(models_dir or settings.paths.models_dir)
    metrics = {}
    for w in settings.windows_weeks:
        path = base / f"w{w}" / "metrics.json"
        if path.exists():
            metrics[w] = pd.read_json(path, typ="series").to_dict()
    return metrics


def pick_best_model(metrics: Dict[int, Dict]) -> int:
    if not metrics:
        raise ValueError("Empty metrics, cannot pick best model")
    best_w = max(metrics.keys(), key=lambda w: metrics[w].get("f1", 0))
    return best_w


def predict(models: Dict[int, joblib], features: pd.DataFrame) -> Dict[int, np.ndarray]:
    preds = {}
    for w, model in models.items():
        proba = model.predict_proba(features)[:, 1]
        preds[w] = proba
    return preds


def ensemble_prob(preds: Dict[int, np.ndarray], metrics: Dict[int, Dict]) -> np.ndarray:
    weights = {}
    for w, proba in preds.items():
        weight = metrics.get(w, {}).get("f1", 0.5)
        weights[w] = max(weight, 1e-3)
    total_weight = sum(weights.values())
    weighted = sum(proba * weights[w] for w, proba in preds.items()) / total_weight
    return weighted


def should_trade(prob: float, threshold: float) -> bool:
    return prob >= threshold


def position_size(equity: float, risk_per_trade: float, atr_pips: float, pip_value: float = 10.0) -> float:
    if atr_pips <= 0:
        return 0.0
    return (equity * risk_per_trade) / (atr_pips * pip_value)


def apply_drawdown_breaker(equity_curve: List[float], cut: float) -> bool:
    if not equity_curve:
        return False
    peak = max(equity_curve)
    if peak == 0:
        return False
    dd = (peak - equity_curve[-1]) / peak
    return dd >= cut


def inference_step(latest_features: pd.DataFrame, equity_curve: List[float]) -> Tuple[float, bool]:
    settings = load_settings()
    models = load_models()
    metrics = load_metrics()
    preds = predict(models, latest_features)
    prob = float(ensemble_prob(preds, metrics)[-1])
    halted = apply_drawdown_breaker(equity_curve, settings.training.drawdown_cut)
    do_trade = should_trade(prob, settings.training.probability_threshold) and not halted
    return prob, do_trade
