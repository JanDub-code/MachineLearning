from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .configs import load_settings
from .features import build_features


def _split_window(df: pd.DataFrame, holdout_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["time"].max() - pd.Timedelta(hours=holdout_hours)
    train_df = df[df["time"] <= cutoff]
    val_df = df[df["time"] > cutoff]
    return train_df, val_df


def _train_single(df: pd.DataFrame, window_weeks: int, holdout_hours: int) -> Dict:
    train_df, val_df = _split_window(df, holdout_hours)
    feature_cols = [c for c in df.columns if c not in {"time", "target", "return_h"}]
    X_train, y_train = train_df[feature_cols], train_df["target"]
    X_val, y_val = val_df[feature_cols], val_df["target"]

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    return {
        "model": clf,
        "metrics": {"accuracy": acc, "f1": f1},
        "feature_importances": dict(zip(feature_cols, clf.feature_importances_.tolist())),
    }


def train_models_from_parquet(parquet_path: Path) -> Dict[int, Dict]:
    settings = load_settings()
    data = pd.read_parquet(parquet_path)
    data["time"] = pd.to_datetime(data["time"])
    results: Dict[int, Dict] = {}

    for window_weeks in settings.windows_weeks:
        window_minutes = window_weeks * 7 * 24 * 60
        window_df = data.tail(window_minutes)
        feats = build_features(window_df, horizon_minutes=settings.training.target_horizon_minutes)
        res = _train_single(feats, window_weeks=window_weeks, holdout_hours=settings.training.holdout_hours)
        results[window_weeks] = res

        out_dir = Path(settings.paths.models_dir) / f"w{window_weeks}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.joblib"
        meta_path = out_dir / "metrics.json"
        joblib.dump(res["model"], model_path)
        pd.Series(res["metrics"]).to_json(meta_path)

    return results


def train_latest_raw(pair: str = "EUR_USD", granularity: str = "M1") -> Dict[int, Dict]:
    settings = load_settings()
    raw_dir = Path(settings.paths.raw_dir) / pair
    if not raw_dir.exists():
        raise FileNotFoundError(f"No raw data found under {raw_dir}")
    latest_date_dirs = sorted(raw_dir.iterdir())
    if not latest_date_dirs:
        raise FileNotFoundError(f"No date subfolders under {raw_dir}")
    latest = latest_date_dirs[-1]
    parquet_path = latest / f"{granularity}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing file {parquet_path}")
    return train_models_from_parquet(parquet_path)
