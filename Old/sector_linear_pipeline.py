#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sector Linear Regression Pipeline (Monthly, Fundamentals + Price) - yfinance only
----------------------------------------------------------------------------------
- Downloads S&P 500 constituents and their sectors from Wikipedia.
- Groups sectors into coarse buckets (Technology, Consumer, Industrials, etc.).
- For selected buckets, picks N tickers per bucket.
- Uses yfinance to pull:
  - Historical prices
  - Fundamental data (P/E, P/B, P/S, etc.) from .info
- Builds MONTHLY dataset with fundamentals and next-month price target.
- Trains a per-sector Linear Regression to predict next-month *log-price*
  from current month fundamentals.
- Saves datasets, models, coefficients, predictions, and metrics.

USAGE (example):
  python sector_linear_pipeline.py --sectors Technology Consumer Industrials \
      --per_sector 50 --start 2015-01-01 --out ./out

Requirements:
  pip install yfinance pandas numpy scikit-learn joblib lxml requests
"""

import os
import time
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump

pd.options.mode.copy_on_write = True


# -----------------------------
# Helpers
# -----------------------------

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def to_month_end(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert dates to their month-end timestamp."""
    return idx.to_period('M').to_timestamp('M')


def get_sp500_constituents() -> pd.DataFrame:
    """
    Try fetching S&P 500 list from Wikipedia.
    If blocked (HTTP 403), load fallback CSV of tickers and sectors.
    """
    import io
    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GPTResearchBot/1.0; +https://openai.com)"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        df = tables[0].copy()
        df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        return df[["ticker", "name", "sector"]]
    except Exception as e:
        print(f"[WARN] Wikipedia access failed ({e}). Using fallback CSV...")

        # === fallback ===
        csv_fallback = """ticker,name,sector
AAPL,Apple Inc,Information Technology
MSFT,Microsoft Corp,Information Technology
GOOGL,Alphabet Inc,Communication Services
AMZN,Amazon.com Inc,Consumer Discretionary
META,Meta Platforms Inc,Communication Services
NVDA,NVIDIA Corp,Information Technology
JPM,JP Morgan Chase & Co,Financials
UNH,UnitedHealth Group Inc,Health Care
XOM,Exxon Mobil Corp,Energy
PG,Procter & Gamble Co,Consumer Staples
HD,Home Depot Inc,Consumer Discretionary
CAT,Caterpillar Inc,Industrials
"""
        df = pd.read_csv(io.StringIO(csv_fallback))
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        return df


SECTOR_BUCKET_MAP = {
    "Information Technology": "Technology",
    "Communication Services": "Communication",
    "Consumer Discretionary": "Consumer",
    "Consumer Staples": "Consumer",
    "Industrials": "Industrials",
    "Health Care": "HealthCare",
    "Financials": "Financials",
    "Energy": "Energy",
    "Materials": "Materials",
    "Real Estate": "RealEstate",
    "Utilities": "Utilities",
}


def filter_tickers_by_buckets(df_const: pd.DataFrame, buckets: List[str], per_sector: int) -> pd.DataFrame:
    df_const = df_const.copy()
    df_const["bucket"] = df_const["sector"].map(SECTOR_BUCKET_MAP)
    df_const = df_const[df_const["bucket"].isin(buckets)]
    # take first N per bucket (you can randomize or sort differently if you prefer)
    out = []
    for b in buckets:
        g = df_const[df_const["bucket"] == b].head(per_sector)
        out.append(g)
    return pd.concat(out, axis=0).reset_index(drop=True)


def yf_download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close prices (monthly sampled later).
    """
    log(f"Downloading daily prices for {len(tickers)} tickers from {start} to {end}...")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.sort_index()
    return df


def quarterly_frames(t: yf.Ticker) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (fin_q, bs_q) with index as quarter dates and columns as items.
    """
    try:
        fin = t.quarterly_financials  # rows=items, cols=dates
        fin_q = fin.T if isinstance(fin, pd.DataFrame) and not fin.empty else pd.DataFrame()
    except Exception:
        fin_q = pd.DataFrame()

    try:
        bs = t.quarterly_balance_sheet
        bs_q = bs.T if isinstance(bs, pd.DataFrame) and not bs.empty else pd.DataFrame()
    except Exception:
        bs_q = pd.DataFrame()

    # Normalize column labels to str
    if not fin_q.empty:
        fin_q.columns = fin_q.columns.astype(str)
        fin_q.index = pd.to_datetime(fin_q.index)
    if not bs_q.empty:
        bs_q.columns = bs_q.columns.astype(str)
        bs_q.index = pd.to_datetime(bs_q.index)

    return fin_q, bs_q


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            s = df[c]
            if s.notna().any():
                return s.astype(float)
    return None


def compute_ratios_for_ticker(ticker: str, px_m: pd.Series) -> pd.DataFrame:
    """
    Compute quarterly ratios P/E, P/B, P/S, EV/EBITDA, ROE, then forward-fill to monthly.
    """
    t = yf.Ticker(ticker)

    # Shares outstanding (snapshot, not historical, but acceptable for student project)
    try:
        info = t.get_info()
    except Exception:
        info = {}
    shares_out = info.get("sharesOutstanding", None)
    if shares_out is not None:
        try:
            shares_out = float(shares_out)
        except Exception:
            shares_out = None

    fin_q, bs_q = quarterly_frames(t)

    # Extract quarterly series
    net_income_q = pick_col(fin_q, ["Net Income", "NetIncome"])
    revenue_q    = pick_col(fin_q, ["Total Revenue", "TotalRevenue"])
    ebitda_q     = pick_col(fin_q, ["Ebitda", "EBITDA"])

    equity_q     = pick_col(bs_q, ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Stockholders Equity"])
    cash_q       = pick_col(bs_q, ["Cash And Cash Equivalents", "Cash And Cash Equivalents USD", "Cash"])
    ltd_q        = pick_col(bs_q, ["Long Term Debt"])
    std_q        = pick_col(bs_q, ["Short Long Term Debt", "Short Term Borrowings"])

    debt_q = None
    if ltd_q is not None and std_q is not None:
        debt_q = (ltd_q.fillna(0) + std_q.fillna(0))
    elif ltd_q is not None:
        debt_q = ltd_q
    elif std_q is not None:
        debt_q = std_q

    # Compute TTM where needed
    def ttm(s: Optional[pd.Series]) -> Optional[pd.Series]:
        if s is None:
            return None
        return s.sort_index().rolling(4, min_periods=1).sum()

    net_income_ttm = ttm(net_income_q)
    revenue_ttm    = ttm(revenue_q)
    ebitda_ttm     = ttm(ebitda_q)

    # Align monthly price to quarter dates (use month-end of quarter date)
    q_idx = fin_q.index if not fin_q.empty else (bs_q.index if not bs_q.empty else pd.DatetimeIndex([]))
    if len(q_idx) == 0:
        return pd.DataFrame()

    q_me = to_month_end(q_idx)
    px_q = px_m.reindex(q_me).ffill()  # month-end price aligned to quarter

    # Market Cap = Price * Shares
    if shares_out is not None:
        mcap_q = px_q * shares_out
    else:
        mcap_q = pd.Series(index=px_q.index, dtype=float)

    # Ratios
    pe_q = None
    if net_income_ttm is not None and shares_out is not None and shares_out > 0:
        eps_ttm = net_income_ttm / shares_out
        with np.errstate(divide="ignore", invalid="ignore"):
            pe_q = px_q / eps_ttm.replace(0, np.nan)

    pb_q = None
    if equity_q is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            if shares_out is not None and shares_out > 0:
                pb_q = (px_q * shares_out) / equity_q.replace(0, np.nan)
            elif mcap_q is not None:
                pb_q = mcap_q / equity_q.replace(0, np.nan)

    ps_q = None
    if revenue_ttm is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            if mcap_q is not None:
                ps_q = mcap_q / revenue_ttm.replace(0, np.nan)

    ev_ebitda_q = None
    if ebitda_ttm is not None:
        # EV = MarketCap + Debt - Cash
        debt_aligned = debt_q.reindex(q_idx).reindex(q_me).ffill() if debt_q is not None else pd.Series(0, index=q_me)
        cash_aligned = cash_q.reindex(q_idx).reindex(q_me).ffill() if cash_q is not None else pd.Series(0, index=q_me)
        ev_q = (mcap_q.fillna(np.nan)) + debt_aligned.fillna(0) - cash_aligned.fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ev_ebitda_q = ev_q / ebitda_ttm.reindex(q_idx).reindex(q_me).ffill().replace(0, np.nan)

    roe_q = None
    if net_income_ttm is not None and equity_q is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            roe_q = (net_income_ttm.reindex(q_idx).reindex(q_me).ffill()) / equity_q.reindex(q_idx).reindex(q_me).ffill().replace(0, np.nan)

    ratios_q = pd.DataFrame({
        "PE": pe_q,
        "PB": pb_q,
        "PS": ps_q,
        "EV_EBITDA": ev_ebitda_q,
        "ROE": roe_q,
    }).sort_index()

    # Forward-fill to monthly frequency
    ratios_m = ratios_q.resample("ME").ffill().reindex(px_m.index).ffill()

    return ratios_m


def build_monthly_dataset(tickers: List[str], sector_name: str, prices_daily: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    For a list of tickers in a sector: build a monthly panel with features and next-month log-price target.
    """
    # Monthly close
    px_m = prices_daily[tickers].resample("ME").last()

    rows = []
    for tk in tickers:
        try:
            ratios_m = compute_ratios_for_ticker(tk, px_m[tk].dropna())
        except Exception as e:
            log(f"  {tk}: failed to compute ratios ({e}), skipping")
            continue
        if ratios_m.empty:
            continue

        df = pd.DataFrame(index=ratios_m.index)
        df["ticker"] = tk
        df["sector"] = sector_name
        df["price"] = px_m[tk].reindex(df.index).ffill()

        # Features at t
        for col in ["PE", "PB", "PS", "EV_EBITDA", "ROE"]:
            df[col] = ratios_m[col]

        # Target: next-month log-price
        df["log_price_t"] = np.log(df["price"].replace(0, np.nan))
        df["log_price_t+1"] = df["log_price_t"].shift(-1)

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    data = pd.concat(rows)

    # Reset index to get date column
    data = data.reset_index()
    # The index should be the date, so rename it properly
    if "index" in data.columns:
        data = data.rename(columns={"index": "date"})
    # Make sure we have a date column
    if "date" not in data.columns:
        # Find the datetime column
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            data = data.rename(columns={datetime_cols[0]: "date"})

    # Drop rows with NaNs in features or target
    feats = ["PE", "PB", "PS", "EV_EBITDA", "ROE"]
    data = data.dropna(subset=feats + ["log_price_t+1"]).copy()

    # Ensure date column is datetime type
    data["date"] = pd.to_datetime(data["date"])
    
    # Filter by time window (start–end)
    mask = data["date"] >= pd.to_datetime(start)
    if end is not None:
        mask = mask & (data["date"] <= pd.to_datetime(end))
    data = data[mask].copy()

    return data.sort_values(["date", "ticker"])


def fit_linear_by_sector(df: pd.DataFrame, sector_name: str, out_dir: str) -> Dict[str, float]:
    """
    Fit LinearRegression on features to predict next-month log-price.
    Save model, coefficients, predictions, and dataset CSV.
    """
    ensure_dir(out_dir)
    feats = ["PE", "PB", "PS", "EV_EBITDA", "ROE"]

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)
    
    # Debug: print date range
    log(f"  Date range: {df['date'].min()} to {df['date'].max()}, total rows: {len(df)}")
    
    # Chronological split: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    tr = df.iloc[:split_idx].copy()
    te = df.iloc[split_idx:].copy()
    
    log(f"  Train: {len(tr)} rows ({tr['date'].min()} to {tr['date'].max()})")
    log(f"  Test: {len(te)} rows ({te['date'].min()} to {te['date'].max()})")

    if len(tr) < 100 or len(te) < 30:
        log(f"  WARNING: not enough data for sector {sector_name} (train={len(tr)}, test={len(te)})")

    pre = ColumnTransformer([("num", StandardScaler(), feats)], remainder="drop")
    model = Pipeline([("pre", pre), ("lr", LinearRegression())])
    model.fit(tr[feats], tr["log_price_t+1"])

    # Predictions on test
    te_pred_log = model.predict(te[feats])
    te_pred_price = np.exp(te_pred_log)
    te_true_price = np.exp(te["log_price_t+1"])

    mae = float(mean_absolute_error(te_true_price, te_pred_price))
    rmse = float(np.sqrt(mean_squared_error(te_true_price, te_pred_price)))

    # Save predictions
    pred = te[["date", "ticker", "sector"]].copy()
    pred["price_true"] = te_true_price.values
    pred["price_pred"] = te_pred_price
    pred_out = os.path.join(out_dir, f"{sector_name}_predictions.csv")
    pred.to_csv(pred_out, index=False)

    # Save dataset (all)
    data_out = os.path.join(out_dir, f"{sector_name}_monthly_dataset.csv")
    df.to_csv(data_out, index=False)

    # Save model
    model_out = os.path.join(out_dir, f"{sector_name}_linear_model.pkl")
    dump(model, model_out)

    # Save coefficients (in standardized feature space)
    lr = model.named_steps["lr"]
    coef = pd.DataFrame({"feature": feats, "coef": lr.coef_})
    coef_out = os.path.join(out_dir, f"{sector_name}_coefficients.csv")
    coef.to_csv(coef_out, index=False)

    log(f"  {sector_name}: MAE={mae:.4f} RMSE={rmse:.4f} | saved to {out_dir}")
    return {"sector": sector_name, "mae": mae, "rmse": rmse,
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "dataset_csv": data_out, "pred_csv": pred_out, "model_pkl": model_out, "coef_csv": coef_out}


def main(args):
    ensure_dir(args.out)
    out_data = os.path.join(args.out, "data")
    out_models = os.path.join(args.out, "models")
    ensure_dir(out_data)
    ensure_dir(out_models)

    log("Fetching S&P 500 constituents...")
    sp500 = get_sp500_constituents()
    sp500["bucket"] = sp500["sector"].map(SECTOR_BUCKET_MAP)

    # Resolve requested buckets
    wanted = args.sectors if args.sectors else ["Technology", "Consumer", "Industrials"]
    log(f"Buckets requested: {wanted}")
    picks = filter_tickers_by_buckets(sp500, wanted, args.per_sector)

    # Price download for all picked
    tickers = sorted(picks["ticker"].unique().tolist())
    prices_daily = yf.download(tickers, start=args.start, end=args.end, auto_adjust=True, progress=False)["Close"]
    if isinstance(prices_daily, pd.Series):
        prices_daily = prices_daily.to_frame()
    prices_daily = prices_daily.sort_index()

    metrics_rows = []
    for b in wanted:
        tks = picks.loc[picks["bucket"] == b, "ticker"].unique().tolist()
        if not tks:
            log(f"No tickers for bucket {b}, skipping.")
            continue

        log(f"Building monthly dataset for {b} with {len(tks)} tickers...")
        df_b = build_monthly_dataset(tks, b, prices_daily, args.start, args.end)

        if df_b.empty:
            log(f"  No data built for {b}, skipping model fit.")
            continue

        # Fit model + save artifacts under out_models/bucket
        out_b = os.path.join(out_models, b)
        ensure_dir(out_b)
        m = fit_linear_by_sector(df_b, b, out_b)
        metrics_rows.append(m)

        # Also save the monthly dataset under out_data
        ds_path = os.path.join(out_data, f"{b}_monthly.csv")
        df_b.to_csv(ds_path, index=False)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(os.path.join(args.out, "metrics_summary.csv"), index=False)
        log("Done. Metrics summary written.")

    log("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sector-wise monthly dataset and linear regression (yfinance).")
    parser.add_argument("--sectors", nargs="*", default=["Technology", "Consumer", "Industrials"],
                        help="Sector buckets to include (e.g., Technology Consumer Industrials Energy Financials ...)")
    parser.add_argument("--per_sector", type=int, default=50, help="Number of tickers per bucket")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", type=str, default="./out", help="Output directory")
    args = parser.parse_args()
    main(args)
