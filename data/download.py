"""S1: download adjusted closes (yfinance), cache raw CSVs, fill gaps (max 1 day), log every fix."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def fill_gaps(prices: pd.DataFrame, limit: int = 1):
    """Forward-fill at most `limit` consecutive missing days per asset.

    Returns (filled_prices, gap_log). Rows that are still missing afterwards (e.g. before an
    ETF's inception, or gaps longer than `limit`) are left as NaN for the caller to trim.
    """
    log = []
    filled = prices.copy()
    for col in prices.columns:
        s = prices[col]
        na = s.isna()
        # only gaps that follow a valid price (not leading NaNs before inception)
        first_valid = s.first_valid_index()
        if first_valid is None:
            continue
        run = 0
        for dt, is_na in na.loc[first_valid:].items():
            if is_na:
                run += 1
                if run <= limit:
                    log.append({"asset": col, "date": str(dt.date()), "action": "ffill"})
                else:
                    log.append({"asset": col, "date": str(dt.date()), "action": "left_missing"})
            else:
                run = 0
        filled[col] = s.ffill(limit=limit)
    return filled, log


def download_basket(tickers, start, end, name="basket", raw_dir="data/raw", force=False):
    """Pull adjusted closes for `tickers`, cache raw CSV, forward-fill <=1 day, trim to the
    first date on which every asset has a price. Writes <raw_dir>/<name>.csv and
    <raw_dir>/<name>_gaplog.json. Returns a clean prices DataFrame (no NaN)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv = raw_dir / f"{name}.csv"
    if csv.exists() and not force:
        raw = pd.read_csv(csv, index_col=0, parse_dates=True)
    else:
        import yfinance as yf  # imported lazily so the rest of the repo works offline

        raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
        raw = raw[list(tickers)]
        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
        raw.to_csv(csv)
    filled, log = fill_gaps(raw, limit=1)
    clean = filled.dropna(how="any")
    n_trim = len(filled) - len(clean)
    left = [g for g in log if g["action"] == "left_missing"]
    summary = {
        "n_rows_raw": len(raw),
        "n_rows_clean": len(clean),
        "n_rows_trimmed": n_trim,
        "first_clean_date": str(clean.index[0].date()),
        "n_ffill": sum(g["action"] == "ffill" for g in log),
        "n_left_missing": len(left),
        "fixes": log,
    }
    (raw_dir / f"{name}_gaplog.json").write_text(json.dumps(summary, indent=1))
    return clean
