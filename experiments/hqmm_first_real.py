"""First real-data run of the HQMM (step Sa10, owner: Sarah).

Three US sector funds (financials, technology, energy) around the 2008 crisis.
A quick stand-in until Shriker's data pipeline and harness land: it downloads
prices itself and uses one train / validation / test split, not the paper's
walk-forward protocol.

Run from the repo root (needs internet):
    python -m experiments.hqmm_first_real
Saves results/figures/first_real_2008.png and logs to results/logs/runs.jsonl.

Symbols: returns are divided by each asset's volatility over the TRAINING period
(a fixed number, known in advance), not by recent volatility. Dividing by recent
volatility would hide exactly the volatility jumps we want to detect.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

# run as `python experiments/hqmm_first_real.py` from anywhere: make the repo root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICKERS = ["XLF", "XLK", "XLE"]
START, END = "1999-01-01", "2011-01-01"
TRAIN_END, VAL_END = "2005-12-31", "2006-12-31"     # test: 2007-2010
K = 3
RAW_CACHE = os.path.join("data", "raw", "first_real_prices.csv")


# ----------------------------------------------------------------------------- data
def download_prices(tickers=TICKERS, start=START, end=END, cache=RAW_CACHE) -> pd.DataFrame:
    """Adjusted daily closes, cached to data/raw so later runs work offline."""
    if os.path.exists(cache):
        cached = pd.read_csv(cache, index_col=0, parse_dates=True)
        if len(cached) > 1000 and all(t in cached.columns for t in tickers):
            return cached
        print(f"ignoring unusable cache {cache} ({len(cached)} rows); downloading again")
    import yfinance as yf
    cols = {}
    for t in tickers:  # one ticker at a time, with retries: parallel downloads can hit "database is locked"
        for attempt in range(4):
            try:
                df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, threads=False)
                close = df["Close"]
                close = close.iloc[:, 0] if hasattr(close, "columns") else close
                if len(close.dropna()) > 0:
                    cols[t] = close
                    break
            except Exception as e:  # noqa: BLE001
                print(f"{t}: attempt {attempt + 1} failed ({type(e).__name__}); retrying")
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"could not download {t}; check your internet connection and retry")
    px = pd.DataFrame(cols)[tickers].dropna()
    if len(px) < 1000:
        raise RuntimeError(f"only {len(px)} usable rows downloaded; nothing was saved. Check your connection and retry")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    px.to_csv(cache)
    return px


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()


def make_symbols(returns: pd.DataFrame, train_mask: np.ndarray, K: int = K):
    """Scale by training-period volatility, then cut at equal-share thresholds of the
    pooled training returns. Everything is fitted on training days only."""
    r = returns.to_numpy()
    scale = r[train_mask].std(axis=0)
    z = r / scale
    thresholds = np.quantile(z[train_mask].ravel(), np.arange(1, K) / K)
    symbols = np.searchsorted(thresholds, z).astype(int)
    return symbols, z, thresholds


def volatility_label(returns: pd.DataFrame, train_mask: np.ndarray, window: int = 21, pct: float = 90):
    """Rough stress label for scoring only (regime definition R2 style): 21-day realised
    volatility of the equal-weight basket above the 90th percentile of training days.
    Uses data up to and including each day; it is never shown to the model."""
    basket = returns.mean(axis=1)
    rv = basket.rolling(window).std() * np.sqrt(252)
    threshold = np.nanpercentile(rv.to_numpy()[train_mask], pct)
    return (rv.to_numpy() > threshold).astype(int), rv, threshold


# ----------------------------------------------------------------------------- run
def run(prices: pd.DataFrame, n_restarts: int = 3, seed: int = 0, save: bool = True) -> dict:
    from sklearn.metrics import roc_auc_score

    from models.base import Fold
    from models.hqmm.engine import extreme_share
    from models.hqmm.generic import GenericBuilder
    from models.hqmm.model import HQMMModel

    ret = log_returns(prices)
    dates = ret.index
    train = np.asarray(dates <= TRAIN_END)
    val = np.asarray((dates > TRAIN_END) & (dates <= VAL_END))
    test = np.asarray(dates > VAL_END)
    symbols, _, thresholds = make_symbols(ret, train)
    label, rv, rv_threshold = volatility_label(ret, train)

    fold = Fold(0, np.flatnonzero(train), np.flatnonzero(val), np.flatnonzero(test))
    model = HQMMModel(lambda n: GenericBuilder(2, n, layers=3), K=K, n_register=2,
                      n_restarts=n_restarts, name="Q-G first real")
    t0 = time.time()
    model.fit(None, symbols, fold, seed=seed)
    train_s = time.time() - t0
    score = model.score(None, symbols, fold.test_idx)

    e = extreme_share(symbols, K)
    ref = pd.Series(e).rolling(10, min_periods=1).mean().to_numpy()[fold.test_idx]
    y = label[fold.test_idx]
    out = {
        "test_days": int(test.sum()),
        "stress_share_test": float(y.mean()),
        "auc_hqmm": float(roc_auc_score(y, score)) if 0 < y.mean() < 1 else float("nan"),
        "auc_reference": float(roc_auc_score(y, ref)) if 0 < y.mean() < 1 else float("nan"),
        "restart_skills": [round(v, 3) for v in model.restart_vals],
        "n_params": model.n_params(),
        "train_seconds": round(train_s),
    }
    print(f"test days {out['test_days']}, stress share {out['stress_share_test']:.2f}")
    print(f"restart validation skills: {out['restart_skills']}")
    print(f"test AUC vs volatility label  HQMM: {out['auc_hqmm']:.3f}   "
          f"10-day extreme-share reference: {out['auc_reference']:.3f}")

    if save:
        from harness.runner import log_run
        log_run(model.name, 0, seed, {"tickers": list(prices.columns), "K": K, "r": 2, "layers": 3,
                                      "train_end": TRAIN_END, "val_end": VAL_END},
                out, author="Sarah")
        _plot(dates[fold.test_idx], score, rv.to_numpy()[fold.test_idx], y, rv_threshold, out["auc_hqmm"])
    return out


def _plot(dates, score, rv, label, rv_threshold, auc):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for ax in axes:
        ax.fill_between(dates, 0, 1, where=label == 1, transform=ax.get_xaxis_transform(),
                        color="tab:red", alpha=0.15, label="high-volatility days (label)")
    axes[0].plot(dates, score, color="black", lw=1, label="HQMM predicted share of big moves")
    axes[0].set_ylabel("HQMM stress score")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(dates, rv, color="tab:blue", lw=1, label="basket 21-day volatility (annualised)")
    axes[1].axhline(rv_threshold, color="tab:blue", ls="--", lw=0.8, label="label threshold")
    axes[1].set_ylabel("volatility")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"First real-data run: {', '.join(TICKERS)}, test 2007–2010 (AUC = {auc:.2f})")
    fig.tight_layout()
    out = os.path.join("results", "figures", "first_real_2008.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    run(download_prices())
