"""S5: exploratory figures. Used by notebooks/eda.ipynb; each function saves a PNG into results/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.discretize import Discretizer

FIG = Path(__file__).resolve().parents[1] / "results" / "figures"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    return FIG / name


def plot_returns_and_vol(md, tag="basket"):
    ew = pd.Series(md.equal_weight_returns(), index=md.dates)
    vol = ew.rolling(21).std() * np.sqrt(252)
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(np.exp(ew.cumsum()), lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_title(f"{tag}: equal-weight index (log)")
    ax[1].plot(ew, lw=0.4)
    ax[1].set_title("daily log returns")
    ax[2].plot(vol, lw=0.8)
    ax[2].set_title("21-day realised volatility (annualised)")
    return _save(fig, f"eda_returns_vol_{tag}.png")


def plot_corr_heatmaps(md, stress, tag="basket"):
    """Average pairwise correlation matrix on calm vs stress days (trailing 63-day correlations)."""
    calm = md.corr[np.asarray(stress) == 0].mean(0)
    hot = md.corr[np.asarray(stress) == 1].mean(0)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, m, t in zip(ax[:2], (calm, hot), ("calm days", "stress days")):
        im = a.imshow(m, vmin=0, vmax=1, cmap="viridis")
        a.set_title(t)
        a.set_xticks(range(len(md.assets)), md.assets, rotation=90)
        a.set_yticks(range(len(md.assets)), md.assets)
    fig.colorbar(im, ax=ax[:2], shrink=0.8)
    off = ~np.eye(len(md.assets), dtype=bool)
    ax[2].plot(md.dates, md.corr[:, off].mean(1), lw=0.8)
    ax[2].set_title("mean pairwise correlation")
    return _save(fig, f"eda_corr_heatmaps_{tag}.png")


def plot_symbol_frequencies(md, K=3, method="quantile", tag="basket"):
    d = Discretizer(K, method).fit(md.std_returns)
    sym = d.transform(md.std_returns)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    freq = np.stack([(sym == k).mean(0) for k in range(K)])
    for k in range(K):
        ax[0].bar(np.arange(len(md.assets)) + 0.8 / K * k, freq[k], width=0.8 / K, label=f"symbol {k}")
    ax[0].set_xticks(np.arange(len(md.assets)) + 0.4, md.assets)
    ax[0].legend()
    ax[0].set_title("symbol frequency by asset")
    extreme = ((sym == 0) | (sym == K - 1)).mean(1)
    ax[1].plot(md.dates, pd.Series(extreme).rolling(63).mean(), lw=0.8)
    ax[1].set_title("share of extreme symbols (63-day mean)")
    return _save(fig, f"eda_symbols_{tag}.png")
