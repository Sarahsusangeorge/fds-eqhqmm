"""Symbols from training-window volatility (shared by HQMMModel and the experiment drivers).

Why not the harness's trailing-volatility standardisation: dividing each return by the volatility of the
previous 21 days removes the volatility jump that marks the start of a stress period (a crash day looks
ordinary once the volatility estimate has already risen). Dividing by the asset's volatility over the
TRAINING window instead keeps it. The scale and the thresholds use training days only, so nothing from
the future leaks in. See docs/engine_notes.md (finding 13) and docs/sarah_findings.md.
"""
from __future__ import annotations

import numpy as np

from data.discretize import DEFAULT_SHARES


def shares_for(K: int) -> list[float]:
    return list(DEFAULT_SHARES.get(K, [1 / K] * K))


def fit_train_vol(returns: np.ndarray, train_idx: np.ndarray, window_idx: np.ndarray, K: int = 3):
    """Per-asset scale from train_idx and pooled quantile thresholds from window_idx."""
    scale = returns[train_idx].std(axis=0, keepdims=True)
    z = returns[window_idx] / scale
    thresholds = np.quantile(z.ravel(), np.cumsum(shares_for(K))[:-1])
    return scale, thresholds


def apply_train_vol(returns: np.ndarray, scale: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.digitize(returns / scale, thresholds).astype(int)


def train_vol_transform(data, K: int = 3):
    """symbol_transform(symbols, fold, seed) for harness.runner.run: gives every model the same symbols."""

    def f(symbols, fold, seed):
        scale, thr = fit_train_vol(data.returns, fold.train_idx, fold.window_idx, K)
        return apply_train_vol(data.returns, scale, thr)

    return f
