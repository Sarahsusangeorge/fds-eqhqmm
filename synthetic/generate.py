"""Synthetic regime-switching data with known switch points (step Sa1).

Owner: Sarah. Used to check that the HQMM engine can find regime switches
before any real market data is involved.

The process is exchangeable across assets by construction (every asset
follows the same rules), so it is also a fair test for the equivariant model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

CALM, STRESS = 0, 1


@dataclass
class SyntheticData:
    symbols: np.ndarray    # (T, n) integers in 0..K-1 (0 = most negative)
    returns: np.ndarray    # (T, n) continuous returns before discretisation
    regimes: np.ndarray    # (T,) 0 = calm, 1 = stress (the hidden truth)
    switches: np.ndarray   # day indices where the regime changes
    thresholds: np.ndarray # (K-1,) cut points used to make symbols


def make_synthetic(
    T: int = 2000,
    n: int = 3,
    K: int = 3,
    p_stay_calm: float = 0.98,
    p_stay_stress: float = 0.95,
    vol_calm: float = 1.0,
    vol_stress: float = 2.5,
    corr_calm: float = 0.2,
    corr_stress: float = 0.6,
    drift_stress: float = -0.3,
    seed: int = 0,
) -> SyntheticData:
    """Generate T days of n asset symbols driven by a hidden two-regime Markov chain.

    Calm: low volatility, weak co-movement, no drift.
    Stress: high volatility, strong co-movement (a shared shock), negative drift.
    Symbol thresholds are equal-probability cut points of a standard normal,
    so in the calm regime each symbol appears about 1/K of the time and in the
    stress regime the extreme symbols appear much more often.
    """
    if not (0 < p_stay_calm < 1 and 0 < p_stay_stress < 1):
        raise ValueError("stay probabilities must be strictly between 0 and 1")
    if K < 2:
        raise ValueError("K must be at least 2")
    rng = np.random.default_rng(seed)

    # 1. Hidden regime path (starts calm).
    regimes = np.empty(T, dtype=int)
    regimes[0] = CALM
    for t in range(1, T):
        stay = p_stay_calm if regimes[t - 1] == CALM else p_stay_stress
        regimes[t] = regimes[t - 1] if rng.random() < stay else 1 - regimes[t - 1]

    # 2. Returns: one shared market shock plus an asset-specific shock.
    vol = np.where(regimes == STRESS, vol_stress, vol_calm)
    corr = np.where(regimes == STRESS, corr_stress, corr_calm)
    drift = np.where(regimes == STRESS, drift_stress, 0.0)
    common = rng.standard_normal(T)
    own = rng.standard_normal((T, n))
    returns = drift[:, None] + vol[:, None] * (
        np.sqrt(corr)[:, None] * common[:, None] + np.sqrt(1 - corr)[:, None] * own
    )

    # 3. Discretise with equal-probability standard-normal cut points.
    thresholds = norm.ppf(np.arange(1, K) / K)
    symbols = np.searchsorted(thresholds, returns).astype(int)

    switches = np.flatnonzero(np.diff(regimes)) + 1
    return SyntheticData(symbols, returns, regimes, switches, thresholds)
