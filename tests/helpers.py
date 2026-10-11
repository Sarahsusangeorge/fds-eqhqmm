"""Synthetic market used by the tests (offline). Calm/stress regimes with a shared shock in stress."""

import numpy as np
import pandas as pd

from data.features import build_market_data


def fake_prices(T=1700, n=5, seed=0, start="1999-01-04"):
    rng = np.random.default_rng(seed)
    state = np.zeros(T, int)
    for t in range(1, T):
        stay = 0.985 if state[t - 1] == 0 else 0.95
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    vol = np.where(state == 1, 0.02, 0.008)
    common = rng.standard_normal(T) * np.where(state == 1, 0.015, 0.002)
    r = rng.standard_normal((T, n)) * vol[:, None] + common[:, None]
    dates = pd.bdate_range(start, periods=T + 1)
    prices = pd.DataFrame(
        100 * np.exp(np.vstack([np.zeros(n), np.cumsum(r, 0)])), index=dates, columns=[f"A{i}" for i in range(n)]
    )
    return prices, state


def fake_market(T=1700, n=5, seed=0):
    prices, state = fake_prices(T, n, seed)
    return build_market_data(prices), state
