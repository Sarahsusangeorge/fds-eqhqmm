"""S2: MarketData from prices. Every quantity at day t uses only information available before day t
(volatility and correlation windows end at t-1); the return at t itself is observed at close of t."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.market_data import MarketData


def build_market_data_from_returns(returns: pd.DataFrame, vol_window: int = 21, corr_window: int = 63) -> MarketData:
    r = returns.to_numpy(float)
    T, n = r.shape
    vol = pd.DataFrame(r).rolling(vol_window).std().shift(1).to_numpy()  # window ends at t-1
    std = r / vol
    corr = np.full((T, n, n), np.nan)
    with np.errstate(all="ignore"):
        for t in range(corr_window, T):
            c = np.corrcoef(r[t - corr_window : t].T)  # days t-63 .. t-1
            c = np.where(np.isfinite(c), c, 0.0)
            np.fill_diagonal(c, 1.0)
            corr[t] = c
    warm = max(vol_window, corr_window)  # first row where everything is defined
    return MarketData(
        dates=pd.DatetimeIndex(returns.index[warm:]),
        assets=list(returns.columns),
        returns=r[warm:],
        std_returns=std[warm:],
        corr=corr[warm:],
    )


def build_market_data(prices: pd.DataFrame, vol_window: int = 21, corr_window: int = 63) -> MarketData:
    """Log returns; trailing 21-day vol (shifted one day); standardised returns; trailing 63-day corr."""
    returns = np.log(prices).diff().dropna()
    return build_market_data_from_returns(returns, vol_window, corr_window)
