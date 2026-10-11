"""S14: risk-off overlay. Hold the equal-weight basket; move to cash while the previous day's score is at or
above the calibrated threshold (decision at close of t-1, applied to day t). Costs: bps per unit of position change."""

from __future__ import annotations

import numpy as np


def risk_off_overlay(ret, scores, h, cost_bps=10.0, days_per_year=252):
    ret, scores = np.asarray(ret, float), np.asarray(scores, float)
    pos = np.ones(len(ret))
    pos[1:] = (scores[:-1] < h).astype(float)
    trades = np.abs(np.diff(np.r_[1.0, pos]))
    net = pos * ret - trades * cost_bps / 1e4
    return {"net_ret": net, **_summary(net, trades.sum(), len(ret), days_per_year)}


def buy_and_hold(ret, days_per_year=252):
    ret = np.asarray(ret, float)
    return _summary(ret, 0.0, len(ret), days_per_year)


def _summary(net, n_trades, n, dpy):
    sd = net.std(ddof=1) if len(net) > 1 else 0.0
    eq = np.exp(np.cumsum(net))
    dd = float(np.max(1 - eq / np.maximum.accumulate(eq))) if len(eq) else float("nan")
    return {
        "sharpe": float(net.mean() / sd * np.sqrt(dpy)) if sd > 0 else float("nan"),
        "max_drawdown": dd,
        "turnover_per_year": float(n_trades / (n / dpy)),
    }
