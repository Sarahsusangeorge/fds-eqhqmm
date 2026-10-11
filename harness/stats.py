"""S13: paired tests, Holm correction, stationary bootstrap, difference-in-differences, Diebold-Mariano."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def paired_wilcoxon(a, b):
    """Wilcoxon signed-rank test on paired per-fold metrics. Returns (statistic, p-value)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if d.size == 0 or np.all(d == 0):
        return float("nan"), 1.0
    r = sps.wilcoxon(d)
    return float(r.statistic), float(r.pvalue)


def holm(pvals, alpha=0.05):
    """Holm step-down correction. Returns (reject, adjusted p-values)."""
    from statsmodels.stats.multitest import multipletests

    rej, adj, _, _ = multipletests(np.asarray(pvals, float), alpha=alpha, method="holm")
    return rej, adj


def _block(x):
    from arch.bootstrap import optimal_block_length

    if len(x) < 10:
        return 2.0
    return float(max(2.0, optimal_block_length(x).iloc[0]["stationary"]))


def stationary_bootstrap_ci(x, func=np.mean, reps=2000, level=0.95, block=None, seed=0):
    """Percentile CI for func(x) under the stationary bootstrap (Politis & Romano 1994)."""
    from arch.bootstrap import StationaryBootstrap

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = StationaryBootstrap(block or _block(x), x, seed=seed)
    ci = bs.conf_int(lambda z: func(z), reps=reps, method="percentile", size=level)
    return float(ci[0, 0]), float(ci[1, 0])


def did_bootstrap(a_gen, a_eq, b_gen, b_eq, reps=2000, level=0.95, seed=0):
    """Interaction (H3): (quantum_eq - quantum_gen) - (classical_eq - classical_gen), on paired per-fold metrics.
    Returns (estimate, ci_low, ci_high) using the stationary bootstrap over folds."""
    from arch.bootstrap import StationaryBootstrap

    arr = np.column_stack([a_gen, a_eq, b_gen, b_eq]).astype(float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    f = lambda z: np.mean((z[:, 1] - z[:, 0]) - (z[:, 3] - z[:, 2]))
    bs = StationaryBootstrap(_block(arr[:, 0]), arr, seed=seed)
    ci = bs.conf_int(f, reps=reps, method="percentile", size=level)
    return float(f(arr)), float(ci[0, 0]), float(ci[1, 0])


def diebold_mariano(loss1, loss2, lags=None, hln=True):
    """DM test of equal predictive accuracy on a loss differential d = loss1 - loss2 (e.g. daily log-loss).
    HAC (Bartlett/Newey-West) long-run variance; Harvey-Leybourne-Newbold small-sample correction (h=1) with a
    t(T-1) reference. Returns (statistic, two-sided p). A negative statistic means model 1 has lower loss."""
    d = np.asarray(loss1, float) - np.asarray(loss2, float)
    d = d[np.isfinite(d)]
    T = len(d)
    L = int(lags) if lags is not None else int(np.floor(T ** (1 / 3)))
    dc = d - d.mean()
    lrv = dc @ dc / T
    for k in range(1, L + 1):
        lrv += 2 * (1 - k / (L + 1)) * (dc[k:] @ dc[:-k]) / T
    if lrv <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(lrv / T)
    if hln:
        stat *= np.sqrt((T - 1) / T)
        return float(stat), float(2 * sps.t.sf(abs(stat), T - 1))
    return float(stat), float(2 * sps.norm.sf(abs(stat)))
