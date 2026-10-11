"""S7: threshold calibration and alarm extraction. Identical for every model."""

from __future__ import annotations

import numpy as np


def alarms_from_scores(scores, h, gap=20) -> np.ndarray:
    """Indices where the score first reaches h. Exceedances less than `gap` days apart (chained) are
    one alarm, raised on the first day of the cluster."""
    ex = np.flatnonzero(np.asarray(scores) >= h)
    if ex.size == 0:
        return ex
    new = np.r_[True, np.diff(ex) > gap]
    return ex[new]


def calibrate_threshold(val_scores, target_far=1.0, gap=20, days_per_year=252) -> float:
    """Most sensitive threshold h whose de-clustered alarm rate on the validation scores is <= target_far per year.

    Scans h downward from the maximum score and stops at the first h that exceeds the target rate. (Taking the global
    minimum h that meets the target would be wrong: a very low h keeps the score above it all the time, which collapses
    everything into one cluster and gives an always-on detector with a tiny 'alarm rate'.)"""
    v = np.asarray(val_scores, float)
    v = v[np.isfinite(v)]
    years = len(v) / days_per_year
    best = float(
        np.nextafter(v.max(), np.inf)
    )  # just above the validation maximum (an additive epsilon can overshoot a bounded score)
    for h in np.unique(v)[::-1]:
        if len(alarms_from_scores(v, h, gap)) / years > target_far:
            break
        best = float(h)
    return best
