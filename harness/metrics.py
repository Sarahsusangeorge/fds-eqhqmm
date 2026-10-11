"""S8: detection metrics. A detection window for an onset is [onset - pre, onset + post]."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from labels.common import index_intervals


def detection_lags(alarm_idx, onsets, pre=20, post=40) -> np.ndarray:
    """Lag (days from onset to first alarm inside the window; negative = early) per onset; NaN if missed."""
    alarm_idx = np.asarray(alarm_idx)
    out = np.full(len(onsets), np.nan)
    for i, o in enumerate(onsets):
        hit = alarm_idx[(alarm_idx >= o - pre) & (alarm_idx <= o + post)]
        if hit.size:
            out[i] = hit.min() - o
    return out


def recall(lags) -> float:
    lags = np.asarray(lags)
    return float("nan") if lags.size == 0 else float(np.mean(~np.isnan(lags)))


def false_alarm_count(alarm_idx, stress_series, n_days, pre=20, post=40) -> int:
    """Alarms not inside [start - pre, end + post] of any stress episode (episodes: runs of 1s)."""
    ok = np.zeros(n_days, bool)
    for a, b in index_intervals(stress_series):
        ok[max(0, a - pre) : min(n_days, b + post + 1)] = True
    alarm_idx = np.asarray(alarm_idx, int)
    return int(np.sum(~ok[alarm_idx])) if alarm_idx.size else 0


def false_alarms_per_year(alarm_idx, stress_series, n_days, n_test_days, pre=20, post=40, days_per_year=252) -> float:
    return false_alarm_count(alarm_idx, stress_series, n_days, pre, post) / (n_test_days / days_per_year)


def auprc(stress, score) -> float:
    stress = np.asarray(stress).astype(int)
    if stress.sum() in (0, len(stress)):
        return float("nan")
    return float(average_precision_score(stress, score))


def cohens_d(stress, score) -> float:
    stress = np.asarray(stress).astype(bool)
    s, c = np.asarray(score)[stress], np.asarray(score)[~stress]
    if len(s) < 2 or len(c) < 2:
        return float("nan")
    sp = np.sqrt(((len(s) - 1) * s.var(ddof=1) + (len(c) - 1) * c.var(ddof=1)) / (len(s) + len(c) - 2))
    return float("nan") if sp == 0 else float((s.mean() - c.mean()) / sp)


def heldout_ll(surprise) -> float:
    """Mean log-likelihood per day = -mean(surprise)."""
    return float("nan") if surprise is None else float(-np.nanmean(surprise))
