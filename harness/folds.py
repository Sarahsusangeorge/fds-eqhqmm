"""S4: walk-forward folds (contract 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray  # first 80% of the training window (models are fitted here)
    val_idx: np.ndarray  # last 20% of the training window (early stopping, threshold calibration)
    test_idx: np.ndarray  # the next 6 months

    @property
    def window_idx(self) -> np.ndarray:
        """The whole training window (train + val), e.g. for fitting the Discretizer."""
        return np.concatenate([self.train_idx, self.val_idx])


def make_folds(
    dates,
    scheme="expanding",
    first_train_end="2004-12-31",
    step_months=6,
    rolling_years=5,
    val_frac=0.2,
    min_test_days=21,
):
    dates = pd.DatetimeIndex(dates)
    assert scheme in ("expanding", "rolling")
    test_start = pd.Timestamp(first_train_end) + pd.Timedelta(days=1)
    folds, k = [], 0
    while test_start <= dates[-1]:
        test_end = test_start + pd.DateOffset(months=step_months)  # exclusive
        test_idx = np.flatnonzero((dates >= test_start) & (dates < test_end))
        if len(test_idx) >= min_test_days:
            if scheme == "expanding":
                win = np.flatnonzero(dates < test_start)
            else:
                lo = test_start - pd.DateOffset(years=rolling_years)
                win = np.flatnonzero((dates >= lo) & (dates < test_start))
            n_val = max(1, int(round(val_frac * len(win))))
            folds.append(Fold(k, win[:-n_val], win[-n_val:], test_idx))
            k += 1
        test_start = test_end
    return folds
