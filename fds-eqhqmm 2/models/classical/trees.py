"""S12: supervised tree baselines (random forest, XGBoost) on rolling features of the equal-weight basket.
Training labels come from `label_fn(window_data) -> 0/1 series`, applied ONLY to the training window, so no hindsight
beyond the end of training leaks in. Score = predicted stress probability at day t (features use data up to t)."""

import numpy as np
import pandas as pd

from models.base import RegimeModel


def make_features(data) -> np.ndarray:
    ew = pd.Series(data.equal_weight_returns())
    px = ew.cumsum()
    f = pd.DataFrame(
        {
            "vol21": ew.rolling(21, min_periods=5).std(),
            "vol63": ew.rolling(63, min_periods=5).std(),
            "ret5": ew.rolling(5, min_periods=1).sum(),
            "ret21": ew.rolling(21, min_periods=5).sum(),
            "dd252": px - px.rolling(252, min_periods=1).max(),
        }
    )
    if data.corr is not None:
        n = data.n_assets
        off = (data.corr.sum(axis=(1, 2)) - n) / (n * (n - 1))
        f["mean_corr"] = off
    return f.fillna(0.0).to_numpy()


class TreeModel(RegimeModel):
    def __init__(self, kind="rf", label_fn=None):
        assert kind in ("rf", "xgb")
        self.kind, self.label_fn = kind, label_fn
        self.name = "RF" if kind == "rf" else "XGB"

    def fit(self, data, symbols, fold, seed):
        lo, hi = int(fold.train_idx[0]), int(fold.train_idx[-1]) + 1
        y = np.asarray(self.label_fn(data.slice(lo, hi)))
        X = make_features(data)[lo:hi]
        self.const = None
        if y.min() == y.max():
            self.const = float(y[0])
            self.clf = None
            return self
        if self.kind == "rf":
            from sklearn.ensemble import RandomForestClassifier

            self.clf = RandomForestClassifier(
                200, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=1
            )
        else:
            from xgboost import XGBClassifier

            self.clf = XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=seed,
                scale_pos_weight=float((y == 0).sum() / max(1, (y == 1).sum())),
                n_jobs=1,
                verbosity=0,
            )
        self.clf.fit(X, y)
        return self

    def score(self, data, symbols, idx):
        if self.clf is None:
            return np.full(len(idx), self.const)
        X = make_features(data)[idx]
        if self.kind == "xgb":  # raw margin = log-odds; the probability saturates at exactly 1.0 in float32
            return self.clf.predict(X, output_margin=True)
        return self.clf.predict_proba(X)[:, 1]

    def n_params(self):
        if self.clf is None:
            return 0
        if self.kind == "rf":
            return int(sum(e.tree_.node_count for e in self.clf.estimators_))
        return int(len(self.clf.get_booster().trees_to_dataframe()))
