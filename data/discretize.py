"""S3: Discretizer. Thresholds are fitted on the POOLED standardised returns of all assets in the
training window, so one set of thresholds is shared by every asset (keeps symbols exchangeable)."""

from __future__ import annotations

import numpy as np

DEFAULT_SHARES = {3: [0.2, 0.6, 0.2], 5: [0.1, 0.2, 0.4, 0.2, 0.1]}


class Discretizer:
    def __init__(self, K: int = 3, method: str = "quantile", shares=None, seed: int = 0):
        assert method in ("quantile", "gmm")
        self.K, self.method, self.seed = K, method, seed
        self.shares = list(shares) if shares is not None else DEFAULT_SHARES.get(K, [1 / K] * K)
        assert len(self.shares) == K and abs(sum(self.shares) - 1) < 1e-9

    def fit(self, x: np.ndarray) -> "Discretizer":
        v = np.asarray(x, float).ravel()
        v = v[np.isfinite(v)]
        if self.method == "quantile":
            self.thresholds_ = np.quantile(v, np.cumsum(self.shares)[:-1])
        else:
            from sklearn.mixture import GaussianMixture

            gm = GaussianMixture(self.K, random_state=self.seed, n_init=3).fit(v.reshape(-1, 1))
            self._gm = gm
            self._order = np.argsort(gm.means_.ravel())  # component -> symbol by ascending mean
            self._rank = np.empty(self.K, int)
            self._rank[self._order] = np.arange(self.K)
        self.train_shares_ = np.bincount(self.transform(x).ravel(), minlength=self.K) / v.size
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        if self.method == "quantile":
            return np.digitize(x, self.thresholds_).astype(int)
        lab = self._gm.predict(x.reshape(-1, 1))
        return self._rank[lab].reshape(x.shape).astype(int)
