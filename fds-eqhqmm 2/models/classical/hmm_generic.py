"""S10: C-G. Generic Gaussian HMM, full covariance, on standardised returns (hmmlearn)."""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal

from models.base import RegimeModel
from models.classical._hmm_utils import forward_filter_log, stationary, stress_log_odds


class GenericGaussianHMM(RegimeModel):
    def __init__(self, n_states=2, n_iter=100, name="C-G", input="std"):
        """input: 'std' = standardised returns (the spec); 'raw' = raw returns in percent (diagnostic variant)."""
        assert input in ("std", "raw")
        self.n_states, self.n_iter, self.name, self.input = n_states, n_iter, name, input

    def _x(self, data):
        return data.std_returns if self.input == "std" else data.returns * 100.0

    def fit(self, data, symbols, fold, seed):
        from hmmlearn.hmm import GaussianHMM

        X = self._x(data)[fold.train_idx]
        best = None
        for i in range(3):
            h = GaussianHMM(self.n_states, covariance_type="full", n_iter=self.n_iter, random_state=seed * 100 + i)
            try:
                h.fit(X)
                ll = h.score(X)
            except Exception:
                continue
            if best is None or ll > best[0]:
                best = (ll, h)
        h = best[1]
        self.A = h.transmat_
        self.means, self.covars = h.means_, h.covars_
        self.pi = stationary(self.A)
        self.stress = int(np.argmax([np.trace(c) for c in self.covars]))
        return self

    def _filter(self, data, upto):
        X = self._x(data)[: upto + 1]
        lb = np.column_stack(
            [
                multivariate_normal.logpdf(X, self.means[k], self.covars[k], allow_singular=True)
                for k in range(self.n_states)
            ]
        )
        return forward_filter_log(lb, self.pi, self.A)

    def score(self, data, symbols, idx):
        """Stress log-odds log P(stress)/P(calm) of the filtered state (monotone in the probability, no saturation)."""
        la, _ = self._filter(data, int(np.max(idx)))
        return stress_log_odds(la, self.stress)[idx]

    def surprise(self, data, symbols, idx):
        _, logc = self._filter(data, int(np.max(idx)))
        return -logc[idx]

    def n_params(self):
        k, n = self.n_states, self.means.shape[1]
        return k * n + k * n * (n + 1) // 2 + k * (k - 1)
