"""C-G-cat: the matched generic classical cell. A categorical HMM on the same discretised symbols as C-E-cat, but every asset
has its own emission table (no exchangeability): P[state, asset, symbol]. It is the classical counterpart of Q-G
for H1 (and, with C-E-cat, for H3)."""

from __future__ import annotations

import numpy as np

from models.classical.hmm_exchangeable import ExchangeableHMM


class GenericCategoricalHMM(ExchangeableHMM):
    def __init__(self, n_states=2, K=3, n_iter=100, n_init=3, name="C-G-cat"):
        super().__init__(n_states=n_states, emission="categorical", K=K, n_iter=n_iter, n_init=n_init, name=name)

    def _stats(self, data, symbols, rows):
        sy = symbols[rows]
        n = sy.shape[1]
        onehot = np.concatenate([(sy == j) for j in range(self.K)], axis=1).astype(float)  # (T, K*n), symbol-major
        self._n_assets = n
        return onehot, n

    def _logb(self, S, n):
        return S @ np.log(self.P).T  # P: (k, K*n)

    def _mstep(self, S, n, gamma):
        w = gamma.sum(0) + 1e-9
        P = (gamma.T @ S) / w[:, None]  # per (symbol, asset) frequency within each state
        P = P + 1e-4
        # normalise over the K symbols of each asset
        P = P.reshape(self.k, self.K, n)
        P = P / P.sum(1, keepdims=True)
        self.P = P.reshape(self.k, self.K * n)

    def _magnitude(self, S, n):
        return (S[:, :n].sum(1) + S[:, (self.K - 1) * n:].sum(1)) / n

    def fit(self, data, symbols, fold, seed):
        super().fit(data, symbols, fold, seed)
        n = self.P.shape[1] // self.K
        ext = self.P[:, :n].sum(1) + self.P[:, (self.K - 1) * n:].sum(1)
        self.stress = int(np.argmax(ext))
        return self

    def n_params(self):
        n = self.P.shape[1] // self.K if getattr(self, "P", None) is not None else 0
        return self.k * (self.k - 1) + self.k * n * (self.K - 1)
