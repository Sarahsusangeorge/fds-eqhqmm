"""S11: C-E. Own small HMM (EM) whose emission is invariant to asset order.
gaussian:    per regime one mean, one variance, one common correlation (compound-symmetric covariance);
             the likelihood depends on a day's returns only through sum(x) and sum(x^2).
categorical: per regime one distribution over the K symbols, shared by all assets (depends on symbol counts)."""

from __future__ import annotations

import numpy as np

from models.base import RegimeModel
from models.classical._hmm_utils import forward_backward, forward_filter_log, stationary, stress_log_odds


class ExchangeableHMM(RegimeModel):
    def __init__(self, n_states=2, emission="gaussian", K=3, n_iter=100, n_init=3, name=None, input="std"):
        """input (gaussian emission only): 'std' = standardised returns (the spec); 'raw' = raw returns in percent."""
        assert emission in ("gaussian", "categorical") and input in ("std", "raw")
        self.input = input
        self.k, self.emission, self.K, self.n_iter, self.n_init = n_states, emission, K, n_iter, n_init
        self.name = name or ("C-E" if emission == "gaussian" else "C-E-cat")

    # ---- sufficient statistics (permutation invariant) ----
    def _stats(self, data, symbols, rows):
        if self.emission == "gaussian":
            x = (data.std_returns if self.input == "std" else data.returns * 100.0)[rows]
            return np.column_stack([x.sum(1), (x**2).sum(1)]), x.shape[1]
        sy = symbols[rows]
        return np.stack([(sy == j).sum(1) for j in range(self.K)], axis=1).astype(float), sy.shape[1]

    def _logb(self, S, n):
        k = self.k
        if self.emission == "categorical":
            return S @ np.log(self.P).T
        s1, s2 = S[:, 0:1], S[:, 1:2]
        mu, var, rho = self.mu[None], self.var[None], self.rho[None]
        a, b = var * (1 - rho), var * rho
        sumc = s1 - n * mu
        sqc = s2 - 2 * mu * s1 + n * mu**2
        Q = sqc / a - b * sumc**2 / (a * (a + n * b))
        return -0.5 * (n * np.log(2 * np.pi) + (n - 1) * np.log(a) + np.log(a + n * b) + Q)

    def _mstep(self, S, n, gamma):
        w = gamma.sum(0) + 1e-9
        if self.emission == "categorical":
            P = (gamma.T @ S) / (n * w[:, None])
            self.P = (P + 1e-4) / (P + 1e-4).sum(1, keepdims=True)
            return
        s1, s2 = S[:, 0], S[:, 1]
        self.mu = (gamma.T @ s1) / (n * w)
        sqc = s2[:, None] - 2 * self.mu[None] * s1[:, None] + n * self.mu[None] ** 2
        sumc = s1[:, None] - n * self.mu[None]
        self.var = np.maximum((gamma * sqc).sum(0) / (n * w), 1e-6)
        cross = (sumc**2 - sqc) / (n * (n - 1))
        rho = (gamma * cross).sum(0) / w / self.var
        self.rho = np.clip(rho, -1 / (n - 1) + 1e-3, 0.999)

    def _magnitude(self, S, n):
        return S[:, 1] / n if self.emission == "gaussian" else (S[:, 0] + S[:, -1]) / n  # extreme-ness per day

    def _fit_once(self, S, n, seed):
        rng = np.random.default_rng(seed)
        T, k = len(S), self.k
        order = np.argsort(np.argsort(self._magnitude(S, n) + 1e-3 * rng.standard_normal(T)))
        g = np.full((T, k), 0.1 / max(k - 1, 1))
        g[np.arange(T), np.minimum(order * k // T, k - 1)] = 0.9
        g = g * rng.uniform(0.9, 1.1, g.shape)
        g /= g.sum(1, keepdims=True)
        A = np.full((k, k), 0.05 / max(k - 1, 1)) + np.eye(k) * (0.95 - 0.05 / max(k - 1, 1))
        self._mstep(S, n, g)
        prev = -np.inf
        for _ in range(self.n_iter):
            pi = stationary(A)
            g, xi, ll = forward_backward(self._logb(S, n), pi, A)
            A = (xi + 1e-6) / (xi + 1e-6).sum(1, keepdims=True)
            self._mstep(S, n, g)
            if abs(ll - prev) < 1e-5 * abs(ll):
                break
            prev = ll
        return ll, A

    def fit(self, data, symbols, fold, seed):
        S, n = self._stats(data, symbols, fold.train_idx)
        best = None
        for i in range(self.n_init):
            ll, A = self._fit_once(S, n, seed * 100 + i)
            params = (
                self.__dict__.get("mu"),
                self.__dict__.get("var"),
                self.__dict__.get("rho"),
                self.__dict__.get("P"),
            )
            if best is None or ll > best[0]:
                best = (ll, A, tuple(None if p is None else p.copy() for p in params))
        _, self.A, (self.mu, self.var, self.rho, self.P) = (best[0], best[1], best[2])
        self.pi = stationary(self.A)
        if self.emission == "gaussian":
            self.stress = int(np.argmax(self.var))
        else:
            self.stress = int(np.argmax(self.P[:, 0] + self.P[:, -1]))
        return self

    def _filter(self, data, symbols, upto):
        S, n = self._stats(data, symbols, np.arange(upto + 1))
        return forward_filter_log(self._logb(S, n), self.pi, self.A)

    def score(self, data, symbols, idx):
        """Stress log-odds of the filtered state (monotone in the probability, no saturation at 1.0)."""
        la, _ = self._filter(data, symbols, int(np.max(idx)))
        return stress_log_odds(la, self.stress)[idx]

    def surprise(self, data, symbols, idx):
        _, logc = self._filter(data, symbols, int(np.max(idx)))
        return -logc[idx]

    def n_params(self):
        k = self.k
        return k * (k - 1) + (k * 3 if self.emission == "gaussian" else k * (self.K - 1))
