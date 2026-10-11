"""Sa11: spectral observable-operator model (OOM) baseline for the quantum models.

Owner: Sarah.

Why it exists: an HQMM is a quantum-flavoured OOM. If a plain spectral OOM (no quantum
structure, learned in closed form from moments, Hsu-Kakade-Zhang 2012) does as well as the
HQMM, the quantum channel adds nothing (hypothesis H1). It is the "unconstrained" classical
sequence model that is closest to the HQMM in form.

Observation per day = how many assets had a big move (symbol != the middle one), 0..n.
This count is the same whatever order the assets are listed in, so the model is permutation
invariant by construction, like the equivariant HQMM. Same big-move target as the HQMM.

Rank 2 is the default: it matches the two-regime synthetic market and was stable; rank 4
overfitted the moments and gave unstable filters (synthetic AUPRC 0.96 / 0.90 / 0.46 for
rank 2 / 3 / 4). Choose the rank on validation data for real data.

Score = predicted share of big moves tomorrow, from the causal OOM filter.
"""
from __future__ import annotations

import numpy as np

from models.base import Fold, MarketData, RegimeModel


def extreme_count(symbols: np.ndarray, K: int) -> np.ndarray:
    """(T, n) symbols -> (T,) number of assets with a big move (not the middle symbol)."""
    mid = K // 2
    return (np.asarray(symbols) != mid).sum(axis=1).astype(int)


class SpectralOOM(RegimeModel):
    def __init__(self, rank: int = 2, K: int = 3, name: str = "OOM-spectral", ridge: float = 1e-6):
        self.rank, self.K, self.name, self.ridge = rank, K, name, ridge
        self.B = None

    # ------------------------------------------------------------------ learning
    def fit(self, data: MarketData | None, symbols: np.ndarray, fold: Fold, seed: int = 0) -> "SpectralOOM":
        x = extreme_count(symbols, self.K)[np.concatenate([fold.train_idx, fold.val_idx])]
        self.n_sym = int(extreme_count(symbols, self.K).max()) + 1
        S = self.n_sym
        m = min(self.rank, S)
        T = len(x)
        oh = np.eye(S)[x]                                   # (T, S) one-hot
        P1 = oh.mean(axis=0)                                # P(x_t)
        P21 = oh[1:].T @ oh[:-1] / (T - 1)                  # P(x_{t+1}, x_t)
        P3x1 = [oh[2:][x[1:-1] == s].T @ oh[:-2][x[1:-1] == s] / (T - 2) for s in range(S)]
        U, _, _ = np.linalg.svd(P21, full_matrices=False)
        U = U[:, :m]
        pinv = np.linalg.pinv(U.T @ P21 + self.ridge * np.eye(m, S))
        self.b1 = U.T @ P1
        self.binf = np.linalg.pinv(P21.T @ U) @ P1
        self.B = [(U.T @ P3x1[s] @ pinv) for s in range(S)]
        self.share = np.arange(S) / max(S - 1, 1)           # share of assets with a big move
        self.m = m
        return self

    # ------------------------------------------------------------------ causal filter
    def _filter(self, x: np.ndarray):
        """Returns predicted share of big moves for the NEXT day, known at the end of each day."""
        b = self.b1.copy()
        pred = np.zeros(len(x))
        logp = np.zeros(len(x))
        for t, s in enumerate(x):
            probs = np.array([max(self.binf @ self.B[o] @ b, 0.0) for o in range(self.n_sym)])
            tot = probs.sum()
            logp[t] = np.log(max(probs[s] / tot, 1e-12)) if tot > 0 else np.log(1e-12)
            num = self.binf @ self.B[s] @ b
            b = self.B[s] @ b
            den = self.binf @ b
            b = b / den if abs(den) > 1e-12 else self.b1.copy()
            nxt = np.array([max(self.binf @ self.B[o] @ b, 0.0) for o in range(self.n_sym)])
            pred[t] = (nxt @ self.share) / nxt.sum() if nxt.sum() > 0 else self.share @ np.ones(self.n_sym) / self.n_sym
        return pred, logp

    def score(self, data, symbols, idx):
        idx = np.asarray(idx)
        return self._filter(extreme_count(symbols[: idx.max() + 1], self.K))[0][idx]

    def surprise(self, data, symbols, idx):
        idx = np.asarray(idx)
        return -self._filter(extreme_count(symbols[: idx.max() + 1], self.K))[1][idx]

    def n_params(self) -> int:
        return self.m * self.m * self.n_sym + 2 * self.m
