"""Reimplementation of Hammond (2026) geometric observables for financial regime detection.

Reference:
  Hammond, Will (2026). "Geometric Observables for Financial Regime Detection."
  arXiv: 2605.17117 [q-fin.ST].

Owner: Abhinav. Build step: A10.

Two geometric observables computed on rolling correlation matrices:
  1. Spectral Entropy:
     Normalized eigenvalues p_i = lambda_i / n of rolling correlation C_t.
     S_t = - (1 / ln n) sum_i p_i ln(p_i).
     Stress score: s_t^{entropy} = 1 - S_t (high in collective crashes, low in calm).
  2. Berry Phase Rate / Geodesic Subspace Curvature:
     Rotation rate of the leading eigenvector u_1(t) of C_t on the projective manifold:
     theta_t = arccos(|u_1(t)^T u_1(t-1)|).
     Stress score: smoothed rotation rate detecting abrupt market geometry shifts.

Deviations from original arXiv paper:
  - We use causal trailing 63-day correlation windows (shifted by 1 day) to strictly
    prevent lookahead leakage.
  - S_t is normalized by ln(n) to bound entropy in [0, 1] across varying basket sizes n.
  - Sign ambiguity in eigenvectors u_1(t) is resolved by taking the absolute inner product
    |<u_1(t), u_1(t-1)>|, representing the Fubini-Study / geodesic distance between 1D subspaces.
"""

from __future__ import annotations

import numpy as np

from models.base import Fold, MarketData, RegimeModel


def compute_spectral_entropy(corr_matrices: np.ndarray) -> np.ndarray:
    """Compute rolling spectral entropy for (T, n, n) correlation matrices.

    Returns:
      entropy: (T,) array in [0, 1].
    """
    T, n, _ = corr_matrices.shape
    if n <= 1:
        return np.ones(T, dtype=np.float64)

    max_entropy = np.log(n)
    entropies = np.zeros(T, dtype=np.float64)

    for t in range(T):
        C = corr_matrices[t]
        # Eigenvalues of symmetric correlation matrix
        eigvals = np.linalg.eigvalsh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        p = eigvals / eigvals.sum()
        ent = -np.sum(p * np.log(p)) / max_entropy
        entropies[t] = np.clip(ent, 0.0, 1.0)

    return entropies


def compute_berry_phase_rate(
    corr_matrices: np.ndarray, smooth_window: int = 10
) -> np.ndarray:
    """Compute discrete Berry phase / subspace geodesic rotation rate of top eigenvector.

    Returns:
      rate: (T,) array of smoothed angular velocity of the leading market mode.
    """
    T, n, _ = corr_matrices.shape
    if n <= 1 or T <= 1:
        return np.zeros(T, dtype=np.float64)

    leading_evecs = np.zeros((T, n), dtype=np.float64)
    for t in range(T):
        w, v = np.linalg.eigh(corr_matrices[t])
        leading_evecs[t] = v[:, -1]  # eigenvector with largest eigenvalue

    # Step-by-step geodesic angle: theta_t = arccos(|<v_t, v_{t-1}>|)
    angles = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        overlap = np.abs(np.dot(leading_evecs[t], leading_evecs[t - 1]))
        overlap = np.clip(overlap, 0.0, 1.0)
        angles[t] = np.arccos(overlap)

    # Rolling smoothed rate
    smoothed = np.zeros(T, dtype=np.float64)
    for t in range(T):
        start = max(0, t - smooth_window + 1)
        smoothed[t] = np.mean(angles[start : t + 1])
    return smoothed


class HammondModel(RegimeModel):
    """Geometric observable regime detector (Hammond 2026)."""

    def __init__(
        self,
        observable: str = "entropy",
        corr_window: int = 63,
        berry_window: int = 10,
        name: str = "hammond",
    ):
        if observable not in ("entropy", "berry", "combined"):
            raise ValueError("observable must be 'entropy', 'berry', or 'combined'")
        self.observable = observable
        self.corr_window = corr_window
        self.berry_window = berry_window
        self.name = f"{name}_{observable}"
        self.threshold: float = 0.5

    def fit(
        self, data: MarketData | None, symbols: np.ndarray, fold: Fold, seed: int
    ) -> "HammondModel":
        # Unsupervised geometric model: calculates thresholds or calibration on training fold
        scores_train = self.score(data, symbols, fold.train_idx)
        self.threshold = float(np.median(scores_train))
        return self

    def _get_corr_matrices(
        self, data: MarketData | None, symbols: np.ndarray, T_needed: int
    ) -> np.ndarray:
        if data is not None and data.corr is not None and len(data.corr) >= T_needed:
            return data.corr[:T_needed]

        # Calculate causal rolling correlation from returns or symbols
        if data is not None and data.returns is not None:
            series = data.returns[:T_needed]
        else:
            series = symbols[:T_needed].astype(np.float64)

        T, n = series.shape
        corrs = np.zeros((T, n, n), dtype=np.float64)
        for t in range(T):
            if t < 2:
                corrs[t] = np.eye(n)
            else:
                start = max(0, t - self.corr_window)
                sub = series[start:t]  # strictly up to t-1
                if len(sub) < 2:
                    corrs[t] = np.eye(n)
                else:
                    std = sub.std(axis=0)
                    if np.any(std < 1e-12):
                        corrs[t] = np.eye(n)
                    else:
                        c = np.corrcoef(sub, rowvar=False)
                        corrs[t] = np.nan_to_num(c, nan=0.0) + np.eye(n) * 1e-6
        return corrs

    def score(
        self, data: MarketData | None, symbols: np.ndarray, idx: np.ndarray
    ) -> np.ndarray:
        idx = np.asarray(idx)
        max_t = int(idx.max()) + 1
        corrs = self._get_corr_matrices(data, symbols, max_t)

        if self.observable == "entropy":
            ent = compute_spectral_entropy(corrs)
            scores = (
                1.0 - ent
            )  # High stress = Low entropy (dominant market crisis mode)
        elif self.observable == "berry":
            scores = compute_berry_phase_rate(corrs, self.berry_window)
        else:  # combined
            ent_score = 1.0 - compute_spectral_entropy(corrs)
            berry_score = compute_berry_phase_rate(corrs, self.berry_window)
            b_norm = berry_score / (np.max(berry_score) + 1e-12)
            scores = 0.5 * ent_score + 0.5 * b_norm

        return scores[idx]

    def n_params(self) -> int:
        return 0  # Analytical geometric observable (zero trainable parameters)
