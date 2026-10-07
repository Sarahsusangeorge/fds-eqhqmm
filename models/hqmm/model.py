"""HQMMModel: the engine wrapped as a RegimeModel (step Sa9).

Owner: Sarah.

Works with any UnitaryBuilder: GenericBuilder gives Q-G, Abhinav's
EquivariantBuilder gives Q-E. Uses the discretised symbols only.

Defaults reflect what worked in the synthetic tests (docs/engine_notes.md):
reset-mode measurement, 2 register qubits (4 regime states), L-BFGS training,
several random restarts kept by validation likelihood, and the stress score =
the model's predicted share of extreme moves for the next day.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from models.base import Fold, MarketData, RegimeModel
from models.hqmm.engine import LBFGSConfig, run_filter, train_lbfgs
from models.hqmm.measurement import LocalMeasurement


class HQMMModel(RegimeModel):
    """Hidden quantum Markov model regime detector.

    builder_factory(n_assets) -> a fresh UnitaryBuilder (also an nn.Module).
    shared_measurement=True for the equivariant model, False for the generic one.
    """

    def __init__(self, builder_factory: Callable, K: int = 3, n_register: int = 2,
                 shared_measurement: bool = True, reset: bool = True,
                 config: LBFGSConfig = LBFGSConfig(), n_restarts: int = 3, name: str = "hqmm"):
        self.builder_factory = builder_factory
        self.K, self.n_register = K, n_register
        self.shared_measurement, self.reset = shared_measurement, reset
        self.config, self.n_restarts, self.name = config, n_restarts, name
        self.builder = self.measurement = None
        self.val_loglik = None
        self.restart_vals: list[float] = []

    # ------------------------------------------------------------------ RegimeModel API
    def fit(self, data: MarketData | None, symbols: np.ndarray, fold: Fold, seed: int) -> "HQMMModel":
        Y = np.asarray(symbols)
        n = Y.shape[1]
        Y_train, Y_val = Y[fold.train_idx], Y[fold.val_idx]
        best, self.restart_vals = None, []
        for r in range(self.n_restarts):
            s = seed * 1000 + r
            torch.manual_seed(s)
            builder = self.builder_factory(n)
            meas = LocalMeasurement(self.K, n_assets=n, shared=self.shared_measurement, reset=self.reset)
            res = train_lbfgs(builder, meas, Y_train, Y_val, self.n_register, self.config, seed=s)
            self.restart_vals.append(res.val_loglik)
            if best is None or res.val_loglik > best[2]:
                best = (builder, meas, res.val_loglik)
        self.builder, self.measurement, self.val_loglik = best
        return self

    def score(self, data: MarketData | None, symbols: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Predicted share of extreme moves for the next day, known at the end of each day in idx."""
        idx = np.asarray(idx)
        return self._filter(np.asarray(symbols)[: idx.max() + 1]).pred_extreme[0].numpy()[idx]

    def n_params(self) -> int:
        if self.builder is None:
            raise RuntimeError("fit the model first")
        return self.builder.n_params() + self.measurement.n_params()

    # ------------------------------------------------------------------ extras
    def posterior(self, symbols: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Register-state probabilities (len(idx), 2^r) using only data up to each day."""
        idx = np.asarray(idx)
        return self._filter(np.asarray(symbols)[: idx.max() + 1]).posterior[0].numpy()[idx]

    def surprise(self, symbols: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """-log p(y_t | past) on each day in idx (secondary alarm score)."""
        idx = np.asarray(idx)
        return (-self._filter(np.asarray(symbols)[: idx.max() + 1]).log_p[0]).numpy()[idx]

    def _filter(self, Y):
        with torch.no_grad():
            return run_filter(self.builder, self.measurement, torch.as_tensor(Y), self.n_register, predict=True)
