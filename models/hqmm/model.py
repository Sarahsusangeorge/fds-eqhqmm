"""HQMMModel: the engine wrapped as a RegimeModel (step Sa9).

Owner: Sarah.

Works with any UnitaryBuilder: GenericBuilder gives Q-G, Abhinav's
EquivariantBuilder gives Q-E. Uses the discretised symbols only.

Defaults reflect what worked in the synthetic tests (docs/engine_notes.md):
reset-mode measurement, 2 register qubits (4 regime states), L-BFGS training on
the big-move pattern only (objective="coarse"), several random restarts kept by
validation forecast skill (selection="skill"), and the stress score = the
model's predicted share of extreme moves for the next day.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from models.base import Fold, MarketData, RegimeModel
from models.hqmm.engine import LBFGSConfig, run_filter, train_coarse, train_lbfgs, train_lbfgs_fast
from models.hqmm.measurement import LocalMeasurement


class HQMMModel(RegimeModel):
    """Hidden quantum Markov model regime detector.

    builder_factory(n_assets) -> a fresh UnitaryBuilder (also an nn.Module).
    shared_measurement=True for the equivariant model, False for the generic one.
    """

    def __init__(self, builder_factory: Callable, K: int = 3, n_register: int = 2,
                 shared_measurement: bool = True, reset: bool = True,
                 config: LBFGSConfig = LBFGSConfig(), n_restarts: int = 3, name: str = "hqmm",
                 objective: str = "coarse", selection: str = "skill", symbol_mode: str = "harness"):
        if objective not in ("coarse", "full"):
            raise ValueError("objective must be 'coarse' or 'full'")
        if selection not in ("skill", "loglik"):
            raise ValueError("selection must be 'skill' or 'loglik'")
        if objective == "coarse" and not reset:
            raise ValueError("objective='coarse' needs reset=True")
        if symbol_mode not in ("harness", "train_vol"):
            raise ValueError("symbol_mode must be 'harness' or 'train_vol'")
        self.objective, self.selection, self.symbol_mode = objective, selection, symbol_mode
        self._scale = self._thresholds = None
        self.builder_factory = builder_factory
        self.K, self.n_register = K, n_register
        self.shared_measurement, self.reset = shared_measurement, reset
        self.config, self.n_restarts, self.name = config, n_restarts, name
        self.builder = self.measurement = None
        self.val_criterion = None
        self.restart_vals: list[float] = []      # selection criterion of each restart

    # ------------------------------------------------------------------ RegimeModel API
    def _symbols(self, data, symbols):
        """Symbols the model actually uses.

        'harness': the symbols the harness passes in (built from trailing-volatility
        standardised returns). 'train_vol': built here from raw returns scaled by the
        TRAINING-window volatility; this keeps the volatility jump of a crash visible
        (trailing-volatility scaling cancels it, see docs/engine_notes.md).
        """
        if self.symbol_mode == "harness":
            return np.asarray(symbols)
        z = data.returns / self._scale
        return np.digitize(z, self._thresholds).astype(int)

    def _fit_symbols(self, data, fold):
        window = np.concatenate([fold.train_idx, fold.val_idx])
        self._scale = data.returns[fold.train_idx].std(axis=0, keepdims=True)
        z = data.returns[window] / self._scale
        shares = np.cumsum([0.2, 0.6, 0.2][: self.K - 1]) if self.K == 3 else np.linspace(0, 1, self.K + 1)[1:-1]
        self._thresholds = np.quantile(z.ravel(), shares)

    def fit(self, data: MarketData | None, symbols: np.ndarray, fold: Fold, seed: int) -> "HQMMModel":
        if self.symbol_mode == "train_vol":
            self._fit_symbols(data, fold)
        Y = self._symbols(data, symbols)
        n = Y.shape[1]
        Y_train, Y_val = Y[fold.train_idx], Y[fold.val_idx]
        best, self.restart_vals = None, []
        for r in range(self.n_restarts):
            s = seed * 1000 + r
            torch.manual_seed(s)
            builder = self.builder_factory(n)
            meas = LocalMeasurement(self.K, n_assets=n, shared=self.shared_measurement, reset=self.reset)
            if self.objective == "coarse":
                res = train_coarse(builder, meas, Y_train, Y_val, self.n_register, self.config, seed=s)
            elif self.reset:
                res = train_lbfgs_fast(builder, meas, Y_train, Y_val, self.n_register, self.config, seed=s)
            else:
                res = train_lbfgs(builder, meas, Y_train, Y_val, self.n_register, self.config, seed=s)
            if self.selection == "skill":
                if "val_skill" in res.history[-1]:
                    crit = res.history[-1]["val_skill"]
                else:
                    with torch.no_grad():
                        f = run_filter(builder, meas, np.concatenate([Y_train, Y_val]), self.n_register, predict=True)
                    from models.hqmm.engine import forecast_skill
                    crit = forecast_skill(f.pred_extreme[0].numpy(), np.concatenate([Y_train, Y_val]),
                                          self.K, np.arange(len(Y_train), len(Y_train) + len(Y_val)))
            else:
                crit = res.val_loglik
            self.restart_vals.append(crit)
            if best is None or crit > best[2]:
                best = (builder, meas, crit)
        self.builder, self.measurement, self.val_criterion = best
        return self

    def score(self, data: MarketData | None, symbols: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Predicted share of extreme moves for the next day, known at the end of each day in idx."""
        idx = np.asarray(idx)
        return self._filter(self._symbols(data, symbols)[: idx.max() + 1]).pred_extreme[0].numpy()[idx]

    def n_params(self) -> int:
        if self.builder is None:
            raise RuntimeError("fit the model first")
        return self.builder.n_params() + self.measurement.n_params()

    # ------------------------------------------------------------------ extras
    def posterior(self, data: MarketData | None, symbols: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Register-state probabilities (len(idx), 2^r) using only data up to each day."""
        idx = np.asarray(idx)
        return self._filter(self._symbols(data, symbols)[: idx.max() + 1]).posterior[0].numpy()[idx]

    def surprise(self, data: MarketData | None, symbols: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """-log p(y_t | past) on each day in idx (secondary alarm score)."""
        idx = np.asarray(idx)
        return (-self._filter(self._symbols(data, symbols)[: idx.max() + 1]).log_p[0]).numpy()[idx]

    def _filter(self, Y):
        with torch.no_grad():
            return run_filter(self.builder, self.measurement, torch.as_tensor(Y), self.n_register, predict=True)
