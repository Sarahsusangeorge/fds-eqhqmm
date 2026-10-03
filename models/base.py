"""Shared contracts for every module. Agreed jointly by all three members.

Change this file only through a pull request approved by both other members:
every model, the harness and the engine depend on it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    """One basket of assets. Produced by data/ (Shriker), read by everyone.

    Every field at day t uses only information available before or at day t.
    """

    dates: pd.DatetimeIndex          # length T
    assets: list[str]                # length n
    returns: np.ndarray              # (T, n) daily log returns
    std_returns: np.ndarray          # (T, n) returns / trailing volatility known before day t
    corr: Optional[np.ndarray] = None  # (T, n, n) trailing 63-day correlation known before day t

    def __post_init__(self) -> None:
        T, n = len(self.dates), len(self.assets)
        if self.returns.shape != (T, n):
            raise ValueError(f"returns must be {(T, n)}, got {self.returns.shape}")
        if self.std_returns.shape != (T, n):
            raise ValueError(f"std_returns must be {(T, n)}, got {self.std_returns.shape}")
        if self.corr is not None and self.corr.shape != (T, n, n):
            raise ValueError(f"corr must be {(T, n, n)}, got {self.corr.shape}")

    @property
    def T(self) -> int:
        return len(self.dates)

    @property
    def n(self) -> int:
        return len(self.assets)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            dates=self.dates.values.astype("datetime64[ns]"),
            assets=np.array(self.assets),
            returns=self.returns,
            std_returns=self.std_returns,
            corr=self.corr if self.corr is not None else np.array([]),
        )

    @classmethod
    def load(cls, path: str) -> "MarketData":
        z = np.load(path, allow_pickle=False)
        corr = z["corr"] if z["corr"].size else None
        return cls(
            dates=pd.DatetimeIndex(z["dates"]),
            assets=[str(a) for a in z["assets"]],
            returns=z["returns"],
            std_returns=z["std_returns"],
            corr=corr,
        )


@dataclass
class Fold:
    """One walk-forward fold. Produced by harness/folds.py (Shriker)."""

    fold_id: int
    train_idx: np.ndarray   # integer day indices
    val_idx: np.ndarray     # last part of the training window
    test_idx: np.ndarray    # the following test period

    def __post_init__(self) -> None:
        if len(self.test_idx) and len(self.train_idx) and self.test_idx.min() <= self.train_idx.max():
            raise ValueError("test days must come after all training days")


class RegimeModel(ABC):
    """Every model (classical, quantum, neural) implements this.

    Models output scores only. Thresholds, alarms and metrics are applied by the
    harness, identically for every model.
    """

    name: str = "unnamed"

    @abstractmethod
    def fit(self, data: MarketData, symbols: np.ndarray, fold: Fold, seed: int) -> "RegimeModel":
        """Fit on fold.train_idx, using fold.val_idx for early stopping and tuning."""

    @abstractmethod
    def score(self, data: MarketData, symbols: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Stress score s_t for each day in idx, using only data up to and including t."""

    @abstractmethod
    def n_params(self) -> int:
        """Number of trainable parameters (used to match model sizes)."""


class UnitaryBuilder(ABC):
    """Builds the unitary V(theta) used inside the HQMM Kraus operators.

    Implemented by GenericBuilder (Sarah) and EquivariantBuilder (Abhinav);
    called by the engine (Sarah).
    Qubit order: the r register qubits first, then one qubit per asset.
    """

    n_register: int
    n_assets: int

    @property
    def n_qubits(self) -> int:
        return self.n_register + self.n_assets

    @abstractmethod
    def n_params(self) -> int: ...

    @abstractmethod
    def build(self, theta, step_inputs=None):
        """Return V as a (2^q, 2^q) complex unitary torch tensor.

        step_inputs carries per-day inputs (e.g. the correlation matrix) for
        time-varying builders; static builders ignore it.
        """


# Optional per-step noise channel for circuit-noise experiments: rho -> rho.
NoiseChannel = Callable[["object"], "object"]
