"""Contract 1: MarketData (Shriker produces, everyone reads)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    dates: pd.DatetimeIndex  # length T
    assets: list  # length n
    returns: np.ndarray  # (T, n) daily log returns
    std_returns: np.ndarray  # (T, n) returns / trailing vol, vol known before day t
    corr: "np.ndarray | None"  # (T, n, n) trailing 63-day correlation, known before day t

    def __len__(self) -> int:
        return len(self.dates)

    @property
    def T(self) -> int:  # same names as the shared models/base.py dataclass
        return len(self.dates)

    @property
    def n(self) -> int:
        return len(self.assets)

    @property
    def n_assets(self) -> int:
        return self.returns.shape[1]

    def equal_weight_returns(self) -> np.ndarray:
        return self.returns.mean(axis=1)

    def slice(self, start: int, stop: int) -> "MarketData":
        """Rows start..stop-1 (used to build labels from a training window only)."""
        return MarketData(
            dates=self.dates[start:stop],
            assets=list(self.assets),
            returns=self.returns[start:stop],
            std_returns=self.std_returns[start:stop],
            corr=None if self.corr is None else self.corr[start:stop],
        )

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            dates=self.dates.values.astype("datetime64[ns]"),
            assets=np.array(self.assets),
            returns=self.returns,
            std_returns=self.std_returns,
            corr=np.empty(0) if self.corr is None else self.corr,
        )

    @classmethod
    def load(cls, path) -> "MarketData":
        z = np.load(path, allow_pickle=False)
        corr = z["corr"]
        return cls(
            dates=pd.DatetimeIndex(z["dates"]),
            assets=[str(a) for a in z["assets"]],
            returns=z["returns"],
            std_returns=z["std_returns"],
            corr=None if corr.size == 0 else corr,
        )
