"""S12: GARCH(1,1) on the equal-weight basket (arch package). Score_t = next-day conditional volatility forecast
sigma_{t+1|t} computed from data up to and including day t with the fitted parameters."""

import warnings

import numpy as np

from models.base import RegimeModel


class GARCH11(RegimeModel):
    name = "GARCH"

    def fit(self, data, symbols, fold, seed):
        from arch import arch_model

        y = data.equal_weight_returns()[fold.train_idx] * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1).fit(disp="off")
        p = res.params
        self.mu, self.omega, self.alpha, self.beta = (
            float(p["mu"]),
            float(p["omega"]),
            float(p["alpha[1]"]),
            float(p["beta[1]"]),
        )
        self.var0 = float(y.var())
        return self

    def _recursion(self, y):
        T = len(y)
        s2 = np.zeros(T + 1)  # s2[t] = variance of day t given data < t
        s2[0] = self.var0
        e2 = (y - self.mu) ** 2
        for t in range(T):
            s2[t + 1] = self.omega + self.alpha * e2[t] + self.beta * s2[t]
        return s2, e2

    def score(self, data, symbols, idx):
        y = data.equal_weight_returns()[: int(np.max(idx)) + 1] * 100
        s2, _ = self._recursion(y)
        return np.sqrt(s2[1:])[idx]  # forecast for t+1 made at the close of t

    def surprise(self, data, symbols, idx):
        y = data.equal_weight_returns()[: int(np.max(idx)) + 1] * 100
        s2, e2 = self._recursion(y)
        return (0.5 * (np.log(2 * np.pi * s2[:-1]) + e2 / s2[:-1]))[idx]

    def n_params(self):
        return 4
