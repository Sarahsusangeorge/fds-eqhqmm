"""S12: Bayesian online change-point detection (Adams & MacKay 2007), Gaussian with unknown mean/variance
(Normal-inverse-gamma prior, Student-t predictive), constant hazard. Score = posterior mass on run lengths < `window`
(a change-point in the last `window` days). Detects changes in either direction, as a pure change-point baseline."""

import numpy as np
from scipy.special import gammaln

from models.base import RegimeModel


def _t_logpdf(x, mu, kappa, alpha, beta):
    df = 2 * alpha
    scale2 = beta * (kappa + 1) / (alpha * kappa)
    z = (x - mu) ** 2 / scale2
    return gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(np.pi * df * scale2) - (df + 1) / 2 * np.log1p(z / df)


class BOCPD(RegimeModel):
    name = "BOCPD"

    def __init__(self, hazard=1 / 250, max_run=300, window=10):
        self.hazard, self.max_run, self.window = hazard, max_run, window

    def fit(self, data, symbols, fold, seed):
        y = data.equal_weight_returns()[fold.train_idx] * 100
        self.mu0, self.var0 = float(y.mean()), float(y.var())
        return self

    def _run(self, y):
        H, R_ = self.hazard, self.max_run + 1
        mu, kappa, alpha, beta = (np.array([v], float) for v in (self.mu0, 1.0, 1.0, self.var0))
        R = np.array([1.0])
        score, surprise = np.zeros(len(y)), np.zeros(len(y))
        for t, x in enumerate(y):
            pred = np.exp(_t_logpdf(x, mu, kappa, alpha, beta))
            joint = R * pred
            ev = joint.sum()
            surprise[t] = -np.log(max(ev, 1e-300))
            newR = np.r_[H * ev, (1 - H) * joint]
            newR /= newR.sum()
            mu_n = np.r_[self.mu0, (kappa * mu + x) / (kappa + 1)]
            beta_n = np.r_[self.var0, beta + kappa * (x - mu) ** 2 / (2 * (kappa + 1))]
            kappa_n = np.r_[1.0, kappa + 1]
            alpha_n = np.r_[1.0, alpha + 0.5]
            if len(newR) > R_:  # truncate: fold the tail into the last bucket
                newR[R_ - 1] += newR[R_:].sum()
                newR, mu_n, beta_n, kappa_n, alpha_n = (a[:R_] for a in (newR, mu_n, beta_n, kappa_n, alpha_n))
            R, mu, kappa, alpha, beta = newR, mu_n, kappa_n, alpha_n, beta_n
            score[t] = R[: self.window].sum()
        return score, surprise

    def score(self, data, symbols, idx):
        y = data.equal_weight_returns()[: int(np.max(idx)) + 1] * 100
        return self._run(y)[0][idx]

    def surprise(self, data, symbols, idx):
        y = data.equal_weight_returns()[: int(np.max(idx)) + 1] * 100
        return self._run(y)[1][idx]

    def n_params(self):
        return 2  # hazard + prior scale (nonparametric otherwise)
