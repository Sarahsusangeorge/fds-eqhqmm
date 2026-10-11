"""S12: Absorption ratio (Kritzman et al. 2011): share of total variance explained by the top-k principal
components of the trailing-window correlation matrix (window includes day t). k=ceil(n/5) by default; k=1 gives the
largest-eigenvalue share. No parameters to fit."""

import numpy as np

from models.base import RegimeModel


class AbsorptionRatio(RegimeModel):
    def __init__(self, window=250, k=None, name="Absorption"):
        self.window, self.k, self.name = window, k, name

    def fit(self, data, symbols, fold, seed):
        self.n = data.n_assets
        return self

    def score(self, data, symbols, idx):
        k = self.k or int(np.ceil(data.n_assets / 5))
        idx = np.asarray(idx)
        ar = {}
        out = np.zeros(len(idx))
        for j, t in enumerate(idx):
            lo = max(0, t - self.window + 1)
            if t - lo + 1 < 30:  # not enough history at the very start of the sample
                out[j] = np.nan
                continue
            C = np.corrcoef(data.returns[lo : t + 1].T)
            w = np.sort(np.linalg.eigvalsh(np.nan_to_num(C)))[::-1]
            out[j] = w[:k].sum() / w.sum()
        if np.isnan(out).any():
            out = np.where(np.isnan(out), np.nanmin(out) if (~np.isnan(out)).any() else 0.0, out)
        return out

    def n_params(self):
        return 0
