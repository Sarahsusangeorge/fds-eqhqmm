"""S12: Hamilton (1989) two-regime Markov-switching mean/variance on the equal-weight basket (statsmodels).
Fitted on the training rows; scored by FILTERING the whole series with the fitted parameters (causal)."""

import warnings

import numpy as np
from scipy.stats import norm

from models.base import RegimeModel
from models.classical._hmm_utils import forward_filter_log, stationary, stress_log_odds


class HamiltonMS(RegimeModel):
    name = "Hamilton"

    def __init__(self, k_regimes=2):
        self.k_regimes = k_regimes

    def _mod(self, y):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        return MarkovRegression(y, k_regimes=self.k_regimes, trend="c", switching_variance=True)

    def _starts(self, y, names, rng, n=3):
        """Deterministic, seed-controlled start values (statsmodels' own random search is not reproducible)."""
        k = self.k_regimes
        v = float(np.var(y))
        out = []
        for i in range(n):
            sp = np.zeros(len(names))
            for j, nm in enumerate(names):
                if nm.startswith("p[") and nm.endswith("->0]"):
                    sp[j] = np.clip(0.95 - 0.05 * rng.random() - (0.0 if nm.startswith("p[0") else 0.85), 0.02, 0.98)
                elif nm.startswith("const["):
                    sp[j] = float(np.mean(y)) + 0.05 * rng.standard_normal()
                elif nm.startswith("sigma2["):
                    r = int(nm[7])
                    sp[j] = v * (0.5 + 1.5 * r / max(k - 1, 1)) * (0.8 + 0.4 * rng.random())
            out.append(sp)
        return out

    def fit(self, data, symbols, fold, seed):
        y = data.equal_weight_returns()[fold.train_idx] * 100
        mod = self._mod(y)
        names = list(mod.param_names)
        rng = np.random.default_rng(seed)
        best = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for sp in self._starts(y, names, rng):
                try:
                    res = mod.fit(start_params=sp, disp=False, search_reps=0, em_iter=0)
                except Exception:
                    continue
                if best is None or res.llf > best.llf:
                    best = res
        self.params = np.asarray(best.params)
        s2 = [self.params[names.index(f"sigma2[{j}]")] for j in range(self.k_regimes)]
        self.stress = int(np.argmax(s2))
        k = self.k_regimes
        self.mu = np.array([self.params[names.index(f"const[{j}]")] for j in range(k)])
        self.s2 = np.array(s2)
        # statsmodels names: p[i->0] = P(next regime 0 | current regime i); two regimes only
        p00, p10 = (self.params[names.index(n)] for n in ("p[0->0]", "p[1->0]"))
        self.A = np.array([[p00, 1 - p00], [p10, 1 - p10]])  # A[from, to]
        self.pi = stationary(self.A)
        return self

    def _filter(self, data, upto):
        """Causal filter in log space with the fitted parameters (Gaussian emission per regime)."""
        y = data.equal_weight_returns()[: upto + 1] * 100
        lb = np.column_stack([norm.logpdf(y, self.mu[j], np.sqrt(self.s2[j])) for j in range(self.k_regimes)])
        return forward_filter_log(lb, self.pi, self.A)

    def _statsmodels_filtered(self, data, upto):
        """Reference only (used in tests): statsmodels' own filtered probabilities."""
        y = data.equal_weight_returns()[: upto + 1] * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(self._mod(y).filter(self.params).filtered_marginal_probabilities)

    def score(self, data, symbols, idx):
        """Stress log-odds of the filtered regime (monotone in the probability, no saturation at 1.0)."""
        la, _ = self._filter(data, int(np.max(idx)))
        return stress_log_odds(la, self.stress)[idx]

    def surprise(self, data, symbols, idx):
        _, logc = self._filter(data, int(np.max(idx)))
        return -logc[idx]

    def n_params(self):
        return len(self.params)
