"""Which assumptions of Proposition 1 are needed?  Break one at a time and measure the change in the posterior and the
log-likelihood under all n! relabellings (n = 4 assets, r = 2, K = 3, random non-trivial parameters, 400 random symbols).
    python experiments/prop1_assumptions.py --out results/prop1_assumptions.csv
"""
from __future__ import annotations
import argparse, itertools, sys
from pathlib import Path
import numpy as np, torch, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.hqmm.engine import run_filter
from models.hqmm.equivariant import EquivariantBuilder, permute_obs
from models.hqmm.generic import GenericBuilder
from models.hqmm.measurement import LocalMeasurement

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/prop1_assumptions.csv"); a = ap.parse_args()
n, r, K, T = 4, 2, 3, 400
rng = np.random.default_rng(0); torch.manual_seed(0)
Y = rng.integers(0, K, size=(T, n))
cases = {
    "all assumptions hold (Q-E circuit, shared measurement)": (lambda: EquivariantBuilder(r, n, layers=2, init_scale=0.5), True),
    "A2 broken: Q-E circuit, separate measurement per asset": (lambda: EquivariantBuilder(r, n, layers=2, init_scale=0.5), False),
    "A1 broken: generic circuit, shared measurement": (lambda: GenericBuilder(r, n, layers=2), True),
    "A1 and A2 broken (Q-G)": (lambda: GenericBuilder(r, n, layers=2), False),
}
rows = []
for name, (mk, shared) in cases.items():
    b = mk()
    with torch.no_grad():
        for p in b.parameters():
            if p.dim() > 0 and "theta" in name: pass
        m = LocalMeasurement(K, n_assets=n, shared=shared, reset=True)
        # make per-asset measurements genuinely different, and the generic circuit non-trivial
        for p in m.parameters(): p.add_(0.7 * torch.randn_like(p))
        if isinstance(b, GenericBuilder):
            for p in b.parameters(): p.add_(0.7 * torch.randn_like(p))
        base = run_filter(b, m, Y, r)
        pd0, ll0 = base.posterior[0].numpy(), float(base.log_p.sum())
        dp, dl = [], []
        for s in itertools.permutations(range(n)):
            f = run_filter(b, m, permute_obs(Y, s), r)
            dp.append(np.abs(f.posterior[0].numpy() - pd0).max())
            dl.append(abs(float(f.log_p.sum()) - ll0))
    rows.append(dict(case=name, max_posterior_change=max(dp), max_loglik_change=max(dl), permutations=len(dp)))
    print(rows[-1], flush=True)
# day-by-day independent relabellings for the full-assumption case
b = EquivariantBuilder(r, n, layers=2, init_scale=0.5); m = LocalMeasurement(K, n_assets=n, shared=True, reset=True)
for p_ in m.parameters():
    with torch.no_grad(): p_.add_(0.7 * torch.randn_like(p_))
with torch.no_grad():
    base = run_filter(b, m, Y, r); worst = 0.0
    for _ in range(50):
        Yp = np.stack([Y[t][rng.permutation(n)] for t in range(T)])
        f = run_filter(b, m, Yp, r); worst = max(worst, float(np.abs(f.posterior[0].numpy() - base.posterior[0].numpy()).max()))
rows.append(dict(case="day-by-day independent relabellings (50 draws), all assumptions hold", max_posterior_change=worst, max_loglik_change=float("nan"), permutations=50))
print(rows[-1])
pd.DataFrame(rows).to_csv(a.out, index=False)
