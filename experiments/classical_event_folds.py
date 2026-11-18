"""Classical models on EVERY fold whose test window holds labelled stress onsets (rolling 5-year windows, seed 0).
    python experiments/classical_event_folds.py --cheap A:cheap_A.jsonl --cheap B:cheap_B.jsonl --out results/v2
Paired Wilcoxon signed-rank tests on per-fold AUPRC, Holm-corrected across all comparisons in the output."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from harness.runner import read_log

ap = argparse.ArgumentParser(); ap.add_argument("--cheap", action="append", required=True); ap.add_argument("--out", default="results/v2"); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
summ, tests = [], []
for c in a.cheap:
    b, f = c.split(":", 1); df = read_log(f); df = df[(df.seed == 0) & (df.model != "OOM-spectral")]  # OOM withheld: suspected leakage (R08)
    for lab in ("R1", "R4"):
        ev = df[df[f"metrics.{lab}.n_onsets"] > 0]
        folds = sorted(set(ev.fold_id))
        p = ev.pivot(index="fold_id", columns="model", values=f"metrics.{lab}.auprc").dropna()
        for m, g in ev.groupby("model"):
            n_on = g[f"metrics.{lab}.n_onsets"].sum()
            rec = (g[f"metrics.{lab}.recall"].fillna(0) * g[f"metrics.{lab}.n_onsets"]).sum() / n_on
            summ.append(dict(basket=b, label=lab, model=m, params=int(g.n_params.iloc[0]), folds=len(p), onsets=int(n_on),
                             mean_auprc=p[m].mean() if m in p else np.nan, recall=rec, far=g[f"metrics.{lab}.far"].mean()))
        for x, y in (("C-E-cat", "C-G"), ("C-E-cat", "C-G-cat")):
            if x in p and y in p and len(p) >= 5:
                diff = p[x] - p[y]
                pv = wilcoxon(p[x], p[y]).pvalue if np.any(diff != 0) else 1.0
                tests.append(dict(basket=b, label=lab, a=x, b=y, folds=len(p), mean_diff=diff.mean(), wins=int((diff > 0).sum()), p_raw=pv))
S = pd.DataFrame(summ); T = pd.DataFrame(tests)
if len(T):
    order = T.p_raw.argsort().values; m = len(T); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * T.p_raw.iloc[i]); adj[i] = min(1.0, run)
    T["p_holm"] = adj
S.to_csv(out / "classical_event_folds_summary.csv", index=False); T.to_csv(out / "classical_event_folds_tests.csv", index=False)
print(S.round(3).to_string(index=False)); print(T.round(4).to_string(index=False))
