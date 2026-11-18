"""Summaries for the v2 real-data runs on event folds (folds whose test window contains labelled stress onsets).
    python experiments/analyze_v2.py --cheap A:cheap_A.jsonl --quantum A:quantum_A.jsonl --out results/v2
Only folds where EVERY listed model has a record are used for the comparison, so all rows are matched fold by fold.
Descriptive only: with a handful of folds no significance test is run."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from harness.runner import read_log

ap = argparse.ArgumentParser()
ap.add_argument("--cheap", action="append", required=True); ap.add_argument("--quantum", action="append", required=True)
ap.add_argument("--out", default="results/v2"); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for c, q in zip(a.cheap, a.quantum):
    b, cf = c.split(":", 1); _, qf = q.split(":", 1)
    df = pd.concat([read_log(cf), read_log(qf)], ignore_index=True)
    df = df[(df.seed == 0) & (df.model != "OOM-spectral")]  # OOM withheld: suspected leakage (R08)
    qmods = sorted(set(read_log(qf).model))
    folds = set.intersection(*[set(df[df.model == m].fold_id) for m in qmods])
    d = df[df.fold_id.isin(folds)].copy()
    rows, per = [], []
    for lab in ("R1", "R4"):
        for m, g in d.groupby("model"):
            g = g.set_index("fold_id").sort_index()
            ok = g[f"metrics.{lab}.auprc"].notna()
            per.append(g.loc[ok, [f"metrics.{lab}.auprc"]].rename(columns={f"metrics.{lab}.auprc": "auprc"}).assign(model=m, label=lab).reset_index())
    per = pd.concat(per) if per else pd.DataFrame()
    per.to_csv(out / f"per_fold_{b}.csv", index=False)
    for lab in ("R1", "R4"):
        p = per[per.label == lab].pivot(index="fold_id", columns="model", values="auprc").dropna()
        for m in p.columns:
            gg = d[(d.model == m) & d.fold_id.isin(p.index)]
            n_on = gg[f"metrics.{lab}.n_onsets"].sum(); rec = (gg[f"metrics.{lab}.recall"] * gg[f"metrics.{lab}.n_onsets"]).sum() / n_on if n_on else np.nan
            rows.append(dict(basket=b, label=lab, model=m, params=int(gg.n_params.iloc[0]), folds=len(p), onsets=int(n_on),
                             mean_auprc=p[m].mean(), median_auprc=p[m].median(), recall=rec, far=gg[f"metrics.{lab}.far"].mean()))
    s = pd.DataFrame(rows); s.to_csv(out / f"summary_{b}.csv", index=False)
    # H1 / H2 / H3 on the matched folds (descriptive): symmetry effect inside each family, and their difference (interaction)
    did = []
    for lab in ("R1", "R4"):
        p = per[per.label == lab].pivot(index="fold_id", columns="model", values="auprc").dropna()
        need = {"Q-E", "Q-G", "C-E-cat", "C-G-cat"}
        if need <= set(p.columns) and len(p):
            dq, dc = p["Q-E"] - p["Q-G"], p["C-E-cat"] - p["C-G-cat"]
            did.append(dict(basket=b, label=lab, folds=len(p), sym_effect_quantum=dq.mean(), sym_effect_classical=dc.mean(),
                            interaction=(dq - dc).mean(), q_wins=int((dq > 0).sum()), c_wins=int((dc > 0).sum()),
                            quantum_vs_classical_generic=(p["Q-G"] - p["C-G-cat"]).mean(), quantum_vs_classical_equivariant=(p["Q-E"] - p["C-E-cat"]).mean()))
    pd.DataFrame(did).to_csv(out / f"did_{b}.csv", index=False)
    if did: print(pd.DataFrame(did).round(3).to_string(index=False))
    print(f"Basket {b}: folds with all of {qmods}: {sorted(folds)}\n", s.round(3).to_string(index=False), "\n")
