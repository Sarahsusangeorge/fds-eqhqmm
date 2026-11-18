"""Learned stress score through two crises (Basket A): refit Q-E, Q-G and the categorical HMM on the folds whose
test window holds 2008 (fold 7) and Feb-Mar 2020 (fold 30); save scores to CSV and draw the figure.
    python experiments/regime_figure.py --out results/regime
"""
from __future__ import annotations
import argparse, copy, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from data.discretize import Discretizer
from data.market_data import MarketData
from harness.alarms import alarms_from_scores, calibrate_threshold
from harness.folds import make_folds
from experiments.run_quantum import build_models
from models.hqmm.symbols import train_vol_transform

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/regime"); ap.add_argument("--folds", default="7,30")
ap.add_argument("--basket", default="us_sectors"); ap.add_argument("--scheme", default="rolling"); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
md = MarketData.load(str(ROOT / "data/processed" / f"{a.basket}.npz"))
fc = yaml.safe_load((ROOT / "config/folds.yaml").read_text())
folds = {f.fold_id: f for f in make_folds(md.dates, a.scheme, fc["first_train_end"],
         fc.get("step_months", fc.get("test_step_months", 6)), fc.get("rolling_years", 5), fc.get("val_frac", fc.get("val_fraction", 0.2)))}
tr = train_vol_transform(md, 3)
models = build_models(["qe", "c-e-cat", "oom"], 3, "train_vol")
ew = md.equal_weight_returns(); rv = pd.Series(ew).rolling(21).std().values * np.sqrt(252)
rows = []
for fid in [int(x) for x in a.folds.split(",")]:
    f = folds[fid]
    disc = Discretizer(3, "quantile").fit(md.std_returns[f.window_idx]); sym = disc.transform(md.std_returns)
    sym = tr(sym, f, 0)
    for m0 in models:
        m = copy.deepcopy(m0); m.K = 3; m.fit(md, sym, f, 0)
        sv = np.asarray(m.score(md, sym, f.val_idx), float); st = np.asarray(m.score(md, sym, f.test_idx), float)
        h = calibrate_threshold(sv, 1.0, 20); al = set(alarms_from_scores(st, h, 20).tolist())
        for j, i in enumerate(f.test_idx):
            rows.append(dict(fold=fid, model=m.name, date=md.dates[i].date(), score=st[j], threshold=h, alarm=int(j in al), ew_vol_ann=rv[i]))
        print("done", fid, m.name, flush=True)
df = pd.DataFrame(rows); df.to_csv(out / "regime_scores.csv", index=False)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fids = sorted(df.fold.unique()); fig, ax = plt.subplots(len(fids), 1, figsize=(8, 3.4 * len(fids)))
ax = np.atleast_1d(ax); col = {"Q-E": "#1f77b4", "OOM-spectral": "#d95f02", "C-E-cat": "#555555"}
for k, fid in enumerate(fids):
    d = df[df.fold == fid]; a1 = ax[k]; a2 = a1.twinx()
    v = d[d.model == "Q-E"]; a2.fill_between(pd.to_datetime(v.date), v.ew_vol_ann, color="#cccccc", alpha=.6, label="realised vol (right)")
    for nm, g in d.groupby("model"):
        a1.plot(pd.to_datetime(g.date), g.score, color=col[nm], lw=1.3, label=nm)
        a1.axhline(g.threshold.iloc[0], color=col[nm], ls=":", lw=.8)
    a1.set_zorder(a2.get_zorder() + 1); a1.patch.set_visible(False)
    a1.set_ylabel("stress score"); a2.set_ylabel("annualised realised vol"); a1.set_title(f"Basket A, fold {fid}: test window {d.date.min()} to {d.date.max()}")
    if k == 0: a1.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(out / "regime_figure.png", dpi=200)
