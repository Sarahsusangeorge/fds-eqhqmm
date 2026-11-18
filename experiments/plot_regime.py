"""Redraw the regime figure from regime_scores.csv. Each model's score is min-max scaled inside the window (display only) because the models' scores are on different scales."""
import sys
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src, dst = (sys.argv + ["results/regime/regime_scores.csv", "results/regime/regime_figure.png"][len(sys.argv) - 1:])[1:3]
df = pd.read_csv(src, parse_dates=["date"]); fids = sorted(df.fold.unique())
col = {"Q-E": "#1f77b4", "OOM-spectral": "#d95f02", "C-E-cat": "#444444"}
fig, ax = plt.subplots(len(fids), 1, figsize=(8, 3.5 * len(fids)))
ax = np.atleast_1d(ax)
for k, fid in enumerate(fids):
    d = df[df.fold == fid]; a1 = ax[k]; a2 = a1.twinx()
    v = d[d.model == "Q-E"]; a2.fill_between(v.date, v.ew_vol_ann, color="#cccccc", alpha=.6, lw=0); a2.set_ylabel("realised volatility, annualised", fontsize=8)
    for nm, g in d.groupby("model"):
        lo, hi = g.score.min(), g.score.max(); sc = lambda x: (x - lo) / (hi - lo)
        a1.plot(g.date, sc(g.score), color=col[nm], lw=1.3, label=nm)
        a1.axhline(sc(g.threshold.iloc[0]), color=col[nm], ls=":", lw=.9)
        al = g[g.alarm == 1]; a1.scatter(al.date, sc(al.score), color=col[nm], s=22, zorder=5)
    a1.set_zorder(a2.get_zorder() + 1); a1.patch.set_visible(False)
    a1.set_ylabel("stress score, scaled 0 to 1", fontsize=8)
    a1.set_title(f"Basket A, test window {d.date.min().date()} to {d.date.max().date()} (fold {fid})", fontsize=9)
    if k == 0: a1.legend(loc="upper left", fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(dst, dpi=200)
