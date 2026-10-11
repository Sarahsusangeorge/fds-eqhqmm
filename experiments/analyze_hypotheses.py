"""H1 / H2 / H3 analysis on logged runs.  Owner: Sarah.

Reads one or more JSONL run logs (quantum log from run_quantum.py + the classical logs from run_pipeline.py),
averages each metric over seeds within a fold, and tests paired fold-level differences:

  H1  quantum channel      Q-G vs C-G   (generic)      Q-E vs C-E-cat (equivariant)
  H2  equivariance         Q-E vs Q-G   (quantum)      C-E-cat vs C-G (classical)
  H3  interaction (DiD)    (Q-E - Q-G) - (C-E - C-G)

Each contrast: mean difference, stationary-bootstrap 95% CI, paired Wilcoxon p, Holm-adjusted p (family = all
contrasts x labels for one metric).  H3 reports the DiD estimate and its bootstrap CI only.
Metrics: auprc, recall (higher better), far (lower better) -- sign is reported as logged (A - B), not flipped.
No speed-up / advantage claim is made; "significant" only ever means "differs in this benchmark".

    python experiments/analyze_hypotheses.py --logs results/logs/runs_quantum_A.jsonl results/logs/runs_A_expanding.jsonl \
        --filter robustness=baseline --out results/tables/hypotheses.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.runner import read_log  # noqa: E402
from harness.stats import did_bootstrap, holm, paired_wilcoxon, stationary_bootstrap_ci  # noqa: E402

CELLS = {"QG": "Q-G", "QE": "Q-E", "CG": "C-G", "CE": "C-E-cat"}
CONTRASTS = [  # (hypothesis, name, A, B)  -> difference A - B
    ("H1", "quantum - classical | generic", "QG", "CG"),
    ("H1", "quantum - classical | equivariant", "QE", "CE"),
    ("H2", "equivariant - generic | quantum", "QE", "QG"),
    ("H2", "equivariant - generic | classical", "CE", "CG"),
]
METRICS = ["auprc", "recall", "far"]
LABELS = ["R1", "R2", "R3", "R4"]


def load(paths, filters=None, cells=CELLS) -> pd.DataFrame:
    df = pd.concat([read_log(p) for p in paths], ignore_index=True)
    for f in filters or []:
        k, v = f.split("=", 1)
        col = f"config.{k}"
        if col not in df:
            raise SystemExit(f"no config field {k!r} in logs")
        df = df[df[col].astype(str) == v]
    return df[df["model"].isin(cells.values())]


def fold_table(df, label, metric):
    """folds x cells matrix of seed-averaged metric (NaN where model/fold missing or metric undefined)."""
    col = f"metrics.{label}.{metric}"
    if col not in df:
        return pd.DataFrame()
    t = df.assign(v=pd.to_numeric(df[col], errors="coerce")).groupby(["fold_id", "model"])["v"].mean().unstack()
    return t.rename(columns={v: k for k, v in CELLS.items()})


def analyse(df, reps=2000, seed=0) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        fam = []
        for label in LABELS:
            t = fold_table(df, label, metric)
            for hyp, name, a, b in CONTRASTS:
                r = dict(metric=metric, label=label, hyp=hyp, contrast=name, n_folds=0, diff=np.nan,
                         ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
                if a in t and b in t:
                    d = (t[a] - t[b]).dropna()
                    r["n_folds"] = len(d)
                    if len(d) >= 3:
                        r["diff"] = float(d.mean())
                        r["ci_lo"], r["ci_hi"] = stationary_bootstrap_ci(d.values, reps=reps, seed=seed)
                        r["p"] = paired_wilcoxon(t.loc[d.index, a], t.loc[d.index, b])[1]
                rows.append(r)
                fam.append(len(rows) - 1)
            r3 = dict(metric=metric, label=label, hyp="H3", contrast="DiD (Q-E - Q-G) - (C-E - C-G)", n_folds=0,
                      diff=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
            if all(c in t for c in CELLS):
                x = t[["QG", "QE", "CG", "CE"]].dropna()
                r3["n_folds"] = len(x)
                if len(x) >= 5:
                    r3["diff"], r3["ci_lo"], r3["ci_hi"] = did_bootstrap(x.QG, x.QE, x.CG, x.CE, reps=reps, seed=seed)
            rows.append(r3)
        # Holm across the Wilcoxon family of this metric
        idx = [i for i in fam if np.isfinite(rows[i]["p"])]
        if idx:
            _, adj = holm([rows[i]["p"] for i in idx])
            for i, a in zip(idx, adj):
                rows[i]["p_holm"] = float(a)
    out = pd.DataFrame(rows)
    if "p_holm" not in out:
        out["p_holm"] = np.nan
    out["ci_excludes_0"] = (out.ci_lo > 0) | (out.ci_hi < 0)
    return out


def to_markdown(res, header="") -> str:
    lines = [header, "Differences are A - B, mean over folds of the seed-averaged metric. "
             "CI = stationary bootstrap 95%; p = paired Wilcoxon, Holm-adjusted within metric. "
             "Nothing here is a quantum-advantage claim.\n"]
    for metric in METRICS:
        lines += [f"### {metric}", "", "| hyp | contrast | label | folds | diff | 95% CI | p | p (Holm) | CI excl. 0 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in res[res.metric == metric].iterrows():
            f = lambda x, n=3: "–" if not np.isfinite(x) else f"{x:.{n}f}"
            lines.append(f"| {r.hyp} | {r.contrast} | {r.label} | {r.n_folds} | {f(r['diff'])} | "
                         f"[{f(r.ci_lo)}, {f(r.ci_hi)}] | {f(r.p, 4)} | {f(r.p_holm, 4)} | "
                         f"{'yes' if r.ci_excludes_0 else 'no'} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--filter", nargs="*", default=[], help="config filters, e.g. robustness=baseline basket=us_sectors")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="results/tables/hypotheses.md")
    a = ap.parse_args(argv)
    df = load(a.logs, a.filter)
    have = sorted(df["model"].unique())
    print("models found:", have, "| rows:", len(df))
    missing = [v for v in CELLS.values() if v not in have]
    if missing:
        print("WARNING: missing cells (those contrasts will be blank):", missing)
    res = analyse(df, reps=a.reps)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(res, f"# H1/H2/H3 results\n\nLogs: {', '.join(a.logs)}; filters: {a.filter or 'none'}\n"))
    res.to_csv(out.with_suffix(".csv"), index=False)
    print(f"wrote {out}")
    return res


if __name__ == "__main__":
    main()
