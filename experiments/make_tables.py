"""Tables for the paper, built ONLY from the run logs (results/logs/*.jsonl). Writes CSV + Markdown to results/tables/.

python experiments/make_tables.py results/logs/runs_A_expanding.jsonl                       # tables 2-5 for basket A
python experiments/make_tables.py results/logs/runs_A_expanding.jsonl results/logs/runs_A_robust.jsonl   # + robustness
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.report import summary_table
from harness.runner import read_log
from harness.stats import holm, paired_wilcoxon

OUT = ROOT / "results" / "tables"


def save(df, name, title):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    with open(OUT / f"{name}.md", "w") as f:
        f.write(f"**{title}**\n\n{df.round(3).to_markdown(index=False)}\n")
    print(f"\n{title}\n{df.round(2).to_string(index=False)}")


def table_main(df):
    """Detection quality per model, pooled over folds (rec = share of onsets caught, lag in days, far per year)."""
    return summary_table(df)


def table_overlay(df, bps=(5, 10, 25)):
    """Risk-off overlay vs buy-and-hold, equal-weight basket, mean over 6-month test folds."""
    rows = []
    bh = df.drop_duplicates("fold_id")
    base = {"sharpe": bh["metrics.buy_hold.sharpe"].mean(), "mdd": bh["metrics.buy_hold.max_drawdown"].mean()}
    rows.append(
        {
            "model": "buy and hold",
            "cost_bps": 0,
            "sharpe": base["sharpe"],
            "max_drawdown": base["mdd"],
            "turnover_per_yr": 0.0,
        }
    )
    for m, d in df.groupby("model", sort=False):
        for b in bps:
            rows.append(
                {
                    "model": m,
                    "cost_bps": b,
                    "sharpe": d[f"metrics.overlay.{b}.sharpe"].mean(),
                    "max_drawdown": d[f"metrics.overlay.{b}.max_drawdown"].mean(),
                    "turnover_per_yr": d[f"metrics.overlay.{b}.turnover_per_year"].mean(),
                }
            )
    return pd.DataFrame(rows)


def table_cost(df):
    g = df.groupby("model", sort=False)
    return pd.DataFrame(
        {
            "model": list(g.groups),
            "params": g["n_params"].mean().astype(int).values,
            "fit_score_sec_per_fold": g["metrics.runtime_s"].mean().values,
            "total_min": (g["metrics.runtime_s"].sum() / 60).values,
        }
    )


def table_stats(df, refs=("C-G", "C-E"), metrics=("auprc_R3", "auprc_R4", "auprc_R1")):
    """Paired Wilcoxon on per-fold AUPRC against reference models, Holm-corrected within each (reference, metric).
    mean_diff > 0 means the model beats the reference."""
    rows = []
    models = list(df["model"].unique())
    for ref in refs:
        for met in metrics:
            lab = met.split("_")[1]
            col = f"metrics.{lab}.auprc"
            if col not in df.columns:
                continue
            wide = df.pivot_table(index=["fold_id"], columns="model", values=col)
            res = []
            for m in models:
                if m == ref or ref not in wide or m not in wide:
                    continue
                pair = wide[[m, ref]].dropna()
                if len(pair) < 6:
                    continue
                stat, p = paired_wilcoxon(pair[m], pair[ref])
                res.append(
                    {
                        "reference": ref,
                        "metric": met,
                        "model": m,
                        "n_folds": len(pair),
                        "mean_diff": float((pair[m] - pair[ref]).mean()),
                        "p_raw": p,
                    }
                )
            if res:
                _, adj = holm([r["p_raw"] for r in res])
                for r, a in zip(res, adj):
                    r["p_holm"] = float(a)
                rows += res
    return pd.DataFrame(rows)


def table_robust(df):
    """Recall, false alarms per year and AUPRC averaged over the labels R2 and R4, which every setting evaluates, so the
    d_ columns (change vs the clean baseline) compare like with like. Label-variant rows use their own labels and
    carry no delta. K and symbol-flip changes affect only symbol-based models (C-E-cat and the quantum models)."""
    d = df.copy()
    d["setting"] = d["config.robustness"].fillna("main").astype(str) + "|" + d["config.level"].fillna("").astype(str)

    def cols(kind, labs):
        return [f"metrics.{l}.{kind}" for l in labs if f"metrics.{l}.{kind}" in d.columns]

    common = ("R2", "R4")
    is_var = d["config.robustness"] == "label_variants"
    for name, kind in (("recall", "recall"), ("far", "far"), ("auprc", "auprc")):
        main_cols = cols(kind, common)
        var_cols = [c for c in d.columns if c.startswith("metrics.") and c.endswith(f".{kind}") and c not in main_cols]
        d[name] = np.where(is_var, d[var_cols].mean(axis=1) if var_cols else np.nan, d[main_cols].mean(axis=1))
    out = d.groupby(["setting", "model"], sort=False)[["recall", "far", "auprc"]].mean().reset_index()
    base = out[out.setting.str.startswith("baseline")].set_index("model")
    for c in ("recall", "far", "auprc"):
        out[f"d_{c}"] = out.apply(
            lambda r: np.nan if r["setting"].startswith("label_variants") else r[c] - base[c].get(r["model"], np.nan),
            axis=1,
        )
    return out


if __name__ == "__main__":
    paths = sys.argv[1:] or ["results/logs/runs.jsonl"]
    main = read_log(paths[0])
    save(table_main(main), "table2_main", "Table 2. Detection results (pooled over folds)")
    save(table_overlay(main), "table_overlay", "Risk-off overlay vs buy and hold (mean over folds)")
    save(table_cost(main), "table5_cost", "Table 5. Cost")
    save(table_stats(main), "table3_stats", "Paired Wilcoxon on per-fold AUPRC vs reference (Holm-corrected)")
    if len(paths) > 1:
        save(
            table_robust(read_log(paths[1])),
            "table4_robust",
            "Table 4. Robustness (mean over labels; d_ = change vs clean baseline)",
        )
