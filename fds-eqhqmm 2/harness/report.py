"""Tables built from results/logs/runs.jsonl only (never from in-memory results)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness.runner import read_log


def _pooled_recall(d, lab):
    n, r = d[f"metrics.{lab}.n_onsets"], d[f"metrics.{lab}.recall"].fillna(0)
    return float((r * n).sum() / n.sum()) if n.sum() else np.nan


def summary_table(df, labels=("R1", "R2", "R3", "R4"), far_label="R2"):
    """One row per model, pooled over folds and seeds.
    rec_X = share of stress onsets detected (pooled, so few onsets means coarse steps); lag_X = mean lag in days
    (negative = early); far = false alarms per year against `far_label`; auprc/d are medians over folds
    (robust to the degenerate folds that make means explode)."""
    rows = []
    for m, d in df.groupby("model", sort=False):
        row = {"model": m, "params": int(d["n_params"].mean())}
        for lab in labels:
            if f"metrics.{lab}.recall" in d:
                row[f"rec_{lab}"] = _pooled_recall(d, lab)
        row["lag_R2"] = d["metrics.R2.mean_lag"].mean() if "metrics.R2.mean_lag" in d else np.nan
        row["far"] = d[f"metrics.{far_label}.far"].mean()
        for lab in labels:
            if f"metrics.{lab}.auprc" in d:
                row[f"auprc_{lab}"] = d[f"metrics.{lab}.auprc"].median()
        row["d_R2"] = d["metrics.R2.cohens_d"].median()
        row["no_alarm_folds"] = int((d["metrics.alarm_idx"].apply(len) == 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def onset_table(df, fold_id, labels=("R2", "R4")):
    cols = (
        ["model", "metrics.threshold"]
        + [f"metrics.{l}.recall" for l in labels]
        + [f"metrics.{l}.auprc" for l in labels]
    )
    out = df[df.fold_id == fold_id][cols].copy()
    out.columns = ["model", "threshold"] + [f"rec_{l}" for l in labels] + [f"auprc_{l}" for l in labels]
    return out


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "results/logs/runs.jsonl"
    df = read_log(path)
    print(f"rows: {len(df)} | models: {df.model.nunique()} | folds: {df.fold_id.nunique()}\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary_table(df).round(2).to_string(index=False))
        for f in sys.argv[2:]:
            print(f"\nFold {f}:")
            print(onset_table(df, int(f)).round(3).to_string(index=False))
