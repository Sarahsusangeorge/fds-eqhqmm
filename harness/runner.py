"""S9: run(models, folds, seeds) -> fit, score, calibrate, compute metrics, log every run (contract 5)."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from data.discretize import Discretizer
from harness.alarms import alarms_from_scores, calibrate_threshold
from harness.metrics import auprc, cohens_d, detection_lags, false_alarms_per_year, heldout_ll, recall
from harness.overlay import buy_and_hold, risk_off_overlay
from labels.common import onsets

DEFAULT_LOG = Path(__file__).resolve().parents[1] / "results" / "logs" / "runs.jsonl"


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def log_run(model, fold_id, seed, config, metrics, author, path=None, n_params=None):
    """Append one JSON line to results/logs/runs.jsonl. Tables and figures are generated from this file only.
    `model` is a RegimeModel or just its name (the shared contract test passes a string)."""
    path = Path(path) if path else DEFAULT_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model if isinstance(model, str) else model.name,
        "n_params": int(n_params if n_params is not None else (0 if isinstance(model, str) else model.n_params())),
        "fold_id": int(fold_id),
        "seed": int(seed),
        "author": author,
        "config": config,
        "metrics": metrics,
    }
    with open(path, "a") as f:
        f.write(json.dumps(_clean(rec)) + "\n")


def fold_metrics(scores, alarm_rel, fold, labels, surprise, pre, post):
    """Metrics for one fold; alarms/scores are relative to fold.test_idx, labels are full-sample LabelResults."""
    T_test = len(fold.test_idx)
    alarm_idx = fold.test_idx[alarm_rel]
    out = {}
    for name, lab in labels.items():
        s = lab.series
        on = onsets(s)
        on = on[(on >= fold.test_idx[0]) & (on <= fold.test_idx[-1])]
        lags = detection_lags(alarm_idx, on, pre, post)
        out[name] = {
            "n_onsets": int(len(on)),
            "lags": lags,
            "recall": recall(lags),
            "mean_lag": float(np.nanmean(lags)) if np.any(~np.isnan(lags)) else float("nan"),
            "far": false_alarms_per_year(alarm_idx, s, len(s), T_test, pre, post),
            "auprc": auprc(s[fold.test_idx], scores),
            "cohens_d": cohens_d(s[fold.test_idx], scores),
        }
    out["heldout_ll"] = heldout_ll(surprise)
    return out


def run(
    models,
    folds,
    seeds,
    data,
    labels,
    K=3,
    disc_method="quantile",
    target_far=1.0,
    gap=20,
    pre=20,
    post=40,
    author="shriker",
    config=None,
    symbol_transform=None,
    overlay_bps=(5, 10, 25),
    log_path=None,
    store_overlay_series=False,
    verbose=True,
):
    """labels: {name: LabelResult}. symbol_transform(symbols, fold, seed) -> symbols (robustness hook)."""
    config = dict(config or {})
    config.update({"K": K, "disc_method": disc_method, "target_far": target_far, "gap": gap, "pre": pre, "post": post})
    ew = data.equal_weight_returns()
    for fold in folds:
        disc = Discretizer(K, disc_method).fit(data.std_returns[fold.window_idx])
        symbols = disc.transform(data.std_returns)
        for model in models:
            for seed in seeds:
                if symbol_transform is not None:
                    symbols_run = symbol_transform(symbols, fold, seed)
                else:
                    symbols_run = symbols
                m = copy.deepcopy(model)
                if hasattr(m, "K"):
                    m.K = K  # symbol-based models must use the discretizer's alphabet size
                t0 = time.time()
                m.fit(data, symbols_run, fold, seed)
                s_val = np.asarray(m.score(data, symbols_run, fold.val_idx), float)
                s_test = np.asarray(m.score(data, symbols_run, fold.test_idx), float)
                h = calibrate_threshold(s_val, target_far, gap)
                alarm_rel = alarms_from_scores(s_test, h, gap)
                met = fold_metrics(
                    s_test,
                    alarm_rel,
                    fold,
                    labels,
                    (m.surprise(data, symbols_run, fold.test_idx) if hasattr(m, "surprise") else None),
                    pre,
                    post,
                )
                met["threshold"] = h
                met["alarm_idx"] = fold.test_idx[alarm_rel]
                met["runtime_s"] = time.time() - t0
                ret = ew[fold.test_idx]
                met["buy_hold"] = buy_and_hold(ret)
                met["overlay"] = {}
                for bps in overlay_bps:
                    o = risk_off_overlay(ret, s_test, h, bps)
                    if not store_overlay_series:
                        o.pop("net_ret")
                    met["overlay"][str(bps)] = o
                log_run(m, fold.fold_id, seed, config, met, author, log_path)
                if verbose:
                    print(
                        f"fold {fold.fold_id:2d} {m.name:<14} seed {seed} alarms={len(alarm_rel)} h={h:.3f} {met['runtime_s']:.1f}s"
                    )


def read_log(path=None):
    import pandas as pd

    path = Path(path) if path else DEFAULT_LOG
    rows = [json.loads(l) for l in open(path)]
    return pd.json_normalize(rows)
