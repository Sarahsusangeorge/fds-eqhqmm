"""S14: data robustness. Each function returns a modified copy; `run_all` re-runs the models under every
perturbation and logs with a `robustness` tag in the config so tables are built from runs.jsonl only."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from data.features import build_market_data_from_returns
from harness.runner import run
from labels.common import load_config
from labels.r1_bear import label_r1
from labels.r2_vol import label_r2
from labels.r4_events import label_r4


def add_return_noise(md, frac, seed=0):
    """Add N(0, (frac * asset vol)^2) noise to returns; features are rebuilt from the noisy returns."""
    rng = np.random.default_rng(seed)
    noisy = md.returns + rng.standard_normal(md.returns.shape) * frac * md.returns.std(0)
    out = build_market_data_from_returns(pd.DataFrame(noisy, index=md.dates, columns=md.assets))
    return out  # note: loses the first 63 rows again, so align labels with out.dates


def flip_symbols(symbols, p, K, rng):
    """Replace a fraction p of symbols with a uniformly random DIFFERENT symbol."""
    s = symbols.copy()
    mask = rng.random(s.shape) < p
    s[mask] = (s[mask] + rng.integers(1, K, size=mask.sum())) % K
    return s


def label_variants(md):
    """Alternative label parameterisations (label-definition sensitivity)."""
    out = {}
    for fall in (0.15, 0.25):
        cfg = copy.deepcopy(load_config())
        cfg["r1_bear"]["fall"] = cfg["r1_bear"]["rise"] = fall
        r = label_r1(md, cfg)
        r.name = f"R1_{int(fall * 100)}"
        out[r.name] = r
    for pct in (85, 95):
        cfg = copy.deepcopy(load_config())
        cfg["r2_vol"]["percentile"] = pct
        r = label_r2(md, cfg)
        r.name = f"R2_p{pct}"
        out[r.name] = r
    return out


def run_all(
    models,
    folds,
    seeds,
    md,
    labels,
    fold_params=None,
    base_config=None,
    log_path=None,
    noise_levels=(0.25, 0.5),
    flip_probs=(0.05, 0.10),
    skip=(),
    **kw,
):
    """fold_params: kwargs for make_folds, used to rebuild folds on the noisy sample (its first rows differ)."""
    from harness.folds import make_folds

    base = dict(base_config or {})
    cfg = lambda **d: {**base, **d}
    if "baseline" not in skip:
        run(models, folds, seeds, md, labels, config=cfg(robustness="baseline"), log_path=log_path, **kw)
    if "K5" not in skip:
        run(
            models, folds, seeds, md, labels, config=cfg(robustness="K5"), log_path=log_path, **{**kw, "K": 5}
        )  # K = 5 instead of 3
    for p in flip_probs:  # symbol flips (return-based models are unaffected by construction)

        def tf(sym, fold, seed, p=p):
            return flip_symbols(sym, p, kw.get("K", 3), np.random.default_rng(1000 + seed))

        run(
            models,
            folds,
            seeds,
            md,
            labels,
            symbol_transform=tf,
            config=cfg(robustness="symbol_flip", level=p),
            log_path=log_path,
            **kw,
        )
    run(models, folds, seeds, md, label_variants(md), config=cfg(robustness="label_variants"), log_path=log_path, **kw)
    for pre, post in ((10, 10), (60, 60)):  # detection-window tolerance
        kw2 = {**kw, "pre": pre, "post": post}
        run(
            models,
            folds,
            seeds,
            md,
            labels,
            config=cfg(robustness="window", level=f"{pre}/{post}"),
            log_path=log_path,
            **kw2,
        )
    for frac in noise_levels:  # return noise: features, labels and folds are rebuilt on the noisy sample
        mdn = add_return_noise(md, frac)
        labn = {"R2": label_r2(mdn), "R4": label_r4(mdn)}
        fl = make_folds(mdn.dates, **(fold_params or {}))
        run(models, fl, seeds, mdn, labn, config=cfg(robustness="return_noise", level=frac), log_path=log_path, **kw)
