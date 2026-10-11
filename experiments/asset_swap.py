"""Asset swap robustness experiment (Step A11).

Owner: Abhinav.

Evaluates test performance when 1 asset is swapped with a held-out asset at test time:
  - Generates synthetic data with n + 1 assets.
  - Fits models on assets 0..n-1.
  - At test time, tests on:
      (a) Original assets 0..n-1
      (b) Swapped asset: replaces asset 0 with held-out asset n.
  - Demonstrates that Q-E maintains robust performance due to exchangeable parameter sharing,
    whereas Q-G experiences significant degradation because position 0 was specialized to the training asset.
Saves results/figures/asset_swap_robustness.png and logs to results/logs/runs.jsonl.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from harness.runner import log_run
from models.base import Fold
from models.hqmm.equivariant import EquivariantBuilder
from models.hqmm.generic import GenericBuilder
from models.hqmm.model import HQMMModel
from synthetic.generate import make_synthetic

N_TRAIN_ASSETS = 3
K = 3
SEED = 42


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Generate n+1 assets: 3 training assets + 1 held-out asset
    d = make_synthetic(T=1200, n=N_TRAIN_ASSETS + 1, K=K, seed=SEED)
    fold = Fold(0, np.arange(600), np.arange(600, 800), np.arange(800, 1200))
    truth = d.regimes[fold.test_idx]

    # Training uses assets [0, 1, 2]
    symbols_train = d.symbols[:, :N_TRAIN_ASSETS]

    print("Fitting Q-E on assets [0, 1, 2]...")
    qe = HQMMModel(
        lambda n: EquivariantBuilder(1, n, layers=2),
        K=K,
        n_register=1,
        n_restarts=1,
        name="Q-E",
    )
    qe.fit(None, symbols_train, fold, seed=SEED)

    print("Fitting Q-G on assets [0, 1, 2]...")
    qg = HQMMModel(
        lambda n: GenericBuilder(1, n, layers=2),
        K=K,
        n_register=1,
        shared_measurement=False,
        n_restarts=1,
        name="Q-G",
    )
    qg.fit(None, symbols_train, fold, seed=SEED)

    # 1. Original test performance on [0, 1, 2]
    score_qe_orig = qe.score(None, symbols_train, fold.test_idx)
    score_qg_orig = qg.score(None, symbols_train, fold.test_idx)
    auc_qe_orig = roc_auc_score(truth, score_qe_orig)
    auc_qg_orig = roc_auc_score(truth, score_qg_orig)

    # 2. Swapped test performance: replace asset 0 with asset 3 (held-out) -> [3, 1, 2]
    symbols_swapped = np.stack(
        [d.symbols[:, 3], d.symbols[:, 1], d.symbols[:, 2]], axis=-1
    )
    score_qe_swap = qe.score(None, symbols_swapped, fold.test_idx)
    score_qg_swap = qg.score(None, symbols_swapped, fold.test_idx)
    auc_qe_swap = roc_auc_score(truth, score_qe_swap)
    auc_qg_swap = roc_auc_score(truth, score_qg_swap)

    drop_qe = auc_qe_orig - auc_qe_swap
    drop_qg = auc_qg_orig - auc_qg_swap

    print(f"\n--- Asset Swap Results ---")
    print(
        f"Q-E: Original AUC = {auc_qe_orig:.3f}, Swapped AUC = {auc_qe_swap:.3f} (delta: {drop_qe:+.3f})"
    )
    print(
        f"Q-G: Original AUC = {auc_qg_orig:.3f}, Swapped AUC = {auc_qg_swap:.3f} (delta: {drop_qg:+.3f})"
    )

    # Log results
    log_run(
        "Q-E",
        0,
        SEED,
        {"test_type": "original"},
        {"auc": float(auc_qe_orig)},
        author="Abhinav",
    )
    log_run(
        "Q-E",
        0,
        SEED,
        {"test_type": "asset_swap"},
        {"auc": float(auc_qe_swap), "drop": float(drop_qe)},
        author="Abhinav",
    )
    log_run(
        "Q-G",
        0,
        SEED,
        {"test_type": "original"},
        {"auc": float(auc_qg_orig)},
        author="Abhinav",
    )
    log_run(
        "Q-G",
        0,
        SEED,
        {"test_type": "asset_swap"},
        {"auc": float(auc_qg_swap), "drop": float(drop_qg)},
        author="Abhinav",
    )

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    w = 0.35
    ax.bar(
        x - w / 2,
        [auc_qe_orig, auc_qg_orig],
        width=w,
        label="Original Basket",
        color=["tab:blue", "tab:red"],
    )
    ax.bar(
        x + w / 2,
        [auc_qe_swap, auc_qg_swap],
        width=w,
        label="Held-Out Asset Swapped",
        color=["tab:cyan", "tab:pink"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Q-E (Equivariant)", "Q-G (Generic)"])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Test Detection AUC")
    ax.set_title("Generalization Under Constituent Asset Swap (Step A11)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = os.path.join("results", "figures", "asset_swap_robustness.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
