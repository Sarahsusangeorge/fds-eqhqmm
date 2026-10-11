"""Data augmentation experiment for symmetry learning (Step A9).

Owner: Abhinav.

Compares:
  1. Q-G without augmentation (breaks symmetry).
  2. Q-G with random asset order augmentation per iteration.
  3. Q-E (equivariant by design, zero augmentation needed).

Logs results to results/logs/runs.jsonl and saves results/figures/augmentation_comparison.png.
"""

from __future__ import annotations

import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from harness.runner import log_run
from models.base import Fold
from models.hqmm.engine import run_filter, train_coarse
from models.hqmm.equivariant import EquivariantBuilder, permute_obs
from models.hqmm.generic import GenericBuilder
from models.hqmm.measurement import LocalMeasurement
from models.hqmm.model import HQMMModel
from synthetic.generate import make_synthetic

N_ASSETS = 3
K = 3
T_TRAIN = 1000
T_TEST = 500
SEED = 42


def train_augmented_qg(
    builder,
    meas,
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    n_register: int,
    rounds: int = 6,
):
    """Train Q-G with random column permutation each round."""
    n = Y_train.shape[1]
    rng = np.random.default_rng(SEED)
    best_loss = np.inf
    best_state = None

    for rnd in range(rounds):
        sigma = rng.permutation(n)
        Y_train_aug = Y_train[:, sigma]
        res = train_coarse(
            builder, meas, Y_train_aug, Y_val, n_register, seed=SEED + rnd
        )
        if res.val_loglik < best_loss:
            best_loss = res.val_loglik
            best_state = (builder.state_dict(), meas.state_dict())

    if best_state is not None:
        builder.load_state_dict(best_state[0])
        meas.load_state_dict(best_state[1])


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    d = make_synthetic(T=T_TRAIN + T_TEST, n=N_ASSETS, K=K, seed=SEED)
    n_val = 200
    fold = Fold(
        0,
        np.arange(T_TRAIN - n_val),
        np.arange(T_TRAIN - n_val, T_TRAIN),
        np.arange(T_TRAIN, T_TRAIN + T_TEST),
    )

    # 1. Q-E (Equivariant)
    print("Training Q-E (Equivariant)...")
    qe = HQMMModel(
        lambda n: EquivariantBuilder(1, n, layers=2),
        K=K,
        n_register=1,
        n_restarts=1,
        name="Q-E",
    )
    qe.fit(None, d.symbols, fold, seed=SEED)
    score_qe = qe.score(None, d.symbols, fold.test_idx)
    auc_qe = roc_auc_score(d.regimes[fold.test_idx], score_qe)

    # 2. Q-G standard (unaugmented)
    print("Training Q-G (standard)...")
    qg = HQMMModel(
        lambda n: GenericBuilder(1, n, layers=2),
        K=K,
        n_register=1,
        shared_measurement=False,
        n_restarts=1,
        name="Q-G",
    )
    qg.fit(None, d.symbols, fold, seed=SEED)
    score_qg = qg.score(None, d.symbols, fold.test_idx)
    auc_qg = roc_auc_score(d.regimes[fold.test_idx], score_qg)

    # 3. Q-G augmented
    print("Training Q-G (augmented)...")
    qg_aug_builder = GenericBuilder(1, N_ASSETS, layers=2)
    qg_aug_meas = LocalMeasurement(K, n_assets=N_ASSETS, shared=False, reset=True)
    train_augmented_qg(
        qg_aug_builder,
        qg_aug_meas,
        d.symbols[fold.train_idx],
        d.symbols[fold.val_idx],
        1,
    )
    with torch.no_grad():
        f = run_filter(
            qg_aug_builder,
            qg_aug_meas,
            d.symbols[: fold.test_idx.max() + 1],
            1,
            predict=True,
        )
        score_qg_aug = f.pred_extreme[0].numpy()[fold.test_idx]
    auc_qg_aug = roc_auc_score(d.regimes[fold.test_idx], score_qg_aug)

    print(f"Results (Test AUC):")
    print(f"  Q-E (Equivariant) : {auc_qe:.3f}")
    print(f"  Q-G (Standard)    : {auc_qg:.3f}")
    print(f"  Q-G (Augmented)   : {auc_qg_aug:.3f}")

    # Log runs
    for name, auc in [("Q-E", auc_qe), ("Q-G", auc_qg), ("Q-G_Augmented", auc_qg_aug)]:
        log_run(
            model_name=name,
            fold_id=0,
            seed=SEED,
            config={
                "n_assets": N_ASSETS,
                "K": K,
                "r": 1,
                "augmented": "Augmented" in name,
            },
            metrics={"test_auc": float(auc)},
            author="Abhinav",
        )

    # Plot summary bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Q-E", "Q-G (Standard)", "Q-G (Augmented)"],
        [auc_qe, auc_qg, auc_qg_aug],
        color=["tab:blue", "tab:red", "tab:orange"],
    )
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Test Regime Detection AUC")
    ax.set_title("Equivariance vs Data Augmentation (Step A9)")
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{yval:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()

    out_path = os.path.join("results", "figures", "augmentation_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
