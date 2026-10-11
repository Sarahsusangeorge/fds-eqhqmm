"""Empirical check of Proposition 1: Permutation Equivariance (Step A5).

Owner: Abhinav.

Compares regime posteriors of Q-E (Equivariant HQMM) vs Q-G (Generic HQMM)
under 100 random asset permutations:
  - Q-E posterior is identical across all permutations to machine precision (< 1e-10).
  - Q-G posterior varies significantly because the generic hardware-efficient ansatz
    breaks permutation symmetry.
Saves results/figures/prop1_invariance.png and logs to results/logs/runs.jsonl.
"""

from __future__ import annotations

import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch

from harness.runner import log_run
from models.hqmm.engine import run_filter
from models.hqmm.equivariant import EquivariantBuilder, permute_obs
from models.hqmm.generic import GenericBuilder
from models.hqmm.measurement import LocalMeasurement
from synthetic.generate import make_synthetic

N_ASSETS = 3
N_REGISTER = 1
K = 3
T = 300
N_PERMS = 100
SEED = 42


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Generate test data
    data = make_synthetic(T=T, n=N_ASSETS, K=K, seed=SEED)
    Y = data.symbols  # (T, n)

    # 2. Build Q-E and Q-G models
    # Q-E: equivariant unitary + shared measurement
    qe_builder = EquivariantBuilder(N_REGISTER, N_ASSETS, layers=2)
    qe_meas = LocalMeasurement(K, n_assets=N_ASSETS, shared=True, reset=True)

    # Q-G: generic ansatz (per-qubit parameters) + measurement
    qg_builder = GenericBuilder(N_REGISTER, N_ASSETS, layers=2)
    qg_meas = LocalMeasurement(K, n_assets=N_ASSETS, shared=False, reset=True)

    # Base posterior curves on original data: (1, T, 2^r) -> take stress/first state
    with torch.no_grad():
        qe_base = run_filter(qe_builder, qe_meas, Y, N_REGISTER).posterior[0].numpy()
        qg_base = run_filter(qg_builder, qg_meas, Y, N_REGISTER).posterior[0].numpy()

    # 3. Evaluate under 100 random asset permutations
    all_perms = list(itertools.permutations(range(N_ASSETS)))
    rng = np.random.default_rng(SEED)
    sampled_indices = rng.choice(len(all_perms), size=N_PERMS, replace=True)

    qe_diffs = []
    qg_diffs = []
    qe_curves = []
    qg_curves = []

    with torch.no_grad():
        for idx in sampled_indices:
            sigma = all_perms[idx]
            Y_perm = permute_obs(Y, sigma)

            qe_post = (
                run_filter(qe_builder, qe_meas, Y_perm, N_REGISTER).posterior[0].numpy()
            )
            qg_post = (
                run_filter(qg_builder, qg_meas, Y_perm, N_REGISTER).posterior[0].numpy()
            )

            qe_diff = np.max(np.abs(qe_post - qe_base))
            qg_diff = np.max(np.abs(qg_post - qg_base))

            qe_diffs.append(qe_diff)
            qg_diffs.append(qg_diff)
            qe_curves.append(qe_post[:, 0])
            qg_curves.append(qg_post[:, 0])

    max_qe_diff = float(np.max(qe_diffs))
    max_qg_diff = float(np.max(qg_diffs))
    mean_qg_diff = float(np.mean(qg_diffs))

    print(f"Proposition 1 Check ({N_PERMS} permutations, T={T}, n={N_ASSETS}):")
    print(
        f"  Q-E max deviation from unpermuted: {max_qe_diff:.2e} (machine precision invariant: {max_qe_diff < 1e-10})"
    )
    print(
        f"  Q-G max deviation from unpermuted: {max_qg_diff:.4f} (breaks symmetry, mean dev: {mean_qg_diff:.4f})"
    )

    assert max_qe_diff < 1e-10, f"Q-E is not invariant: max diff = {max_qe_diff}"
    assert (
        max_qg_diff > 1e-2
    ), f"Q-G should vary under permutations: max diff = {max_qg_diff}"

    # 4. Plot Figure 5: Q-E vs Q-G posterior trajectories under permutations
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)
    t = np.arange(min(150, T))

    # Q-E panel
    axes[0].plot(
        t, qe_base[t, 0], color="black", lw=2, label="Original order (Q-E)", zorder=10
    )
    for c in qe_curves[:30]:
        axes[0].plot(t, c[t], color="tab:blue", alpha=0.3, lw=0.8)
    axes[0].set_title(
        f"Q-E (Equivariant): Invariant under all 100 permutations (max dev = {max_qe_diff:.1e})",
        fontsize=11,
    )
    axes[0].set_ylabel("Regime posterior P(R=0)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Q-G panel
    axes[1].plot(
        t, qg_base[t, 0], color="black", lw=2, label="Original order (Q-G)", zorder=10
    )
    for c in qg_curves[:30]:
        axes[1].plot(t, c[t], color="tab:red", alpha=0.3, lw=0.8)
    axes[1].set_title(
        f"Q-G (Generic): Symmetry broken by ansatz order (max dev = {max_qg_diff:.3f})",
        fontsize=11,
    )
    axes[1].set_xlabel("Day t")
    axes[1].set_ylabel("Regime posterior P(R=0)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join("results", "figures", "prop1_invariance.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")

    # 5. Log run
    log_run(
        model_name="Prop1_Equivariance_Check",
        fold_id=0,
        seed=SEED,
        config={
            "n_assets": N_ASSETS,
            "n_register": N_REGISTER,
            "n_perms": N_PERMS,
            "T": T,
        },
        metrics={
            "max_qe_diff": max_qe_diff,
            "max_qg_diff": max_qg_diff,
            "mean_qg_diff": mean_qg_diff,
            "is_qe_invariant": bool(max_qe_diff < 1e-10),
        },
        author="Abhinav",
    )


if __name__ == "__main__":
    main()
