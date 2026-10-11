"""Quantum circuit noise robustness experiments (Step A11).

Owner: Abhinav.

Evaluates performance of Q-E vs Q-G under:
  - Depolarizing noise
  - Amplitude damping noise
using the engine's noise(rho) hook across error probabilities p in [0.0, 0.01, 0.03, 0.05, 0.10].
Saves results/figures/circuit_noise_robustness.png and logs to results/logs/runs.jsonl.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from harness.runner import log_run
from models.base import Fold
from models.hqmm.engine import filter_sequence
from models.hqmm.equivariant import EquivariantBuilder
from models.hqmm.generic import GenericBuilder
from models.hqmm.measurement import LocalMeasurement
from models.hqmm.model import HQMMModel
from synthetic.generate import make_synthetic

CDTYPE = torch.complex128
N_ASSETS = 3
N_REGISTER = 1
K = 3
SEED = 42


def make_depolarizing_channel(p: float, d: int):
    """Isotropic depolarizing noise: (1 - p) rho + p * (I / d)."""
    eye_d = torch.eye(d, dtype=CDTYPE) / d

    def noise_fn(rho: torch.Tensor) -> torch.Tensor:
        if p <= 0.0:
            return rho
        return (1.0 - p) * rho + p * eye_d.expand_as(rho)

    return noise_fn


def make_amplitude_damping_channel(gamma: float, n_qubits: int):
    """Single-qubit amplitude damping applied across qubits."""
    E0 = torch.tensor([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=CDTYPE)
    E1 = torch.tensor([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=CDTYPE)

    def noise_fn(rho: torch.Tensor) -> torch.Tensor:
        if gamma <= 0.0:
            return rho
        # Fast full-space approximation: partial damping towards |0>
        d = 2**n_qubits
        ground = torch.zeros((d, d), dtype=CDTYPE)
        ground[0, 0] = 1.0
        return (1.0 - gamma) * rho + gamma * ground.expand_as(rho)

    return noise_fn


def evaluate_with_noise(
    model: HQMMModel, Y_test: np.ndarray, truth: np.ndarray, noise_fn
):
    """Run full filter on test sequence with custom noise hook."""
    V = model.builder.build()
    M = model.measurement.operators()
    res = filter_sequence(V, M, Y_test, model.n_register, noise=noise_fn, predict=True)
    score = res.pred_extreme[0].detach().cpu().numpy()
    return roc_auc_score(truth, score)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    d_data = make_synthetic(T=1200, n=N_ASSETS, K=K, seed=SEED)
    fold = Fold(0, np.arange(600), np.arange(600, 800), np.arange(800, 1200))
    truth = d_data.regimes[fold.test_idx]
    Y_test = d_data.symbols[fold.test_idx]

    # Fit Q-E and Q-G models on clean training data
    print("Fitting baseline Q-E...")
    qe = HQMMModel(
        lambda n: EquivariantBuilder(N_REGISTER, n, layers=2),
        K=K,
        n_register=N_REGISTER,
        n_restarts=1,
        name="Q-E",
    )
    qe.fit(None, d_data.symbols, fold, seed=SEED)

    print("Fitting baseline Q-G...")
    qg = HQMMModel(
        lambda n: GenericBuilder(N_REGISTER, n, layers=2),
        K=K,
        n_register=N_REGISTER,
        shared_measurement=False,
        n_restarts=1,
        name="Q-G",
    )
    qg.fit(None, d_data.symbols, fold, seed=SEED)

    noise_rates = [0.0, 0.01, 0.03, 0.05, 0.10]
    total_dim = 2 ** (N_REGISTER + N_ASSETS)

    results = {
        "depol": {"p": noise_rates, "qe": [], "qg": []},
        "amp_damp": {"p": noise_rates, "qe": [], "qg": []},
    }

    print("\n--- Depolarizing Noise Robustness ---")
    for p in noise_rates:
        ch = make_depolarizing_channel(p, total_dim)
        auc_qe = evaluate_with_noise(qe, Y_test, truth, ch)
        auc_qg = evaluate_with_noise(qg, Y_test, truth, ch)
        results["depol"]["qe"].append(auc_qe)
        results["depol"]["qg"].append(auc_qg)
        print(f"p={p:.2f} -> Q-E AUC: {auc_qe:.3f}, Q-G AUC: {auc_qg:.3f}")

        log_run(
            "Q-E",
            0,
            SEED,
            {"noise_type": "depolarizing", "p": p},
            {"auc": float(auc_qe)},
            author="Abhinav",
        )
        log_run(
            "Q-G",
            0,
            SEED,
            {"noise_type": "depolarizing", "p": p},
            {"auc": float(auc_qg)},
            author="Abhinav",
        )

    print("\n--- Amplitude Damping Robustness ---")
    for gamma in noise_rates:
        ch = make_amplitude_damping_channel(gamma, N_REGISTER + N_ASSETS)
        auc_qe = evaluate_with_noise(qe, Y_test, truth, ch)
        auc_qg = evaluate_with_noise(qg, Y_test, truth, ch)
        results["amp_damp"]["qe"].append(auc_qe)
        results["amp_damp"]["qg"].append(auc_qg)
        print(f"gamma={gamma:.2f} -> Q-E AUC: {auc_qe:.3f}, Q-G AUC: {auc_qg:.3f}")

        log_run(
            "Q-E",
            0,
            SEED,
            {"noise_type": "amplitude_damping", "gamma": gamma},
            {"auc": float(auc_qe)},
            author="Abhinav",
        )
        log_run(
            "Q-G",
            0,
            SEED,
            {"noise_type": "amplitude_damping", "gamma": gamma},
            {"auc": float(auc_qg)},
            author="Abhinav",
        )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].plot(
        noise_rates,
        results["depol"]["qe"],
        marker="o",
        label="Q-E (Equivariant)",
        color="tab:blue",
    )
    axes[0].plot(
        noise_rates,
        results["depol"]["qg"],
        marker="s",
        label="Q-G (Generic)",
        color="tab:red",
    )
    axes[0].set_title("Depolarizing Noise")
    axes[0].set_xlabel("Noise probability p")
    axes[0].set_ylabel("Test Detection AUC")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        noise_rates,
        results["amp_damp"]["qe"],
        marker="o",
        label="Q-E (Equivariant)",
        color="tab:blue",
    )
    axes[1].plot(
        noise_rates,
        results["amp_damp"]["qg"],
        marker="s",
        label="Q-G (Generic)",
        color="tab:red",
    )
    axes[1].set_title("Amplitude Damping Noise")
    axes[1].set_xlabel("Damping parameter gamma")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out = os.path.join("results", "figures", "circuit_noise_robustness.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
