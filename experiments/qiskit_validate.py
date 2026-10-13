"""Run the trained HQMM as a Qiskit circuit and check it against the PyTorch engine.  Owner: Sarah.

    python experiments/qiskit_validate.py                  # synthetic data, 3 assets, local Qiskit/Aer simulators
    python experiments/qiskit_validate.py --data prices.npz --model qe --assets 3
    python experiments/qiskit_validate.py --azure          # also submit one circuit to Azure Quantum (needs an account)

Checks, for the generic (Q-G) and equivariant (Q-E) model:
  1. Kraus operators obtained by simulating the Qiskit circuit (Statevector) == the engine's operators.
  2. Log-likelihood of a real symbol sequence from the Qiskit-derived operators == the engine's.
  3. Aer shot sampling with mid-circuit measurement reproduces the exact outcome probabilities.
  4. Proposition 1 at circuit level: relabelling assets leaves the Q-E circuit likelihood unchanged.
  5. Resource estimate (qubits, depth, CX count) after transpiling to {rz, sx, x, cx}.
Writes results/tables/qiskit_validation.md.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.base import Fold, MarketData  # noqa: E402
from models.hqmm import qiskit_backend as qb  # noqa: E402
from models.hqmm.engine import LBFGSConfig, register_kraus_table, unitary_columns  # noqa: E402
from models.hqmm.equivariant import EquivariantBuilder  # noqa: E402
from models.hqmm.generic import GenericBuilder  # noqa: E402
from models.hqmm.model import HQMMModel  # noqa: E402


def loglik(table, Y, K, n_register=2):
    """log p(Y) from register Kraus operators A_y (numpy), register starts maximally mixed."""
    dR = 2 ** n_register
    rho, ll = np.eye(dR, dtype=complex) / dR, 0.0
    for y in Y:
        j = 0
        for s in y:
            j = j * K + int(s)
        rho = table[j] @ rho @ table[j].conj().T
        p = np.trace(rho).real
        ll += np.log(p)
        rho = rho / p
    return ll


def make_model(kind, K, fast):
    cfg = LBFGSConfig(rounds=2, max_iter=3) if fast else LBFGSConfig()
    if kind == "qe":
        return HQMMModel(lambda n: EquivariantBuilder(2, n, layers=2), K=K, name="Q-E", n_restarts=1,
                         shared_measurement=True, symbol_mode="train_vol", config=cfg)
    return HQMMModel(lambda n: GenericBuilder(2, n, layers=3), K=K, name="Q-G", n_restarts=1,
                     shared_measurement=False, symbol_mode="train_vol", config=cfg)


def azure_submit(builder, measurement, n_register, n_assets, shots, target):
    """Submit ONE step circuit (basis-state input, terminal measurement only) to Azure Quantum via Qiskit.
    Needs: pip install azure-quantum[qiskit]; env AZURE_QUANTUM_RESOURCE_ID and AZURE_QUANTUM_LOCATION.
    Not exercised in CI - it needs your Azure workspace."""
    from azure.quantum.qiskit import AzureQuantumProvider
    from qiskit import transpile

    provider = AzureQuantumProvider(resource_id=os.environ["AZURE_QUANTUM_RESOURCE_ID"],
                                    location=os.environ["AZURE_QUANTUM_LOCATION"])
    print("available targets:", [b.name() for b in provider.backends()])
    backend = provider.get_backend(target)
    qc = qb.build_step_circuit(builder, measurement, n_register, n_assets, measure=True)
    t = transpile(qc, backend, optimization_level=1)
    job = backend.run(t, shots=shots)
    print("job id:", job.id())
    return job.result().get_counts()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="saved MarketData .npz (default: synthetic)")
    ap.add_argument("--assets", type=int, default=3)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--days", type=int, default=250, help="length of the sequence used for the likelihood check")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--fast", action="store_true", help="tiny training budget")
    ap.add_argument("--azure", action="store_true")
    ap.add_argument("--azure-target", default="ionq.simulator")
    ap.add_argument("--out", default="results/tables/qiskit_validation.md")
    a = ap.parse_args(argv)

    if a.data:
        md = MarketData.load(a.data)
        md = MarketData(**{**md.__dict__, **{k: getattr(md, k)[:, : a.assets] for k in ("returns", "std_returns")}}) \
            if md.n > a.assets else md
    else:
        from tests.helpers import fake_market
        md, _ = fake_market(T=1700, n=a.assets, seed=0)
    T = md.T
    fold = Fold(0, np.arange(int(T * 0.6)), np.arange(int(T * 0.6), int(T * 0.75)), np.arange(int(T * 0.75), T))
    sym0 = np.zeros((T, md.n), dtype=int)
    rows = ["| model | Kraus max abs diff | loglik engine | loglik Qiskit | Aer max prob diff | qubits | CX | depth |",
            "|---|---|---|---|---|---|---|---|"]
    n, K = md.n, a.K
    for kind in ("qg", "qe"):
        m = make_model(kind, K, a.fast).fit(md, sym0, fold, seed=0)
        b, meas = m.builder, m.measurement
        Y = m._symbols(md, sym0)[fold.test_idx][: a.days]
        with torch.no_grad():
            ref = register_kraus_table(unitary_columns(b), meas.rows(), 2, n).numpy()
        got = qb.kraus_table_from_circuit(b, meas, 2, n)
        d_k = np.abs(got - ref).max()
        ll_e, ll_q = loglik(ref, Y, K), loglik(got, Y, K)
        # Aer: first-day outcome frequencies vs exact probabilities
        rho = np.eye(4) / 4
        p = np.array([np.trace(ref[y] @ rho @ ref[y].conj().T).real for y in range(K ** n)])
        S = qb.sample_sequences(b, meas, 2, n, steps=1, shots=a.shots, seed=1)
        y = np.zeros(len(S), int)
        for i in range(n):
            y = y * K + S[:, 0, i]
        d_p = np.abs(np.bincount(y, minlength=K ** n) / len(y) - p).max()
        res = qb.circuit_resources(qb.build_step_circuit(b, meas, 2, n))
        rows.append(f"| {m.name} | {d_k:.1e} | {ll_e:.4f} | {ll_q:.4f} | {d_p:.4f} | {res['qubits']} | {res['cx']} | {res['depth']} |")
        print(rows[-1])
        if kind == "qe":
            perm = np.random.default_rng(0).permutation(n)
            ll_perm = loglik(ref, Y[:, perm], K)
            print(f"Proposition 1 (Q-E, relabelled assets): loglik {ll_e:.6f} -> {ll_perm:.6f}")
            rows.append(f"\nProposition 1 check on the Q-E circuit: log-likelihood {ll_e:.6f} before, {ll_perm:.6f} after a random relabelling of the assets.\n")
        if a.azure and kind == "qe":
            print("Azure counts:", azure_submit(b, meas, 2, n, a.shots, a.azure_target))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Qiskit validation of the HQMM\n\nSame trained model run as a Qiskit circuit and as the PyTorch engine.\n\n"
                   + "\n".join(rows) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
