import numpy as np
import pytest
import torch

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from models.hqmm.engine import register_kraus_table, unitary_columns  # noqa: E402
from models.hqmm.equivariant import EquivariantBuilder  # noqa: E402
from models.hqmm.generic import GenericBuilder  # noqa: E402
from models.hqmm.measurement import LocalMeasurement  # noqa: E402
from models.hqmm import qiskit_backend as qb  # noqa: E402


def _model(kind, n=2, K=3, seed=0):
    torch.manual_seed(seed)
    if kind == "qe":
        b, shared = EquivariantBuilder(2, n, layers=2), True
    else:
        b, shared = GenericBuilder(2, n, layers=2), False
    m = LocalMeasurement(K, n_assets=n, shared=shared, reset=True)
    return b, m


def test_readout_unitary_is_unitary_and_embeds_rows():
    R = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 2)) + 1j * np.random.default_rng(1).normal(size=(3, 2)))[0]
    U = qb.readout_unitary(R)
    assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-12)
    for s in range(2):
        col = U[:, s * 4]
        assert np.allclose(col[:3], R[:, s], atol=1e-12) and abs(col[3:]).max() < 1e-12


@pytest.mark.parametrize("kind", ["qg", "qe"])
def test_qiskit_circuit_reproduces_engine_kraus_operators(kind):
    n = 2
    b, m = _model(kind, n)
    with torch.no_grad():
        ref = register_kraus_table(unitary_columns(b), m.rows(), 2, n).numpy()
    got = qb.kraus_table_from_circuit(b, m, 2, n)
    assert np.abs(got - ref).max() < 1e-8   # engine uses the fast columns path; circuit uses dense build()


def test_aer_sampling_matches_exact_probabilities():
    n, K = 2, 3
    b, m = _model("qe", n, K)
    with torch.no_grad():
        A = register_kraus_table(unitary_columns(b), m.rows(), 2, n).numpy()
    rho = np.eye(4) / 4
    p = np.array([np.trace(A[y] @ rho @ A[y].conj().T).real for y in range(K ** n)])
    assert abs(p.sum() - 1) < 1e-10
    S = qb.sample_sequences(b, m, 2, n, steps=1, shots=20000, seed=1)
    y = S[:, 0, 0] * K + S[:, 0, 1]
    freq = np.bincount(y, minlength=K ** n) / len(y)
    assert np.abs(freq - p).max() < 0.02


def test_two_step_sampling_uses_memory():
    n, K = 2, 3
    b, m = _model("qg", n, K)
    with torch.no_grad():
        A = register_kraus_table(unitary_columns(b), m.rows(), 2, n).numpy()
    rho = np.eye(4) / 4
    exp = np.zeros((K ** n, K ** n))
    for y1 in range(K ** n):
        r1 = A[y1] @ rho @ A[y1].conj().T
        for y2 in range(K ** n):
            exp[y1, y2] = np.trace(A[y2] @ r1 @ A[y2].conj().T).real
    S = qb.sample_sequences(b, m, 2, n, steps=2, shots=40000, seed=2)
    J = (S[:, :, 0] * K + S[:, :, 1])
    freq = np.zeros_like(exp)
    np.add.at(freq, (J[:, 0], J[:, 1]), 1)
    freq /= len(J)
    assert np.abs(freq - exp).max() < 0.02


def test_resource_estimate_runs():
    b, m = _model("qe", 2)
    qc = qb.build_step_circuit(b, m, 2, 2)
    r = qb.circuit_resources(qc)
    assert r["cx"] > 0 and r["depth"] > 0
