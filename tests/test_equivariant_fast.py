"""Fast column builder of the equivariant HQMM (owner of the builder: Abhinav; fast path: Sarah)."""
import numpy as np
import pytest
import torch

from models.hqmm.engine import register_kraus_table, total_loglik_coarse, unitary_columns
from models.hqmm.equivariant import EquivariantBuilder
from models.hqmm.measurement import LocalMeasurement


def _exact_columns(b):
    """Reference with exact (eigendecomposition) exponentials; torch.matrix_exp drifts by ~1e-10."""
    d = 2 ** b.n_qubits
    W = torch.eye(d, dtype=torch.complex128)
    for layer in range(b.layers):
        for g in range(b.n_generators):
            ev, U = torch.linalg.eigh(b.generators[g])
            W = (U * torch.exp(-1j * b.theta[layer, g].detach() * ev)) @ U.conj().T @ W
    return W[:, torch.arange(2 ** b.n_register) * 2 ** b.n_assets]


@pytest.mark.parametrize("r,n", [(1, 1), (1, 2), (2, 3), (1, 4)])
def test_fast_columns_match_exact_unitary(r, n):
    torch.manual_seed(1)
    b = EquivariantBuilder(r, n, layers=2, init_scale=0.8)
    S = b.build_columns()
    assert S.shape == (2 ** (r + n), 2 ** r)
    assert torch.allclose(S, _exact_columns(b), atol=1e-12)
    assert torch.allclose(S, b.build()[:, torch.arange(2 ** r) * 2 ** n], atol=1e-8)   # old path, less accurate
    assert torch.allclose(S.conj().T @ S, torch.eye(2 ** r, dtype=torch.complex128), atol=1e-12)


def test_fast_path_gradient_matches_finite_differences():
    Y = np.random.default_rng(0).integers(0, 3, size=(50, 3))
    torch.manual_seed(0)
    b = EquivariantBuilder(1, 3, layers=1, init_scale=0.4)
    m = LocalMeasurement(3, n_assets=3, shared=True, reset=True)

    def loss():
        return -total_loglik_coarse(register_kraus_table(unitary_columns(b), m.rows(), 1, 3), Y, 3, 1)

    loss().backward()
    analytic = b.theta.grad.clone()
    eps = 1e-6
    for g in (0, 3, 4, 7):
        with torch.no_grad():
            b.theta[0, g] += eps
            up = loss().item()
            b.theta[0, g] -= 2 * eps
            dn = loss().item()
            b.theta[0, g] += eps
        assert abs((up - dn) / (2 * eps) - analytic[0, g].item()) < 1e-6


def test_large_basket_builds_without_dense_generators():
    b = EquivariantBuilder(2, 9, layers=2)          # dense generators for 9 assets would need about 5 GB
    S = b.build_columns()
    assert S.shape == (2 ** 11, 4)
    assert b._generators_cache is None
    assert torch.allclose(S.conj().T @ S, torch.eye(4, dtype=torch.complex128), atol=1e-12)
