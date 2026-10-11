"""Tests for permutation operators, symmetric generators, and EquivariantBuilder (step A4).

Owner: Abhinav.
Verifies:
  1. Permutation operators: unitarity and group homomorphism P_sigma P_tau = P_{sigma tau}.
  2. Observation permutation: permute_obs consistency.
  3. Symmetric generators: Hermiticity and exact commutation [G, P_sigma] = 0.
  4. EquivariantBuilder: unitarity and [V(theta), P_sigma] = 0 to 1e-10 for random theta and sigma.
  5. CorrWeightedBuilder: equivariance under joint permutation of assets and correlation matrix.
  6. Commutant dimension: rank of the S_n-averaging map on random matrices equals 4^r * C(n+3, 3) for n=2..5.
  7. Dynamical Lie algebra (DLA) dimension: nested commutators generate the full traceless commutant for n <= 4.
"""

import itertools
import math

import numpy as np
import pytest
import torch

from models.hqmm.equivariant import (
    CorrWeightedBuilder,
    EquivariantBuilder,
    perm_operator,
    permute_obs,
    symmetric_generators,
)

C = torch.complex128


def test_perm_operator_unitarity_and_homomorphism():
    """Step A1: P_sigma is unitary and P_sigma P_tau = P_{sigma tau}."""
    for n in (2, 3):
        for r in (0, 1):
            d = 2 ** (r + n)
            sigmas = list(itertools.permutations(range(n)))
            for s in sigmas:
                Ps = perm_operator(s, n, r)
                # Unitarity
                assert torch.allclose(
                    Ps.conj().T @ Ps, torch.eye(d, dtype=C), atol=1e-12
                )
            # Homomorphism on pairs
            for s in sigmas[:4]:
                Ps = perm_operator(s, n, r)
                for t in sigmas[:4]:
                    Pt = perm_operator(t, n, r)
                    st = [s[t[i]] for i in range(n)]
                    Pst = perm_operator(st, n, r)
                    assert torch.allclose(Ps @ Pt, Pst, atol=1e-12)


def test_permute_obs_consistency():
    """Check permute_obs relabels observations consistently for 1D, 2D, and 3D arrays."""
    y = np.array([10, 20, 30])
    sigma = [1, 2, 0]  # asset 0->1, 1->2, 2->0
    y_perm = permute_obs(y, sigma)
    # Output position sigma[i] should receive y[i]
    # sigma[0]=1 -> y_perm[1] == 10
    # sigma[1]=2 -> y_perm[2] == 20
    # sigma[2]=0 -> y_perm[0] == 30
    np.testing.assert_array_equal(y_perm, [30, 10, 20])

    # Batched 2D
    Y = np.array([[10, 20, 30], [1, 2, 3]])
    Y_perm = permute_obs(Y, sigma)
    np.testing.assert_array_equal(Y_perm, [[30, 10, 20], [3, 1, 2]])

    # Torch tensor
    Y_t = torch.tensor([[10, 20, 30], [1, 2, 3]])
    Y_t_perm = permute_obs(Y_t, sigma)
    assert torch.equal(Y_t_perm, torch.tensor([[30, 10, 20], [3, 1, 2]]))


def test_symmetric_generators_hermitian_and_commute():
    """Step A2: Each generator Hermitian and [G, P_sigma] = 0."""
    for n in (2, 3):
        for r in (0, 1):
            gens = symmetric_generators(n, r)
            for G in gens:
                assert torch.allclose(G, G.conj().T, atol=1e-12)

            sigmas = list(itertools.permutations(range(n)))
            for s in sigmas:
                P = perm_operator(s, n, r)
                for G in gens:
                    comm = G @ P - P @ G
                    assert torch.allclose(comm, torch.zeros_like(comm), atol=1e-12)


def test_equivariant_builder_commutes_with_permutations():
    """Step A3: [V(theta), P_sigma] = 0 to 1e-10 for random theta and sigma."""
    torch.manual_seed(42)
    for n in (2, 3):
        for r in (0, 1):
            builder = EquivariantBuilder(r, n, layers=2)
            d = 2 ** (r + n)
            V = builder.build()
            # Unitary
            assert torch.allclose(V.conj().T @ V, torch.eye(d, dtype=C), atol=1e-9)

            sigmas = list(itertools.permutations(range(n)))
            for s in sigmas:
                P = perm_operator(s, n, r)
                comm = V @ P - P @ V
                assert torch.allclose(comm, torch.zeros_like(comm), atol=1e-10)


def test_corr_weighted_builder_equivariance():
    """Step A6: P_sigma V(C) P_sigma^dag = V(C') under joint asset and correlation permutation."""
    torch.manual_seed(42)
    n = 3
    r = 1
    builder = CorrWeightedBuilder(r, n, layers=1)
    d = 2 ** (r + n)

    # Random correlation matrix C
    rng = np.random.default_rng(123)
    A = rng.normal(size=(n, n))
    C_mat = np.corrcoef(A)

    V = builder.build(step_inputs=C_mat)
    assert torch.allclose(V.conj().T @ V, torch.eye(d, dtype=C), atol=1e-9)

    sigmas = list(itertools.permutations(range(n)))
    for s in sigmas:
        P = perm_operator(s, n, r)
        # Permuted correlation matrix: C'[s[i], s[j]] = C[i, j]
        # Equivalently: C' = P_mat @ C @ P_mat^T
        C_perm = np.zeros_like(C_mat)
        for i in range(n):
            for j in range(n):
                C_perm[s[i], s[j]] = C_mat[i, j]

        V_perm = builder.build(step_inputs=C_perm)
        # Check P_sigma @ V(C) @ P_sigma^dag == V(C_perm)
        V_transformed = P @ V @ P.conj().T
        assert torch.allclose(V_transformed, V_perm, atol=1e-10)


def test_commutant_dimension():
    """Step A4: Rank of S_n-averaging map equals 4^r * C(n+3, 3) for n=2..5."""
    for r in (0, 1):
        for n in range(2, 6):
            expected_dim = (4**r) * math.comb(n + 3, 3)
            d = 2 ** (r + n)
            sigmas = list(itertools.permutations(range(n)))
            n_fact = len(sigmas)
            Ps = [perm_operator(s, n, r) for s in sigmas]

            K = expected_dim + 12
            torch.manual_seed(0)
            projected = []
            for _ in range(K):
                M = torch.randn(d, d, dtype=C)
                P_M = sum(P @ M @ P.conj().T for P in Ps) / n_fact
                projected.append(P_M.reshape(-1))

            stacked = torch.stack(projected, dim=0)
            svals = torch.linalg.svdvals(stacked)
            rank = int((svals > svals[0] * 1e-6).sum())
            assert (
                rank == expected_dim
            ), f"Failed for r={r}, n={n}: got {rank}, expected {expected_dim}"


def _compute_dla_dimension(generators: list[torch.Tensor], tol: float = 1e-8) -> int:
    d = generators[0].shape[0]

    def to_real_vec(M: torch.Tensor) -> torch.Tensor:
        return torch.cat([M.real.reshape(-1), M.imag.reshape(-1)])

    def from_real_vec(v: torch.Tensor) -> torch.Tensor:
        half = d * d
        return torch.complex(v[:half].reshape(d, d), v[half:].reshape(d, d))

    elements = [-1j * G for G in generators]
    basis = []

    for elem in elements:
        v = to_real_vec(elem)
        for b in basis:
            v = v - torch.dot(b, v) * b
        norm = torch.linalg.norm(v)
        if norm > tol:
            basis.append(v / norm)

    new_found = True
    while new_found:
        new_found = False
        n_curr = len(basis)
        basis_matrices = [from_real_vec(b) for b in basis]
        candidates = []
        for i in range(n_curr):
            for j in range(i + 1, n_curr):
                comm = (
                    basis_matrices[i] @ basis_matrices[j]
                    - basis_matrices[j] @ basis_matrices[i]
                )
                v = to_real_vec(comm)
                for b in basis:
                    v = v - torch.dot(b, v) * b
                for c in candidates:
                    v = v - torch.dot(c, v) * c
                norm = torch.linalg.norm(v)
                if norm > tol:
                    candidates.append(v / norm)
        if candidates:
            for c in candidates:
                for b in basis:
                    c = c - torch.dot(b, c) * b
                norm = torch.linalg.norm(c)
                if norm > tol:
                    basis.append(c / norm)
                    new_found = True
    return len(basis)


def test_dynamical_lie_algebra_dimension():
    """Step A4: DLA dimension equals the traceless commutant dimension 4^r * C(n+3, 3) - 1 for n <= 3."""
    expected = {
        (0, 2): 9,
        (0, 3): 19,
        (1, 2): 39,
        (1, 3): 79,
    }
    for (r, n), exp_dim in expected.items():
        gens = symmetric_generators(n, r)
        dla_dim = _compute_dla_dimension(gens)
        assert (
            dla_dim == exp_dim
        ), f"Failed DLA for r={r}, n={n}: got {dla_dim}, expected {exp_dim}"
