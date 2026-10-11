"""Permutation operators, symmetric generators, EquivariantBuilder and CorrWeightedBuilder.

Owner: Abhinav. Build steps: A1-A3, A6 (see the Implementation Split doc).
"""

from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn

from models.base import UnitaryBuilder

CDTYPE = torch.complex128
FDTYPE = torch.float64

# Single-qubit Pauli matrices
_I2 = torch.eye(2, dtype=CDTYPE)
_X = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE)
_Y = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE)
_Z = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE)
_PAULIS_1Q = [_I2, _X, _Y, _Z]


def register_paulis(r: int) -> list[torch.Tensor]:
    """Basis {I, X, Y, Z}^r of the 2^r x 2^r register operator space."""
    if r == 0:
        return [torch.ones((1, 1), dtype=CDTYPE)]
    paulis = [_I2, _X, _Y, _Z]
    res = paulis
    for _ in range(r - 1):
        res = [torch.kron(a, b) for a in res for b in paulis]
    return res


def _single_qubit_op(op: torch.Tensor, i: int, n: int) -> torch.Tensor:
    res = torch.ones((1, 1), dtype=CDTYPE)
    for k in range(n):
        res = torch.kron(res, op if k == i else _I2)
    return res


def _two_qubit_op(
    op1: torch.Tensor, op2: torch.Tensor, i: int, j: int, n: int
) -> torch.Tensor:
    res = torch.ones((1, 1), dtype=CDTYPE)
    for k in range(n):
        if k == i:
            res = torch.kron(res, op1)
        elif k == j:
            res = torch.kron(res, op2)
        else:
            res = torch.kron(res, _I2)
    return res


# --------------------------------------------------------------------------- Step A1
def perm_operator(sigma: Sequence[int] | np.ndarray, n: int, r: int) -> torch.Tensor:
    """Dense matrix P_sigma that permutes asset qubits, leaving register untouched.

    Qubit ordering follows models/base.py: r register qubits first, then n asset qubits.
    sigma: permutation of 0..n-1, mapping asset i to sigma[i].
    Guarantees:
      1. P_sigma is unitary: P_sigma^dag @ P_sigma = I.
      2. P_sigma @ P_tau = P_{sigma tau} where (sigma tau)(i) = sigma[tau[i]].
    """
    sigma = list(sigma)
    if len(sigma) != n:
        raise ValueError(f"sigma must have length n={n}, got {len(sigma)}")
    inv_sigma = [0] * n
    for i, s in enumerate(sigma):
        inv_sigma[s] = i

    q = r + n
    d = 2**q
    idx = torch.arange(d)
    reg_bits = [(idx >> (q - 1 - k)) & 1 for k in range(r)]
    asset_bits = [(idx >> (n - 1 - k)) & 1 for k in range(n)]

    new_idx = torch.zeros(d, dtype=torch.long)
    for k in range(r):
        new_idx |= reg_bits[k] << (q - 1 - k)
    for k in range(n):
        new_idx |= asset_bits[inv_sigma[k]] << (n - 1 - k)

    P = torch.zeros((d, d), dtype=CDTYPE)
    P[new_idx, idx] = 1.0
    return P


def permute_obs(y, sigma: Sequence[int] | np.ndarray):
    """Relabel observation vector y according to asset permutation sigma.

    y: (..., n) array or tensor.
    If asset i moves to sigma[i], then position sigma[i] receives y[..., i].
    """
    sigma = list(sigma)
    n = len(sigma)
    inv_sigma = [0] * n
    for i, s in enumerate(sigma):
        inv_sigma[s] = i
    if isinstance(y, torch.Tensor):
        return y[..., inv_sigma]
    y_arr = np.asarray(y)
    return y_arr[..., inv_sigma]


# --------------------------------------------------------------------------- Step A2
def symmetric_generators(n: int, r: int) -> list[torch.Tensor]:
    """Generators of the S_n-equivariant unitary group on r register + n asset qubits.

    For each register Pauli B in {I, X, Y, Z}^r:
      B (x) sum_i X_i
      B (x) sum_i Y_i
      B (x) sum_i Z_i
      B (x) sum_{i<j} Z_i Z_j
      B (x) sum_{i<j} (X_i X_j + Y_i Y_j)

    Each generator is Hermitian and commutes with P_sigma for any permutation sigma.
    """
    paulis_reg = register_paulis(r)

    sum_X = sum(_single_qubit_op(_X, i, n) for i in range(n))
    sum_Y = sum(_single_qubit_op(_Y, i, n) for i in range(n))
    sum_Z = sum(_single_qubit_op(_Z, i, n) for i in range(n))

    if n >= 2:
        sum_ZZ = sum(
            _two_qubit_op(_Z, _Z, i, j, n) for i in range(n) for j in range(i + 1, n)
        )
        sum_XX_YY = sum(
            _two_qubit_op(_X, _X, i, j, n) + _two_qubit_op(_Y, _Y, i, j, n)
            for i in range(n)
            for j in range(i + 1, n)
        )
        asset_ops = [sum_X, sum_Y, sum_Z, sum_ZZ, sum_XX_YY]
    else:
        asset_ops = [sum_X, sum_Y, sum_Z]

    generators = []
    for B in paulis_reg:
        for A in asset_ops:
            G = torch.kron(B, A)
            generators.append(G)
    return generators


# --------------------------------------------------------------------------- Step A3
class EquivariantBuilder(nn.Module, UnitaryBuilder):
    """S_n-equivariant unitary builder for the Q-E model.

    Unitary ansatz:
      V(theta) = product_{layer} product_{g} exp(-1j * theta_{layer, g} * G_g)
    over layers and symmetric generators G_g.
    Because every generator G_g commutes with P_sigma, V(theta) strictly commutes
    with P_sigma: [V(theta), P_sigma] = 0 for any theta and sigma.
    """

    def __init__(
        self, n_register: int, n_assets: int, layers: int = 2, init_scale: float = 0.05
    ):
        super().__init__()
        self.n_register = n_register
        self.n_assets = n_assets
        self.layers = layers
        gens = symmetric_generators(n_assets, n_register)
        self.n_generators = len(gens)
        # Store generators as a stacked buffer (n_gen, d, d)
        self.register_buffer("generators", torch.stack(gens, dim=0))
        self.theta = nn.Parameter(
            init_scale * torch.randn(layers, self.n_generators, dtype=FDTYPE)
        )
        self.time_varying = False

    def n_params(self) -> int:
        return self.theta.numel()

    def build(self, theta=None, step_inputs=None) -> torch.Tensor:
        theta = self.theta if theta is None else theta
        d = 2**self.n_qubits
        V = torch.eye(d, dtype=CDTYPE, device=self.generators.device)
        for layer in range(self.layers):
            for g in range(self.n_generators):
                G = self.generators[g]
                th = theta[layer, g].to(CDTYPE)
                Ug = torch.matrix_exp(-1j * th * G)
                V = Ug @ V
        return V


# --------------------------------------------------------------------------- Step A6
class CorrWeightedBuilder(nn.Module, UnitaryBuilder):
    """Correlation-weighted equivariant unitary builder (Step A6).

    Adds a time-varying pairwise asset coupling:
      H_corr(t) = sum_{i<j} w_ij(t) (I_R (x) Z_i Z_j)
    with w_ij(t) = a * corr_ij(t) + b, where a and b are learned scalar parameters.
    V(t) = exp(-1j * H_corr(t)) @ V_base.
    Equivariant when assets and correlation matrix are permuted together.
    """

    def __init__(
        self, n_register: int, n_assets: int, layers: int = 2, init_scale: float = 0.05
    ):
        super().__init__()
        self.n_register = n_register
        self.n_assets = n_assets
        self.layers = layers
        self.base_builder = EquivariantBuilder(
            n_register, n_assets, layers=layers, init_scale=init_scale
        )
        self.a = nn.Parameter(torch.tensor(0.05, dtype=FDTYPE))
        self.b = nn.Parameter(torch.tensor(0.05, dtype=FDTYPE))
        self.time_varying = True

        # Precompute pairwise ZZ diagonals on (2^(r+n)) space
        # Z_i Z_j is diagonal in computational basis, diagonal values are (+1 or -1)
        q = n_register + n_assets
        d = 2**q
        idx = torch.arange(d)
        asset_bits = [(idx >> (n_assets - 1 - k)) & 1 for k in range(n_assets)]

        pairs = []
        zz_diags = []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                pairs.append((i, j))
                # Z eigenvalues: bit 0 -> +1, bit 1 -> -1. Product: (1 - 2*b_i)*(1 - 2*b_j)
                sign = (1 - 2 * asset_bits[i]) * (1 - 2 * asset_bits[j])
                zz_diags.append(sign.to(FDTYPE))
        self.pairs = pairs
        if zz_diags:
            self.register_buffer(
                "zz_diags", torch.stack(zz_diags, dim=0)
            )  # (n_pairs, d)
        else:
            self.register_buffer("zz_diags", torch.zeros((0, d), dtype=FDTYPE))

    def n_params(self) -> int:
        return self.base_builder.n_params() + 2

    def build(self, theta=None, step_inputs=None) -> torch.Tensor:
        V_base = self.base_builder.build(theta=theta)
        if len(self.pairs) == 0:
            return V_base

        # Extract correlation matrix from step_inputs
        corr = None
        if step_inputs is not None:
            if isinstance(step_inputs, dict):
                corr = step_inputs.get("corr", None)
            elif isinstance(step_inputs, (torch.Tensor, np.ndarray)):
                corr = step_inputs

        d = 2**self.n_qubits
        if corr is not None:
            if isinstance(corr, np.ndarray):
                corr = torch.as_tensor(corr, dtype=FDTYPE, device=self.zz_diags.device)
            else:
                corr = corr.to(dtype=FDTYPE, device=self.zz_diags.device)
            # w_ij = a * corr_ij + b
            weights = []
            for i, j in self.pairs:
                w_ij = self.a * corr[i, j] + self.b
                weights.append(w_ij)
            weights = torch.stack(weights)  # (n_pairs,)
        else:
            # Default when no corr provided: w_ij = b
            weights = self.b.expand(len(self.pairs))

        # H_corr diagonal = sum_pairs w_ij * zz_diag_ij
        diag_H = (weights[:, None] * self.zz_diags).sum(dim=0)  # (d,)
        # exp(-1j * diag_H)
        U_corr_diag = torch.exp(torch.complex(torch.zeros_like(diag_H), -diag_H)).to(
            CDTYPE
        )
        # V = diag(U_corr_diag) @ V_base
        return U_corr_diag[:, None] * V_base


__all__ = [
    "perm_operator",
    "permute_obs",
    "symmetric_generators",
    "register_paulis",
    "EquivariantBuilder",
    "CorrWeightedBuilder",
]
