"""Generic (non-equivariant) unitary builder for the Q-G model (step Sa6).

Owner: Sarah.

Hardware-efficient ansatz on r register qubits + n asset qubits:
each layer applies RY then RZ to every qubit with its OWN angles, then a ring
of CZ gates. Because every qubit has different parameters and the ring has an
order, this circuit does NOT respect asset permutations: it is the matched
"no symmetry" comparison for the equivariant builder.

Qubit order (shared convention): register qubits first, then assets; qubit 0
is the most significant bit of the basis index.
"""
from __future__ import annotations

import torch
from torch import nn

from models.base import UnitaryBuilder

CDTYPE = torch.complex128

def _ry(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a / 2), torch.sin(a / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(CDTYPE)

def _rz(b: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((), dtype=torch.float64)
    e = torch.exp(torch.complex(z, -b / 2))
    return torch.diag(torch.stack([e, e.conj()]))

def cz_ring_diagonal(n_qubits: int) -> torch.Tensor:
    """Diagonal (+1/-1) of a ring of CZ gates between neighbouring qubits."""
    d = 2 ** n_qubits
    idx = torch.arange(d)
    bits = [(idx >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
    sign = torch.ones(d, dtype=torch.float64)
    if n_qubits < 2:
        return sign.to(CDTYPE)
    pairs = [(q, q + 1) for q in range(n_qubits - 1)]
    if n_qubits > 2:
        pairs.append((n_qubits - 1, 0))
    for a, b in pairs:
        sign = sign * (1 - 2 * (bits[a] & bits[b])).to(torch.float64)
    return sign.to(CDTYPE)

class GenericBuilder(nn.Module, UnitaryBuilder):
    """Q-G unitary: `layers` x (per-qubit RY, RZ, then CZ ring).

    Small initial angles (init_scale=0.05) matter: starting close to "do nothing"
    lets the regime memory form during training; large random starts got stuck
    learning only same-day co-movement (see docs/engine_notes.md).
    """

    def __init__(self, n_register: int, n_assets: int, layers: int = 3, init_scale: float = 0.05):
        super().__init__()
        self.n_register, self.n_assets, self.layers = n_register, n_assets, layers
        q = n_register + n_assets
        self.theta = nn.Parameter(init_scale * torch.randn(layers, q, 2, dtype=torch.float64))
        self.register_buffer("cz", cz_ring_diagonal(q))
        self.time_varying = False

    def n_params(self) -> int:
        return self.theta.numel()

    def build(self, theta=None, step_inputs=None) -> torch.Tensor:
        theta = self.theta if theta is None else theta
        q = self.n_qubits
        V = torch.eye(2 ** q, dtype=CDTYPE)
        for layer in range(self.layers):
            U = torch.ones((1, 1), dtype=CDTYPE)
            for k in range(q):
                U = torch.kron(U, _rz(theta[layer, k, 1]) @ _ry(theta[layer, k, 0]))
            V = self.cz[:, None] * (U @ V)
        return V

def matched_layers(target_params: int, n_register: int, n_assets: int) -> int:
    """Number of GenericBuilder layers whose parameter count is closest to target."""
    per_layer = 2 * (n_register + n_assets)
    return max(1, round(target_params / per_layer))

__all__ = ["GenericBuilder", "cz_ring_diagonal", "matched_layers"]
