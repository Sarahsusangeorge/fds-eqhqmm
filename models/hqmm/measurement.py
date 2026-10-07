"""Shared local measurement {M_k} for the HQMM (step Sa2).

Owner: Sarah.

Each asset qubit is "read" with K operators M_0..M_{K-1} (one per symbol:
for K = 3, down / flat / up). For the model to give valid probabilities they
must satisfy  sum_k M_k^dagger M_k = I  (completeness).

Two forms are available:
  general (reset=False): any K operators satisfying completeness;
  reset   (reset=True) : M_k = |0><v_k|, i.e. read the asset, then reset it to |0>.
                         All memory then lives in the regime register, and the
                         engine can track the small register state only (fast).

We guarantee completeness exactly by stacking the K operators into a (2K x 2) matrix W
with orthonormal columns (an isometry, i.e. a point on the Stiefel manifold):
W^dagger W = I  is the same equation as completeness. W is obtained from a
free complex matrix A by QR decomposition, so A can be trained with ordinary
gradient descent and completeness never breaks.
"""
from __future__ import annotations

import torch
from torch import nn

CDTYPE = torch.complex128


def _isometry(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    """Map free real tensors (..., 2K, 2) to an isometry W with W^dagger W = I."""
    A = torch.complex(re, im)
    Q, R = torch.linalg.qr(A, mode="reduced")
    # Fix the sign/phase freedom of QR so the map is continuous: make diag(R) real positive.
    d = torch.diagonal(R, dim1=-2, dim2=-1)
    phase = d / d.abs().clamp_min(1e-12)
    return Q * phase.unsqueeze(-2)


class LocalMeasurement(nn.Module):
    """K single-qubit Kraus operators, shared by all assets or one set per asset.

    shared=True  (equivariant model): one set {M_k} used for every asset.
    shared=False (generic model):     a separate set for each of n_assets assets.
    """

    def __init__(self, K: int, n_assets: int | None = None, shared: bool = True,
                 init_scale: float = 1.0, reset: bool = False):
        super().__init__()
        if K < 2:
            raise ValueError("K must be at least 2")
        if not shared and n_assets is None:
            raise ValueError("n_assets is required when shared=False")
        self.K, self.shared, self.n_assets, self.reset = K, shared, n_assets, reset
        rows = K if reset else 2 * K
        shape = (rows, 2) if shared else (n_assets, rows, 2)
        self.re = nn.Parameter(init_scale * torch.randn(shape, dtype=torch.float64))
        self.im = nn.Parameter(init_scale * torch.randn(shape, dtype=torch.float64))

    def rows(self) -> torch.Tensor:
        """Reset mode only: the read-out rows <v_k|, shape (K, 2) or (n, K, 2).

        M_k = |0><v_k|; completeness sum_k M_k^dag M_k = R^dag R = I holds because
        R has orthonormal columns.
        """
        if not self.reset:
            raise RuntimeError("rows() is only defined in reset mode")
        return _isometry(self.re, self.im)

    def operators(self) -> torch.Tensor:
        """Return M with shape (K, 2, 2) if shared, else (n_assets, K, 2, 2)."""
        if self.reset:
            R = self.rows()
            M = torch.zeros(R.shape[:-1] + (2, 2), dtype=R.dtype)
            M = M.clone()
            M[..., 0, :] = R                        # M_k = |0><v_k|: read, then reset to |0>
            return M
        W = _isometry(self.re, self.im)
        return W.reshape(W.shape[:-2] + (self.K, 2, 2)).contiguous()

    def n_params(self) -> int:
        return self.re.numel() + self.im.numel()
