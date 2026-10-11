"""DeepSets-GRU permutation-invariant neural baseline (Step A8).

Owner: Abhinav.

Architecture:
  1. Shared per-asset MLP phi: processes each asset identically.
  2. Sum over assets: z_t = sum_i phi(x_{t, i}) (strictly permutation-invariant).
  3. GRU temporal model: h_t = GRU(z_t, h_{t-1}).
  4. Output head: s_t = Sigmoid(MLP(h_t)).

Guarantees output unchanged under any asset column order.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.base import Fold, MarketData, RegimeModel


class _DeepSetsGRUNet(nn.Module):
    def __init__(
        self,
        in_per_asset: int,
        phi_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 1,
    ):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_per_asset, phi_dim),
            nn.ReLU(),
            nn.Linear(phi_dim, phi_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(phi_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n, in_per_asset)
        B, T, n, d = x.shape
        # Process each asset through shared phi
        x_flat = x.reshape(B * T * n, d)
        phi_out = self.phi(x_flat).reshape(B, T, n, -1)  # (B, T, n, phi_dim)
        # Sum over asset dimension (permutation invariant pooling)
        z = phi_out.sum(dim=2)  # (B, T, phi_dim)
        # Temporal GRU
        gru_out, _ = self.gru(z)  # (B, T, hidden_dim)
        return self.head(gru_out).squeeze(-1)  # (B, T)


class DeepSetsGRUModel(RegimeModel):
    """Permutation-invariant DeepSets + GRU sequence model."""

    def __init__(
        self,
        K: int = 3,
        phi_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 1,
        lr: float = 0.01,
        epochs: int = 30,
        batch_size: int = 32,
        window: int = 50,
        name: str = "deepsets_gru",
    ):
        self.K = K
        self.phi_dim = phi_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.window = window
        self.name = name
        self.net: _DeepSetsGRUNet | None = None

    def _prepare_inputs(
        self, data: MarketData | None, symbols: np.ndarray
    ) -> np.ndarray:
        """Shape: (T, n, d_per_asset)."""
        Y = np.asarray(symbols)
        T, n = Y.shape
        # One-hot encode per asset: (T, n, K)
        one_hot = np.zeros((T, n, self.K), dtype=np.float32)
        for k in range(self.K):
            one_hot[:, :, k] = (Y == k).astype(np.float32)

        if data is not None and data.std_returns is not None:
            # Append std_returns for each asset: (T, n, 1)
            std_r = data.std_returns[:, :, None].astype(np.float32)
            feats = np.concatenate([one_hot, std_r], axis=-1)
        else:
            feats = one_hot
        return feats

    def _extreme_targets(self, symbols: np.ndarray) -> np.ndarray:
        Y = np.asarray(symbols)
        extreme = ((Y == 0) | (Y == self.K - 1)).mean(axis=-1).astype(np.float32)
        targets = np.zeros_like(extreme)
        targets[:-1] = extreme[1:]
        targets[-1] = extreme[-1]
        return targets

    def fit(
        self, data: MarketData | None, symbols: np.ndarray, fold: Fold, seed: int
    ) -> "DeepSetsGRUModel":
        torch.manual_seed(seed)
        np.random.seed(seed)
        X_all = self._prepare_inputs(data, symbols)  # (T, n, d)
        targets_all = self._extreme_targets(symbols)

        train_idx = fold.train_idx
        in_per_asset = X_all.shape[-1]
        self.net = _DeepSetsGRUNet(
            in_per_asset, self.phi_dim, self.hidden_dim, self.num_layers
        )

        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=1e-4
        )
        criterion = nn.MSELoss()

        T_train = len(train_idx)
        w = min(self.window, T_train)
        X_train_sub = X_all[train_idx]
        y_train_sub = targets_all[train_idx]

        windows_X, windows_y = [], []
        for i in range(T_train - w + 1):
            windows_X.append(X_train_sub[i : i + w])
            windows_y.append(y_train_sub[i : i + w])

        if len(windows_X) == 0:
            windows_X.append(X_train_sub)
            windows_y.append(y_train_sub)

        dataset = TensorDataset(
            torch.as_tensor(np.array(windows_X), dtype=torch.float32),
            torch.as_tensor(np.array(windows_y), dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for _ in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                pred = self.net(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()

        return self

    def score(
        self, data: MarketData | None, symbols: np.ndarray, idx: np.ndarray
    ) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Fit model first before scoring")
        self.net.eval()
        idx = np.asarray(idx)
        max_idx = int(idx.max())

        X_all = self._prepare_inputs(data, symbols)[: max_idx + 1]
        tx = torch.as_tensor(X_all[None], dtype=torch.float32)
        with torch.no_grad():
            preds = self.net(tx)[0].cpu().numpy()
        return preds[idx]

    def n_params(self) -> int:
        if self.net is None:
            raise RuntimeError("Fit model first")
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
