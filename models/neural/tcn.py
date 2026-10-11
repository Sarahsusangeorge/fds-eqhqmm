"""Temporal Convolutional Network (TCN) baseline for market regime detection (Step A7).

Owner: Abhinav.
Uses dilated causal 1D convolutions ensuring strict no-lookahead causality.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.base import Fold, MarketData, RegimeModel


class _CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, : -self.padding]
        return out


class _TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.conv1 = _CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.conv2 = _CausalConv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        return out + res


class _TCNNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        num_levels: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        layers = []
        c_in = in_features
        for i in range(num_levels):
            dilation = 2**i
            layers.append(
                _TemporalBlock(c_in, hidden_dim, kernel_size, dilation=dilation)
            )
            c_in = hidden_dim
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_features) -> permute to (B, in_features, T)
        x_t = x.transpose(1, 2)
        out_t = self.network(x_t)  # (B, hidden_dim, T)
        out = out_t.transpose(1, 2)  # (B, T, hidden_dim)
        return self.head(out).squeeze(-1)


class TCNModel(RegimeModel):
    """TCN model for regime stress scoring."""

    def __init__(
        self,
        K: int = 3,
        hidden_dim: int = 32,
        num_levels: int = 3,
        kernel_size: int = 3,
        lr: float = 0.01,
        epochs: int = 30,
        batch_size: int = 32,
        window: int = 50,
        name: str = "tcn",
    ):
        self.K = K
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.kernel_size = kernel_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.window = window
        self.name = name
        self.net: _TCNNet | None = None

    def _prepare_inputs(
        self, data: MarketData | None, symbols: np.ndarray
    ) -> np.ndarray:
        Y = np.asarray(symbols)
        T, n = Y.shape
        one_hot = np.zeros((T, n, self.K), dtype=np.float32)
        for k in range(self.K):
            one_hot[:, :, k] = (Y == k).astype(np.float32)
        feats = one_hot.reshape(T, n * self.K)
        if data is not None and data.std_returns is not None:
            feats = np.concatenate(
                [feats, data.std_returns.astype(np.float32)], axis=-1
            )
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
    ) -> "TCNModel":
        torch.manual_seed(seed)
        np.random.seed(seed)
        X_all = self._prepare_inputs(data, symbols)
        targets_all = self._extreme_targets(symbols)

        train_idx = fold.train_idx
        in_dim = X_all.shape[-1]
        self.net = _TCNNet(in_dim, self.hidden_dim, self.num_levels, self.kernel_size)

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
