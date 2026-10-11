"""Causal Transformer baseline for market regime detection (Step A7).

Owner: Abhinav.
Uses multi-head self-attention with causal triangular mask preventing future leakage.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.base import Fold, MarketData, RegimeModel


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(
        self, in_features: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_features)
        T = x.size(1)
        h = self.pos_enc(self.in_proj(x))
        # Causal mask: mask[i, j] = True (masked) when j > i
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        out = self.transformer(h, mask=mask, is_causal=True)
        return self.head(out).squeeze(-1)


class TransformerModel(RegimeModel):
    """Causal Transformer sequence model for regime stress scoring."""

    def __init__(
        self,
        K: int = 3,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        lr: float = 0.005,
        epochs: int = 30,
        batch_size: int = 32,
        window: int = 50,
        name: str = "transformer",
    ):
        self.K = K
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.window = window
        self.name = name
        self.net: _TransformerNet | None = None

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
    ) -> "TransformerModel":
        torch.manual_seed(seed)
        np.random.seed(seed)
        X_all = self._prepare_inputs(data, symbols)
        targets_all = self._extreme_targets(symbols)

        train_idx = fold.train_idx
        in_dim = X_all.shape[-1]
        self.net = _TransformerNet(in_dim, self.d_model, self.nhead, self.num_layers)

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
