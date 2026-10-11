"""Tests for neural baselines and Hammond geometric observable model (Steps A7, A8, A10).

Owner: Abhinav.
"""

import numpy as np
import pytest
import torch

from models.base import Fold, MarketData
from models.geometric.hammond import HammondModel
from models.neural.deepsets_gru import DeepSetsGRUModel
from models.neural.gru import GRUModel
from models.neural.lstm import LSTMModel
from models.neural.tcn import TCNModel
from models.neural.transformer import TransformerModel
from synthetic.generate import make_synthetic


@pytest.fixture
def synthetic_sample():
    d = make_synthetic(T=120, n=3, K=3, seed=0)
    fold = Fold(0, np.arange(80), np.arange(80, 100), np.arange(100, 120))
    return d, fold


def test_deepsets_gru_permutation_invariance(synthetic_sample):
    """Step A8: DeepSets-GRU score is unchanged under any column permutation."""
    d, fold = synthetic_sample
    model = DeepSetsGRUModel(K=3, epochs=3, hidden_dim=16, phi_dim=8)
    model.fit(None, d.symbols, fold, seed=42)

    score_orig = model.score(None, d.symbols, fold.test_idx)

    # Permute columns of symbols: [1, 2, 0]
    perm = [1, 2, 0]
    symbols_perm = d.symbols[:, perm]
    score_perm = model.score(None, symbols_perm, fold.test_idx)

    # Should be identical to floating point tolerance
    np.testing.assert_allclose(score_orig, score_perm, atol=1e-5)


def test_standard_neural_models_run(synthetic_sample):
    """Step A7: LSTM, GRU, TCN, and Transformer fit and score properly."""
    d, fold = synthetic_sample

    models = [
        LSTMModel(K=3, epochs=2, hidden_dim=16),
        GRUModel(K=3, epochs=2, hidden_dim=16),
        TCNModel(K=3, epochs=2, hidden_dim=16, num_levels=2),
        TransformerModel(K=3, epochs=2, d_model=16, nhead=2, num_layers=1),
    ]

    for m in models:
        m.fit(None, d.symbols, fold, seed=0)
        scores = m.score(None, d.symbols, fold.test_idx)
        assert len(scores) == len(fold.test_idx)
        assert not np.isnan(scores).any()
        assert m.n_params() > 0


def test_hammond_geometric_observables(synthetic_sample):
    """Step A10: Hammond spectral entropy and Berry phase rate run causally."""
    d, fold = synthetic_sample

    for obs in ("entropy", "berry", "combined"):
        model = HammondModel(observable=obs, corr_window=20, berry_window=5)
        model.fit(None, d.symbols, fold, seed=0)
        assert model.n_params() == 0

        scores = model.score(None, d.symbols, fold.test_idx)
        assert len(scores) == len(fold.test_idx)
        assert not np.isnan(scores).any()
        assert np.all(scores >= 0.0)


def test_no_future_leakage_on_neural_and_hammond(synthetic_sample):
    """Crucial test: modifying data at day t > t_eval does NOT alter score at t_eval."""
    d, fold = synthetic_sample
    t_eval = fold.test_idx[5]

    models = [
        DeepSetsGRUModel(K=3, epochs=1, hidden_dim=8, phi_dim=8),
        LSTMModel(K=3, epochs=1, hidden_dim=8),
        HammondModel(observable="entropy", corr_window=20),
    ]

    for m in models:
        m.fit(None, d.symbols, fold, seed=0)
        score_before = m.score(None, d.symbols, np.array([t_eval]))[0]

        # Modify all symbols AFTER t_eval
        symbols_corrupted = d.symbols.copy()
        symbols_corrupted[t_eval + 1 :] = 0  # wipe out future
        score_after = m.score(None, symbols_corrupted, np.array([t_eval]))[0]

        assert np.isclose(
            score_before, score_after, atol=1e-6
        ), f"Leakage detected in {m.name}: score changed from {score_before} to {score_after}"
