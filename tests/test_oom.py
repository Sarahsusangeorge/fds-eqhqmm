import numpy as np
from sklearn.metrics import average_precision_score

from models.base import Fold
from models.oom.spectral import SpectralOOM, extreme_count
from tests.helpers import fake_market


def _setup(T=2500, n=3, seed=2):
    md, state = fake_market(T=T, n=n, seed=seed)
    state = state[len(state) - md.T:]
    sc = md.returns[:1800].std(axis=0, keepdims=True)
    z = md.returns / sc
    sym = np.digitize(z, np.quantile(z[:1800].ravel(), [0.2, 0.8]))
    fold = Fold(0, np.arange(1500), np.arange(1500, 1800), np.arange(1800, md.T))
    return md, state, sym, fold


def test_extreme_count_ignores_asset_order():
    s = np.array([[0, 1, 2], [1, 1, 1], [2, 0, 1]])
    assert extreme_count(s, 3).tolist() == [2, 0, 2]
    assert (extreme_count(s[:, ::-1], 3) == extreme_count(s, 3)).all()


def test_oom_finite_and_beats_chance():
    md, state, sym, fold = _setup()
    m = SpectralOOM().fit(md, sym, fold, 0)
    s = m.score(md, sym, fold.test_idx)
    assert np.isfinite(s).all()
    assert average_precision_score(state[fold.test_idx], s) > state[fold.test_idx].mean() + 0.2


def test_oom_score_is_causal():
    md, state, sym, fold = _setup()
    m = SpectralOOM().fit(md, sym, fold, 0)
    idx = np.arange(1800, 1900)
    a = m.score(md, sym, idx)
    sym2 = sym.copy()
    sym2[1901:] = 2
    assert np.allclose(a, m.score(md, sym2, idx))
    assert np.allclose(m.surprise(md, sym, idx), m.surprise(md, sym2, idx))
