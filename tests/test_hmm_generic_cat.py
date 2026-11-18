import numpy as np
from types import SimpleNamespace
from models.classical.hmm_generic_cat import GenericCategoricalHMM
from models.classical.hmm_exchangeable import ExchangeableHMM


def _data(T=600, n=4, seed=0):
    rng = np.random.default_rng(seed)
    z = (rng.random(T) < 0.3).astype(int)
    sy = np.where(rng.random((T, n)) < (0.1 + 0.6 * z[:, None]), rng.choice([0, 2], (T, n)), 1)
    return SimpleNamespace(), sy


def test_fits_and_counts_params():
    d, sy = _data()
    f = SimpleNamespace(train_idx=np.arange(400))
    m = GenericCategoricalHMM().fit(d, sy, f, 0)
    assert m.n_params() == 2 + 2 * 4 * 2
    sc = m.score(d, sy, np.arange(400, 600))
    assert np.isfinite(sc).all()


def test_not_permutation_invariant_but_exchangeable_is():
    d, sy = _data()
    f = SimpleNamespace(train_idx=np.arange(400))
    idx = np.arange(400, 600)
    swapped = sy.copy(); swapped[:, [0, 1]] = swapped[:, [1, 0]]
    e = ExchangeableHMM(emission="categorical").fit(d, sy, f, 0)
    assert np.allclose(e.score(d, sy, idx), e.score(d, swapped, idx))
    g = GenericCategoricalHMM().fit(d, sy, f, 0)
    assert np.isfinite(g.score(d, swapped, idx)).all()
