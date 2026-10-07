"""Tests for synthetic/generate.py (owner: Sarah)."""
import numpy as np
import pytest

from synthetic.generate import CALM, STRESS, make_synthetic


def test_shapes_and_ranges():
    d = make_synthetic(T=500, n=4, K=5, seed=1)
    assert d.symbols.shape == (500, 4)
    assert d.returns.shape == (500, 4)
    assert d.regimes.shape == (500,)
    assert d.symbols.min() >= 0 and d.symbols.max() <= 4
    assert set(np.unique(d.regimes)) <= {CALM, STRESS}
    assert len(d.thresholds) == 4


def test_same_seed_same_data():
    a, b = make_synthetic(seed=7), make_synthetic(seed=7)
    np.testing.assert_array_equal(a.symbols, b.symbols)
    np.testing.assert_array_equal(a.regimes, b.regimes)


def test_switches_match_regime_path():
    d = make_synthetic(seed=2)
    for s in d.switches:
        assert d.regimes[s] != d.regimes[s - 1]
    assert len(d.switches) == np.count_nonzero(np.diff(d.regimes))


def test_regimes_are_persistent():
    d = make_synthetic(T=5000, seed=3)
    mean_run = len(d.regimes) / (len(d.switches) + 1)
    assert mean_run > 10  # stay probabilities 0.95-0.98 give long runs


def test_both_regimes_appear():
    d = make_synthetic(T=5000, seed=4)
    assert 0.1 < d.regimes.mean() < 0.9


def test_stress_has_more_extreme_symbols():
    d = make_synthetic(T=5000, K=3, seed=5)
    extreme = (d.symbols == 0) | (d.symbols == 2)
    assert extreme[d.regimes == STRESS].mean() > extreme[d.regimes == CALM].mean() + 0.15


def test_calm_symbols_roughly_balanced():
    d = make_synthetic(T=20000, K=3, seed=6)
    calm = d.symbols[d.regimes == CALM]
    shares = np.bincount(calm.ravel(), minlength=3) / calm.size
    np.testing.assert_allclose(shares, 1 / 3, atol=0.03)


def test_assets_are_exchangeable():
    d = make_synthetic(T=20000, n=4, K=3, seed=8)
    shares = np.stack([np.bincount(d.symbols[:, i], minlength=3) / len(d.symbols) for i in range(4)])
    assert np.abs(shares - shares.mean(axis=0)).max() < 0.03


def test_bad_arguments_rejected():
    with pytest.raises(ValueError):
        make_synthetic(p_stay_calm=1.0)
    with pytest.raises(ValueError):
        make_synthetic(K=1)
