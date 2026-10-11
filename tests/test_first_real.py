"""Tests for the data preparation in experiments/hqmm_first_real.py (owner: Sarah).

Uses made-up prices, so no internet is needed."""
import numpy as np
import pandas as pd

from experiments.hqmm_first_real import log_returns, make_symbols, volatility_label


def fake_prices(T=600, n=3, seed=0):
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(T) > 400, 0.03, 0.01)[:, None]          # volatility jumps late on
    r = rng.normal(0, 1, (T, n)) * vol
    idx = pd.bdate_range("2004-01-01", periods=T)
    return pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=idx, columns=["A", "B", "C"])


def test_symbols_shape_range_and_balance_on_training():
    ret = log_returns(fake_prices())
    train = np.arange(len(ret)) < 300
    s, z, thr = make_symbols(ret, train, K=3)
    assert s.shape == ret.shape and s.min() >= 0 and s.max() <= 2
    shares = np.bincount(s[train].ravel(), minlength=3) / s[train].size
    np.testing.assert_allclose(shares, 1 / 3, atol=0.02)


def test_symbols_ignore_data_after_training():
    ret = log_returns(fake_prices())
    train = np.arange(len(ret)) < 300
    s1, _, _ = make_symbols(ret, train)
    ret2 = ret.copy()
    ret2.iloc[300:] *= 5                                              # change only the future
    s2, _, _ = make_symbols(ret2, train)
    np.testing.assert_array_equal(s1[:300], s2[:300])


def test_volatility_jump_shows_up_as_more_extreme_symbols():
    ret = log_returns(fake_prices())
    train = np.arange(len(ret)) < 300
    s, _, _ = make_symbols(ret, train)
    extreme = ((s == 0) | (s == 2)).mean(1)
    assert extreme[450:].mean() > extreme[:300].mean() + 0.15         # not cancelled by scaling


def test_volatility_label_is_causal():
    ret = log_returns(fake_prices())
    train = np.arange(len(ret)) < 300
    a, _, _ = volatility_label(ret, train)
    ret2 = ret.copy()
    ret2.iloc[500:] *= 3
    b, _, _ = volatility_label(ret2, train)
    np.testing.assert_array_equal(a[:500], b[:500])
    assert a[450:].mean() > 0.5
