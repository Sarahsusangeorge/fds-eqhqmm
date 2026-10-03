"""Checks that the shared contracts in models/base.py behave as agreed."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from models.base import Fold, MarketData, RegimeModel
from harness.runner import log_run


def _toy(T=10, n=3, with_corr=True):
    rng = np.random.default_rng(0)
    r = rng.normal(size=(T, n))
    return MarketData(
        dates=pd.bdate_range("2020-01-01", periods=T),
        assets=[f"A{i}" for i in range(n)],
        returns=r,
        std_returns=r,
        corr=np.tile(np.eye(n), (T, 1, 1)) if with_corr else None,
    )


def test_marketdata_shapes_checked():
    d = _toy()
    assert (d.T, d.n) == (10, 3)
    with pytest.raises(ValueError):
        MarketData(d.dates, d.assets, d.returns[:, :2], d.std_returns)


def test_marketdata_roundtrip():
    for with_corr in (True, False):
        d = _toy(with_corr=with_corr)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "toy.npz")
            d.save(p)
            e = MarketData.load(p)
        assert e.assets == d.assets
        np.testing.assert_allclose(e.returns, d.returns)
        assert (e.corr is None) == (d.corr is None)


def test_fold_rejects_leakage():
    with pytest.raises(ValueError):
        Fold(0, np.arange(10), np.arange(8, 10), np.arange(5, 12))
    Fold(0, np.arange(10), np.arange(8, 10), np.arange(10, 15))


def test_regimemodel_is_abstract():
    with pytest.raises(TypeError):
        RegimeModel()


def test_log_run_writes_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "runs.jsonl")
        log_run("toy", 0, 1, {"k": 3}, {"far": 0.5}, author="test", path=p)
        with open(p) as f:
            assert len(f.readlines()) == 1
