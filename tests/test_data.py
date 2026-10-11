import numpy as np
import pandas as pd

from data.discretize import Discretizer
from data.download import fill_gaps
from data.features import build_market_data
from data.market_data import MarketData
from harness.folds import make_folds
from tests.helpers import fake_market, fake_prices


def test_no_leakage_in_features():
    """Changing prices after day t must not change any feature up to day t."""
    prices, _ = fake_prices(400)
    full = build_market_data(prices)
    t = 250
    pos = prices.index.get_loc(full.dates[t])  # price row of day t
    cut = prices.copy()
    cut.iloc[pos + 1 :] *= np.random.default_rng(1).uniform(0.5, 2, size=cut.iloc[pos + 1 :].shape)
    other = build_market_data(cut)
    np.testing.assert_array_equal(full.returns[: t + 1], other.returns[: t + 1])
    np.testing.assert_array_equal(full.std_returns[: t + 1], other.std_returns[: t + 1])
    np.testing.assert_array_equal(full.corr[: t + 1], other.corr[: t + 1])
    assert not np.array_equal(full.returns[t + 1 :], other.returns[t + 1 :])  # the test really changed the future


def test_vol_known_before_day_t():
    prices, _ = fake_prices(300)
    md = build_market_data(prices)
    r = np.log(prices).diff().dropna()
    t = 100
    day = md.dates[t]
    pos = r.index.get_loc(day)
    vol = r.iloc[pos - 21 : pos].std().to_numpy()  # days t-21..t-1 only
    np.testing.assert_allclose(md.std_returns[t], md.returns[t] / vol)


def test_market_data_roundtrip(tmp_path):
    md, _ = fake_market(300)
    md.save(tmp_path / "x.npz")
    md2 = MarketData.load(tmp_path / "x.npz")
    assert list(md2.dates) == list(md.dates) and md2.assets == md.assets
    np.testing.assert_array_equal(md.corr, md2.corr)
    assert not np.isnan(md.std_returns).any() and not np.isnan(md.corr).any()


def test_gap_fill_at_most_one_day():
    idx = pd.bdate_range("2020-01-01", periods=8)
    df = pd.DataFrame({"a": [1, 2, np.nan, 4, np.nan, np.nan, 7, 8.0]}, index=idx)
    filled, log = fill_gaps(df, limit=1)
    assert filled["a"].iloc[2] == 2  # 1-day gap filled
    assert np.isnan(filled["a"].iloc[5])  # second consecutive day left missing
    assert sum(g["action"] == "ffill" for g in log) == 2 and sum(g["action"] == "left_missing" for g in log) == 1


def test_discretizer_shares_and_exchangeability():
    x = np.random.default_rng(0).standard_t(4, size=(3000, 6))
    for method in ("quantile", "gmm"):
        d = Discretizer(3, method).fit(x)
        assert d.transform(x).min() == 0 and d.transform(x).max() == 2
    d = Discretizer(3, "quantile").fit(x)
    np.testing.assert_allclose(d.train_shares_, [0.2, 0.6, 0.2], atol=0.005)
    d5 = Discretizer(5, "quantile").fit(x)
    np.testing.assert_allclose(d5.train_shares_, [0.1, 0.2, 0.4, 0.2, 0.1], atol=0.005)
    perm = np.random.default_rng(1).permutation(6)
    np.testing.assert_array_equal(d.transform(x)[:, perm], d.transform(x[:, perm]))


def test_folds_expanding_no_overlap_and_count():
    dates = pd.bdate_range("1999-03-01", "2026-09-30")
    folds = make_folds(dates, "expanding", "2004-12-31", 6)
    assert 42 <= len(folds) <= 45
    for f in folds:
        assert f.train_idx.max() < f.val_idx.min() and f.val_idx.max() < f.test_idx.min()
        assert len(np.intersect1d(f.window_idx, f.test_idx)) == 0
        assert abs(len(f.val_idx) / len(f.window_idx) - 0.2) < 0.01
        assert (dates[f.test_idx[-1]] - dates[f.test_idx[0]]).days < 190
    assert all(len(b.window_idx) >= len(a.window_idx) for a, b in zip(folds[:-1], folds[1:]))
    # tests tile the calendar without overlap
    assert all(a.test_idx.max() < b.test_idx.min() for a, b in zip(folds[:-1], folds[1:]))


def test_folds_rolling_window_length():
    dates = pd.bdate_range("1999-03-01", "2026-09-30")
    folds = make_folds(dates, "rolling", "2004-12-31", 6, rolling_years=5)
    lens = [len(f.window_idx) for f in folds]
    assert max(lens) - min(lens) < 30 and 1200 < np.mean(lens) < 1320
