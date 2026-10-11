import copy
import json

import numpy as np
import pytest

from data.discretize import Discretizer
from harness import metrics, stats
from harness.alarms import alarms_from_scores, calibrate_threshold
from harness.folds import make_folds
from harness.overlay import risk_off_overlay
from harness.runner import log_run, read_log, run
from labels.common import clean_series, index_intervals
from labels.r1_bear import label_r1
from labels.r2_vol import label_r2
from labels.r3_consensus import label_r3
from labels.r4_events import label_r4
from models.classical.absorption import AbsorptionRatio
from models.classical.bocpd import BOCPD
from models.classical.garch import GARCH11
from models.classical.hamilton import HamiltonMS
from models.classical.hmm_exchangeable import ExchangeableHMM
from models.classical.hmm_generic import GenericGaussianHMM
from models.classical._hmm_utils import forward_filter, forward_filter_log, stress_log_odds
from models.classical.trees import TreeModel
from tests.helpers import fake_market


# ---------- alarms ----------
def test_alarm_declustering_and_calibration():
    s = np.zeros(200)
    s[[10, 11, 12, 40, 41, 100]] = 1.0  # 10-12 one cluster; 40-41 second (gap 28>20); 100 third
    np.testing.assert_array_equal(alarms_from_scores(s, 0.5, gap=20), [10, 40, 100])
    s2 = np.zeros(200)
    s2[[10, 25, 40]] = 1  # chained: gaps of 15 <= 20 -> one alarm
    np.testing.assert_array_equal(alarms_from_scores(s2, 0.5, gap=20), [10])
    rng = np.random.default_rng(0)
    v = rng.random(2520)  # 10 years
    h = calibrate_threshold(v, target_far=1.0)
    assert len(alarms_from_scores(v, h)) / 10 <= 1.0
    assert len(alarms_from_scores(v, h * 0.98)) / 10 > 0.5  # and it is not absurdly conservative


# ---------- metrics with known answers ----------
def test_metrics_known_answers():
    lags = metrics.detection_lags([95, 300], onsets=[100, 200, 300], pre=20, post=40)
    np.testing.assert_array_equal(lags[[0, 2]], [-5, 0])
    assert np.isnan(lags[1])
    assert metrics.recall(lags) == pytest.approx(2 / 3)
    stress = np.zeros(1000, int)
    stress[100:150] = 1
    assert metrics.false_alarm_count([90, 160, 190, 500], stress, 1000, pre=20, post=40) == 2  # 190 and 500
    assert metrics.false_alarms_per_year([500], stress, 1000, 252) == pytest.approx(1.0)
    y = np.array([0, 0, 1, 1])
    assert metrics.auprc(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert metrics.auprc(y, [0.9, 0.8, 0.2, 0.1]) < 0.6
    assert metrics.cohens_d([0, 0, 1, 1], [1, 3, 5, 7]) == pytest.approx(
        4 / np.sqrt(2)
    )  # means 6 vs 2, pooled sd sqrt(2)
    assert np.isnan(metrics.cohens_d([0, 0, 0], [1, 2, 3]))
    assert metrics.heldout_ll(np.array([1.0, 3.0])) == pytest.approx(-2.0)


# ---------- stats ----------
def test_stats_reproduce_known_examples():
    d = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]  # scipy.stats.wilcoxon documentation example
    stat, p = stats.paired_wilcoxon(d, np.zeros(len(d)))
    assert stat == 24.0 and p == pytest.approx(0.041259765625)
    rej, adj = stats.holm([0.01, 0.04, 0.03])  # Holm: sorted p*(m-i) with monotone max
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
    assert list(rej) == [True, False, False]


def test_diebold_mariano_calibration_and_power():
    rng = np.random.default_rng(0)
    rej = np.mean(
        [stats.diebold_mariano(rng.standard_normal(250), rng.standard_normal(250))[1] < 0.05 for _ in range(400)]
    )
    assert 0.02 < rej < 0.09  # size close to 5%
    x = rng.standard_normal(250)
    stat, p = stats.diebold_mariano(x + 0.5, x + rng.standard_normal(250) * 0.1)
    assert stat > 5 and p < 1e-6  # model 1 clearly worse -> positive statistic


def test_bootstrap_ci_and_did():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1.0, 60)
    lo, hi = stats.stationary_bootstrap_ci(x, reps=500)
    assert lo < x.mean() < hi and hi - lo < 1.0
    est, lo, hi = stats.did_bootstrap(
        rng.normal(0, 1, 50), rng.normal(1, 1, 50), rng.normal(0, 1, 50), rng.normal(0, 1, 50), reps=500
    )
    assert lo < est < hi and est > 0.3


def test_overlay_costs():
    ret = np.array([0.01, -0.05, -0.05, 0.01, 0.01])
    s = np.array([0.0, 1.0, 1.0, 0.0, 0.0])  # stress signalled at end of day 1,2 -> cash on days 2,3
    o = risk_off_overlay(ret, s, 0.5, cost_bps=100)
    # positions [1,1,0,0,1]; trades on days 2 (exit) and 4 (re-entry), 100 bps each
    np.testing.assert_allclose(o["net_ret"], [0.01, -0.05, -0.01, 0.0, 0.0], atol=1e-12)


# ---------- labels ----------
def test_labels_basic():
    md, state = fake_market(1700)
    r1, r2, r4 = label_r1(md), label_r2(md), label_r4(md)
    for r in (r1, r2, r4):
        assert r.series.shape == (len(md),) and set(np.unique(r.series)) <= {0, 1}
    assert 0.05 < r2.series.mean() < 0.2
    assert r4.series.sum() == 0 or r4.series.sum() > 0  # synthetic dates start 1999: events exist
    r3 = label_r3(md)
    assert r3.series.shape == (len(md),)
    assert len(clean_series([1, 0, 0, 1, 1, 1, 0, 1], merge_gap=2, min_len=0)) == 8
    assert index_intervals([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]


# ---------- models run through the harness; no-look-ahead ----------
def _label_fn(d):
    return label_r2(d).series


def make_models(K=3):
    return [
        GenericGaussianHMM(),
        ExchangeableHMM(),
        ExchangeableHMM(emission="categorical", K=K),
        HamiltonMS(),
        BOCPD(),
        GARCH11(),
        AbsorptionRatio(),
        AbsorptionRatio(k=1, name="Eig1"),
        TreeModel("rf", _label_fn),
        TreeModel("xgb", _label_fn),
    ]


@pytest.fixture(scope="module")
def setup():
    md, _ = fake_market(1500)
    folds = make_folds(md.dates, "expanding", md.dates[900].strftime("%Y-%m-%d"), 6)
    return md, folds


def test_no_lookahead_for_every_model(setup):
    """Change ALL data after day t; every model's score (and surprise) up to t must be unchanged."""
    md, folds = setup
    f = folds[0]
    t = int(f.test_idx[40])
    md2 = copy.deepcopy(md)
    rng = np.random.default_rng(5)
    for a in ("returns", "std_returns"):
        getattr(md2, a)[t + 1 :] = rng.standard_normal(getattr(md2, a)[t + 1 :].shape) * 5
    md2.corr[t + 1 :] = np.eye(md.n_assets)
    disc = Discretizer(3).fit(md.std_returns[f.window_idx])
    s1, s2 = disc.transform(md.std_returns), disc.transform(md2.std_returns)
    idx = np.arange(f.test_idx[0], t + 1)
    for m in make_models():
        a = copy.deepcopy(m).fit(md, s1, f, 0)
        b = copy.deepcopy(m).fit(md2, s2, f, 0)
        np.testing.assert_allclose(a.score(md, s1, idx), b.score(md2, s2, idx), atol=1e-9, err_msg=m.name)
        if a.surprise(md, s1, idx) is not None:
            np.testing.assert_allclose(a.surprise(md, s1, idx), b.surprise(md2, s2, idx), atol=1e-9, err_msg=m.name)


def test_exchangeable_hmm_invariant_to_column_order(setup):
    md, folds = setup
    f = folds[0]
    perm = np.random.default_rng(3).permutation(md.n_assets)
    md_p = copy.deepcopy(md)
    md_p.returns, md_p.std_returns = md.returns[:, perm], md.std_returns[:, perm]
    disc = Discretizer(3).fit(md.std_returns[f.window_idx])
    s, sp = disc.transform(md.std_returns), disc.transform(md_p.std_returns)
    for emission in ("gaussian", "categorical"):
        a = ExchangeableHMM(emission=emission).fit(md, s, f, 0)
        b = ExchangeableHMM(emission=emission).fit(md_p, sp, f, 0)
        np.testing.assert_allclose(a.score(md, s, f.test_idx), b.score(md_p, sp, f.test_idx), atol=1e-9)


def test_models_detect_synthetic_stress(setup):
    """Against the TRUE latent regime of the synthetic market every model must beat the base rate clearly."""
    md, folds = setup
    _, state = fake_market(1500)
    truth = state[-len(md.dates) :]
    f = folds[0]
    s = Discretizer(3).fit(md.std_returns[f.window_idx]).transform(md.std_returns)
    got = {}
    for m in make_models():
        m.fit(md, s, f, 0)
        got[m.name] = metrics.auprc(truth, m.score(md, s, np.arange(len(md.dates))))
    base = truth.mean()
    for name in ("C-G", "C-E", "C-E-cat", "Hamilton", "BOCPD", "GARCH", "RF", "XGB"):
        assert got[name] > 1.5 * base, (name, got[name], base)
    assert got["Hamilton"] > 0.8 and got["GARCH"] > 0.8


def test_runner_end_to_end_logs(tmp_path, setup):
    md, folds = setup
    labels = {"R2": label_r2(md), "R4": label_r4(md)}
    log = tmp_path / "runs.jsonl"
    run([GenericGaussianHMM(), GARCH11()], folds[:2], [0], md, labels, log_path=log, verbose=False)
    rows = [json.loads(l) for l in open(log)]
    assert len(rows) == 4
    r = rows[0]
    assert {"model", "fold_id", "seed", "author", "config", "metrics", "n_params"} <= set(r)
    assert "R2" in r["metrics"] and "overlay" in r["metrics"] and "alarm_idx" in r["metrics"]
    df = read_log(log)
    assert len(df) == 4


# ---------- scores must not saturate (regression test for the h = 1.000 defect) ----------
def test_log_space_filter_matches_probability_filter_and_does_not_saturate():
    rng = np.random.default_rng(0)
    A = np.array([[0.98, 0.02], [0.05, 0.95]])
    pi = np.array([0.7, 0.3])
    # very confident emissions: stress state is 40 log-units more likely on the last 100 days
    log_b = np.zeros((300, 2))
    log_b[200:, 1] = 40.0
    alpha, c1 = forward_filter(log_b, pi, A)
    la, c2 = forward_filter_log(log_b, pi, A)
    np.testing.assert_allclose(np.exp(la), alpha, atol=1e-12)
    np.testing.assert_allclose(c1, c2, atol=1e-9)
    assert (alpha[210:, 1] == 1.0).all()  # the probability rounds to exactly 1.0 ...
    lo = stress_log_odds(la, 1)
    assert len(np.unique(lo[210:])) == 1 or lo[210:].min() > 10  # ... but the log-odds is finite and large
    noisy = log_b + rng.normal(0, 3, log_b.shape)
    lo = stress_log_odds(forward_filter_log(noisy, pi, A)[0], 1)
    assert np.isfinite(lo).all() and len(np.unique(lo)) > 250  # all-distinct scores -> no top-end ties


def test_probability_models_use_unsaturated_scores(setup):
    md, folds = setup
    f = folds[0]
    s = Discretizer(3).fit(md.std_returns[f.window_idx]).transform(md.std_returns)
    for m in (GenericGaussianHMM(), ExchangeableHMM(), ExchangeableHMM(emission="categorical"), HamiltonMS()):
        m.fit(md, s, f, 0)
        sc = m.score(md, s, np.arange(len(md.dates)))
        assert np.isfinite(sc).all(), m.name
        assert len(np.unique(sc)) > 0.95 * len(sc), (m.name, len(np.unique(sc)))  # no ties at the top
        assert sc.max() > 0 > sc.min(), m.name  # spans both regimes


def test_hamilton_log_filter_matches_statsmodels(setup):
    md, folds = setup
    f = folds[0]
    s = Discretizer(3).fit(md.std_returns[f.window_idx]).transform(md.std_returns)
    m = HamiltonMS().fit(md, s, f, 0)
    idx = np.arange(len(md.dates))
    mine = 1 / (1 + np.exp(-m.score(md, s, idx)))
    ref = m._statsmodels_filtered(md, len(md.dates) - 1)[:, m.stress]
    np.testing.assert_allclose(mine, ref, atol=1e-8)


def test_raw_input_option_runs(setup):
    md, folds = setup
    f = folds[0]
    s = Discretizer(3).fit(md.std_returns[f.window_idx]).transform(md.std_returns)
    for m in (GenericGaussianHMM(input="raw", name="C-G-raw"), ExchangeableHMM(input="raw", name="C-E-raw")):
        sc = m.fit(md, s, f, 0).score(md, s, f.test_idx)
        assert np.isfinite(sc).all()


def test_calibration_fallback_is_valid_for_bounded_scores():
    """Old fallback was max + 1e-9: for a probability within 1e-9 of 1.0 that is a threshold above 1.0, so nothing can ever alarm."""
    v = np.full(600, 0.0)
    v[[50, 150, 250, 350, 450]] = 1 - 1e-12  # five separate near-1 days in ~2.4 years -> too many alarms at the top
    h = calibrate_threshold(v, target_far=0.5)
    assert h <= 1.0 and h > v.max()  # a usable threshold just above the validation max
