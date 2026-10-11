"""R3: statistical consensus = PELT high-variance segments  AND  smoothed Markov-switching P(high variance) > 0.5.
Uses the FULL sample (two-sided); acceptable because it is a scoring reference, never a model input."""

import warnings

import numpy as np

from labels.common import clean_series, load_config, make_result


def pelt_stress(y, p):
    import ruptures as rpt

    T = len(y)
    algo = rpt.Pelt(model="normal", min_size=p["pelt_min_size"], jump=p["pelt_jump"]).fit(y.reshape(-1, 1))
    bkps = algo.predict(pen=p["pelt_pen_scale"] * np.log(T))
    seg_std = np.zeros(T)
    lo = 0
    for hi in bkps:
        seg_std[lo:hi] = y[lo:hi].std()
        lo = hi
    return (seg_std >= p["seg_vol_ratio"] * np.median(seg_std)).astype(int)


def markov_switching_stress(y, p):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    mod = MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True)
    np.random.seed(0)
    res = mod.fit(disp=False, search_reps=5)
    names = list(mod.param_names)
    s2 = [res.params[names.index(f"sigma2[{k}]")] for k in range(2)]
    hi = int(np.argmax(s2))
    prob = np.asarray(res.smoothed_marginal_probabilities)[:, hi]
    return (prob > p["ms_prob"]).astype(int)


def label_r3(data, cfg=None):
    p = (cfg or load_config())["r3_consensus"]
    y = data.equal_weight_returns() * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pelt_stress(y, p)
        b = markov_switching_stress(y, p)
    series = clean_series(a & b, p["merge_gap"], p["min_len"])
    return make_result("R3", data.dates, series, p)
