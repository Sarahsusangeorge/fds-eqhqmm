"""R1: bear-market dating on the equal-weight basket (peak-to-trough rules, Lunde-Timmermann style)."""

import numpy as np

from labels.common import load_config, make_result


def label_r1(data, cfg=None):
    p = (cfg or load_config())["r1_bear"]
    px = np.exp(np.cumsum(data.equal_weight_returns()))
    T = len(px)
    series = np.zeros(T, int)
    state, peak, trough, start = "bull", 0, 0, 0
    episodes = []
    for t in range(T):
        if state == "bull":
            if px[t] > px[peak]:
                peak = t
            elif px[t] <= px[peak] * (1 - p["fall"]):
                state, trough = "bear", t
                start = peak if p["start"] == "peak" else t
        else:
            if px[t] < px[trough]:
                trough = t
            elif px[t] >= px[trough] * (1 + p["rise"]):
                episodes.append((start, trough))
                state, peak = "bull", t
    if state == "bear":
        episodes.append((start, trough))
    for a, b in episodes:
        series[a : b + 1] = 1
    return make_result("R1", data.dates, series, p)
