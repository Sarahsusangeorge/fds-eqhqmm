"""R2: volatility regime. 21-day realised vol of the equal-weight basket above a fixed full-sample
percentile; variant: VIX above a level."""

import numpy as np
import pandas as pd

from labels.common import clean_series, load_config, make_result


def label_r2(data, cfg=None, vix=None, variant="vol"):
    p = (cfg or load_config())["r2_vol"]
    if variant == "vix":
        assert vix is not None, "VIX series required for the VIX variant"
        v = pd.Series(vix).reindex(data.dates).ffill().to_numpy()
        raw = (v >= p["vix_level"]).astype(int)
        name = "R2_vix"
    else:
        rv = pd.Series(data.equal_weight_returns()).rolling(p["window"], min_periods=p["window"]).std().to_numpy()
        thr = np.nanpercentile(rv, p["percentile"])
        raw = np.nan_to_num(rv >= thr).astype(int)
        name = "R2"
    series = clean_series(raw, p["merge_gap"], p["min_len"])
    return make_result(name, data.dates, series, p)
