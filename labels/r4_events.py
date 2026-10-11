"""R4: narrative crisis windows from an event list fixed in advance (config/labels.yaml)."""

import numpy as np
import pandas as pd

from labels.common import load_config, make_result


def label_r4(data, cfg=None):
    p = (cfg or load_config())["r4_events"]
    series = np.zeros(len(data.dates), int)
    for ev in p["events"]:
        m = (data.dates >= pd.Timestamp(ev["start"])) & (data.dates <= pd.Timestamp(ev["end"]))
        series[m] = 1
    return make_result("R4", data.dates, series, p)
