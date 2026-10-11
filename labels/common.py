from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass
class LabelResult:
    name: str
    intervals: list  # [(start_date, end_date)] inclusive
    series: np.ndarray  # (T,) 0/1 daily stress series
    params: dict = field(default_factory=dict)


def load_config(path=None) -> dict:
    path = Path(path) if path else Path(__file__).resolve().parents[1] / "config" / "labels.yaml"
    return yaml.safe_load(path.read_text())


def index_intervals(series) -> list:
    """Inclusive (a, b) index pairs of the runs of 1s in a 0/1 series."""
    s = np.asarray(series).astype(int)
    d = np.diff(np.r_[0, s, 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def onsets(series) -> np.ndarray:
    return np.array([a for a, _ in index_intervals(series)], dtype=int)


def clean_series(series, merge_gap=0, min_len=0) -> np.ndarray:
    """Join episodes separated by < merge_gap days, then drop episodes shorter than min_len."""
    s = np.asarray(series).astype(int).copy()
    iv = index_intervals(s)
    for (a0, b0), (a1, b1) in zip(iv[:-1], iv[1:]):
        if a1 - b0 - 1 < merge_gap:
            s[b0:a1] = 1
    for a, b in index_intervals(s):
        if b - a + 1 < min_len:
            s[a : b + 1] = 0
    return s


def make_result(name, dates, series, params=None) -> LabelResult:
    s = np.asarray(series).astype(int)
    iv = [(dates[a], dates[b]) for a, b in index_intervals(s)]
    return LabelResult(name, iv, s, params or {})
