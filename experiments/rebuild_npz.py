"""Rebuild the processed .npz arrays from the shipped adjusted-close CSVs (no network needed) and compare them with the
arrays the experiments used.
    python experiments/rebuild_npz.py --csv-dir 4_Dataset/real --out-dir data/processed [--check data/processed]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from data.features import build_market_data
from data.market_data import MarketData

ap = argparse.ArgumentParser(); ap.add_argument("--csv-dir", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--check", default=None)
a = ap.parse_args()
for tag, name in (("basketA_us_sectors", "us_sectors"), ("basketB_developed_countries", "developed_countries")):
    px = pd.read_csv(Path(a.csv_dir) / f"{tag}_adj_close.csv", index_col=0, parse_dates=True)
    md = build_market_data(px.ffill(limit=1).dropna(), 21, 63)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); md.save(str(out / f"{name}.npz"))
    msg = f"{name}: {len(md)} days, {md.dates[0].date()} to {md.dates[-1].date()}"
    if a.check:
        ref = MarketData.load(str(Path(a.check) / f"{name}.npz"))
        same = (len(ref) == len(md)) and np.allclose(ref.returns, md.returns, equal_nan=True) and np.allclose(ref.std_returns, md.std_returns, equal_nan=True, rtol=1e-9)
        msg += f"; matches reference arrays: {same}"
    print(msg)
