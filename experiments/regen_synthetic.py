"""Regenerate a synthetic CSV (deterministic by seed):  python experiments/regen_synthetic.py --n 9 --seed 1 --out synthetic_n9_seed1.csv"""
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from synthetic.generate import make_synthetic
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
d = make_synthetic(T=3000, n=a.n, seed=a.seed)
df = pd.DataFrame({"day": range(3000), "regime": d.regimes})
for i in range(a.n): df[f"ret_{i}"] = d.returns[:, i]
for i in range(a.n): df[f"sym_{i}"] = d.symbols[:, i]
df.to_csv(a.out, index=False); print("wrote", a.out)
