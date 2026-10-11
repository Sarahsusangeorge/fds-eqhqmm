"""Run the HQMM inside Shriker's harness next to his classical baselines (step Sa9 integration check).

Owner: Sarah. Uses a synthetic market with KNOWN stress periods, so it works offline.
Real data: replace `make_data()` by MarketData.load('data/processed/<basket>.npz') and the
label by the R1-R4 labels from experiments/run_pipeline.py.

    python experiments/hqmm_in_harness.py --folds 3 --seeds 1
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.folds import make_folds
from harness.runner import read_log, run
from labels.common import make_result
from models.classical.hmm_exchangeable import ExchangeableHMM
from models.classical.hmm_generic import GenericGaussianHMM
from models.hqmm.engine import LBFGSConfig
from models.hqmm.equivariant import EquivariantBuilder
from models.hqmm.generic import GenericBuilder, matched_layers
from models.hqmm.model import HQMMModel
from models.hqmm.symbols import train_vol_transform
from models.oom.spectral import SpectralOOM
from tests.helpers import fake_market


def make_data(T=3600, n=3, seed=0):
    md, state = fake_market(T=T, n=n, seed=seed)
    off = len(state) - md.T
    return md, state[off:]


def hqmm(name, symbol_mode):
    return HQMMModel(lambda n: GenericBuilder(2, n, layers=3), name=name, n_restarts=2,
                     shared_measurement=False, symbol_mode=symbol_mode,
                     config=LBFGSConfig())


def hqmm_equivariant(name="Q-E/train-vol-symbols"):
    """Equivariant HQMM: S_n-symmetric gates and ONE measurement shared by all assets."""
    return HQMMModel(lambda n: EquivariantBuilder(2, n, layers=2), name=name, n_restarts=2,
                     shared_measurement=True, symbol_mode="train_vol", config=LBFGSConfig())


def hqmm_generic_matched(n_params=160, name="Q-G(matched)/train-vol-symbols"):
    """Generic HQMM with about as many gate parameters as the equivariant one (fair H2 test)."""
    return HQMMModel(lambda n: GenericBuilder(2, n, layers=matched_layers(n_params, 2, n)), name=name,
                     n_restarts=2, shared_measurement=False, symbol_mode="train_vol", config=LBFGSConfig())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--h2", action="store_true", help="equivariant vs parameter-matched generic HQMM (H2)")
    a = ap.parse_args()
    md, state = make_data()
    labels = {"TRUE": make_result("TRUE", md.dates, state)}
    folds = make_folds(md.dates, "expanding", "2004-12-31", 6, 5, 0.2)[: a.folds]
    models = [GenericGaussianHMM(), ExchangeableHMM(emission="categorical"),
              hqmm("Q-G/harness-symbols", "harness"), hqmm("Q-G/train-vol-symbols", "train_vol")]
    if a.h2:
        models = [hqmm("Q-G/train-vol-symbols", "train_vol"), hqmm_generic_matched(), hqmm_equivariant()]
    log = Path(tempfile.mkdtemp()) / "runs.jsonl"
    run(models, folds, list(range(a.seeds)), md, labels, log_path=log, author="sarah")
    # fairness check: the classical categorical HMM given the SAME training-volatility symbols
    fair = ExchangeableHMM(emission="categorical")
    fair.name = "C-E-cat/train-vol-symbols"
    oom = SpectralOOM()
    oom.name = "OOM-spectral/train-vol-symbols"
    run([fair, oom], folds, list(range(a.seeds)), md, labels, log_path=log, author="sarah",
        symbol_transform=train_vol_transform(md))
    df = read_log(log)
    cols = ["model", "metrics.TRUE.auprc", "metrics.TRUE.recall", "metrics.TRUE.far", "metrics.runtime_s"]
    out = df[cols].groupby("model").mean(numeric_only=True)
    out.columns = ["auprc", "recall", "false_alarms_per_yr", "runtime_s"]
    base = float(np.mean([state[f.test_idx].mean() for f in folds]))
    print(f"\nstress share in test periods (= chance AUPRC): {base:.2f}")
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
