"""Why do C-G / C-E show NEGATIVE Cohen's d? Hypothesis: standardising returns by their trailing volatility removes the
persistent volatility level, so a Gaussian HMM on standardised returns finds brief shock states (vol onsets), not stress
regimes. This runs the HMMs on standardised ('std') and raw returns ('raw') over the same folds and prints both.

  python experiments/diagnose_hmm.py A        # basket A, all folds, ~5 min
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.market_data import MarketData
from harness.folds import make_folds
from harness.report import summary_table
from harness.runner import read_log, run
from labels.common import load_config
from labels.r1_bear import label_r1
from labels.r2_vol import label_r2
from labels.r3_consensus import label_r3
from labels.r4_events import label_r4
from models.classical.hmm_exchangeable import ExchangeableHMM
from models.classical.hmm_generic import GenericGaussianHMM

basket = sys.argv[1] if len(sys.argv) > 1 else "A"
bc = yaml.safe_load((ROOT / "config" / "baskets.yaml").read_text())
fc = yaml.safe_load((ROOT / "config" / "folds.yaml").read_text())
name = bc["baskets"][basket]["name"]
md = MarketData.load(ROOT / "data" / "processed" / f"{name}.npz")
cfg = load_config()
labels = {r.name: r for r in (label_r1(md, cfg), label_r2(md, cfg), label_r3(md, cfg), label_r4(md, cfg))}
folds = make_folds(md.dates, "expanding", fc["first_train_end"], fc["step_months"], fc["rolling_years"], fc["val_frac"])
models = [
    GenericGaussianHMM(),
    GenericGaussianHMM(input="raw", name="C-G-raw"),
    ExchangeableHMM(),
    ExchangeableHMM(input="raw", name="C-E-raw"),
]
log = ROOT / "results" / "logs" / "diag_hmm.jsonl"
log.unlink(missing_ok=True)
run(
    models,
    folds,
    [0],
    md,
    labels,
    K=fc["discretizer"]["K"],
    target_far=fc["target_far"],
    gap=fc["alarm_gap"],
    pre=fc["window_pre"],
    post=fc["window_post"],
    log_path=log,
    config={"diag": "std_vs_raw"},
    verbose=False,
)
with pd.option_context("display.width", 200, "display.max_columns", 30):
    print(summary_table(read_log(log)).round(2).to_string(index=False))
