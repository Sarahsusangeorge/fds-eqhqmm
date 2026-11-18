"""Synthetic market (3 assets, known stress periods): the two classical categorical cells on the same training-volatility symbols,
same 4 expanding folds and 3 seeds as docs/sarah_findings.md section 4.   python experiments/synthetic_matched_classical.py"""
import sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from experiments.hqmm_in_harness import make_data
from harness.folds import make_folds
from harness.runner import read_log, run
from labels.common import make_result
from models.classical.hmm_exchangeable import ExchangeableHMM
from models.classical.hmm_generic_cat import GenericCategoricalHMM
from models.hqmm.symbols import train_vol_transform

md, state = make_data()
labels = {"TRUE": make_result("TRUE", md.dates, state)}
folds = make_folds(md.dates, "expanding", "2004-12-31", 6, 5, 0.2)[:4]
log = Path(tempfile.mkdtemp()) / "runs.jsonl"
run([ExchangeableHMM(emission="categorical"), GenericCategoricalHMM()], folds, [0, 1, 2], md, labels, log_path=log, author="sarah", symbol_transform=train_vol_transform(md))
df = read_log(log)
print(df.groupby("model")[["metrics.TRUE.auprc", "metrics.TRUE.recall", "metrics.TRUE.far"]].mean().round(3).to_string())
print("n_params:", df.groupby("model").n_params.first().to_dict())
