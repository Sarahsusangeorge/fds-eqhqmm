"""Can the HQMM find regime switches it was never told about? (owner: Sarah)

Trains the generic HQMM (Q-G) on synthetic data, then checks its stress score
on unseen test days against the hidden truth. Run from the repo root:
    python -m experiments.synthetic_recovery
Takes about 2 minutes (3 restarts, big-move objective). Saves results/figures/synthetic_recovery.png
and logs to results/logs/runs.jsonl.
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from harness.runner import log_run
from models.base import Fold
from models.hqmm.engine import extreme_share
from models.hqmm.generic import GenericBuilder
from models.hqmm.model import HQMMModel
from synthetic.generate import make_synthetic

N_ASSETS, K, T_TRAIN, T_TEST, SEED, RESTARTS = 3, 3, 2000, 1000, 0, 3


def main():
    d = make_synthetic(T=T_TRAIN + T_TEST, n=N_ASSETS, K=K, seed=SEED)
    n_val = 400
    fold = Fold(0, np.arange(T_TRAIN - n_val), np.arange(T_TRAIN - n_val, T_TRAIN),
                np.arange(T_TRAIN, T_TRAIN + T_TEST))
    model = HQMMModel(lambda n: GenericBuilder(2, n, layers=3), K=K, n_register=2,
                      n_restarts=RESTARTS, name="Q-G synthetic")
    t0 = time.time()
    model.fit(None, d.symbols, fold, seed=SEED)
    train_s = time.time() - t0

    score = model.score(None, d.symbols, fold.test_idx)
    truth = d.regimes[fold.test_idx]
    auc = roc_auc_score(truth, score)
    e = extreme_share(d.symbols, K)
    ref = np.array([e[max(0, t - 9): t + 1].mean() for t in range(len(e))])[fold.test_idx]
    auc_ref = roc_auc_score(truth, ref)
    print(f"restart validation forecast skills: {np.round(model.restart_vals, 3)}")
    print(f"test AUC  HQMM: {auc:.3f}   10-day extreme-share reference: {auc_ref:.3f}")
    print(f"chosen restart skill {model.val_criterion:+.3f}  params {model.n_params()}  train time {train_s:.0f}s")

    log_run(model.name, 0, SEED, {"n": N_ASSETS, "K": K, "r": 2, "layers": 3, "restarts": RESTARTS},
            {"auc": auc, "auc_reference": auc_ref, "val_skill": model.val_criterion,
             "n_params": model.n_params(), "train_seconds": train_s}, author="Sarah")

    t = fold.test_idx
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(t, 0, 1, where=truth == 1, transform=ax.get_xaxis_transform(),
                    color="tab:red", alpha=0.15, label="true stress (hidden)")
    ax.plot(t, score, color="black", lw=1, label="HQMM predicted share of big moves")
    ax.set_xlabel("day (unseen test period)")
    ax.set_title(f"HQMM stress score on unseen synthetic data (AUC = {auc:.2f})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = os.path.join("results", "figures", "synthetic_recovery.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
