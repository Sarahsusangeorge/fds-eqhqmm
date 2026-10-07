"""Figure: synthetic regimes and symbols (owner: Sarah).

Run from the repo root:  python -m experiments.plot_synthetic
Saves results/figures/synthetic_regimes.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from synthetic.generate import STRESS, make_synthetic


def main():
    d = make_synthetic(T=1000, n=3, K=3, seed=0)
    t = np.arange(len(d.regimes))
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)

    for ax in axes:
        ax.fill_between(t, 0, 1, where=d.regimes == STRESS, transform=ax.get_xaxis_transform(),
                        color="tab:red", alpha=0.15, label="stress regime (true)")

    for i in range(d.returns.shape[1]):
        axes[0].plot(t, d.returns[:, i], lw=0.6, label=f"asset {i + 1}")
    axes[0].set_ylabel("return")
    axes[0].legend(loc="upper right", fontsize=8, ncol=4)

    extreme = ((d.symbols == 0) | (d.symbols == 2)).mean(axis=1)
    axes[1].plot(t, pd.Series(extreme).rolling(20, min_periods=1).mean(), color="black", lw=1)
    axes[1].set_ylabel("share of extreme\nsymbols (20-day avg)")
    axes[1].set_xlabel("day")

    fig.suptitle("Synthetic two-regime data with known switch points")
    fig.tight_layout()
    out = os.path.join("results", "figures", "synthetic_regimes.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
