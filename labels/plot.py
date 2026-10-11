"""S6 done-when: overlay figure of R1-R4 on the basket price."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_label_overlay(data, results, path):
    px = np.exp(np.cumsum(data.equal_weight_returns()))
    fig, axes = plt.subplots(
        len(results) + 1,
        1,
        figsize=(12, 1.6 * (len(results) + 1) + 1.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3] + [1] * len(results)},
    )
    axes[0].plot(data.dates, px, lw=0.8, color="k")
    axes[0].set_yscale("log")
    axes[0].set_title("Equal-weight basket (log scale) and stress labels")
    for ax, r in zip(axes[1:], results):
        for a, b in r.intervals:
            axes[0].axvspan(a, b, alpha=0.06, color="r")
            ax.axvspan(a, b, color="r", alpha=0.6)
        ax.set_yticks([])
        ax.set_ylabel(r.name, rotation=0, ha="right", va="center")
        ax.text(
            0.995,
            0.5,
            f"{100 * r.series.mean():.1f}% of days",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
