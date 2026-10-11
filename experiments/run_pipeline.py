"""Shriker's end-to-end driver: download -> features -> labels -> folds -> classical models -> logs.

python experiments/run_pipeline.py --download            # S1-S3: fetch both baskets, build data/processed/*.npz
python experiments/run_pipeline.py --basket A --run      # S4-S12: labels, folds, all classical models
python experiments/run_pipeline.py --basket A --robust   # S14
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.download import download_basket
from data.features import build_market_data
from data.market_data import MarketData
from harness.folds import make_folds
from harness.runner import run
from labels.common import load_config
from labels.plot import plot_label_overlay
from labels.r1_bear import label_r1
from labels.r2_vol import label_r2
from labels.r3_consensus import label_r3
from labels.r4_events import label_r4
from models.classical.absorption import AbsorptionRatio
from models.classical.bocpd import BOCPD
from models.classical.garch import GARCH11
from models.classical.hamilton import HamiltonMS
from models.classical.hmm_exchangeable import ExchangeableHMM
from models.classical.hmm_generic import GenericGaussianHMM
from models.classical.trees import TreeModel


def classical_models(label_fn):
    return [
        GenericGaussianHMM(),
        ExchangeableHMM(),
        ExchangeableHMM(emission="categorical"),
        HamiltonMS(),
        BOCPD(),
        GARCH11(),
        AbsorptionRatio(),
        AbsorptionRatio(k=1, name="Eig1"),
        TreeModel("rf", label_fn),
        TreeModel("xgb", label_fn),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--basket", default="A", choices=["A", "B"])
    ap.add_argument("--scheme", default="expanding", choices=["expanding", "rolling"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--robust-skip", nargs="*", default=[], help="robustness settings to skip, e.g. baseline")
    a = ap.parse_args()
    bc = yaml.safe_load((ROOT / "config" / "baskets.yaml").read_text())
    fc = yaml.safe_load((ROOT / "config" / "folds.yaml").read_text())
    if a.download:
        for key, b in bc["baskets"].items():
            prices = download_basket(b["tickers"], bc["start"], bc["end"], b["name"], ROOT / "data" / "raw")
            md = build_market_data(prices, bc["vol_window"], bc["corr_window"])
            md.save(ROOT / "data" / "processed" / f"{b['name']}.npz")
            print(key, b["name"], len(md), md.dates[0].date(), md.dates[-1].date())
    if a.run or a.robust:
        name = bc["baskets"][a.basket]["name"]
        md = MarketData.load(ROOT / "data" / "processed" / f"{name}.npz")
        cfg = load_config()
        labels = {r.name: r for r in (label_r1(md, cfg), label_r2(md, cfg), label_r3(md, cfg), label_r4(md, cfg))}
        plot_label_overlay(md, list(labels.values()), ROOT / "results" / "figures" / f"labels_{name}.png")
        folds = make_folds(
            md.dates, a.scheme, fc["first_train_end"], fc["step_months"], fc["rolling_years"], fc["val_frac"]
        )
        models = classical_models(lambda d: label_r2(d, cfg).series)  # supervised baselines train on R2 labels
        kw = dict(
            K=fc["discretizer"]["K"],
            disc_method=fc["discretizer"]["method"],
            target_far=fc["target_far"],
            gap=fc["alarm_gap"],
            pre=fc["window_pre"],
            post=fc["window_post"],
            config={"basket": name, "scheme": a.scheme},
        )
        if a.run:
            run(models, folds, list(range(a.seeds)), md, labels, **kw)
        if a.robust:
            from experiments.robustness_data import run_all

            run_all(
                models,
                folds,
                list(range(a.seeds)),
                md,
                labels,
                skip=tuple(a.robust_skip),
                fold_params=dict(
                    scheme=a.scheme,
                    first_train_end=fc["first_train_end"],
                    step_months=fc["step_months"],
                    rolling_years=fc["rolling_years"],
                    val_frac=fc["val_frac"],
                ),
                base_config=kw["config"],
                K=kw["K"],
                disc_method=kw["disc_method"],
                target_far=kw["target_far"],
                gap=kw["gap"],
            )


if __name__ == "__main__":
    main()
