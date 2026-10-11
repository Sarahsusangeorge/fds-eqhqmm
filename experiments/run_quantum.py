"""Full real-data experiment: quantum models next to classical and neural baselines, same harness.

Owner: Sarah.  Uses Shriker's folds, labels R1-R4, alarm calibration and metrics (harness/), so every model
is scored the same way, and writes to results/logs/ (one JSON line per model x fold x seed).

    python experiments/run_quantum.py --basket A --download            # fetch prices, then run (default models)
    python experiments/run_quantum.py --basket A --quick               # about 1/4 of the folds, 1 seed
    python experiments/run_quantum.py --basket A --models qg,qe,oom,c-e-cat,gru --seeds 5
    python experiments/run_quantum.py --data my_market.npz             # any saved MarketData file

It is RESUMABLE: records already in the log are skipped, so you can stop it (Ctrl+C) and start it again.
Models: qg (generic HQMM), qgm (generic HQMM with as many gate parameters as qe), qe (equivariant HQMM),
oom (spectral OOM), c-e-cat (Shriker's categorical HMM), c-g (Gaussian HMM), lstm, gru, tcn, transformer, deepsets.
Rough cost per fit on a laptop (9 assets, 2 restarts): qg/qe about 3-6 minutes, oom/c-e-cat seconds.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.folds import make_folds  # noqa: E402
from harness.runner import read_log, run  # noqa: E402
from labels.common import load_config  # noqa: E402
from labels.r1_bear import label_r1  # noqa: E402
from labels.r2_vol import label_r2  # noqa: E402
from labels.r3_consensus import label_r3  # noqa: E402
from labels.r4_events import label_r4  # noqa: E402
from models.hqmm.symbols import train_vol_transform  # noqa: E402

BASKETS = {"A": "us_sectors", "B": "developed_countries"}
ALL_MODELS = ["qg", "qgm", "qe", "oom", "c-e-cat", "c-g", "lstm", "gru", "tcn", "transformer", "deepsets"]


# ----------------------------------------------------------------------------------------------- data
def fetch_prices(tickers, start, end, csv: Path, retries: int = 4) -> pd.DataFrame:
    """Adjusted closes, one ticker at a time with retries; never caches a failed or empty download."""
    if csv.exists():
        cached = pd.read_csv(csv, index_col=0, parse_dates=True)
        if len(cached) > 1000 and all(t in cached.columns for t in tickers):
            return cached
        print(f"ignoring unusable cache {csv} ({len(cached)} rows)")
    import yfinance as yf

    cols = {}
    for t in tickers:
        for attempt in range(retries):
            try:
                df = yf.download(t, start=start, end=end, auto_adjust=True, progress=False, threads=False)
                close = df["Close"]
                close = close.iloc[:, 0] if hasattr(close, "columns") else close
                if len(close.dropna()) > 1000:
                    cols[t] = close
                    break
            except Exception as e:  # noqa: BLE001
                print(f"{t}: attempt {attempt + 1} failed ({type(e).__name__})")
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"could not download {t}; check your connection and rerun")
    px = pd.DataFrame(cols)[list(tickers)]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    csv.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(csv)
    return px


def load_market(args, bc):
    from data.features import build_market_data
    from data.market_data import MarketData

    if args.data:
        return MarketData.load(args.data), Path(args.data).stem
    name = BASKETS[args.basket]
    npz = ROOT / "data" / "processed" / f"{name}.npz"
    if npz.exists() and not args.download:
        return MarketData.load(str(npz)), name
    spec = bc[name]
    if not spec.get("tickers"):
        raise SystemExit(f"basket {args.basket} ({name}) has no tickers in config/baskets.yaml; fill them in first")
    px = fetch_prices(spec["tickers"], spec.get("start", "1999-01-01"), bc.get("end", bc.get("data_freeze")),
                      ROOT / "data" / "raw" / f"{name}_prices.csv")
    fill = px.ffill(limit=1).dropna()
    md = build_market_data(fill, bc.get("vol_window", 21), bc.get("corr_window", 63))
    npz.parent.mkdir(parents=True, exist_ok=True)
    md.save(str(npz))
    print(f"saved {npz}: {len(md)} days, {len(md.assets)} assets, {md.dates[0].date()} to {md.dates[-1].date()}")
    return md, name


# ----------------------------------------------------------------------------------------------- models
def build_models(names, K, symbol_mode, fast_test=False, n_restarts=2):
    from models.hqmm.engine import LBFGSConfig
    from models.hqmm.equivariant import EquivariantBuilder
    from models.hqmm.generic import GenericBuilder, matched_layers
    from models.hqmm.model import HQMMModel
    from models.oom.spectral import SpectralOOM

    cfg = LBFGSConfig(rounds=2, max_iter=3) if fast_test else LBFGSConfig()
    out = []
    for nm in names:
        if nm == "qg":
            out.append(HQMMModel(lambda n: GenericBuilder(2, n, layers=3), K=K, name="Q-G", n_restarts=n_restarts,
                                 shared_measurement=False, symbol_mode=symbol_mode, config=cfg))
        elif nm == "qgm":
            out.append(HQMMModel(lambda n: GenericBuilder(2, n, layers=matched_layers(
                4 ** 2 * 5 * 2, 2, n)), K=K, name="Q-G(matched)", n_restarts=n_restarts,
                shared_measurement=False, symbol_mode=symbol_mode, config=cfg))
        elif nm == "qe":
            out.append(HQMMModel(lambda n: EquivariantBuilder(2, n, layers=2), K=K, name="Q-E", n_restarts=n_restarts,
                                 shared_measurement=True, symbol_mode=symbol_mode, config=cfg))
        elif nm == "oom":
            out.append(SpectralOOM(K=K))
        elif nm == "c-e-cat":
            from models.classical.hmm_exchangeable import ExchangeableHMM
            out.append(ExchangeableHMM(emission="categorical"))
        elif nm == "c-g":
            from models.classical.hmm_generic import GenericGaussianHMM
            out.append(GenericGaussianHMM())
        elif nm in ("lstm", "gru", "tcn", "transformer", "deepsets"):
            kw = dict(epochs=3) if fast_test else {}
            if nm == "lstm":
                from models.neural.lstm import LSTMModel as M
            elif nm == "gru":
                from models.neural.gru import GRUModel as M
            elif nm == "tcn":
                from models.neural.tcn import TCNModel as M
            elif nm == "transformer":
                from models.neural.transformer import TransformerModel as M
            else:
                from models.neural.deepsets_gru import DeepSetsGRUModel as M
            out.append(M(K=K, **kw))
        else:
            raise SystemExit(f"unknown model {nm!r}; choose from {ALL_MODELS}")
    return out


# ----------------------------------------------------------------------------------------------- main
def done_keys(log: Path) -> set:
    if not log.exists() or log.stat().st_size == 0:
        return set()
    df = read_log(log)
    return set(zip(df["model"], df["fold_id"].astype(int), df["seed"].astype(int)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basket", default="A", choices=list(BASKETS))
    ap.add_argument("--data", help="saved MarketData .npz (overrides --basket)")
    ap.add_argument("--download", action="store_true", help="(re)download prices and rebuild the .npz")
    ap.add_argument("--models", default="qg,qe,oom,c-e-cat")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fold-stride", type=int, default=2, help="use every k-th walk-forward fold (default 2)")
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="every 4th fold, 1 seed")
    ap.add_argument("--K", type=int, default=3, choices=[3, 5], help="symbols per return (3 or 5)")
    ap.add_argument("--symbols", default="train_vol", choices=["train_vol", "harness"])
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--log", default=None)
    ap.add_argument("--fast-test", action="store_true", help="tiny training budget (for tests)")
    a = ap.parse_args(argv)
    if a.quick:
        a.fold_stride, a.seeds = 4, 1

    bc = yaml.safe_load((ROOT / "config" / "baskets.yaml").read_text())
    fc = yaml.safe_load((ROOT / "config" / "folds.yaml").read_text())
    md, name = load_market(a, bc)
    lab_cfg = load_config()
    labels = {r.name: r for r in (label_r1(md, lab_cfg), label_r2(md, lab_cfg), label_r3(md, lab_cfg),
                                  label_r4(md, lab_cfg))}
    folds = make_folds(md.dates, fc.get("scheme", "expanding"), fc["first_train_end"],
                       fc.get("step_months", fc.get("test_step_months", 6)), fc.get("rolling_years", 5),
                       fc.get("val_frac", fc.get("val_fraction", 0.2)))
    folds = folds[:: a.fold_stride][: a.max_folds]
    names = [m.strip() for m in a.models.split(",") if m.strip()]
    models = build_models(names, a.K, "train_vol" if a.symbols == "train_vol" else "harness", a.fast_test, a.restarts)
    log = Path(a.log) if a.log else ROOT / "results" / "logs" / f"runs_quantum_{name}.jsonl"
    done = done_keys(log)
    transform = train_vol_transform(md, a.K) if a.symbols == "train_vol" else None
    disc = fc.get("discretizer", {})
    kw = dict(K=a.K, disc_method=disc.get("method", "quantile"), target_far=fc.get("target_far", 1.0),
              gap=fc.get("alarm_gap", 20), pre=fc.get("window_pre", 20), post=fc.get("window_post", 40),
              symbol_transform=transform, log_path=log, author="sarah", verbose=False,
              config={"basket": name, "scheme": fc.get("scheme", "expanding"), "symbols": a.symbols, "K": a.K})
    todo = [(f, m, s) for f in folds for m in models for s in range(a.seeds) if (m.name, f.fold_id, s) not in done]
    print(f"{len(models)} models x {len(folds)} folds x {a.seeds} seeds; {len(done)} records already logged; "
          f"{len(todo)} runs to do -> {log}")
    t0 = time.time()
    for i, (fold, model, seed) in enumerate(todo, 1):
        t = time.time()
        run([model], [fold], [seed], md, labels, **kw)
        el, per = time.time() - t0, (time.time() - t0) / i
        print(f"[{i}/{len(todo)}] {model.name:<14} fold {fold.fold_id:2d} seed {seed}  {time.time() - t:5.0f}s  "
              f"elapsed {el / 60:5.1f} min  eta {(len(todo) - i) * per / 60:5.1f} min", flush=True)
    print("done. Summary: python harness/report.py", log)
    return log


if __name__ == "__main__":
    main()
