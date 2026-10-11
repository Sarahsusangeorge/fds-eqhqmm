"""The real-data driver, run on a synthetic market with a tiny training budget (no internet)."""
import json

from data.market_data import MarketData  # noqa: F401  (the saved .npz is read back as this type)
from experiments.run_quantum import main
from tests.helpers import fake_market


def test_driver_runs_logs_and_resumes(tmp_path):
    md, _ = fake_market(T=2700, n=3, seed=0)
    npz = tmp_path / "synthetic.npz"
    md.save(str(npz))
    log = tmp_path / "runs.jsonl"
    args = ["--data", str(npz), "--models", "oom,qg,qe,c-e-cat", "--seeds", "1", "--fold-stride", "50",
            "--max-folds", "1", "--restarts", "1", "--fast-test", "--log", str(log)]
    main(args)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert {r["model"] for r in rows} == {"OOM-spectral", "Q-G", "Q-E", "C-E-cat"}
    assert all(r["author"] == "sarah" for r in rows)
    assert all(r["config"]["symbols"] == "train_vol" for r in rows)
    n_before = len(rows)
    main(args)                                   # second call: everything already logged
    assert len(log.read_text().splitlines()) == n_before
