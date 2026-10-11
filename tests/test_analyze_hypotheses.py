import json

import numpy as np

from experiments.analyze_hypotheses import analyse, load, main, to_markdown


def _write(path, effect_eq=0.1, nfold=12, seeds=(0, 1)):
    rng = np.random.default_rng(0)
    with open(path, "w") as f:
        for m, base, eq in [("Q-G", .5, 0), ("Q-E", .5, 1), ("C-G", .5, 0), ("C-E-cat", .5, 1)]:
            for fold in range(nfold):
                for s in seeds:
                    v = base + effect_eq * eq * (m == "Q-E") + rng.normal(0, .01)
                    rec = {"model": m, "fold_id": fold, "seed": s, "config": {"robustness": "baseline"},
                           "metrics": {l: {"auprc": v, "recall": v, "far": 1 - v} for l in ["R1", "R2", "R3", "R4"]}}
                    f.write(json.dumps(rec) + "\n")


def test_pipeline_detects_h2_quantum_and_h3(tmp_path):
    p = tmp_path / "l.jsonl"; _write(p)
    res = analyse(load([p], ["robustness=baseline"]), reps=200)
    r = res[(res.metric == "auprc") & (res.label == "R1")].set_index("contrast")
    assert r.loc["equivariant - generic | quantum", "diff"] > 0.08
    assert r.loc["equivariant - generic | quantum", "ci_excludes_0"]
    assert not r.loc["equivariant - generic | classical", "ci_excludes_0"]
    h3 = r[r.hyp == "H3"].iloc[0]
    assert h3["diff"] > 0.08 and h3.ci_excludes_0
    assert "auprc" in to_markdown(res)


def test_missing_cell_is_blank_and_cli_writes(tmp_path):
    p = tmp_path / "l.jsonl"; _write(p)
    lines = [l for l in open(p) if '"C-G"' not in l]
    open(p, "w").writelines(lines)
    out = tmp_path / "t" / "h.md"
    res = main(["--logs", str(p), "--reps", "100", "--out", str(out)])
    assert out.exists() and out.with_suffix(".csv").exists()
    assert res[res.hyp == "H3"]["diff"].isna().all()
