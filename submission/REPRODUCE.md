# Reproducing the v2 results

All commands run on a laptop CPU, from the folder `6_Code/`. Nothing needs a network, a GPU or a quantum service.

## 1. Software used for the v2 runs

Python 3.13.16 · PyTorch 2.14.1 · NumPy 2.5.3 · pandas 3.0.5 · SciPy 1.18.1 · scikit-learn 1.9.1 · statsmodels 0.15.0 · hmmlearn 0.3.3 · arch 8.0.0 · matplotlib 3.11.2 · Qiskit 2.5.2 · Qiskit Aer 0.17.2 · 2 CPU cores.
The Qiskit notebook outputs were also reproduced on a second machine (Qiskit 2.6.1, Aer 0.18.2). `pip install -r requirements.txt` installs compatible versions; exact numbers can differ in the last digits with other versions.
Code origin: group repository github.com/Sarahsusangeorge/fds-eqhqmm, branch `sarah/qiskit-backend`, commit `f0650ab`, plus the files added for v2 (`experiments/analyze_v2.py`, `classical_event_folds.py`, `prop1_assumptions.py`, `regime_figure.py`, `rebuild_npz.py` and the `--scheme` / `--folds` options of `run_quantum.py`), which are in this snapshot.

## 2. Data

```
python experiments/rebuild_npz.py --csv-dir ../4_Dataset/real --out-dir data/processed
```
Rebuilds `us_sectors.npz` and `developed_countries.npz` (6,913 days each, 1999-04-07 to 2026-09-29) from the adjusted-close CSVs. On the authors' machine the rebuilt arrays equal the arrays used for all runs (`--check <dir>` compares them).

Synthetic data: `python experiments/regen_synthetic.py --n 9 --seed 1 --out synthetic_n9_seed1.csv` recreates any synthetic file (the four that are not shipped were checked equal to their originals).

## 3. Tests

```
PYTHONPATH=. python -m pytest -q
```
101 tests, all pass (output of our run: `7_Results_v2/pytest_output.txt`).

## 4. Proposition 1 and its assumptions (Section 4.4)

```
python experiments/prop1_assumptions.py --out results/prop1_assumptions.csv
```
Seeds are fixed in the script (numpy 0, torch 0). Expected: change below 1e-12 when all assumptions hold, 0.9 or more when A1 or A2 is removed. Saved output: `7_Results_v2/prop1_assumptions.csv`.

## 5. Real-data runs (Section 9.8). Rolling 5-year training window, seed 0, K = 3 symbols, 2 restarts

Classical models on all folds (seconds):
```
python experiments/run_quantum.py --basket A --scheme rolling --models oom,c-e-cat,c-g --seeds 1 --fold-stride 1 --log results/logs/cheap_A.jsonl
python experiments/run_quantum.py --basket B --scheme rolling --models oom,c-e-cat,c-g --seeds 1 --fold-stride 1 --log results/logs/cheap_B.jsonl
python experiments/classical_event_folds.py --cheap A:results/logs/cheap_A.jsonl --cheap B:results/logs/cheap_B.jsonl --out results/v2
```
Quantum models on the stress-event folds (about 10 to 20 minutes per fit; the command is resumable, so stop and restart freely):
```
python experiments/run_quantum.py --basket A --scheme rolling --models qe,qg,qgm --seeds 1 --folds 7,30 --log results/logs/quantum_A.jsonl
python experiments/run_quantum.py --basket A --scheme rolling --models qe,qg,qgm --seeds 1 --folds 8,36 --log results/logs/quantum_A2.jsonl
```
Folds that hold labelled onsets: Basket A R1 {5, 8, 30}, R4 {7, 13, 21, 27, 30, 34, 36, 39, 40}; Basket B R1 {5, 7, 8, 12, 19, 26, 30, 32}, R4 as for A. Fold 34 cannot be scored on R4 (no stress days in the window). Fold k covers six months starting January 2005 + 6k months. To extend the study, pass more fold ids to `--folds`, add `--seeds 5`, or use `--basket B`; every other number stays unchanged because records are keyed by (model, fold, seed).
Tables:
```
cat results/logs/quantum_A.jsonl results/logs/quantum_A2.jsonl > results/logs/quantum_A_all.jsonl
python experiments/analyze_v2.py --cheap A:results/logs/cheap_A.jsonl --quantum A:results/logs/quantum_A_all.jsonl --out results/v2
```
(`analyze_v2.py` uses only folds on which every quantum model has a record, so the rows are matched fold by fold. `quantum_A.jsonl` also holds one Q-E record for fold 5 from an interrupted first pass; it is ignored because the other models were not run on that fold.)

## 6. Learned-regime figure (Section 9.10)

```
python experiments/regime_figure.py --folds 30,7 --out results/regime
```
Output: `regime_scores.csv`, `regime_figure.png` (also in `7_Results_v2/`).

## 7. Qiskit checks (Section 8.5)

```
python experiments/qiskit_validate.py
```
or open `5_Notebooks/01_simulator_qiskit.ipynb`. `02_azure_quantum_hardware.ipynb` runs only as a local dry run unless an Azure Quantum workspace is configured; no live Azure run was made.

## 8. Run identifiers and integrity

* `7_Results_v2/runs_index.csv` has one run id per saved log line (`<file>:<line>`) with basket, window scheme, model, fold, seed, parameter count and timestamp.
* `CHECKSUMS.sha256` lists a SHA256 for every file in the package; `python verify_checksums.py` (from the package root) checks them all.

## 9. Where each new number in the paper comes from

| Paper item | File |
|---|---|
| Section 4.4 invariance and ablation numbers | `7_Results_v2/prop1_assumptions.csv` |
| Table 9.8a | `7_Results_v2/classical_event_folds_summary.csv`, tests in `classical_event_folds_tests.csv` |
| Tables 9.8b, 9.8c | `7_Results_v2/summary_A.csv`, `per_fold_A.csv` (built from the `*.jsonl` logs) |
| Table 9.9 | `n_params` field in the logs; gate and read-out split from the builders in `6_Code/models/hqmm/` |
| Figure 9.10 | `7_Results_v2/regime_scores.csv`, `regime_figure.png` |
| 101 tests | `7_Results_v2/pytest_output.txt` |
| Sections 9.2 to 9.7 (older development runs) | `7_Results_v2/earlier_tables/` (group repository tables); not regenerated here |
