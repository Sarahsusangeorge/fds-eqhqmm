# Sarah: integration, scaling and H1 findings (Sa9-Sa12)

All numbers are from this repo on synthetic data with known stress periods (no internet in the
build environment, so real-data HQMM runs still need to be done on a laptop). Reproduce with the two
scripts named below.

## 1. HQMM now runs inside Shriker's harness
`HQMMModel.surprise()` and `.posterior()` took (symbols, idx); the harness calls (data, symbols, idx).
Fixed; `tests/test_engine.py::test_hqmm_model_works_in_the_harness_signature` checks the call and that
scores never use data after day t.

## 2. Scaling (Sa12) - `python experiments/runtime_table.py`
Reset mode only needs the columns of the gate matrix V whose asset qubits are |0>, so the engine now
builds those directly (`GenericBuilder.build_columns`, identical to the full matrix to 1e-16, gradients
identical) instead of the whole 2^(r+n) x 2^(r+n) matrix. One loss + gradient step, r=2, K=3, 1500 days,
2 CPU cores:

| assets | before (s) | after (s) |
|---|---|---|
| 3 | 0.83 | 0.06 |
| 6 | 1.06 | 0.05 |
| 9 | 11.25 | 0.23 |
| 10 | not run | 0.69 |
| 11 | not run | 2.1 |

9 assets (Basket A) is now practical. 12 assets ran out of memory on a 7 GB machine (the table of K^n
register operators and the 2^(r+n)-length vectors). Limit for the current design: about 11 assets.
Abhinav's EquivariantBuilder can add a `build_columns` method for the same speed-up; without it the
engine falls back to the full matrix (correct, just slow above ~8 assets).

## 3. H1 on synthetic data (Sa11/Sa12) - `python experiments/hqmm_in_harness.py --folds 4 --seeds 3`
Mean AUPRC for the true stress periods over 4 walk-forward folds x 3 seeds (chance = 0.24):

| model | symbols | AUPRC |
|---|---|---|
| C-G (Gaussian HMM, Shriker) | harness std_returns | 0.75 |
| C-E-cat (categorical HMM, Shriker) | harness symbols | 0.67 |
| Q-G (generic HQMM) | harness symbols | 0.46 |
| C-E-cat | training-vol symbols | 0.97 |
| OOM-spectral (new, rank 2) | training-vol symbols | 0.95 |
| Q-G (generic HQMM) | training-vol symbols | 0.94 |

Two things follow:
1. **The symbols matter more than the model.** The harness scales each return by trailing 21-day
   volatility. That cancels the volatility jump at the start of a stress period. Every model gets
   much better when returns are scaled by training-window volatility instead (C-E-cat 0.67 -> 0.97).
   This must be checked on real data before it is adopted: on the synthetic market regimes switch
   volatility abruptly, real volatility rises more gradually.
2. **With matched inputs the quantum model does not beat the classical ones** (0.94 vs 0.97 and 0.95).
   H1 is not supported on synthetic data. This is consistent with the paper's no-advantage framing, and
   should be reported as it is. Equivariant (Q-E) vs generic (Q-G) cannot be tested until Q-E exists.
Caveats: synthetic data only, 4 folds, 3 seeds (the plan says 10), Q-G only.

## 4. H2 (equivariant vs generic) - `python experiments/hqmm_in_harness.py --folds 4 --seeds 3 --h2`
Same synthetic market, 3 assets, 4 folds x 3 seeds, training-volatility symbols. Q-E = Abhinav's
EquivariantBuilder + one shared measurement (160 gate parameters). Q-G (matched) = generic builder with
about the same number of gate parameters.

| model | AUPRC | seconds per fit |
|---|---|---|
| Q-E (equivariant) | 0.944 | 136 |
| Q-G (matched size) | 0.912 | 45 |
| Q-G (small, 30 gate parameters) | 0.940 | 29 |
| C-E-cat (classical) | 0.970 | 0.4 |

Q-E is 0.03 above the size-matched generic model and level with the small generic one. With 12 runs and
no significance test this is not evidence for H2; use the 10-seed Wilcoxon + Holm run for that. What is
solid: relabelling the assets changes Q-E scores by ~1e-15 and Q-G scores by ~0.3 (test added in
tests/test_engine.py), so Proposition 1 holds for a trained model, not only for the gates.

Cost: EquivariantBuilder forms dense matrix exponentials for 80 generators per layer. One loss+gradient
step: 0.2 s at 3 assets, 3.8 s at 5, 211 s at 7; 8 assets runs out of memory (7 GB). H2 on a 9-asset
basket needs a faster builder (apply the gates to the thin column matrix as `GenericBuilder.build_columns`
does, or work in the symmetric subspace), otherwise use a 3-5 asset sub-basket for the quantum models.
