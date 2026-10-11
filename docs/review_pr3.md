# Review of PR #3 (Shriker: classical baselines, labels, harness)

Checked: all 64 tests passed on merged main; read data/, harness/, labels/, the classical models, and the
logged results in results/.

## Works well
- Discretizer is fitted on train+val only (`fold.window_idx`), never on test days.
- Thresholds are calibrated on validation days; models only output scores, as the contract says.
- Leakage test in tests/test_harness.py (changing data after day t leaves scores up to t unchanged).
- `log_run` records author, model, fold, seed; tables come from the log.

## Needs a decision (affects fairness and every result)
1. **Trailing-volatility standardisation** (`data/features.py`, `std_returns`). It cancels the
   volatility jump of a crash, which is the signal the symbol models need. On synthetic data the same
   categorical HMM goes from AUPRC 0.67 to 0.97 when returns are scaled by training-window volatility
   (docs/sarah_findings.md). Please test on the real baskets: add a training-vol option to the
   Discretizer/harness and compare. If it holds, the real-data baseline numbers will change.
2. **R2 is circular for volatility models.** R2 is defined from realised volatility, so GARCH
   (AUPRC 0.99) and RF/XGB trained on R2 labels (1.00) are being scored against a function of their own
   inputs. Report them on R1, R3, R4, and treat R2 as a sanity check or exclude trained-on-R2 models.
3. **Many folds produce no alarms** (`no_alarm_folds` is 15-30 of 44 for every model). Recall is then
   decided by the few folds that do alarm. Check the calibration at target_far = 1 per year (C-G ends up
   with a threshold of about 32 on the log-odds) and also report a threshold-free metric next to recall.
4. **Matched objective.** The HQMM is trained on the big-move pattern only. C-E-cat is fitted on the full
   3-symbol likelihood. For a like-for-like comparison, add a classical categorical HMM trained on the
   same extreme / not-extreme sequence (same number of states as the HQMM's 4) or state in the paper that
   the objectives differ.

## Small things
- `harness/folds.py` defines its own `Fold`, separate from `models/base.py::Fold`. It works (same fields
  plus `window_idx`), but base.py says it should be the shared one: import it or add `window_idx` there.
- `results/logs/*.jsonl` is about 5,000 lines of generated output in the repo; consider git-ignoring it
  and committing only the tables.

---
# Note for PR #4 (Abhinav: symmetry, neural, geometric)
- Proposition 1 confirmed on a trained model (`tests/test_engine.py`): Q-E scores change by ~1e-15 when
  assets are relabelled, Q-G by ~0.3. 81 tests pass with PR #4 merged.
- Neural baselines train on the same big-move target as the HQMM: a fair comparison.
- **Speed:** `EquivariantBuilder.build()` multiplies dense matrix exponentials of 80 generators per layer.
  Per training step: 0.2 s at 3 assets, 3.8 s at 5, 211 s at 7, out of memory at 8. Please add a
  `build_columns()` (see `GenericBuilder.build_columns` and `engine.unitary_columns`) that applies the
  gates to the 2^q x 2^r thin matrix instead; the engine uses it automatically when it exists.

- **Update:** the speed problem above is fixed in `sarah/equivariant-fast` (`build_columns()` in the symmetric
  subspace, dense generators built lazily). Abhinav: please review `models/hqmm/equivariant.py` and
  `tests/test_equivariant_fast.py`.
