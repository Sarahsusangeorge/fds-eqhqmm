# Shriker's modules — Group 15 Eq-HQMM (S1–S14)

Built to the Implementation Split. Everything runs offline on synthetic data; **real-data runs need network access to Yahoo Finance**
(`python experiments/run_pipeline.py --download`).

## Run order
```
pip install -r requirements.txt
pytest                                                    # 18 tests, ~15 s
python experiments/run_pipeline.py --download             # S1-S3: both baskets -> data/processed/*.npz
python experiments/run_pipeline.py --basket A --run       # S4-S12: labels, folds, 10 classical models, logs
python experiments/run_pipeline.py --basket A --robust    # S14
python notebooks/make_eda_notebook.py                     # (re)generate notebooks/eda.ipynb  (S5)
```

## Map
| Step | Where |
|---|---|
| S1 | data/download.py |
| S2 | data/features.py, data/market_data.py |
| S3 | data/discretize.py |
| S4 | harness/folds.py |
| S5 | data/eda.py + notebooks/eda.ipynb |
| S6 | labels/r1_bear.py … r4_events.py, labels/plot.py, config/labels.yaml |
| S7 | harness/alarms.py |
| S8 | harness/metrics.py |
| S9 | harness/runner.py |
| S10 / S11 | models/classical/hmm_generic.py (C-G) / hmm_exchangeable.py (C-E, gaussian + categorical) |
| S12 | hamilton.py, bocpd.py, garch.py, absorption.py, trees.py |
| S13 | harness/stats.py |
| S14 | experiments/robustness_data.py, harness/overlay.py |

## Decisions the team must confirm (they change results)
1. **Fold contract.** `train_idx` = first 80% of the training window, `val_idx` = last 20% (disjoint). `Fold.window_idx` = both. Models fit on `train_idx`; the harness calibrates thresholds on `val_idx`; the Discretizer is fitted on `window_idx`. Sarah's `train(Y_train, Y_val)` maps directly onto this.
2. **Optional `RegimeModel.surprise()`** added to models/base.py (returns None by default) so generative models can report held-out log-likelihood.
3. **Supervised models** (trees; Abhinav's supervised neural mode) get training labels through a `label_fn(window_data)` that is applied to the training window only, so no hindsight beyond the end of training leaks in.
4. **Alarm de-clustering** is chained: exceedances less than 20 days apart are one alarm, raised on the first day.
5. **False alarm** = an alarm outside [start − 20, end + 40] of every stress episode (not only outside [onset − 20, onset + 40]), so an alarm raised during a long crisis is not penalised.
6. **R1 dates the episode from the peak** (config `start: peak`); with a 20-day pre-window most models will miss R1 onsets. `start: crossing` is the alternative.
7. **R4 event dates and basket B tickers are PROPOSALS** in config/*.yaml — freeze them before any experiment.
8. **Overlay rule:** hold the basket, go to cash while the previous day's score ≥ the calibrated threshold; costs per unit of position change.
9. Hamilton uses fixed seeded start values (statsmodels' own random search was not reproducible).

## Update after the first full run (10 Oct)
- **Fixed:** probability-valued scores (C-G, C-E, C-E-cat, Hamilton) saturated at exactly 1.0, which collapsed the calibrated
  threshold (h = 1.000 in every fold) and gave 0 recall in COVID. These models now score with stress **log-odds** computed in
  log space; XGBoost scores with its raw margin; the calibration fallback uses `nextafter` instead of `+1e-9`.
  Hamilton now uses our own log-space filter (verified equal to statsmodels to 1e-8 in tests/test_harness.py).
- **Caveat on R2:** R2 is a rule on 21-day realised volatility of the same basket. GARCH, absorption and especially the
  supervised RF/XGB (trained on R2 labels from vol features) are close to circular against R2. Report the supervised models
  against R3/R4, and treat their R2 columns as an upper bound, not a result.
- **Open question for the team:** C-G and C-E had negative Cohen's d on standardised returns. Dividing by trailing volatility
  removes the persistent volatility level, so a Gaussian HMM can latch onto brief shock days. `experiments/diagnose_hmm.py`
  compares standardised vs raw inputs. The discretised symbols used by the quantum models come from the same standardised returns.
- `python harness/report.py results/logs/runs.jsonl 30` prints the summary table and the fold-30 (COVID) breakdown.
