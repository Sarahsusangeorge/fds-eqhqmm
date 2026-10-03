# Eq-HQMM: Permutation-Equivariant Hidden Quantum Markov Models for Market Regime Detection

Foundations of Data Science course project, Group 15, VIT Vellore.

| Member | Reg. no. | Owns |
| --- | --- | --- |
| CH Shriker Verma | 23BCE0332 | `data/`, `labels/`, `harness/`, `models/classical/` |
| Sarah Susan George | 23BCE2178 | `models/hqmm/engine.py`, `measurement.py`, `generic.py`, `model.py`, `models/oom/`, `synthetic/` |
| Abhinav Kumar Sinha | 23BCE0565 | `models/hqmm/equivariant.py`, `models/neural/`, `models/geometric/` |

Shared (agreed jointly, changed only by a reviewed pull request): `models/base.py`, `config/`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                           # all tests should pass
```

## How the pieces fit

1. `data/` builds a `MarketData` object per basket (returns, standardised returns, correlations).
2. `harness/` makes walk-forward folds, discretises returns per fold, and runs every model.
3. Every model implements `RegimeModel` (`models/base.py`): `fit()` and `score()`.
4. Quantum models use Sarah's engine; Abhinav's `EquivariantBuilder` and Sarah's `GenericBuilder` both implement `UnitaryBuilder`.
5. The harness turns scores into alarms and metrics, and logs every run to `results/logs/runs.jsonl`.

See `CONTRIBUTING.md` for the rules on ownership, branches and commits.
