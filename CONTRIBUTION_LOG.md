# Contribution log

One entry per member per week. Link commits, pull requests and figures.

## Week 1 (from YYYY-MM-DD)

### CH Shriker Verma
- 

### Sarah Susan George
- 

### Abhinav Kumar Sinha
- Implemented permutation operators and observation relabeling (`perm_operator`, `permute_obs`, Step A1).
- Constructed S_n-symmetric generators over register Pauli basis and asset permutation algebra (`symmetric_generators`, Step A2).
- Built `EquivariantBuilder` ansatz strictly commuting with asset permutations (`[V, P_\sigma] = 0`, Step A3).
- Implemented comprehensive symmetry test suite (`tests/test_symmetry.py`) validating commutant dimension $4^r \cdot \binom{n+3}{3}$ for $n=2..5$ and dynamical Lie algebra dimension for $n \le 4$ (Step A4).
- Implemented Proposition 1 empirical verification (`experiments/prop1_check.py`, Step A5), confirming Q-E invariance to $1.55 \times 10^{-14}$ vs Q-G deviation of $0.536$; generated Figure 5 (`results/figures/prop1_invariance.png`).
- Developed time-varying `CorrWeightedBuilder` coupling $w_{ij}(t) = a \cdot \text{corr}_{ij}(t) + b$ (Step A6).
- Implemented neural baselines: LSTM, GRU, causal TCN, causal Transformer (`models/neural/`, Step A7).
- Implemented permutation-invariant `DeepSetsGRUModel` using shared asset MLP and symmetric summation pooling (Step A8).
- Implemented data augmentation benchmark (`experiments/augmentation.py`, Step A9).
- Reimplemented Hammond (2026) geometric observables: spectral entropy and Berry phase rate (`models/geometric/hammond.py`, Step A10).
- Implemented quantum noise robustness and constituent asset swap experiments (`experiments/circuit_noise.py`, `experiments/asset_swap.py`, Step A11).
- Implemented H2/H3 evaluation script with Difference-in-Differences and bootstrap confidence intervals (`experiments/h2_h3_equivariance.py`, Step A12).
