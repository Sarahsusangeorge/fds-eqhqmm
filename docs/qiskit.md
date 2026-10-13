# Running the HQMM on Qiskit (and Azure Quantum)

Faculty asked for the model to be simulated in Qiskit or Azure Quantum. `models/hqmm/qiskit_backend.py` expresses the trained
HQMM as a gate-level Qiskit circuit; `experiments/qiskit_validate.py` checks it against the PyTorch engine.

## The circuit (one trading day)
Qubits: 2 register qubits (4 regimes) + n asset qubits (start in |0>) + a = ceil(log2 K) read-out ancillas per asset.
1. `V` (trained unitary) acts on register + assets.
2. For each asset a small unitary maps |psi>|0..0> to sum_k <v_k|psi> |0>|k>, i.e. the instrument M_k = |0><v_k|.
   Measuring the ancillas returns the symbol k and leaves the asset in |0> (the reset of reset-mode measurement).
3. The register carries the filtered regime state to the next day (ancillas are reset between days).

## What is checked (`python experiments/qiskit_validate.py`)
1. Register Kraus operators from simulating the circuit (Statevector) equal the engine's (about 1e-10 for Q-E, 1e-16 for Q-G).
2. Log-likelihood of a real symbol sequence is identical from both.
3. Aer shot sampling with mid-circuit measurement reproduces the exact outcome probabilities (tests/test_qiskit_backend.py).
4. Proposition 1 holds at circuit level: relabelling assets leaves the Q-E log-likelihood unchanged.
5. Resource estimate after transpiling to {rz, sx, x, cx}. V is synthesised as a generic unitary, so the CX count is an upper
   bound; a structured Pauli-evolution synthesis of the equivariant generators would be cheaper.

## Azure Quantum
`python experiments/qiskit_validate.py --azure` submits one step circuit (basis-state register input, terminal measurement)
through `azure-quantum[qiskit]`. Set `AZURE_QUANTUM_RESOURCE_ID` and `AZURE_QUANTUM_LOCATION`; the script prints the available
targets, then run with `--azure-target <name>`. This path needs a workspace and has not been run in CI.

## Scope
Still a classical simulation of a quantum model on ideal (noise-free) simulators; no hardware run, no speed-up claim.
Training uses the PyTorch engine (exact gradients); Qiskit executes the trained model.
