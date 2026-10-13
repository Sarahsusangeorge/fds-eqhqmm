"""Qiskit realisation of the HQMM (reset mode).  Owner: Sarah.

The PyTorch engine is a classical simulator of the quantum model.  This module expresses the SAME trained
model as a gate-level Qiskit circuit and checks that Qiskit's simulators reproduce the engine exactly.

One day of data = one circuit "step" on  r register qubits + n asset qubits + n*a read-out ancillas
(a = ceil(log2 K)):

  1. V            the trained unitary on register + assets (UnitaryGate of builder.build()),
  2. read-out_i   per asset i, a unitary on (asset_i, ancillas_i) implementing the instrument M_k = |0><v_k|:
                  |psi>|0..0>  ->  sum_k <v_k|psi> |0>|k>,   so measuring the ancillas gives the symbol k
                  and leaves the asset in |0> (the "reset" of the reset-mode measurement).
  3. measure the ancillas -> the day's symbols; the register keeps the filtered state.

Ancilla convention: the symbol k is written on the ancillas of its asset with the first ancilla as the
most significant bit.

Qubit convention: our matrices have qubit 0 as the most significant bit; Qiskit is little-endian, so every
matrix is applied with its qargs reversed.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.linalg import null_space

from models.hqmm.engine import register_kraus_table, unitary_columns  # noqa: F401  (re-exported for tests)


def n_ancilla_bits(K: int) -> int:
    return max(1, math.ceil(math.log2(K)))


def readout_unitary(rows: np.ndarray) -> np.ndarray:
    """Unitary on (asset, a ancilla qubits), asset = most significant bit, with
    U |s>|0..0> = sum_k rows[k, s] |0>|k>   (rows = the read-out rows <v_k|, shape (K, 2))."""
    K = rows.shape[0]
    a = n_ancilla_bits(K)
    dim = 2 ** (1 + a)
    cols = np.zeros((dim, 2), dtype=complex)
    for s in range(2):
        for k in range(K):
            cols[k, s] = rows[k, s]                       # asset=0 (MSB 0) -> index = k
    U = np.zeros((dim, dim), dtype=complex)
    in_idx = [s * 2 ** a for s in range(2)]               # |s>|0..0>
    others = [i for i in range(dim) if i not in in_idx]
    U[:, in_idx] = cols
    U[:, others] = null_space(cols.conj().T)
    return U


def build_step_circuit(builder, measurement, n_register: int, n_assets: int, measure: bool = True):
    """Qiskit circuit for one day.  Qubits: [register..., assets..., ancillas of asset 0, asset 1, ...]."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import UnitaryGate

    K, a = measurement.K, n_ancilla_bits(measurement.K)
    reg, ast = QuantumRegister(n_register, "reg"), QuantumRegister(n_assets, "asset")
    anc = [QuantumRegister(a, f"anc{i}") for i in range(n_assets)]
    cl = [ClassicalRegister(a, f"y{i}") for i in range(n_assets)] if measure else []
    qc = QuantumCircuit(reg, ast, *anc, *cl)
    with torch.no_grad():
        V = builder.build().detach().numpy()
        R = measurement.rows().detach().numpy()
    R = np.broadcast_to(R, (n_assets,) + R.shape) if R.ndim == 2 else R
    qc.append(UnitaryGate(V, label="V"), (list(reg) + list(ast))[::-1])
    for i in range(n_assets):
        qc.append(UnitaryGate(readout_unitary(R[i]), label=f"read{i}"), (list(ast)[i:i + 1] + list(anc[i]))[::-1])
        if measure:
            qc.measure(anc[i], cl[i])
    return qc


def kraus_table_from_circuit(builder, measurement, n_register: int, n_assets: int) -> np.ndarray:
    """Register Kraus operators A_y (K^n, dR, dR) obtained by simulating the Qiskit circuit with Statevector
    on every register basis input.  y is indexed with asset 0 as the most significant symbol, as in the engine."""
    from qiskit.quantum_info import Statevector

    K, a, n, dR = measurement.K, n_ancilla_bits(measurement.K), n_assets, 2 ** n_register
    qc = build_step_circuit(builder, measurement, n_register, n, measure=False)
    N = qc.num_qubits
    table = np.zeros((K ** n, dR, dR), dtype=complex)
    for s in range(dR):
        init = np.zeros(2 ** N, dtype=complex)
        # qiskit index: bit q of the integer is qubit q (qubit 0 = register qubit 0 = our MSB of the register)
        idx = sum(((s >> (n_register - 1 - j)) & 1) << j for j in range(n_register))
        init[idx] = 1
        out = Statevector(init).evolve(qc).data.reshape((2,) * N)       # axes: qubit N-1 ... qubit 0
        for y in range(K ** n):
            ys = [(y // K ** (n - 1 - i)) % K for i in range(n)]       # symbol of asset i
            sl = [slice(None)] * N
            for q in range(n_register, n_register + n):                 # assets must be |0>
                sl[N - 1 - q] = 0
            for i in range(n):
                for b in range(a):
                    q = n_register + n + i * a + b
                    sl[N - 1 - q] = (ys[i] >> (a - 1 - b)) & 1
            amp = out[tuple(sl)]                                        # remaining axes: register qubits n_register-1..0
            amp = amp.reshape(dR)                                       # little-endian register index
            for t in range(dR):                                         # convert to our big-endian register index
                tt = sum(((t >> j) & 1) << (n_register - 1 - j) for j in range(n_register))
                table[y, tt, s] = amp[t]
    return table


def circuit_resources(qc, basis=("rz", "sx", "x", "cx"), level: int = 1) -> dict:
    """Gate counts and depth after transpiling to a hardware-style basis (a resource estimate, not a run)."""
    from qiskit import transpile

    t = transpile(qc.remove_final_measurements(inplace=False), basis_gates=list(basis), optimization_level=level)
    ops = dict(t.count_ops())
    return {"qubits": qc.num_qubits, "depth": t.depth(), "cx": ops.get("cx", 0), "ops": ops}


def sample_sequences(builder, measurement, n_register: int, n_assets: int, steps: int, shots: int = 20000, seed: int = 0):
    """Run `steps` consecutive days on Qiskit's Aer simulator with real mid-circuit measurement and shots.
    The register starts maximally mixed (a random computational basis state per shot).
    Returns an int array (shots, steps, n_assets) of sampled symbols."""
    from qiskit import ClassicalRegister, QuantumCircuit
    from qiskit_aer import AerSimulator

    step = build_step_circuit(builder, measurement, n_register, n_assets, measure=False)
    a, n = n_ancilla_bits(measurement.K), n_assets
    nq = step.num_qubits
    anc_qubits = list(range(n_register + n, nq))
    cl = ClassicalRegister(steps * n * a, "c")
    rng = np.random.default_rng(seed)
    sim = AerSimulator(method="statevector", seed_simulator=seed)
    out = np.zeros((shots, steps, n), dtype=int)
    starts = rng.integers(0, 2 ** n_register, size=shots)
    for s0 in range(2 ** n_register):
        mask = starts == s0
        m = int(mask.sum())
        if m == 0:
            continue
        run = QuantumCircuit(nq)
        run.add_register(cl)
        for j in range(n_register):              # prepare basis state s0 (qubit j holds bit of s0, MSB first)
            if (s0 >> (n_register - 1 - j)) & 1:
                run.x(j)
        for t in range(steps):
            run.compose(step, inplace=True)
            for i in range(n):
                for b in range(a):
                    run.measure(n_register + n + i * a + b, cl[(t * n + i) * a + b])
            if t < steps - 1:
                for q in anc_qubits:
                    run.reset(q)
        res = sim.run(run, shots=m).result().get_counts()
        pos = 0
        for bits, c in res.items():
            word = bits.replace(" ", "")[::-1]            # classical bit 0 first
            for _ in range(c):
                for t in range(steps):
                    for i in range(n):
                        y = sum(int(word[(t * n + i) * a + b]) << (a - 1 - b) for b in range(a))
                        out[np.flatnonzero(mask)[pos], t, i] = y
                pos += 1
    return out
