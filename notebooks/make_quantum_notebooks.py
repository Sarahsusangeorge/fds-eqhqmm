"""Generate the two Qiskit notebooks:  python notebooks/make_quantum_notebooks.py

  01_simulator_qiskit.ipynb      - runs the trained HQMM as a gate-level circuit on Qiskit's simulators (no account needed)
  02_azure_quantum_hardware.ipynb - runs a SMALL version on Azure Quantum (needs a workspace; dry-runs on the simulator without one)

Both are written to run in a web notebook (Colab, Azure Quantum portal, IBM Quantum Lab) or locally from the repo.
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = '''# --- setup: find (or clone) the repo and make sure the packages are installed ---
import importlib, os, pathlib, subprocess, sys

REPO_URL = "https://github.com/Sarahsusangeorge/fds-eqhqmm"
BRANCH = "sarah/qiskit-backend"          # change to "main" once the Qiskit branch is merged

def has_repo(p):
    return (p / "models" / "hqmm" / "qiskit_backend.py").exists()

root = pathlib.Path.cwd()
if not has_repo(root):
    if has_repo(root.parent):
        root = root.parent
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, "fds-eqhqmm"], check=True)
        root = pathlib.Path("fds-eqhqmm").resolve()
os.chdir(root)
sys.path.insert(0, str(root))

need = {"qiskit": "qiskit", "qiskit_aer": "qiskit-aer", "torch": "torch", "scipy": "scipy",
        "pandas": "pandas", "yaml": "pyyaml", "sklearn": "scikit-learn", "matplotlib": "matplotlib"}
missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
print("repo:", root)
'''

COMMON = '''import numpy as np
import torch
import matplotlib.pyplot as plt

from models.base import Fold
from models.hqmm import qiskit_backend as qb
from models.hqmm.engine import LBFGSConfig, register_kraus_table, unitary_columns
from models.hqmm.equivariant import EquivariantBuilder
from models.hqmm.generic import GenericBuilder
from models.hqmm.model import HQMMModel
from tests.helpers import fake_market


def make_model(kind, K=3, fast=True):
    cfg = LBFGSConfig(rounds=2, max_iter=3) if fast else LBFGSConfig()
    if kind == "qe":
        return HQMMModel(lambda n: EquivariantBuilder(2, n, layers=2), K=K, name="Q-E", n_restarts=1,
                         shared_measurement=True, symbol_mode="train_vol", config=cfg)
    return HQMMModel(lambda n: GenericBuilder(2, n, layers=3), K=K, name="Q-G", n_restarts=1,
                     shared_measurement=False, symbol_mode="train_vol", config=cfg)


def fit(kind, n_assets, T=1700, seed=0):
    md, _ = fake_market(T=T, n=n_assets, seed=seed)
    T = md.T                      # a few days are lost to the feature warm-up
    n_tr, n_va = int(T * 0.6), int(T * 0.75)
    fold = Fold(0, np.arange(n_tr), np.arange(n_tr, n_va), np.arange(n_va, T))
    sym0 = np.zeros((T, md.n), dtype=int)
    m = make_model(kind).fit(md, sym0, fold, seed=seed)
    return m, md, fold, sym0


def exact_table(model, n_assets):
    with torch.no_grad():
        return register_kraus_table(unitary_columns(model.builder), model.measurement.rows(), 2, n_assets).numpy()
'''

# ------------------------------------------------------------------ notebook 1
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell("""# Notebook 1: the HQMM as a Qiskit circuit on a simulator

**Project:** Permutation-Equivariant Hidden Quantum Markov Models for Detecting Market Regime Transitions (Group 15, FDS).

This notebook takes the model we train with our PyTorch engine and runs it as a **gate-level Qiskit circuit** on Qiskit's simulators. No quantum computer and no account are needed.

What it shows, for the generic model (Q-G) and the equivariant model (Q-E):
1. The Kraus operators read out of the Qiskit circuit equal the engine's operators.
2. The log-likelihood of a symbol sequence from the circuit equals the engine's.
3. Shot sampling with real mid-circuit measurement on Aer reproduces the exact outcome probabilities.
4. **Proposition 1 at circuit level:** relabelling the assets does not change the Q-E circuit's likelihood.
5. How many qubits and gates the circuit needs.

The circuit and the checks come from `models/hqmm/qiskit_backend.py`. The model is trained on a small synthetic market with a tiny training budget, so the point is the agreement between the two implementations, not the forecasting quality."""),
    nbf.v4.new_code_cell(SETUP),
    nbf.v4.new_code_cell(COMMON + '''
import qiskit, qiskit_aer
print("qiskit", qiskit.__version__, "| qiskit-aer", qiskit_aer.__version__)
N = 3        # assets in the basket (the number of qubits grows with N)
K = 3        # symbols per asset: down / ordinary / up
'''),
    nbf.v4.new_markdown_cell("## 1. Train the two models (small budget, about a minute)"),
    nbf.v4.new_code_cell('''models = {}
for kind in ("qg", "qe"):
    models[kind] = fit(kind, N)
    print(models[kind][0].name, "trained")
'''),
    nbf.v4.new_markdown_cell("## 2. One trading day as a circuit\n\nQubits: the 2 register qubits, the N asset qubits, and 2 readout ancilla qubits per asset. `V` is the model's unitary; each `read` block turns an asset qubit into its symbol, which is measured into the classical bits."),
    nbf.v4.new_code_cell('''m_e, md_e, fold_e, sym_e = models["qe"]
qc = qb.build_step_circuit(m_e.builder, m_e.measurement, 2, N)
print(qc.draw("text", fold=110))
'''),
    nbf.v4.new_markdown_cell("## 3. Checks 1 and 2: Kraus operators and log-likelihood"),
    nbf.v4.new_code_cell('''def loglik(table, Y, K, n_register=2):
    dR = 2 ** n_register
    rho, ll = np.eye(dR, dtype=complex) / dR, 0.0
    for y in Y:
        j = 0
        for s in y:
            j = j * K + int(s)
        rho = table[j] @ rho @ table[j].conj().T
        p = np.trace(rho).real
        ll += np.log(p)
        rho = rho / p
    return ll

rows = []
for kind, (m, md, fold, sym0) in models.items():
    ref = exact_table(m, N)
    got = qb.kraus_table_from_circuit(m.builder, m.measurement, 2, N)
    Y = m._symbols(md, sym0)[fold.test_idx][:250]
    rows.append((m.name, np.abs(got - ref).max(), loglik(ref, Y, K), loglik(got, Y, K)))
print(f"{'model':6s} {'max |Kraus diff|':>18s} {'loglik engine':>15s} {'loglik Qiskit':>15s}")
for r in rows:
    print(f"{r[0]:6s} {r[1]:18.2e} {r[2]:15.4f} {r[3]:15.4f}")
assert all(r[1] < 1e-6 for r in rows), "circuit and engine disagree"
'''),
    nbf.v4.new_markdown_cell("## 4. Check 3: shots with mid-circuit measurement on Aer\n\nThe circuit is run 20,000 times. The frequency of each outcome (one symbol per asset) is compared with the exact probability from the Kraus operators."),
    nbf.v4.new_code_cell('''m, md, fold, sym0 = models["qe"]
ref = exact_table(m, N)
rho0 = np.eye(4) / 4
p_exact = np.array([np.trace(ref[y] @ rho0 @ ref[y].conj().T).real for y in range(K ** N)])
S = qb.sample_sequences(m.builder, m.measurement, 2, N, steps=1, shots=20000, seed=1)
y = np.zeros(len(S), int)
for i in range(N):
    y = y * K + S[:, 0, i]
p_aer = np.bincount(y, minlength=K ** N) / len(y)
print("largest probability difference:", round(float(np.abs(p_aer - p_exact).max()), 4))
x = np.arange(K ** N)
plt.figure(figsize=(9, 3))
plt.bar(x - 0.2, p_exact, 0.4, label="exact (engine)")
plt.bar(x + 0.2, p_aer, 0.4, label="Aer, 20,000 shots")
plt.xlabel("outcome (one symbol per asset)"); plt.ylabel("probability"); plt.legend(); plt.tight_layout(); plt.show()
'''),
    nbf.v4.new_markdown_cell("## 5. Check 4: Proposition 1 on the circuit\n\nShuffle the order of the assets in a symbol sequence. The Q-E circuit's log-likelihood should not move."),
    nbf.v4.new_code_cell('''m, md, fold, sym0 = models["qe"]
ref = exact_table(m, N)
got = qb.kraus_table_from_circuit(m.builder, m.measurement, 2, N)
Y = m._symbols(md, sym0)[fold.test_idx][:250]
perm = np.random.default_rng(0).permutation(N)
print("permutation:", perm)
print("log-likelihood, original order  :", round(loglik(got, Y, K), 6))
print("log-likelihood, relabelled order:", round(loglik(got, Y[:, perm], K), 6))
mg, mdg, fg, sg = models["qg"]
gg = qb.kraus_table_from_circuit(mg.builder, mg.measurement, 2, N)
Yg = mg._symbols(mdg, sg)[fg.test_idx][:250]
print("for comparison, the generic Q-G circuit: ", round(loglik(gg, Yg, K), 6), "->", round(loglik(gg, Yg[:, perm], K), 6))
'''),
    nbf.v4.new_markdown_cell("## 6. Resources\n\nCircuit size after compiling to the basic gates `rz, sx, x, cx` (an upper bound; a real compiler would do better)."),
    nbf.v4.new_code_cell('''for kind, (m, md, fold, sym0) in models.items():
    res = qb.circuit_resources(qb.build_step_circuit(m.builder, m.measurement, 2, N))
    print(f"{m.name}: {res['qubits']} qubits, {res['cx']} CX gates, depth {res['depth']}")
'''),
    nbf.v4.new_markdown_cell("""## What this shows

The Qiskit circuit is the same model as the PyTorch engine: same operators, same likelihood, same outcome frequencies, and the symmetry result holds on the circuit. The simulator is exact, so this is a correctness check, not evidence about hardware. Notebook 2 runs a small version on Azure Quantum."""),
]

# ------------------------------------------------------------------ notebook 2
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell("""# Notebook 2: a small HQMM step on Azure Quantum

This notebook sends **one day** of a **2-asset** Q-E model to Azure Quantum through Qiskit, and compares the measured outcome frequencies with the exact probabilities.

**Before it can run on Azure you need** (ask Sir / the course for these):
1. An Azure subscription and an **Azure Quantum workspace**, with at least one provider added (for example IonQ, Quantinuum or Rigetti).
2. The workspace **resource ID** and **location**, from the workspace overview page in the Azure portal.
3. Credits for the hardware target. Simulator targets are usually free or very cheap, so start with one.

**Until then** the notebook still runs: with no credentials it does a **dry run** on Qiskit's local Aer simulator, so everything except the Azure call can be checked. The Azure cells have not been tested against a live workspace.

**What hardware can and cannot do here.** Real devices today are noisy and have few qubits, and most do not allow measuring in the middle of a circuit. So this runs only a single day with measurement at the end, and it starts the register in a fixed basis state instead of the maximally mixed state the full model uses. It is a demonstration, not a regime-detection run. We make no claim of quantum advantage."""),
    nbf.v4.new_code_cell(SETUP),
    nbf.v4.new_code_cell(COMMON + '''
import os
N = 2          # keep the circuit small for real devices
K = 3
SHOTS = 1000   # hardware shots cost credits; raise only if needed

# ---- Azure settings: fill in, or set these as environment variables ----
RESOURCE_ID = os.environ.get("AZURE_QUANTUM_RESOURCE_ID", "")
LOCATION    = os.environ.get("AZURE_QUANTUM_LOCATION", "")
TARGET      = os.environ.get("AZURE_QUANTUM_TARGET", "ionq.simulator")   # start with a simulator target, then a QPU
USE_AZURE   = bool(RESOURCE_ID and LOCATION)
print("mode:", "Azure Quantum, target " + TARGET if USE_AZURE else "DRY RUN on the local Aer simulator (no Azure credentials set)")
'''),
    nbf.v4.new_markdown_cell("## 1. Train a small Q-E model and build one circuit\n\nThe register starts in basis state |00> and the asset outcomes are measured once, at the end."),
    nbf.v4.new_code_cell('''m, md, fold, sym0 = fit("qe", N)
ref = exact_table(m, N)
qc = qb.build_step_circuit(m.builder, m.measurement, 2, N, measure=True)
print(qc.draw("text", fold=110))
est = qb.circuit_resources(qc)
print(f"size in basic gates (rz, sx, x, cx): {est['qubits']} qubits, {est['cx']} CX gates, depth {est['depth']}")
print("Real devices have error rates of roughly 0.1% to 1% per two-qubit gate, so this many CX gates will push the "
      "measured result well away from the exact one. That is expected and is the point of reporting both.")
s0 = 0
psi0 = np.zeros(4); psi0[s0] = 1
p_exact = np.array([np.linalg.norm(ref[y] @ psi0) ** 2 for y in range(K ** N)])
print("exact outcome probabilities (sum %.4f):" % p_exact.sum(), np.round(p_exact, 3))
'''),
    nbf.v4.new_markdown_cell("## 2. Choose the backend"),
    nbf.v4.new_code_cell('''from qiskit import transpile

if USE_AZURE:
    # pip install azure-quantum[qiskit]   (done automatically below if missing)
    if importlib.util.find_spec("azure.quantum") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "azure-quantum[qiskit]"], check=True)
    from azure.quantum.qiskit import AzureQuantumProvider
    provider = AzureQuantumProvider(resource_id=RESOURCE_ID, location=LOCATION)
    print("targets in this workspace:", [b.name() for b in provider.backends()])
    backend = provider.get_backend(TARGET)
else:
    from qiskit_aer import AerSimulator
    backend = AerSimulator()

tqc = transpile(qc, backend, optimization_level=1)
ops = tqc.count_ops()
print(f"after compiling for the backend: {tqc.num_qubits} qubits, depth {tqc.depth()}, operations {dict(ops)}")
if not USE_AZURE:
    print("(the local simulator keeps the large unitary blocks as they are; a real target breaks them into native gates, "
          "so use the basic-gate size printed above as the estimate)")
'''),
    nbf.v4.new_markdown_cell("## 3. Run and compare\n\nThe register qubits start in |00>, so the first circuit run is exactly the case we computed above."),
    nbf.v4.new_code_cell('''job = backend.run(tqc, shots=SHOTS)
if USE_AZURE:
    print("job id:", job.id())          # the job is also visible in the Azure portal under Job management
counts = job.result().get_counts()

a = qb.n_ancilla_bits(K)
freq = np.zeros(K ** N)
for key, c in counts.items():
    regs = key.split()[::-1]            # y0, y1, ... (Qiskit prints the last register first)
    j = 0
    for r in regs:
        bits = r[::-1]                  # classical bit 0 first
        y = sum(int(bits[b]) << (a - 1 - b) for b in range(a))
        j = j * K + y
    freq[j] += c
freq /= freq.sum()

tv = 0.5 * np.abs(freq - p_exact).sum()
print("total variation distance, measured vs exact:", round(float(tv), 4))
x = np.arange(K ** N)
plt.figure(figsize=(7, 3))
plt.bar(x - 0.2, p_exact, 0.4, label="exact")
plt.bar(x + 0.2, freq, 0.4, label=("Azure " + TARGET) if USE_AZURE else "Aer (dry run)")
plt.xlabel("outcome (one symbol per asset)"); plt.ylabel("probability"); plt.legend(); plt.tight_layout(); plt.show()
'''),
    nbf.v4.new_markdown_cell("""## Notes for the write-up

- On an ideal simulator the distance is only shot noise (about 0.03 at 1,000 shots). On a real device, noise adds to it; report both numbers and keep the device name, date and job ID.
- The `ionq.simulator`, `quantinuum` emulator and `rigetti` simulator targets run the same circuit without hardware noise or with a noise model. Use them to check the pipeline before spending hardware credits.
- Full mid-circuit-measurement runs over many days, as in notebook 1, need devices that support it and far more shots; they are future work.
- We cannot claim anything about regime detection from this run. It shows that the circuit can be sent to and read back from Azure Quantum, and how close a device gets to the exact probabilities for one step."""),
]

for name, nb in (("01_simulator_qiskit.ipynb", nb1), ("02_azure_quantum_hardware.ipynb", nb2)):
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / name)
    print("wrote", HERE / name)
