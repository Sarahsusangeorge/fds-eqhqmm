"""Sa12: runtime and memory of the HQMM engine as the basket grows (paper 6.7, scaling claim).

Measures, for n assets (r=2 register qubits, K=3): one loss + gradient evaluation, and the
size of the (2^(r+n))^2 unitary that the engine must build. Run: python experiments/runtime_table.py
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.hqmm.engine import register_kraus_table, total_loglik_coarse, unitary_columns
from models.hqmm.generic import GenericBuilder
from models.hqmm.measurement import LocalMeasurement


def one_step(n, r=2, K=3, T=1500, layers=3, seed=0):
    torch.manual_seed(seed)
    Y = np.random.default_rng(seed).integers(0, K, size=(T, n))
    b = GenericBuilder(r, n, layers=layers)
    m = LocalMeasurement(K, n_assets=n, shared=False, reset=True)
    t0 = time.time()
    table = register_kraus_table(unitary_columns(b), m.rows(), r, n)
    loss = -total_loglik_coarse(table, Y, K, r)
    t1 = time.time()
    loss.backward()
    t2 = time.time()
    return t1 - t0, t2 - t1


def main(sizes=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12)):
    print(f"{'assets':>6} {'matrix dim':>10} {'full-V MB (avoided)':>10} {'fwd s':>7} {'bwd s':>7} {'peak RSS MB':>11}")
    for n in sizes:
        d = 2 ** (2 + n)
        try:
            f, g = one_step(n)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"{n:>6} {d:>10} {d * d * 16 / 1e6:>10.1f} {f:>7.2f} {g:>7.2f} {rss:>11.0f}", flush=True)
        except Exception as e:  # out of memory etc.
            print(f"{n:>6} {d:>10} {d * d * 16 / 1e6:>10.1f}  failed: {type(e).__name__}", flush=True)
            break


# * memory the old full-unitary method needed per matrix; the current method never builds it.
if __name__ == "__main__":
    main()
