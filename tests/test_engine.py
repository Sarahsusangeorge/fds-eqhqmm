"""Correctness tests for the HQMM engine (owner: Sarah).

Each test compares the fast engine against the slow, obvious formula, or
checks a property any valid quantum model must have.
"""
import itertools

import numpy as np
import torch

from models.hqmm.engine import (
    apply_local, filter_sequence, filter_step, identify_stress_state,
    maximally_mixed, register_posterior, select_operators, train, TrainConfig,
)
from models.hqmm.generic import GenericBuilder
from models.hqmm.measurement import LocalMeasurement

C = torch.complex128
torch.manual_seed(0)


def dense_L(M, y, n_register):
    """Slow reference: I_R (x) M_{y1} (x) ... built as a full matrix."""
    L = torch.eye(2 ** n_register, dtype=C)
    for i, k in enumerate(y):
        Mi = M[k] if M.dim() == 3 else M[i, k]
        L = torch.kron(L, Mi.contiguous())
    return L


def random_state(q):
    A = torch.randn(2 ** q, 2 ** q, dtype=C)
    rho = A @ A.conj().T
    return (rho / torch.trace(rho))[None]


def setup(n_register=1, n=2, K=3, shared=True, seed=0):
    torch.manual_seed(seed)
    b = GenericBuilder(n_register, n, layers=2)
    m = LocalMeasurement(K, n_assets=n, shared=shared)
    return b, m, b.build().detach(), m.operators().detach()


def test_measurement_completeness_shared_and_per_asset():
    for shared in (True, False):
        M = LocalMeasurement(3, n_assets=4, shared=shared).operators()
        Ms = M[None] if shared else M
        for Mi in Ms:
            S = sum(Mi[k].conj().T @ Mi[k] for k in range(3))
            assert torch.allclose(S, torch.eye(2, dtype=C), atol=1e-12)


def test_builder_is_unitary():
    V = GenericBuilder(1, 3, layers=3).build()
    assert torch.allclose(V.conj().T @ V, torch.eye(16, dtype=C), atol=1e-12)


def test_full_kraus_completeness():
    """Sum over every joint observation y of K_y^dagger K_y = I."""
    for shared in (True, False):
        _, _, V, M = setup(shared=shared)
        S = sum((dense_L(M, y, 1) @ V).conj().T @ (dense_L(M, y, 1) @ V)
                for y in itertools.product(range(3), repeat=2))
        assert torch.allclose(S, torch.eye(8, dtype=C), atol=1e-10)


def test_apply_local_matches_dense():
    for shared in (True, False):
        _, _, _, M = setup(n=3, shared=shared)
        rho = random_state(4)
        y = torch.tensor([[2, 0, 1]])
        fast = apply_local(rho, select_operators(M, y), 1, 3)[0]
        L = dense_L(M, [2, 0, 1], 1)
        assert torch.allclose(fast, L @ rho[0] @ L.conj().T, atol=1e-12)


def test_probabilities_over_all_symbols_sum_to_one():
    _, _, V, M = setup()
    rho = random_state(3)
    total = sum(filter_step(rho, V, M, torch.tensor([list(y)]), 1, 2)[1].item()
                for y in itertools.product(range(3), repeat=2))
    assert abs(total - 1) < 1e-10


def test_state_stays_valid():
    _, _, V, M = setup(n=3)
    Y = torch.randint(0, 3, (50, 3))
    rho = maximally_mixed(4)
    for t in range(50):
        rho, _ = filter_step(rho, V, M, Y[t:t + 1], 1, 3)
        r = rho[0]
        assert abs(torch.trace(r).real.item() - 1) < 1e-10
        assert torch.allclose(r, r.conj().T, atol=1e-10)
        assert torch.linalg.eigvalsh(r).min().item() > -1e-10


def test_sequence_likelihood_matches_brute_force():
    _, _, V, M = setup()
    Y = [[0, 2], [1, 1], [2, 0], [0, 0]]
    res = filter_sequence(V, M, torch.tensor(Y), 1)
    rho = maximally_mixed(3)[0]
    for y in Y:
        Ky = dense_L(M, y, 1) @ V
        rho = Ky @ rho @ Ky.conj().T
    assert abs(res.log_p.sum().item() - torch.log(torch.trace(rho).real).item()) < 1e-10


def test_all_sequences_of_length_two_sum_to_one():
    _, _, V, M = setup()
    ys = list(itertools.product(range(3), repeat=2))
    total = sum(filter_sequence(V, M, torch.tensor([a, b]), 1).log_p.sum().exp().item()
                for a in ys for b in ys)
    assert abs(total - 1) < 1e-10


def test_batched_equals_one_by_one():
    _, _, V, M = setup()
    Y = torch.randint(0, 3, (4, 30, 2))
    batched = filter_sequence(V, M, Y, 1)
    for b in range(4):
        single = filter_sequence(V, M, Y[b], 1)
        assert torch.allclose(batched.log_p[b], single.log_p[0], atol=1e-12)
        assert torch.allclose(batched.posterior[b], single.posterior[0], atol=1e-12)


def test_register_posterior_is_a_distribution():
    _, _, V, M = setup(n_register=2, n=2)
    res = filter_sequence(GenericBuilder(2, 2).build().detach(), M, torch.randint(0, 3, (20, 2)), 2)
    p = res.posterior[0]
    assert p.shape == (20, 4)
    assert torch.allclose(p.sum(-1), torch.ones(20, dtype=p.dtype), atol=1e-10)
    assert p.min() >= -1e-12


def test_register_posterior_reads_the_right_qubit():
    """Register |1> tensor anything must give posterior [0, 1]."""
    one = torch.tensor([[0, 0], [0, 1]], dtype=C)
    rho = torch.kron(one, random_state(2)[0])[None]
    assert torch.allclose(register_posterior(rho, 1, 2)[0], torch.tensor([0.0, 1.0], dtype=torch.float64))


def test_gradients_flow():
    b, m, _, _ = setup()
    res = filter_sequence(b.build(), m.operators(), torch.randint(0, 3, (2, 20, 2)), 1)
    (-res.log_p.mean()).backward()
    grads = [p.grad for p in list(b.parameters()) + list(m.parameters())]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_training_improves_likelihood():
    from synthetic.generate import make_synthetic
    d = make_synthetic(T=500, n=2, K=3, seed=1)
    b, m, _, _ = setup(n=2, seed=1)
    from models.hqmm.engine import mean_loglik
    before = mean_loglik(b, m, d.symbols, 1, score_from=400)
    cfg = TrainConfig(window=100, batch=4, steps_per_epoch=3, max_epochs=8, patience=8)
    res = train(b, m, d.symbols[:400], d.symbols[400:], 1, cfg, seed=1)
    assert res.val_loglik > before


def test_identify_stress_state():
    T = 200
    truth = (np.arange(T) // 50) % 2                     # alternating calm/stress blocks
    Y = np.where(truth[:, None] == 1, 0, 1).repeat(3, 1)  # stress days all extreme
    post = np.stack([1 - truth, truth], 1).astype(float)  # register state 1 = stress
    assert identify_stress_state(post, Y, K=3) == 1
    assert identify_stress_state(post[:, ::-1], Y, K=3) == 0


def test_dense_and_contraction_paths_agree():
    for shared in (True, False):
        _, _, V, M = setup(n=3, shared=shared)
        Y = torch.randint(0, 3, (2, 40, 3))
        a = filter_sequence(V, M, Y, 1, dense=True)
        b = filter_sequence(V, M, Y, 1, dense=False)
        assert torch.allclose(a.log_p, b.log_p, atol=1e-10)
        assert torch.allclose(a.posterior, b.posterior, atol=1e-10)


def test_asset_marginals_match_brute_force():
    """P(asset i = k) from reduced states equals summing joint probabilities."""
    from models.hqmm.engine import asset_marginals
    for shared in (True, False):
        _, _, V, M = setup(n=3, shared=shared)
        sigma = random_state(4)
        p = asset_marginals(sigma, M, 1, 3)[0]
        joint = {}
        for y in itertools.product(range(3), repeat=3):
            L = dense_L(M, y, 1)
            joint[y] = torch.trace(L @ sigma[0] @ L.conj().T).real.item()
        for i in range(3):
            for k in range(3):
                ref = sum(v for y, v in joint.items() if y[i] == k)
                assert abs(p[i, k].item() - ref) < 1e-10


def test_predicted_extreme_share_is_causal_and_in_range():
    _, _, V, M = setup(n=2)
    Y = torch.randint(0, 3, (60, 2))
    a = filter_sequence(V, M, Y, 1, predict=True).pred_extreme[0]
    Y2 = Y.clone(); Y2[40:] = (Y2[40:] + 1) % 3          # change the future only
    b = filter_sequence(V, M, Y2, 1, predict=True).pred_extreme[0]
    assert torch.allclose(a[:40], b[:40])
    assert a.min() >= 0 and a.max() <= 1


# ---------------------------------------------------------------- reset mode
def test_reset_measurement_is_complete_and_resets():
    for shared in (True, False):
        m = LocalMeasurement(3, n_assets=4, shared=shared, reset=True)
        M = m.operators()
        Ms = M[None] if shared else M
        for Mi in Ms:
            S = sum(Mi[k].conj().T @ Mi[k] for k in range(3))
            assert torch.allclose(S, torch.eye(2, dtype=C), atol=1e-12)
            assert torch.allclose(Mi[:, 1, :], torch.zeros(3, 2, dtype=C))   # output is always |0>


def test_register_filter_matches_full_filter():
    """Reset mode on the 4x4 register equals the full engine exactly."""
    from models.hqmm.engine import filter_register, register_kraus_table, reset_start_state
    for shared in (True, False):
        torch.manual_seed(1)
        b = GenericBuilder(2, 3, layers=2)
        m = LocalMeasurement(3, n_assets=3, shared=shared, reset=True)
        Y = torch.randint(0, 3, (2, 50, 3))
        with torch.no_grad():
            full = filter_sequence(b.build(), m.operators(), Y, 2, rho0=reset_start_state(2, 3, 2), predict=True)
            table = register_kraus_table(b.build(), m.rows(), 2, 3)
            fast = filter_register(table, Y, 2, 3, predict=True)
        assert torch.allclose(full.log_p, fast.log_p, atol=1e-10)
        assert torch.allclose(full.posterior, fast.posterior, atol=1e-10)
        assert torch.allclose(full.pred_extreme, fast.pred_extreme, atol=1e-10)


def test_register_table_completeness():
    from models.hqmm.engine import register_kraus_table
    b = GenericBuilder(2, 3, layers=2)
    m = LocalMeasurement(3, n_assets=3, reset=True)
    A = register_kraus_table(b.build(), m.rows(), 2, 3)
    S = torch.einsum("yji,yjk->ik", A.conj(), A)
    assert torch.allclose(S, torch.eye(4, dtype=C), atol=1e-10)


def test_lbfgs_training_improves_likelihood():
    from models.hqmm.engine import LBFGSConfig, run_filter, train_lbfgs
    from synthetic.generate import make_synthetic
    d = make_synthetic(T=300, n=2, K=3, seed=2)
    torch.manual_seed(3)
    b = GenericBuilder(1, 2, layers=2)
    m = LocalMeasurement(3, n_assets=2, reset=True)
    with torch.no_grad():
        before = run_filter(b, m, d.symbols, 1).log_p[0, 250:].mean().item()
    res = train_lbfgs(b, m, d.symbols[:250], d.symbols[250:], 1, LBFGSConfig(rounds=2, max_iter=10))
    assert res.history[-1]["train_loss"] < res.history[0]["train_loss"] + 1e-9
    assert res.val_loglik > before


def test_hqmm_model_fit_score_is_causal():
    from models.base import Fold
    from models.hqmm.engine import LBFGSConfig
    from models.hqmm.model import HQMMModel
    from synthetic.generate import make_synthetic
    d = make_synthetic(T=260, n=2, K=3, seed=4)
    fold = Fold(0, np.arange(160), np.arange(160, 200), np.arange(200, 260))
    model = HQMMModel(lambda n: GenericBuilder(1, n, layers=1), n_register=1, n_restarts=1,
                      config=LBFGSConfig(rounds=1, max_iter=3)).fit(None, d.symbols, fold, seed=0)
    s = model.score(None, d.symbols, fold.test_idx)
    assert s.shape == (60,) and np.all((s >= 0) & (s <= 1))
    changed = d.symbols.copy()
    changed[230:] = (changed[230:] + 1) % 3                     # alter the future only
    s2 = model.score(None, changed, fold.test_idx)
    np.testing.assert_allclose(s[:30], s2[:30])
    assert model.n_params() > 0
