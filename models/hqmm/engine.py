"""HQMM engine: density-matrix filtering, likelihood, training, stress scores.

Owner: Sarah. Steps Sa3, Sa4, Sa7, Sa8.

Model (paper section 4):
  hidden space  = r register qubits (the market regime) + one qubit per asset
  each day t    : rho <- V rho V^dagger                    (mixing, V from a builder)
                  rho <- L_y rho L_y^dagger,  L_y = I_R (x) M_{y^1} (x) ... (x) M_{y^n}
                  p(y_t | past) = trace(rho);  rho <- rho / p
Shapes: B = batch of sequences, T = days, n = assets, d = 2^(r+n).
Everything is batched over B so training can use several windows at once.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

CDTYPE = torch.complex128
_TINY = 1e-300


# --------------------------------------------------------------------------- state helpers
def maximally_mixed(n_qubits: int, batch: int = 1) -> torch.Tensor:
    """I/d for each sequence in the batch: (B, d, d). Permutation-invariant start state."""
    d = 2 ** n_qubits
    return (torch.eye(d, dtype=CDTYPE) / d).expand(batch, d, d).clone()


def register_posterior(rho: torch.Tensor, n_register: int, n_assets: int) -> torch.Tensor:
    """Probabilities of the 2^r register basis states (the regimes): (B, 2^r), real.

    Partial trace over the asset qubits, then the diagonal of what is left.
    """
    B = rho.shape[0]
    dR, dA = 2 ** n_register, 2 ** n_assets
    t = rho.reshape(B, dR, dA, dR, dA)
    reduced = torch.diagonal(t, dim1=2, dim2=4).sum(-1)          # (B, dR, dR)
    return torch.diagonal(reduced, dim1=1, dim2=2).real


# --------------------------------------------------------------------------- one filtering step
def _contract(t: torch.Tensor, M: torch.Tensor, axis: int) -> torch.Tensor:
    """Apply a batched 2x2 matrix M (B,2,2) to one qubit axis of tensor t."""
    t = torch.movedim(t, axis, -1)
    shape = t.shape
    t = (t.reshape(shape[0], -1, 2) @ M.transpose(-1, -2)).reshape(shape)
    return torch.movedim(t, -1, axis)


def apply_local(rho: torch.Tensor, Ms: torch.Tensor, n_register: int, n_assets: int) -> torch.Tensor:
    """rho -> L rho L^dagger with L = I_R (x) Ms[:,0] (x) ... (x) Ms[:,n-1].

    Done qubit by qubit (cost ~ n d^2) instead of building the d x d matrix L (cost d^3).
    rho: (B, d, d). Ms: (B, n, 2, 2), the operator chosen for each asset today.
    """
    B, d = rho.shape[0], rho.shape[-1]
    dR = 2 ** n_register
    t = rho.reshape((B, dR) + (2,) * n_assets + (dR,) + (2,) * n_assets)
    for i in range(n_assets):
        t = _contract(t, Ms[:, i], 2 + i)                       # ket side: M
        t = _contract(t, Ms[:, i].conj(), 3 + n_assets + i)     # bra side: conj(M)
    return t.reshape(B, d, d)


def select_operators(M: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pick today's operator for each asset. M: (K,2,2) shared or (n,K,2,2). y: (B, n) ints."""
    if M.dim() == 3:
        return M[y]                                              # (B, n, 2, 2)
    n = M.shape[0]
    return M[torch.arange(n, device=y.device)[None, :], y]


def filter_step(rho, V, M, y, n_register, n_assets, noise=None):
    """One day of the HQMM. Returns (new rho, p(y_t | past)) with shapes (B,d,d), (B,)."""
    rho = V @ rho @ V.conj().transpose(-1, -2)
    if noise is not None:
        rho = noise(rho)
    rho = apply_local(rho, select_operators(M, y), n_register, n_assets)
    p = torch.diagonal(rho, dim1=-2, dim2=-1).real.sum(-1).clamp_min(_TINY)
    return rho / p[:, None, None].to(CDTYPE), p


def asset_marginals(sigma: torch.Tensor, M: torch.Tensor, n_register: int, n_assets: int) -> torch.Tensor:
    """P(asset i shows symbol k) for the state sigma (before reading): (B, n, K).

    Uses the reduced 2x2 state of each asset qubit and E_k = M_k^dagger M_k.
    """
    B = sigma.shape[0]
    dR = 2 ** n_register
    t = sigma.reshape((B, dR) + (2,) * n_assets + (dR,) + (2,) * n_assets)
    E = M.conj().transpose(-1, -2) @ M                     # (K,2,2) or (n,K,2,2)
    out = []
    letters = "abcdefghijklmnopqrstuvwxyz"
    for i in range(n_assets):
        ket = [letters[j] for j in range(n_assets + 1)]
        bra = list(ket)
        bra[1 + i] = "Z"                                   # keep asset i open on the bra side
        spec = "B" + "".join(ket) + "".join(bra) + "->B" + ket[1 + i] + "Z"
        rho_i = torch.einsum(spec, t)                     # (B, 2, 2): rho_i[a, b] = <a|rho_i|b>
        Ei = E if E.dim() == 3 else E[i]
        out.append(torch.einsum("kba,Bab->Bk", Ei, rho_i).real)   # Tr(E_k rho_i)
    return torch.stack(out, 1)


def predicted_extreme_share(rho, V, M, n_register, n_assets) -> torch.Tensor:
    """Expected share of assets with an extreme symbol TOMORROW, given today's state: (B,)."""
    sigma = V @ rho @ V.conj().transpose(-1, -2)
    p = asset_marginals(sigma, M, n_register, n_assets)    # (B, n, K)
    return (p[..., 0] + p[..., -1]).mean(-1)


# --------------------------------------------------------------------------- whole sequences
@dataclass
class FilterResult:
    log_p: torch.Tensor       # (B, T) log p(y_t | y_<t)
    posterior: torch.Tensor   # (B, T, 2^r) regime probabilities after seeing day t
    rho: torch.Tensor         # (B, d, d) final state
    pred_extreme: torch.Tensor | None = None   # (B, T) expected extreme share for day t+1, known at day t


# Use the fast "all joint operators at once" path when it fits in memory.
DENSE_LIMIT = 4_000_000   # max entries in the (K^n, d, d) operator table


def joint_kraus(V: torch.Tensor, M: torch.Tensor, n_register: int, n_assets: int) -> torch.Tensor:
    """All joint Kraus operators K_y = (I_R (x) M_{y^1} (x) ... (x) M_{y^n}) V, shape (K^n, d, d).

    Index of y in the table: y^1 * K^(n-1) + ... + y^n (first asset most significant).
    """
    Ms = M[None].expand(n_assets, *M.shape) if M.dim() == 3 else M      # (n, K, 2, 2)
    K = Ms.shape[1]
    L = torch.eye(2 ** n_register, dtype=CDTYPE)[None]                   # (1, dR, dR)
    for i in range(n_assets):
        L = torch.einsum("aij,bkl->abikjl", L, Ms[i])
        L = L.reshape(L.shape[0] * K, L.shape[2] * 2, L.shape[4] * 2)
    return L @ V


def joint_index(Y: torch.Tensor, K: int) -> torch.Tensor:
    n = Y.shape[-1]
    weights = K ** torch.arange(n - 1, -1, -1, device=Y.device)
    return (Y * weights).sum(-1)


def filter_sequence(V, M, Y, n_register: int, rho0=None, noise=None, dense=None,
                    predict: bool = False) -> FilterResult:
    """Run the filter over sequences.

    V : (d,d) tensor, or a callable t -> (d,d) tensor for time-varying builders.
    M : measurement operators (K,2,2) or (n,K,2,2).
    Y : (T, n) or (B, T, n) integer symbols.
    dense: force (True) or forbid (False) the fast table path; None chooses automatically.
    predict: also compute the expected share of extreme symbols for the next day (stress score).
    """
    Y = torch.as_tensor(Y, dtype=torch.long)
    if Y.dim() == 2:
        Y = Y[None]
    B, T, n = Y.shape
    K = M.shape[-3]
    d = 2 ** (n_register + n)
    rho = maximally_mixed(n_register + n, B) if rho0 is None else rho0
    if dense is None:
        dense = (not callable(V)) and noise is None and (K ** n) * d * d <= DENSE_LIMIT
    log_p, post, pred = [], [], []
    if dense:
        table = joint_kraus(V, M, n_register, n)
        table_h = table.conj().transpose(-1, -2)
        J = joint_index(Y, K)                                             # (B, T)
        for t in range(T):
            rho = table[J[:, t]] @ rho @ table_h[J[:, t]]
            p = torch.diagonal(rho, dim1=-2, dim2=-1).real.sum(-1).clamp_min(_TINY)
            rho = rho / p[:, None, None].to(CDTYPE)
            log_p.append(torch.log(p))
            post.append(register_posterior(rho, n_register, n))
            if predict:
                pred.append(predicted_extreme_share(rho, V, M, n_register, n))
    else:
        for t in range(T):
            Vt = V(t) if callable(V) else V
            rho, p = filter_step(rho, Vt, M, Y[:, t], n_register, n, noise)
            log_p.append(torch.log(p))
            post.append(register_posterior(rho, n_register, n))
            if predict:
                Vn = V(t + 1) if callable(V) else V
                pred.append(predicted_extreme_share(rho, Vn, M, n_register, n))
    return FilterResult(torch.stack(log_p, 1), torch.stack(post, 1), rho,
                        torch.stack(pred, 1) if predict else None)


def extreme_share(Y: np.ndarray, K: int) -> np.ndarray:
    """Share of assets with an extreme symbol (lowest or highest) each day: (T,)."""
    Y = np.asarray(Y)
    return ((Y == 0) | (Y == K - 1)).mean(axis=-1)


def identify_stress_state(posterior: np.ndarray, Y: np.ndarray, K: int) -> int:
    """Which register state is 'stress': the one whose days have the most extreme moves.

    posterior: (T, 2^r) from training data. This fixes label switching, as for HMMs.
    """
    e = extreme_share(Y, K)
    w = posterior / posterior.sum(axis=0, keepdims=True).clip(min=1e-12)
    return int(np.argmax((w * e[:, None]).sum(axis=0)))


# --------------------------------------------------------------------------- training
@dataclass
class TrainConfig:
    window: int = 250          # days per training window
    burn_in: int = 20          # first days of each window not scored (state still settling)
    batch: int = 8             # windows per gradient step
    lr: float = 0.03
    steps_per_epoch: int = 5
    max_epochs: int = 60
    patience: int = 10         # stop after this many epochs without validation improvement
    verbose: bool = False


@dataclass
class TrainResult:
    val_loglik: float                    # mean log-likelihood per day on validation
    history: list = field(default_factory=list)


def _sample_windows(Y: np.ndarray, window: int, batch: int, rng: np.random.Generator) -> torch.Tensor:
    T = len(Y)
    w = min(window, T)
    starts = rng.integers(0, T - w + 1, size=batch)
    return torch.as_tensor(np.stack([Y[s:s + w] for s in starts]), dtype=torch.long)


def mean_loglik(builder, measurement, Y, n_register, score_from=0) -> float:
    """Average log p per day over Y[score_from:], filtering from the start of Y."""
    with torch.no_grad():
        V = builder.build()
        res = filter_sequence(V, measurement.operators(), Y, n_register)
        return res.log_p[0, score_from:].mean().item()


def train(builder, measurement, Y_train: np.ndarray, Y_val: np.ndarray,
          n_register: int, config: TrainConfig = TrainConfig(), seed: int = 0) -> TrainResult:
    """Fit builder + measurement by maximum likelihood with early stopping.

    Validation filters through train then val (so the state is warmed up) and
    scores the validation days only. The best parameters are restored at the end.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    params = list(builder.parameters()) + list(measurement.parameters())
    opt = torch.optim.Adam(params, lr=config.lr)
    Y_all = np.concatenate([Y_train, Y_val])
    best = -np.inf
    best_state = None
    bad = 0
    history = []
    for epoch in range(config.max_epochs):
        for _ in range(config.steps_per_epoch):
            Yb = _sample_windows(Y_train, config.window, config.batch, rng)
            res = filter_sequence(builder.build(), measurement.operators(), Yb, n_register)
            loss = -res.log_p[:, config.burn_in:].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = mean_loglik(builder, measurement, Y_all, n_register, score_from=len(Y_train))
        history.append({"epoch": epoch, "train_loss": loss.item(), "val_loglik": val})
        if config.verbose:
            print(f"epoch {epoch:3d}  train -loglik {loss.item():.4f}  val loglik {val:.4f}")
        if val > best + 1e-5:
            best, bad = val, 0
            best_state = (copy.deepcopy(builder.state_dict()), copy.deepcopy(measurement.state_dict()))
        else:
            bad += 1
            if bad >= config.patience:
                break
    if best_state is not None:
        builder.load_state_dict(best_state[0])
        measurement.load_state_dict(best_state[1])
    return TrainResult(best, history)


# --------------------------------------------------------------------------- reset mode (fast)
def reset_start_state(n_register: int, n_assets: int, batch: int = 1) -> torch.Tensor:
    """Full-space start state for reset mode: register maximally mixed, assets |0...0>."""
    dR, dA = 2 ** n_register, 2 ** n_assets
    zero = torch.zeros(dA, dA, dtype=CDTYPE)
    zero[0, 0] = 1
    return torch.kron(torch.eye(dR, dtype=CDTYPE) / dR, zero).expand(batch, dR * dA, dR * dA).clone()


def register_kraus_table(V: torch.Tensor, R: torch.Tensor, n_register: int, n_assets: int) -> torch.Tensor:
    """Reset mode: the register-only Kraus operators A_y, shape (K^n, dR, dR).

    A_y = (I_R (x) <v_{y^1}| (x) ... (x) <v_{y^n}|) V (I_R (x) |0...0>).
    Because every asset is reset to |0> after being read, the full model is
    exactly an HQMM on the small register with these operators.
    R: read-out rows, (K, 2) shared or (n, K, 2) per asset.
    """
    dR = 2 ** n_register
    Rs = R[None].expand(n_assets, *R.shape) if R.dim() == 2 else R      # (n, K, 2)
    K = Rs.shape[1]
    # columns of V whose asset input is |0...0>: (dR_out, 2,...,2, dR_in)
    W = V.reshape(dR, 2 ** n_assets, dR, 2 ** n_assets)[..., 0]
    W = W.reshape((dR,) + (2,) * n_assets + (dR,))
    for i in range(n_assets):
        W = torch.movedim(W, 1 + i, -1)                                  # bring asset i axis last
        W = W @ Rs[i].transpose(-1, -2)                                  # contract with <v_k|  -> axis of size K
        W = torch.movedim(W, -1, 1 + i)
    W = W.reshape(dR, K ** n_assets, dR)
    return W.permute(1, 0, 2).contiguous()                               # (K^n, dR, dR)


def extreme_weights(K: int, n_assets: int) -> torch.Tensor:
    """Share of extreme symbols for every joint observation y, in table order: (K^n,)."""
    grids = torch.cartesian_prod(*[torch.arange(K)] * n_assets) if n_assets > 1 else torch.arange(K)[:, None]
    return ((grids == 0) | (grids == K - 1)).double().mean(-1)


def filter_register(table: torch.Tensor, Y, n_register: int, K: int, predict: bool = False,
                    rho0: torch.Tensor | None = None) -> FilterResult:
    """Filter in reset mode using only the register state (B, dR, dR)."""
    Y = torch.as_tensor(Y, dtype=torch.long)
    if Y.dim() == 2:
        Y = Y[None]
    B, T, n = Y.shape
    dR = 2 ** n_register
    rho = (torch.eye(dR, dtype=CDTYPE) / dR).expand(B, dR, dR).clone() if rho0 is None else rho0
    table_h = table.conj().transpose(-1, -2)
    if predict:   # G = sum_y ext(y) A_y^dag A_y, so E[ext tomorrow] = Tr(G rho)
        G = torch.einsum("y,yij,yjk->ik", extreme_weights(K, n).to(CDTYPE), table_h, table)
    J = joint_index(Y, K)
    log_p, post, pred = [], [], []
    for t in range(T):
        rho = table[J[:, t]] @ rho @ table_h[J[:, t]]
        p = torch.diagonal(rho, dim1=-2, dim2=-1).real.sum(-1).clamp_min(_TINY)
        rho = rho / p[:, None, None].to(CDTYPE)
        log_p.append(torch.log(p))
        post.append(torch.diagonal(rho, dim1=-2, dim2=-1).real)
        if predict:
            pred.append(torch.einsum("ij,bji->b", G, rho).real)
    return FilterResult(torch.stack(log_p, 1), torch.stack(post, 1), rho,
                        torch.stack(pred, 1) if predict else None)


# --------------------------------------------------------------------------- one interface for both modes
def run_filter(builder, measurement, Y, n_register: int, predict: bool = False) -> FilterResult:
    """Filter with whichever path fits the measurement: register-only if reset, full otherwise."""
    Y = torch.as_tensor(Y, dtype=torch.long)
    n = Y.shape[-1]
    if getattr(measurement, "reset", False):
        table = register_kraus_table(builder.build(), measurement.rows(), n_register, n)
        return filter_register(table, Y, n_register, measurement.K, predict)
    return filter_sequence(builder.build(), measurement.operators(), Y, n_register, predict=predict)


@dataclass
class LBFGSConfig:
    rounds: int = 15           # outer L-BFGS calls
    max_iter: int = 20         # iterations per call
    burn_in: int = 20
    tol: float = 1e-6          # stop when the training loss improves less than this per round
    verbose: bool = False


def train_lbfgs(builder, measurement, Y_train: np.ndarray, Y_val: np.ndarray, n_register: int,
                config: LBFGSConfig = LBFGSConfig(), seed: int = 0) -> TrainResult:
    """Full-sequence maximum likelihood with L-BFGS (much more reliable than Adam here)."""
    torch.manual_seed(seed)
    params = list(builder.parameters()) + list(measurement.parameters())
    opt = torch.optim.LBFGS(params, lr=1, max_iter=config.max_iter, history_size=20,
                            line_search_fn="strong_wolfe")
    Yt = torch.as_tensor(Y_train, dtype=torch.long)
    Y_all = torch.as_tensor(np.concatenate([Y_train, Y_val]), dtype=torch.long)

    def closure():
        opt.zero_grad()
        loss = -run_filter(builder, measurement, Yt, n_register).log_p[0, config.burn_in:].mean()
        loss.backward()
        return loss

    history, prev = [], np.inf
    for rnd in range(config.rounds):
        loss = opt.step(closure).item()
        with torch.no_grad():
            val = run_filter(builder, measurement, Y_all, n_register).log_p[0, len(Y_train):].mean().item()
        history.append({"round": rnd, "train_loss": loss, "val_loglik": val})
        if config.verbose:
            print(f"round {rnd:2d}  train -loglik {loss:.4f}  val loglik {val:.4f}", flush=True)
        if prev - loss < config.tol:
            break
        prev = loss
    return TrainResult(history[-1]["val_loglik"], history)


# --------------------------------------------------------------------------- fast total likelihood (training)
def _superoperators(table: torch.Tensor) -> torch.Tensor:
    """S_y = A_y (x) conj(A_y): acts on row-major vec(rho) as vec(A rho A^dag). (K^n, dR^2, dR^2)."""
    k, dR, _ = table.shape
    return torch.einsum("yij,ykl->yikjl", table, table.conj()).reshape(k, dR * dR, dR * dR)


def _log_product(S: torch.Tensor):
    """Product S[T-1] @ ... @ S[0] by pairwise (tree) multiplication with rescaling.

    Returns (P, log_scale) with the true product = exp(log_scale) * P.
    About log2(T) batched matrix multiplications instead of T sequential steps.
    """
    log_scale = torch.zeros((), dtype=torch.float64)
    eye = torch.eye(S.shape[-1], dtype=S.dtype)
    while S.shape[0] > 1:
        if S.shape[0] % 2:
            S = torch.cat([S, eye[None]], 0)
        S = S[1::2] @ S[0::2]                                  # later day on the left
        norms = torch.linalg.matrix_norm(S).clamp_min(_TINY)
        S = S / norms[:, None, None].to(S.dtype)
        log_scale = log_scale + torch.log(norms).sum()
    return S[0], log_scale


def total_loglik(table: torch.Tensor, Y, K: int, n_register: int, burn_in: int = 0) -> torch.Tensor:
    """log p(y_{burn_in+1..T} | y_{1..burn_in}) for one sequence, computed in parallel.

    Equals the sum of filter_register's per-day log_p over days >= burn_in.
    """
    Y = torch.as_tensor(Y, dtype=torch.long)
    dR = 2 ** n_register
    S = _superoperators(table)[joint_index(Y, K)]              # (T, dR^2, dR^2)
    rho0 = (torch.eye(dR, dtype=CDTYPE) / dR).reshape(-1)
    tr = torch.eye(dR, dtype=CDTYPE).reshape(-1)               # vec(I): trace of a vec'd matrix

    def log_prob(Sseq):
        P, ls = _log_product(Sseq)
        return torch.log((tr @ (P @ rho0)).real.clamp_min(_TINY)) + ls

    full = log_prob(S)
    return full - log_prob(S[:burn_in]) if burn_in > 0 else full


def train_lbfgs_fast(builder, measurement, Y_train: np.ndarray, Y_val: np.ndarray, n_register: int,
                     config: LBFGSConfig = LBFGSConfig(), seed: int = 0) -> TrainResult:
    """Same as train_lbfgs (reset mode only) but with the parallel likelihood: much faster."""
    if not getattr(measurement, "reset", False):
        raise ValueError("train_lbfgs_fast needs a reset-mode measurement")
    torch.manual_seed(seed)
    n = Y_train.shape[1]
    params = list(builder.parameters()) + list(measurement.parameters())
    opt = torch.optim.LBFGS(params, lr=1, max_iter=config.max_iter, history_size=20,
                            line_search_fn="strong_wolfe")
    Yt = torch.as_tensor(Y_train, dtype=torch.long)
    n_scored = len(Y_train) - config.burn_in
    Y_all = torch.as_tensor(np.concatenate([Y_train, Y_val]), dtype=torch.long)

    def closure():
        opt.zero_grad()
        table = register_kraus_table(builder.build(), measurement.rows(), n_register, n)
        loss = -total_loglik(table, Yt, measurement.K, n_register, config.burn_in) / n_scored
        loss.backward()
        return loss

    history, prev = [], np.inf
    for rnd in range(config.rounds):
        loss = opt.step(closure).item()
        with torch.no_grad():
            table = register_kraus_table(builder.build(), measurement.rows(), n_register, n)
            val = ((total_loglik(table, Y_all, measurement.K, n_register)
                    - total_loglik(table, Yt, measurement.K, n_register)) / len(Y_val)).item()
        history.append({"round": rnd, "train_loss": loss, "val_loglik": val})
        if config.verbose:
            print(f"round {rnd:2d}  train -loglik {loss:.4f}  val loglik {val:.4f}", flush=True)
        if prev - loss < config.tol:
            break
        prev = loss
    return TrainResult(history[-1]["val_loglik"], history)


# --------------------------------------------------------------------------- coarse (big-move-only) likelihood
def coarse_superoperators(table: torch.Tensor, K: int, n_assets: int) -> torch.Tensor:
    """Superoperators for the coarse observation 'which assets had an extreme move'.

    Sums S_y over all joint symbols y with the same extreme/not-extreme pattern,
    i.e. the same quantum model observed with direction information removed.
    Output (2^n, dR^2, dR^2), indexed like joint_index of the 0/1 pattern with base 2.
    """
    S = _superoperators(table)
    grids = torch.cartesian_prod(*[torch.arange(K)] * n_assets) if n_assets > 1 else torch.arange(K)[:, None]
    pattern = joint_index(((grids == 0) | (grids == K - 1)).long(), 2)          # (K^n,)
    out = torch.zeros((2 ** n_assets,) + S.shape[1:], dtype=S.dtype)
    return out.index_add(0, pattern, S)


def total_loglik_coarse(table: torch.Tensor, Y, K: int, n_register: int, burn_in: int = 0) -> torch.Tensor:
    """Like total_loglik, but only for the extreme / not-extreme pattern of each day."""
    Y = torch.as_tensor(Y, dtype=torch.long)
    n = Y.shape[-1]
    dR = 2 ** n_register
    flags = ((Y == 0) | (Y == K - 1)).long()
    S = coarse_superoperators(table, K, n)[joint_index(flags, 2)]
    rho0 = (torch.eye(dR, dtype=CDTYPE) / dR).reshape(-1)
    tr = torch.eye(dR, dtype=CDTYPE).reshape(-1)

    def log_prob(Sseq):
        P, ls = _log_product(Sseq)
        return torch.log((tr @ (P @ rho0)).real.clamp_min(_TINY)) + ls

    full = log_prob(S)
    return full - log_prob(S[:burn_in]) if burn_in > 0 else full


def pretrain_coarse(builder, measurement, Y_train: np.ndarray, n_register: int,
                    rounds: int = 8, max_iter: int = 20, burn_in: int = 20) -> list:
    """Stage 1: fit on big-move patterns only, so the register learns volatility memory
    instead of the day's direction. Returns the training-loss history."""
    n = Y_train.shape[1]
    params = list(builder.parameters()) + list(measurement.parameters())
    opt = torch.optim.LBFGS(params, lr=1, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe")
    Yt = torch.as_tensor(Y_train, dtype=torch.long)

    def closure():
        opt.zero_grad()
        table = register_kraus_table(builder.build(), measurement.rows(), n_register, n)
        loss = -total_loglik_coarse(table, Yt, measurement.K, n_register, burn_in) / (len(Yt) - burn_in)
        loss.backward()
        return loss

    hist, prev = [], np.inf
    for _ in range(rounds):
        loss = opt.step(closure).item()
        hist.append(loss)
        if prev - loss < 1e-6:
            break
        prev = loss
    return hist


def forecast_skill(pred_extreme: np.ndarray, Y: np.ndarray, K: int, idx: np.ndarray) -> float:
    """Label-free quality check: correlation between the prediction made on day t and the
    actual share of extreme moves on day t+1, over days t in idx (t+1 must exist).

    Used to choose among random restarts: in synthetic tests it tracked regime-detection
    quality (Spearman 0.77 with AUC) far better than validation likelihood (0.14).
    """
    idx = np.asarray(idx)
    idx = idx[idx + 1 < len(Y)]
    e = extreme_share(Y, K)
    p, a = pred_extreme[idx], e[idx + 1]
    if p.std() < 1e-12 or a.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(p, a)[0, 1])


def train_coarse(builder, measurement, Y_train: np.ndarray, Y_val: np.ndarray, n_register: int,
                 config: LBFGSConfig = LBFGSConfig(), seed: int = 0) -> TrainResult:
    """Fit on big-move patterns only (direction removed). Default training objective.

    Why: with direction included, training was dominated by "everyone moved up / down
    together today" and usually missed the volatility memory (synthetic test AUC 0.36-0.79).
    With this objective every start reached 0.74-0.81. Reports validation forecast skill
    in history[-1]["val_skill"] for choosing between restarts.
    """
    torch.manual_seed(seed)
    hist = pretrain_coarse(builder, measurement, Y_train, n_register, rounds=config.rounds,
                           max_iter=config.max_iter, burn_in=config.burn_in)
    Y_all = np.concatenate([Y_train, Y_val])
    with torch.no_grad():
        f = run_filter(builder, measurement, Y_all, n_register, predict=True)
        table = register_kraus_table(builder.build(), measurement.rows(), n_register, Y_train.shape[1])
        val_ll = ((total_loglik_coarse(table, Y_all, measurement.K, n_register)
                   - total_loglik_coarse(table, Y_train, measurement.K, n_register)) / len(Y_val)).item()
    skill = forecast_skill(f.pred_extreme[0].numpy(), Y_all, measurement.K,
                           np.arange(len(Y_train), len(Y_all)))
    history = [{"round": i, "train_loss": l} for i, l in enumerate(hist)]
    history[-1].update({"val_loglik_coarse": val_ll, "val_skill": skill})
    if config.verbose:
        print(f"coarse fit: train -loglik {hist[-1]:.4f}  val loglik {val_ll:.4f}  val skill {skill:+.3f}")
    return TrainResult(val_ll, history)
