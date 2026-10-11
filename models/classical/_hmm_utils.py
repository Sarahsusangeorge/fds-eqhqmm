from __future__ import annotations

import numpy as np


def stationary(A):
    w, v = np.linalg.eig(A.T)
    p = np.real(v[:, np.argmax(np.real(w))])
    p = np.abs(p)
    return p / p.sum()


def forward_filter(log_b, pi, A):
    """Causal filtering. Returns (alpha[t] = P(state_t | y_<=t), log p(y_t | y_<t))."""
    T, k = log_b.shape
    alpha = np.zeros((T, k))
    logc = np.zeros(T)
    prev = pi
    for t in range(T):
        pred = prev if t == 0 else prev @ A
        m = log_b[t].max()
        w = pred * np.exp(log_b[t] - m)
        s = w.sum()
        alpha[t] = w / s
        logc[t] = np.log(s) + m
        prev = alpha[t]
    return alpha, logc


def forward_backward(log_b, pi, A):
    """Scaled forward-backward for EM. Returns gamma, xi_sum, loglik."""
    T, k = log_b.shape
    m = log_b.max(axis=1, keepdims=True)
    b = np.exp(log_b - m)
    alpha = np.zeros((T, k))
    c = np.zeros(T)
    alpha[0] = pi * b[0]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * b[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.ones((T, k))
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (b[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((k, k))
    for t in range(1, T):
        xi += alpha[t - 1][:, None] * A * (b[t] * beta[t])[None, :] / c[t]
    return gamma, xi, float(np.log(c).sum() + m.sum())


def forward_filter_log(log_b, pi, A):
    """Causal filtering entirely in log space.
    Returns (log_alpha[t] = log P(state_t | y_<=t), log p(y_t | y_<t)). Unlike the probability version, the
    filtered probabilities can never round to exactly 0 or 1, so confident days keep distinct scores."""
    T, k = log_b.shape
    log_A = np.log(np.clip(A, 1e-300, None))
    log_pi = np.log(np.clip(pi, 1e-300, None))
    la = np.zeros((T, k))
    logc = np.zeros(T)
    prev = None
    for t in range(T):
        pred = log_pi if t == 0 else np.logaddexp.reduce(prev[:, None] + log_A, axis=0)
        w = pred + log_b[t]
        logc[t] = np.logaddexp.reduce(w)
        la[t] = w - logc[t]
        prev = la[t]
    return la, logc


def stress_log_odds(log_alpha, stress, clip=1000.0):
    """log P(stress) - log P(any other state), from log filtered probabilities (rows = days).
    Monotone in P(stress) but without the saturation at 1.0; clipped to +-clip only to avoid infinities."""
    others = [j for j in range(log_alpha.shape[1]) if j != stress]
    rest = np.logaddexp.reduce(log_alpha[:, others], axis=1)
    return np.clip(log_alpha[:, stress] - rest, -clip, clip)
