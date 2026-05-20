"""Pure-Python translation of ctmm 1.3.0 ``R/QP.R`` simplex QP surface."""

from __future__ import annotations

import numpy as np


PQP_ENV: dict[str, object] = {"EMPTY": True}


def T_C_embed(M):
    m = np.asarray(M, dtype=float).ravel()
    if m.size <= 1:
        return np.fft.fft(m)
    return np.real(np.fft.fft(np.r_[m, 0.0, m[:0:-1]]))


def _interp_matrix(floor, p, N):
    floor = np.asarray(floor, dtype=int).ravel()
    if floor.size and floor.min() >= 1:
        floor = floor - 1
    p = np.zeros(floor.size, dtype=float) if p is None else np.asarray(p, dtype=float).ravel()
    q = 1.0 - p
    B = np.zeros((floor.size, int(N)), dtype=float)
    for i, f in enumerate(floor):
        f = int(np.clip(f, 0, N - 1))
        B[i, f] += q[i]
        if f + 1 < N:
            B[i, f + 1] += p[i]
        else:
            B[i, f] += p[i]
    return B


def _as_matrix(G, FLOOR=None, p=None):
    g = np.asarray(G, dtype=float)
    if g.ndim == 2:
        return 0.5 * (g + g.T)
    if FLOOR is None:
        idx = np.arange(g.size)
        return g[np.abs(idx[:, None] - idx[None, :])]
    B = _interp_matrix(FLOOR, p, g.size)
    idx = np.arange(g.size)
    T = g[np.abs(idx[:, None] - idx[None, :])]
    return B @ T @ B.T


def G_VEC(G, V, FLOOR=None, p=None):
    M = _as_matrix(G, FLOOR=FLOOR, p=p)
    return M @ np.asarray(V, dtype=float)


def PC_VEC(V, *args, **kwargs):
    del args, kwargs
    return np.asarray(V, dtype=float)


def PC_UPDATE(*args, **kwargs):
    del args, kwargs
    return None


def BAND_SOLVE(VEC, DIAG=1.0, BAND=None):
    v = np.asarray(VEC, dtype=float).reshape(-1)
    n = v.size
    if BAND is None:
        return v / float(DIAG)
    band = np.asarray(BAND, dtype=float).reshape(-1)
    diag = np.full(n, float(DIAG), dtype=float) if np.ndim(DIAG) == 0 else np.asarray(DIAG, dtype=float).reshape(-1)
    a = np.zeros(n, dtype=float)
    b = diag.copy()
    c = np.zeros(n, dtype=float)
    a[1:] = band[: n - 1]
    c[:-1] = band[: n - 1]
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        v[i] -= m * v[i - 1]
    x = np.zeros(n, dtype=float)
    x[-1] = v[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (v[i] - c[i] * x[i + 1]) / b[i]
    return x


def MARKOV_SOLVE(G, V, FLOOR=None, p=None):
    M = _as_matrix(G, FLOOR=FLOOR, p=p)
    try:
        return np.linalg.solve(M, np.asarray(V, dtype=float))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M) @ np.asarray(V, dtype=float)


def KKT(G, P, FLOOR=None, p=None, error=np.finfo(float).eps):
    M = _as_matrix(G, FLOOR=FLOOR, p=p)
    prob = np.asarray(P, dtype=float).reshape(-1)
    grad = M @ prob
    free = prob > max(float(error), 0.0)
    lam = float(np.mean(grad[free])) if np.any(free) else float(np.min(grad))
    return {"gradient": grad, "lambda": lam, "free": free, "KKT": bool(np.all(grad[~free] >= lam - max(float(error), 0.0)))}


def _solve_free(G, free):
    idx = np.where(free)[0]
    if idx.size == 0:
        return np.zeros(G.shape[0], dtype=float)
    sub = G[np.ix_(idx, idx)]
    one = np.ones(idx.size, dtype=float)
    try:
        q = np.linalg.solve(sub, one)
    except np.linalg.LinAlgError:
        q = np.linalg.pinv(sub) @ one
    out = np.zeros(G.shape[0], dtype=float)
    s = float(np.sum(q))
    if not np.isfinite(s) or abs(s) <= np.finfo(float).eps:
        q = np.ones(idx.size, dtype=float)
        s = float(idx.size)
    out[idx] = q / s
    return out


def PQP_solve(G, FLOOR=None, p=None, lag=None, error=np.finfo(float).eps, PC="circulant", IG=None, MARKOV=None, trace: bool = False, **kwargs):
    """Solve ``min p'Gp`` subject to ``p>=0`` and ``sum(p)=1``."""
    del lag, PC, IG, MARKOV, trace, kwargs
    M = _as_matrix(G, FLOOR=FLOOR, p=p)
    n = M.shape[0]
    if n == 0:
        return {"P": np.array([], dtype=float), "MISE": 0.0, "STEPS": 0, "CHANGES": 0}
    free = np.ones(n, dtype=bool)
    steps = 0
    changes = 0
    best_p = np.full(n, 1.0 / n, dtype=float)
    best_val = float(best_p @ M @ best_p)
    while True:
        changes += 1
        pvec = _solve_free(M, free)
        neg = (pvec < -max(float(error), 0.0)) & free
        if np.any(neg):
            free[np.argmin(pvec)] = False
            steps += 1
            continue
        pvec = np.clip(pvec, 0.0, np.inf)
        s = float(np.sum(pvec))
        pvec = pvec / s if s > 0 else np.full(n, 1.0 / n)
        grad = M @ pvec
        active = ~free
        if np.any(active):
            lambda_free = float(np.mean(grad[free])) if np.any(free) else float(np.min(grad))
            release = active & (grad < lambda_free - max(float(error), np.finfo(float).eps))
            if np.any(release):
                free[np.argmin(np.where(release, grad, np.inf))] = True
                steps += 1
                continue
        val = float(pvec @ M @ pvec)
        if val <= best_val or changes == 1:
            best_val = val
            best_p = pvec
        break
    PQP_ENV["P"] = best_p.copy()
    PQP_ENV["EMPTY"] = False
    return {"P": best_p, "MISE": best_val, "STEPS": steps, "CHANGES": changes}


def empty_env(ENV=None):
    env = PQP_ENV if ENV is None else ENV
    env.clear()
    env["EMPTY"] = True


def QP_solve(G, **kwargs):
    return PQP_solve(G, **kwargs)


__all__ = [
    "BAND_SOLVE",
    "G_VEC",
    "KKT",
    "MARKOV_SOLVE",
    "PC_UPDATE",
    "PC_VEC",
    "PQP_ENV",
    "PQP_solve",
    "QP_solve",
    "T_C_embed",
    "empty_env",
]
