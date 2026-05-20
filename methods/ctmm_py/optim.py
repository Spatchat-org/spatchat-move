"""Partial parity translation of ctmm 1.3.0 ``R/optim.R``."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .misc_ops import optimizer as _optimizer


def mc_min(min_: int, cores: int = 1) -> int:
    cores = max(int(cores), 1)
    return int(np.ceil(float(min_) / cores) * cores)


def line_boxer(dp, p0=None, lower=-np.inf, upper=np.inf, period=False, period_max: float = 0.5, tol: float | None = None):
    del period, period_max
    d = np.asarray(dp, dtype=float)
    p0 = np.asarray(p0 if p0 is not None else np.zeros(d.shape[0]), dtype=float)
    lo = np.broadcast_to(np.asarray(lower, dtype=float), p0.shape)
    hi = np.broadcast_to(np.asarray(upper, dtype=float), p0.shape)
    p = p0[:, None] + (d if d.ndim == 2 else d[:, None])
    p = np.clip(p, lo[:, None], hi[:, None])
    if tol is not None:
        p[np.abs(p) <= tol] = 0.0
    return p if dp is not None and np.asarray(dp).ndim == 2 else p[:, 0]


def box_search(p0, grad, hess, cov=None, lower=-np.inf, upper=np.inf, period=False, period_max: float = 0.5):
    del period, period_max
    p0 = np.asarray(p0, dtype=float)
    g = np.asarray(grad, dtype=float)
    h = np.asarray(hess, dtype=float)
    if cov is None:
        try:
            cov = np.linalg.pinv(h)
        except Exception:
            cov = np.eye(len(p0), dtype=float)
    step = -np.asarray(cov, dtype=float) @ g
    return line_boxer(step, p0=p0, lower=lower, upper=upper)


def quad_solve(P0, P1, P2, DIR, F0, F1, F2):
    p0 = np.asarray(P0, dtype=float)
    p1 = np.asarray(P1, dtype=float) - p0
    p2 = np.asarray(P2, dtype=float) - p0
    d = np.asarray(DIR, dtype=float)
    d = d if d.ndim == 2 else d[:, None]
    m1 = np.sum(d * p1, axis=0)
    m2 = np.sum(d * p2, axis=0)
    g1 = (np.asarray(F1) - F0) / np.maximum(m1, 1e-18)
    g2 = (np.asarray(F2) - F0) / np.maximum(m2, 1e-18)
    hdiag = 2.0 * (g2 - g1) / np.maximum(m2 - m1, 1e-18)
    grad = (g2 / np.maximum(m2, 1e-18) - g1 / np.maximum(m1, 1e-18)) / np.maximum(1.0 / np.maximum(m2, 1e-18) - 1.0 / np.maximum(m1, 1e-18), 1e-18)
    return {"gradient": np.asarray(grad, dtype=float), "hessian": np.asarray(hdiag, dtype=float)}


def QuadSolve(*args, **kwargs):
    return quad_solve(*args, **kwargs)


def QuadTest(x, y, MIN=None, thresh: float = 0.5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if MIN is None:
        MIN = int(np.nanargmin(y))
    if MIN <= 0 or MIN >= y.size - 1:
        return False
    q = quad_solve([x[MIN]], [x[MIN - 1]], [x[MIN + 1]], [1.0], y[MIN], y[MIN - 1], y[MIN + 1])
    h = float(np.asarray(q["hessian"]).reshape(-1)[0])
    return bool(np.isfinite(h) and h > 0 and np.nanmin(y) == y[MIN])


def WolfeTest(*args, **kwargs):
    del args, kwargs
    return True


def Gram_Schmidt(X):
    q, _ = np.linalg.qr(np.asarray(X, dtype=float))
    return q


def get_theta(par):
    return np.asarray(par, dtype=float)


def put_theta(theta, par=None):
    del par
    return np.asarray(theta, dtype=float)


def is_boxed(par, lower=-np.inf, upper=np.inf):
    p = np.asarray(par, dtype=float)
    return (p <= np.asarray(lower, dtype=float)) | (p >= np.asarray(upper, dtype=float))


def is_toosmallf(value, old=np.inf, precision: float = 0.5):
    return abs(float(old) - float(value)) <= np.finfo(float).eps ** precision


def is_toosmallp(par, old, precision: float = 0.5):
    return np.linalg.norm(np.asarray(par, dtype=float) - np.asarray(old, dtype=float)) <= np.finfo(float).eps ** precision


def m_box(par, lower=-np.inf, upper=np.inf):
    return np.clip(np.asarray(par, dtype=float), lower, upper)


def pmap(par, lower=-np.inf, upper=np.inf):
    return m_box(par, lower=lower, upper=upper)


def rank1update(A, x, y=None):
    x = np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    return np.asarray(A, dtype=float) + np.outer(x, y)


def numderiv_diff(fn, par, eps: float = 1e-6):
    p = np.asarray(par, dtype=float)
    grad = np.zeros_like(p)
    for i in range(p.size):
        e = np.zeros_like(p)
        e[i] = eps
        grad[i] = (fn(p + e) - fn(p - e)) / (2.0 * eps)
    return grad


def y_func(x, y0=0.0, hessian=1.0, x0=0.0):
    return y0 + 0.5 * hessian * (np.asarray(x) - x0) ** 2


def single_boxer(dp, p0=None, lower=-np.inf, upper=np.inf, period=False, period_max: float = 0.5, tol=None):
    return line_boxer(dp, p0=p0, lower=lower, upper=upper, period=period, period_max=period_max, tol=tol)


def mc_optim(*args, **kwargs):
    return optimizer(*args, **kwargs)


def func(fn, par, *args, **kwargs):
    try:
        value = fn(par, *args, **kwargs)
        return float(value) if np.isfinite(value) else np.inf
    except Exception:
        return np.inf


def optimizer(par, fn: Callable, *args, method: str = "pNewton", lower=-np.inf, upper=np.inf, period=False, reset=lambda x: x, control=None, **kwargs):
    del period, reset
    control = {} if control is None else dict(control)
    return _optimizer(par, fn, *args, method=method, lower=lower, upper=upper, control=control, **kwargs)


__all__ = [
    "Gram_Schmidt",
    "QuadSolve",
    "QuadTest",
    "WolfeTest",
    "box_search",
    "func",
    "get_theta",
    "is_boxed",
    "is_toosmallf",
    "is_toosmallp",
    "line_boxer",
    "m_box",
    "mc_min",
    "mc_optim",
    "numderiv_diff",
    "optimizer",
    "pmap",
    "put_theta",
    "quad_solve",
    "rank1update",
    "single_boxer",
    "y_func",
]
