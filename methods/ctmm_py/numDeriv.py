"""Partial parity translation of ctmm 1.3.0 ``R/numDeriv.R``."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .optim import quad_solve


def func(fn: Callable, par, *args, **kwargs):
    return fn(par, *args, **kwargs)


def quad2lin(M, diag: bool = False):
    m = np.asarray(M, dtype=float)
    m = np.atleast_2d(m)
    n, p = m.shape
    tri = np.triu_indices(p)
    q = len(tri[0])
    out = np.zeros((n, q), dtype=float) if diag else np.zeros((q, q), dtype=float)
    for i in range(q):
        e = np.zeros((p, p), dtype=float)
        e[tri[0][i], tri[1][i]] = 1.0
        e = e + e.T - np.diag(np.diag(e))
        if diag:
            out[:, i] = np.einsum("ij,jk,ik->i", m, e, m)
        else:
            t = m @ e @ m.T
            out[:, i] = t[np.triu_indices(n)]
    return out


def genD(
    par,
    fn: Callable,
    zero: bool | float = False,
    lower=-np.inf,
    upper=np.inf,
    step: float | None = None,
    precision: float = 0.5,
    parscale=None,
    mc_cores: int = 1,
    Richardson: int = 2,
    order: int = 2,
    drop: bool = True,
    control=None,
    **kwargs,
):
    del zero, lower, upper, mc_cores, Richardson, drop, control, kwargs
    p = np.asarray(par, dtype=float)
    n = p.size
    s = np.asarray(parscale, dtype=float) if parscale is not None else np.ones(n, dtype=float)
    s[s == 0] = 1.0
    p0 = p / s
    h = float(step) if step is not None else float(np.sqrt(2.0 * np.finfo(float).eps**precision))

    def f(u):
        return float(fn(u * s))

    f0 = f(p0)
    grad = np.zeros(n, dtype=float)
    hess = np.zeros((n, n), dtype=float)

    for i in range(n):
        e = np.zeros(n, dtype=float)
        e[i] = 1.0
        fp = f(p0 + h * e)
        fm = f(p0 - h * e)
        grad[i] = (fp - fm) / (2.0 * h)
        if order >= 2:
            hess[i, i] = (fp - 2.0 * f0 + fm) / (h * h)

    if order >= 2 and n > 1:
        for i in range(n):
            for j in range(i + 1, n):
                ei = np.zeros(n, dtype=float); ei[i] = 1.0
                ej = np.zeros(n, dtype=float); ej[j] = 1.0
                fpp = f(p0 + h * ei + h * ej)
                fpm = f(p0 + h * ei - h * ej)
                fmp = f(p0 - h * ei + h * ej)
                fmm = f(p0 - h * ei - h * ej)
                hij = (fpp - fpm - fmp + fmm) / (4.0 * h * h)
                hess[i, j] = hess[j, i] = hij

    grad = grad / s
    hess = (hess / s).T / s
    return {"gradient": grad, "hessian": hess}


def genD_mcDeriv(*args, **kwargs):
    return genD(*args, **kwargs)


__all__ = ["func", "quad2lin", "genD", "genD_mcDeriv", "quad_solve"]
