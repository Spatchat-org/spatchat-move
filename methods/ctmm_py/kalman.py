"""Partial parity translation of ctmm 1.3.0 ``R/kalman.R``."""

from __future__ import annotations

import numpy as np

from .core_math import dexp1, dexp2


def langevin(dt: float, CTMM, DIM: int = 1):
    tau = list(CTMM.params.get("tau_list", []))
    k = max(1, len(tau))
    green = np.eye(k * DIM, dtype=float)
    sigma = np.eye(k * DIM, dtype=float)

    if len(tau) == 1 and np.isfinite(tau[0]) and tau[0] > 0 and np.isfinite(dt):
        dtau = float(dt) / float(tau[0])
        exp_term = np.exp(-dtau)
        green[...] = exp_term * np.eye(k * DIM)
        sigma[...] = dexp2(dtau, Exp=exp_term) * np.eye(k * DIM)
    elif len(tau) == 1 and (not np.isfinite(tau[0])):
        sigma[...] = max(2.0 * float(dt), 0.0) * np.eye(k * DIM)
    return {"Green": green, "Sigma": sigma}


def Langevin(t, CTMM, DIM: int = 1):
    t = np.asarray(t, dtype=float)
    dt = np.empty_like(t)
    if t.size:
        dt[0] = np.inf
        if t.size > 1:
            dt[1:] = np.diff(t)
    greens = []
    sigmas = []
    for d in dt:
        lg = langevin(float(d), CTMM, DIM=DIM)
        greens.append(lg["Green"])
        sigmas.append(lg["Sigma"])
    return {"Green": np.asarray(greens), "Sigma": np.asarray(sigmas)}


def kalman(z, u, t=None, CTMM=None, error=None, DIM: int = 1, smooth: bool = False, sample: bool = False, residual: bool = False, precompute: bool | int = False):
    """
    Minimal Kalman parity scaffold: returns shape-compatible containers.
    """
    del u, CTMM, error, DIM, smooth, sample, residual, precompute
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    n, p = z.shape
    return {
        "mu": np.zeros((1, p), dtype=float),
        "W": np.eye(1, dtype=float),
        "iW": np.eye(1, dtype=float),
        "sigma": np.eye(p, dtype=float),
        "logdet": 0.0,
        "state": np.zeros((n, p), dtype=float),
    }


__all__ = ["langevin", "Langevin", "kalman"]
