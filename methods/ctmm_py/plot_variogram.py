"""Translation of ctmm 1.3.0 ``R/plot.variogram.R`` model SVF helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .covm import Covm, var_covm
from .types import CTMMModel
from .viz_ops import plot


def _tau_values(CTMM: CTMMModel) -> np.ndarray:
    tau = CTMM.params.get("tau", {})
    if isinstance(tau, dict):
        vals = [float(v) for v in tau.values()]
    elif tau is None:
        vals = []
    else:
        vals = [float(v) for v in np.asarray(tau, dtype=float).ravel()]
    vals = [v for v in vals if np.isfinite(v) or np.isposinf(v)]
    vals = sorted(vals, reverse=True)
    vals = [v for v in vals if v > 0]
    return np.asarray(vals, dtype=float)


def _sigma_trace_variance(CTMM: CTMMModel) -> float:
    sigma = CTMM.params.get("sigma")
    if isinstance(sigma, Covm):
        return var_covm(sigma, ave=True)
    matrix = CTMM.params.get("sigma_matrix")
    if matrix is not None:
        s = np.asarray(matrix, dtype=float)
        if s.ndim == 2 and s.shape[0] == s.shape[1]:
            return float(np.trace(s) / s.shape[0])
    return 1.0


def svf_func(CTMM: CTMMModel, moment: bool = False) -> dict[str, Callable]:
    """Port of ``svf.func`` for stationary CTMMs.

    The AKDE bandwidth path uses ``moment=FALSE`` and no drift contribution;
    stationary mean SVF is exactly zero in ctmm's ``R/mean.R``.
    """
    del moment
    tau = _tau_values(CTMM)
    finite_tau = tau[np.isfinite(tau)]
    range_ = bool(CTMM.params.get("range", True))
    omega = float(CTMM.params.get("omega", 0.0) or 0.0)
    circle = float(CTMM.params.get("circle", 0.0) or 0.0)
    sigma = _sigma_trace_variance(CTMM)

    K = int(finite_tau.size)

    def _as_array(t):
        return np.asarray(t, dtype=float)

    if K == 0 and range_:
        def acf(t):
            tt = _as_array(t)
            return np.where(tt == 0, 1.0, 0.0)
    elif K == 0:
        def acf(t):
            tt = _as_array(t)
            return 1.0 - tt
    elif K == 1 and range_:
        tau1 = float(finite_tau[0])

        def acf(t):
            tt = _as_array(t)
            return np.exp(-tt / tau1)
    elif K == 1:
        tau1 = float(finite_tau[0])

        def acf(t):
            tt = _as_array(t)
            return 1.0 - (tt - tau1 * (1.0 - np.exp(-tt / tau1)))
    else:
        tau2 = finite_tau[:2].astype(float)
        if tau2[0] > tau2[1]:
            def acf(t):
                tt = _as_array(t)
                vals = tau2 * np.exp(-tt[..., None] / tau2)
                return (vals[..., 1] - vals[..., 0]) / (tau2[1] - tau2[0])
        elif not omega:
            tau_c = float(tau2[0])

            def acf(t):
                tt = _as_array(t)
                return (1.0 + tt / tau_c) * np.exp(-tt / tau_c)
        else:
            f = float(np.mean(1.0 / tau2))
            nu = float(omega)

            def acf(t):
                tt = _as_array(t)
                return (np.cos(nu * tt) + (f / nu) * np.sin(nu * tt)) * np.exp(-f * tt)

    if not circle:
        def ACF(t):
            return acf(t)
    else:
        def ACF(t):
            tt = _as_array(t)
            return np.cos(circle * tt) * acf(tt)

    def SVF(t):
        return sigma * (1.0 - ACF(t))

    def VAR(t):
        tt = _as_array(t)
        return np.zeros_like(tt, dtype=float)

    def DOF(t):
        tt = _as_array(t)
        return np.full_like(tt, np.inf, dtype=float)

    return {"svf": SVF, "VAR": VAR, "DOF": DOF, "ACF": ACF}


def acf_grad(CTMM: CTMMModel, t):
    tau = _tau_values(CTMM)
    finite_tau = tau[np.isfinite(tau)]
    tt = np.asarray(t, dtype=float)
    if finite_tau.size == 0:
        return np.zeros((tt.size if tt.ndim else 1, 0), dtype=float)
    grads = []
    for tau_i in finite_tau:
        base = np.exp(-tt / tau_i)
        grads.append((tt / (tau_i * tau_i)) * base)
    return np.vstack([np.asarray(g, dtype=float).reshape(-1) for g in grads]).T


def plot_svf(lag, CTMM: CTMMModel, alpha=0.05, col="red", type="l", **kwargs):
    del alpha, col, type, kwargs
    lag = np.asarray(lag, dtype=float)
    svf = svf_func(CTMM, moment=True)["svf"]
    return {"lag": lag, "svf": svf(lag)}


def plot_variogram(x, CTMM=None, level: float = 0.95, units: bool = True, fraction: float = 0.5, **kwargs):
    del level, units, fraction, kwargs
    return {"type": "variogram", "variogram": x, "CTMM": CTMM}


def zoom_variogram(x, fraction: float = 1.0, **kwargs):
    del kwargs
    if hasattr(x, "iloc") and "lag" in x:
        mx = float(np.nanmax(x["lag"].to_numpy(dtype=float))) * float(fraction)
        return x[x["lag"] <= mx]
    return x


__all__ = ["acf_grad", "plot", "plot_svf", "plot_variogram", "svf_func", "zoom_variogram"]
