"""Partial parity translation of ctmm 1.3.0 ``R/outlier.R``."""

from __future__ import annotations

from math import gcd as _gcd

import numpy as np
import pandas as pd
from scipy import optimize, special

from .outlie_ops import outlie as _outlie


def outlie(data, plot: bool = True, by: str = "d", units: bool = True, **kwargs):
    return _outlie(data, plot=plot, by=by, units=units, **kwargs)


def time_res(DT):
    dt = np.asarray(DT, dtype=float)
    pos = dt[dt > 0]
    if pos.size == 0:
        return np.array([1.0, 1.0], dtype=float)
    if pos.size == 1:
        return np.array([float(pos[0]), min(1.0, float(pos[0]) / 2.0)], dtype=float)

    m = int(max(np.ceil(-np.log10(np.min(pos))), 0))
    scale = 10**m
    ints = np.round(pos * scale).astype(int)
    base = gcd_vec(ints) / scale

    z = dt == 0
    if np.any(z):
        run = 0
        best = 0
        for flag in z:
            run = run + 1 if flag else 0
            best = max(best, run)
        fix = base / (1 + best)
    else:
        fix = 0.0
    return np.array([float(base), float(fix)], dtype=float)


def gcd_vec(vec):
    v = np.asarray(vec, dtype=int)
    v = v[v != 0]
    if v.size == 0:
        return 1
    g = int(abs(v[0]))
    for x in v[1:]:
        g = _gcd(g, int(abs(x)))
    return g


def gcd(x: int, y: int) -> int:
    return _gcd(int(x), int(y))


BESSEL_LIMIT = 2**16


def BesselSolver(z):
    z = np.asarray(z, dtype=float)
    zz = np.clip(z, 0.0, BESSEL_LIMIT)
    return special.i0e(zz) / np.maximum(special.i1e(zz), np.finfo(float).eps)


def TanhSolver(z):
    z = np.asarray(z, dtype=float)
    return np.tanh(z)


def distanceMLE(d, UERE=0, axes=("x", "y"), return_VAR: bool = False, **kwargs):
    del axes, kwargs
    d = np.asarray(d, dtype=float)
    err = np.asarray(UERE, dtype=float)
    if err.ndim >= 2:
        err = np.nanmean(np.diagonal(err, axis1=-2, axis2=-1), axis=-1)
    err = np.resize(err.reshape(-1), d.size) if err.size else np.zeros(d.size)
    est = np.sqrt(np.maximum(d * d - err, 0.0))
    var = np.maximum(err, 0.0)
    if return_VAR:
        return np.column_stack([est, var])
    return est


def speedMLE(data, UERE=0, DT=None, axes=("x", "y"), **kwargs):
    del kwargs
    df = data.data if hasattr(data, "data") else pd.DataFrame(data)
    cols = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    vals = df[cols].to_numpy(dtype=float)
    if DT is None:
        if "t" in df:
            DT = np.diff(pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float))
        else:
            DT = np.ones(max(len(vals) - 1, 0), dtype=float)
    dt = np.asarray(DT, dtype=float).reshape(-1)
    d = np.sqrt(np.sum(np.diff(vals, axis=0) ** 2, axis=1))
    n = min(d.size, dt.size)
    v = d[:n] / np.maximum(dt[:n], np.finfo(float).eps)
    err = np.asarray(UERE, dtype=float)
    var = np.zeros_like(v) if err.size == 0 else np.full_like(v, float(np.nanmean(err)) / np.maximum(np.nanmean(dt[:n]) ** 2, np.finfo(float).eps))
    return {"X": v, "VAR": var}


def assign_speeds(data, DT=None, UERE=0, method: str = "max", axes=("x", "y")):
    if method not in {"max", "min"}:
        raise ValueError("method must be 'max' or 'min'")
    sp = speedMLE(data, UERE=UERE, DT=DT, axes=axes)
    v_dt = np.asarray(sp["X"], dtype=float)
    var_dt = np.asarray(sp["VAR"], dtype=float)
    if v_dt.size == 0:
        return {"v.t": np.array([], dtype=float), "VAR.t": np.array([], dtype=float), "v.dt": v_dt, "VAR.dt": var_dt}
    if v_dt.size == 1:
        return {"v.t": np.repeat(v_dt[0], 2), "VAR.t": np.repeat(var_dt[0], 2), "v.dt": v_dt, "VAR.dt": var_dt}
    reducer = np.maximum if method == "max" else np.minimum
    v_t = np.r_[v_dt[0], reducer(v_dt[:-1], v_dt[1:]), v_dt[-1]]
    var_t = np.r_[var_dt[0], np.maximum(var_dt[:-1], var_dt[1:]), var_dt[-1]]
    return {"v.t": v_t, "VAR.t": var_t, "v.dt": v_dt, "VAR.dt": var_dt}


def mid(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim:
        return arr[1]
    return arr


def plot_outlie(x, *args, **kwargs):
    del args, kwargs
    return {"type": "outlie", "object": x}


__all__ = [
    "BESSEL_LIMIT",
    "BesselSolver",
    "TanhSolver",
    "assign_speeds",
    "distanceMLE",
    "gcd",
    "gcd_vec",
    "mid",
    "outlie",
    "plot_outlie",
    "speedMLE",
    "time_res",
]
