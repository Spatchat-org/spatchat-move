"""Parity translation of ctmm 1.3.0 ``R/modes.R`` helpers."""

from __future__ import annotations

import numpy as np


def modes_numeric(x, method: str = "exact", na_rm: bool = True):
    """``modes.numeric`` sample mode with R's log-distance tie breaker."""
    del method
    arr = np.asarray(x, dtype=float).ravel()
    if na_rm:
        arr = arr[~np.isnan(arr)]
    arr = np.sort(arr)
    if arr.size == 0:
        return np.nan
    unique, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    if unique.size < arr.size:
        max_count = np.max(counts)
        candidates = unique[counts == max_count]
        if candidates.size == 1:
            return float(candidates[0])
        include = np.isin(arr, candidates)
    else:
        include = np.ones(arr.size, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.log(np.abs(arr[:, None] - arr[None, :]))
    np.fill_diagonal(d, 0.0)
    if np.sum(include) < arr.size:
        d[np.ix_(include, include)] = 0.0
        d[np.ix_(~include, ~include)] = float("inf")
    return float(arr[int(np.argmin(np.sum(d, axis=1)))])


def DiffGrid(SUB, dx, dy, dx2=None, dy2=None, dr2=None, dxdydr2=None):
    """Second-order 2D gradient/Hessian from a 3x3 grid."""
    sub = np.asarray(SUB, dtype=float)
    dx = float(dx)
    dy = float(dy)
    dx2 = dx * dx if dx2 is None else float(dx2)
    dy2 = dy * dy if dy2 is None else float(dy2)
    dr2 = dx2 + dy2 if dr2 is None else float(dr2)
    dxdydr2 = dx * dy / dr2 if dxdydr2 is None else float(dxdydr2)

    grad = np.zeros(2, dtype=float)
    grad[0] = np.mean([sub[1, 1] - sub[0, 1], sub[2, 1] - sub[1, 1]]) / dx
    grad[1] = np.mean([sub[1, 1] - sub[1, 0], sub[1, 2] - sub[1, 1]]) / dy

    hess = np.eye(2, dtype=float)
    hess[0, 0] = (sub[2, 1] - 2.0 * sub[1, 1] + sub[0, 1]) / dx2
    hess[1, 1] = (sub[1, 2] - 2.0 * sub[1, 1] + sub[1, 0]) / dy2
    h12a = (sub[2, 0] - 2.0 * sub[1, 1] + sub[0, 2]) / dr2
    h12b = (sub[2, 2] - 2.0 * sub[1, 1] + sub[0, 0]) / dr2
    hess[0, 1] = hess[1, 0] = dxdydr2 * (h12b - h12a)
    return {"GRAD": grad, "HESS": hess}


def modes(object, **kwargs):
    if np.isscalar(object) or isinstance(object, (list, tuple, np.ndarray)):
        return modes_numeric(object, **kwargs)
    params = getattr(object, "params", None)
    if isinstance(params, dict) and "mu" in params:
        return np.asarray(params["mu"], dtype=float)
    raise TypeError("modes currently supports numeric arrays and CTMM-like objects with params['mu']")


def modes_UD(object, **kwargs):
    return modes(object, **kwargs)


def modes_ctmm(object, **kwargs):
    return modes(object, **kwargs)


__all__ = ["DiffGrid", "modes", "modes_numeric", "modes_UD", "modes_ctmm"]
