"""Parity translation of ctmm 1.3.0 ``R/int.R`` interpolation helpers."""

from __future__ import annotations

import numpy as np


def intensity(values, weights=None):
    x = np.asarray(values, dtype=float).reshape(-1)
    if weights is None:
        return float(np.nanmean(x)) if x.size else float("nan")
    w = np.asarray(weights, dtype=float).reshape(-1)
    m = min(x.size, w.size)
    if m == 0:
        return float("nan")
    sw = float(np.sum(w[:m]))
    return float(np.sum(x[:m] * w[:m]) / sw) if sw > 0 else float(np.nanmean(x[:m]))


def vint(vec, ind, return_ind: bool = False):
    v = np.asarray(vec, dtype=float)
    ind = np.asarray(ind, dtype=float)
    n = v.size
    lo = np.floor(ind).astype(int)
    hi = np.ceil(ind).astype(int)
    lo = np.minimum(np.maximum(lo, 1), max(n - 1, 1))
    hi = np.minimum(np.maximum(hi, 2), max(n, 2))
    if return_ind:
        return np.vstack([lo, hi])
    lo0 = lo - 1
    hi0 = hi - 1
    out = v[lo0] + (v[hi0] - v[lo0]) * (ind - lo)
    return out


def mint(mat, ind):
    m = np.asarray(mat, dtype=float)
    idx = vint(m[0, :], ind, return_ind=True)
    lo = idx[0, :] - 1
    hi = idx[1, :] - 1
    return m[:, lo] + (m[:, hi] - m[:, lo]) * (np.asarray(ind) - idx[0, :])


def bint(M, ind, ext=True):
    arr = np.asarray(M, dtype=float)
    ind = np.asarray(ind, dtype=float)
    if ind.ndim == 1:
        ind = ind.reshape(2, -1)
    ix = vint(arr[:, 0], ind[0, :], return_ind=True)
    iy = vint(arr[0, :], ind[1, :], return_ind=True)
    out = np.empty(ind.shape[1], dtype=float)
    for k in range(ind.shape[1]):
        wx = np.array([ix[1, k] - ind[0, k], ind[0, k] - ix[0, k]], dtype=float)
        wy = np.array([iy[1, k] - ind[1, k], ind[1, k] - iy[0, k]], dtype=float)
        wx = np.nan_to_num(wx / np.sum(wx), nan=0.5)
        wy = np.nan_to_num(wy / np.sum(wy), nan=0.5)
        out[k] = wx @ arr[np.ix_(ix[:, k] - 1, iy[:, k] - 1)] @ wy
    if ext is np.nan:
        bad = (ind[0, :] < 0.5) | (ind[0, :] > arr.shape[0] + 0.5) | (ind[1, :] < 0.5) | (ind[1, :] > arr.shape[1] + 0.5)
        out[bad] = np.nan
    return out


def BINT(i, M, ind, INDx=None, INDy=None):
    del INDx, INDy
    return bint(M, np.asarray(ind)[:, [i]])[0]


def tint(M, ind, ext=True):
    arr = np.asarray(M, dtype=float)
    ind = np.asarray(ind, dtype=float)
    if ind.ndim == 1:
        ind = ind.reshape(3, -1)
    ix = vint(arr[:, 0, 0], ind[0, :], return_ind=True)
    iy = vint(arr[0, :, 0], ind[1, :], return_ind=True)
    iz = vint(arr[0, 0, :], ind[2, :], return_ind=True)
    out = np.empty(ind.shape[1], dtype=float)
    for k in range(ind.shape[1]):
        wx = np.array([ix[1, k] - ind[0, k], ind[0, k] - ix[0, k]], dtype=float)
        wy = np.array([iy[1, k] - ind[1, k], ind[1, k] - iy[0, k]], dtype=float)
        wz = np.array([iz[1, k] - ind[2, k], ind[2, k] - iz[0, k]], dtype=float)
        wx = np.nan_to_num(wx / np.sum(wx), nan=0.5)
        wy = np.nan_to_num(wy / np.sum(wy), nan=0.5)
        wz = np.nan_to_num(wz / np.sum(wz), nan=0.5)
        block = arr[np.ix_(ix[:, k] - 1, iy[:, k] - 1, iz[:, k] - 1)]
        out[k] = np.einsum("i,j,k,ijk->", wx, wy, wz, block)
    if ext is np.nan:
        bad = (ind[0, :] < 0.5) | (ind[0, :] > arr.shape[0] + 0.5) | (ind[1, :] < 0.5) | (ind[1, :] > arr.shape[1] + 0.5) | (ind[2, :] < 0.5) | (ind[2, :] > arr.shape[2] + 0.5)
        out[bad] = np.nan
    return out


def CINT(i, M, ind, *args):
    del args
    return tint(M, np.asarray(ind)[:, [i]])[0]


__all__ = ["BINT", "CINT", "bint", "intensity", "mint", "tint", "vint"]
