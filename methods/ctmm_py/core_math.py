from __future__ import annotations

import numpy as np


def composite(n: int) -> int:
    n = int(max(n, 1))
    return int(2 ** np.ceil(np.log2(n)))


def clamp(num, min_val=0.0, max_val=1.0):
    return np.where(num < min_val, min_val, np.where(num > max_val, max_val, num))


def dexp2(x, Exp=None):
    """``R/dexp.R`` ``dexp2``: numerically stable ``1 - exp(-x)^2``."""
    x = np.asarray(x, dtype=float)
    if Exp is None:
        Exp = np.exp(-x)
    else:
        Exp = np.asarray(Exp, dtype=float)
    out = np.empty(np.broadcast(x, Exp).shape, dtype=float)
    xb = np.broadcast_to(x, out.shape)
    eb = np.broadcast_to(Exp, out.shape)
    mask = eb < 0.7071068
    out[mask] = 1.0 - eb[mask] ** 2
    out[~mask] = 2.0 * eb[~mask] * np.sinh(xb[~mask])
    return out.item() if out.shape == () else out


def dexp1(x, Exp=None):
    """``R/dexp.R`` ``dexp1``: numerically stable ``1 - exp(-x)``."""
    x = np.asarray(x, dtype=float)
    if Exp is None:
        Exp = np.exp(-x)
    else:
        Exp = np.asarray(Exp, dtype=float)
    out = np.empty(np.broadcast(x, Exp).shape, dtype=float)
    xb = np.broadcast_to(x, out.shape)
    eb = np.broadcast_to(Exp, out.shape)
    mask = eb < 0.5
    out[mask] = 1.0 - eb[mask]
    out[~mask] = 2.0 * np.sqrt(eb[~mask]) * np.sinh(xb[~mask] / 2.0)
    return out.item() if out.shape == () else out


def pad(vec, size=None, diff=None, padding=0.0, side=+1):
    arr = np.asarray(vec)
    if size is None:
        size = arr.shape[0]
    if diff is None:
        diff = int(size - arr.shape[0])
    if diff <= 0:
        return arr
    padv = np.full((diff,), padding, dtype=arr.dtype)
    if side > 0:
        return np.concatenate([arr, padv])
    if side < 0:
        return np.concatenate([padv, arr])
    return arr


def rpad(mat, size=None, diff=None, padding=0.0, side=+1):
    arr = np.asarray(mat)
    if arr.ndim == 1:
        arr = arr[:, None]
    if size is None:
        size = arr.shape[0]
    if diff is None:
        diff = int(size - arr.shape[0])
    if diff <= 0:
        return arr
    padm = np.full((diff, arr.shape[1]), padding, dtype=arr.dtype)
    if side > 0:
        return np.vstack([arr, padm])
    if side < 0:
        return np.vstack([padm, arr])
    return arr


def FFT(x, inverse: bool = False):
    arr = np.asarray(x)
    if arr.ndim <= 1:
        if not inverse:
            return np.fft.fft(arr)
        return np.fft.ifft(arr)
    if not inverse:
        return np.fft.fft(arr, axis=0)
    return np.fft.ifft(arr, axis=0)


def IFFT(x):
    return FFT(x, inverse=True)
