"""Parity translation of ctmm 1.3.0 ``R/quant.R``."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def quant(x, p: float = 0.5, low: float = -float("inf"), high: float = float("inf")):
    arr = np.sort(np.asarray(x, dtype=float).ravel())
    n = arr.size
    if n == 0:
        return np.array([], dtype=float)
    q = np.arange(1, n + 1, dtype=float) / (n + 1.0)
    fn = PchipInterpolator(arr, q, extrapolate=True)
    idx = float(p) * (n + 1.0)
    lo_i = int(np.floor(idx))
    hi_i = int(np.ceil(idx))
    lo = arr[lo_i - 1] if lo_i > 0 else float(low)
    hi = arr[hi_i - 1] if hi_i <= n else float(high)
    if lo == -float("inf"):
        X = arr[0] - 1.0
    elif hi == float("inf"):
        X = arr[-1] + 1.0
    else:
        X = (lo + hi) / 2.0
    c = np.array([float(fn(X) - p), float(fn(X, 1)), float(fn(X, 2)) / 2.0, float(fn(X, 3)) / 6.0], dtype=float)
    roots = np.roots(c[::-1])
    out = X + roots
    out = out[np.abs(np.imag(out)) <= np.finfo(float).eps * n].real
    out = out[(out >= lo) & (out <= hi)]
    return out


__all__ = ["quant"]
