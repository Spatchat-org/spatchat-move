"""Partial parity translation of ctmm 1.3.0 ``R/sigfig.R``."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .stats import NAMES_CI
from .units import dimfig


def sigfig(est, VAR=None, SD=None, level: float = 0.95, digits: int = 2, **kwargs):
    del kwargs
    if VAR is not None:
        SD = np.sqrt(np.asarray(VAR, dtype=float))

    if SD is not None:
        est_arr = np.asarray(est, dtype=float).reshape(-1)
        sd_arr = np.asarray(SD, dtype=float).reshape(-1)
        alpha = 1.0 - level
        z = norm.ppf(1.0 - alpha / 2.0)
        ci = np.column_stack([est_arr - z * sd_arr, est_arr, est_arr + z * sd_arr])
        out = sigfig(ci, digits=digits)
        return out

    arr = np.asarray(est, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        return np.asarray([str(v) for v in arr.ravel()], dtype=object)

    out = []
    for row in arr:
        lo, mid, hi = map(float, row[:3])
        dif = np.array([mid - lo, hi - mid], dtype=float)
        dref = min(abs(dif[0]), abs(dif[1])) if np.all(np.isfinite(dif)) and np.all(np.abs(dif) > 0) else max(np.abs(dif).max(initial=1.0), 1.0)
        p = np.floor(np.log10(dref) + np.finfo(float).eps) - digits + 1
        first = np.floor(np.log10(abs(mid))) if mid != 0 else 0.0
        sig = int(first - p + 1)
        sig = max(sig, 1)

        def fmt(v: float) -> str:
            if np.isposinf(v):
                return "∞"
            if np.isneginf(v):
                return "-∞"
            s = np.format_float_positional(float(np.round(v, decimals=max(0, -int(p)))), trim="-")
            if "e" not in s.lower() and "." in s:
                s = s.rstrip("0").rstrip(".")
            return s

        ms = fmt(mid)
        ls = fmt(lo)
        hs = fmt(hi)
        out.append(f"{ms} ({ls}—{hs})")
    return np.asarray(out, dtype=object)


def FORMAT(x, *args, **kwargs):
    del args, kwargs
    return sigfig(x)


__all__ = ["FORMAT", "sigfig", "dimfig", "NAMES_CI"]
