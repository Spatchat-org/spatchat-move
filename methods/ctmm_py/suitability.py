"""Partial parity translation of ctmm 1.3.0 ``R/suitability.R``."""
from __future__ import annotations
import numpy as np

def suitability(values):
    v = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(v)
    return (v-lo)/(hi-lo)


def R_suit(values, weights=None):
    s = suitability(values)
    if weights is None:
        return float(np.nanmean(s))
    w = np.asarray(weights, dtype=float)
    w = w / np.nansum(w)
    return float(np.nansum(w * s))


__all__ = ["R_suit", "suitability"]
