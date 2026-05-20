"""Partial parity translation of ctmm 1.3.0 ``R/circle.R``."""

from __future__ import annotations

import numpy as np


def quantile_longitude(x, probs=(0.0, 1.0), na_rm: bool = False, **kwargs):
    del kwargs
    arr = np.asarray(x, dtype=float).reshape(-1)
    if na_rm:
        arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.full(len(np.atleast_1d(probs)), np.nan)

    angle = np.sort(arr)
    wrap = np.concatenate([angle, angle[:1]])
    delta = np.mod(np.diff(wrap), 360.0)
    max_i = int(np.argmax(delta))
    compact = angle[: max_i + 1]
    if max_i + 1 < angle.size:
        compact = np.concatenate([angle[max_i + 1 :], compact])
    m0 = compact[0]
    centered = np.mod(compact - m0, 360.0)
    q = np.quantile(centered, probs)
    out = q + m0
    out = np.mod(out + 180.0, 360.0) - 180.0
    return out


def median_longitude(x, na_rm: bool = False, **kwargs):
    return float(np.ravel(quantile_longitude(x, probs=(0.5,), na_rm=na_rm, **kwargs))[0])


__all__ = ["quantile_longitude", "median_longitude"]
