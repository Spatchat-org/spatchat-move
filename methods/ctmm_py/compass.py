"""Partial parity translation of ctmm 1.3.0 ``R/compass.R``."""

from __future__ import annotations

import numpy as np


def compass(loc=None, cex: float = 3.0, **kwargs):
    del cex, kwargs
    if loc is None:
        return {"x": np.nan, "y": np.nan, "srt": 0.0, "label": "➢"}
    arr = np.asarray(loc, dtype=float).reshape(-1)
    if arr.size < 2:
        return {"x": np.nan, "y": np.nan, "srt": 0.0, "label": "➢"}
    return {"x": float(arr[0]), "y": float(arr[1]), "srt": 0.0, "label": "➢"}


__all__ = ["compass"]
