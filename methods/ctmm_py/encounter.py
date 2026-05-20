"""Partial parity translation of ctmm 1.3.0 ``R/encounter.R``."""
from __future__ import annotations
import numpy as np


def encounter_ecdf(data, UD=None, level: float = 0.95, debias: bool = True, res_time: float = 1.0, r=None, **kwargs):
    del UD, debias, res_time, r, kwargs
    tracks = data if isinstance(data, list) else [data]
    vals = []
    for tr in tracks:
        arr = np.asarray(tr.data[[tr.x_col, tr.y_col]], dtype=float) if hasattr(tr, "data") else np.asarray(tr, dtype=float)
        if arr.ndim == 2 and arr.shape[0] > 1:
            vals.extend(np.sqrt(np.sum(np.diff(arr[:, :2], axis=0) ** 2, axis=1)).tolist())
    vals = np.sort(np.asarray(vals, dtype=float))
    if vals.size == 0:
        return {"x": np.array([], dtype=float), "y": np.array([], dtype=float), "level": level}
    return {"x": vals, "y": np.arange(1, vals.size + 1, dtype=float) / vals.size, "level": level}


def encounter(a, b, threshold: float = 0.0):
    ax = np.asarray(a, dtype=float)
    bx = np.asarray(b, dtype=float)
    n = min(len(ax), len(bx))
    if n == 0:
        return {"count": 0, "fraction": 0.0}
    d = np.linalg.norm(ax[:n] - bx[:n], axis=1) if ax.ndim == 2 else np.abs(ax[:n]-bx[:n])
    hit = d <= threshold
    return {"count": int(np.sum(hit)), "fraction": float(np.mean(hit)), "distance": d}

__all__ = ["encounter", "encounter_ecdf"]
