"""Partial parity translation of ctmm 1.3.0 ``R/corridor.R``."""

from __future__ import annotations

import numpy as np


def _interp_track(df, tcol: str = "t", xcol: str = "x", ycol: str = "y", res_time: int = 100):
    t = np.asarray(df[tcol], dtype=float)
    x = np.asarray(df[xcol], dtype=float)
    y = np.asarray(df[ycol], dtype=float)
    ti = np.linspace(t[0], t[-1], res_time)
    xi = np.interp(ti, t, x)
    yi = np.interp(ti, t, y)
    return ti, xi, yi


def search(t, i, j, WIN=None):
    """Find indices in track ``j`` inside the time window around ``t[i]``."""
    tt = np.asarray(t, dtype=float)
    ti = tt[int(i)]
    if WIN is None:
        return np.asarray([int(j)], dtype=int)
    win = float(WIN)
    return np.where(np.abs(tt - ti) <= win)[0]


def corridor(data, CTMM, res_space: int = 10, res_time: int = 100, window: float = 86400.0, grid=None, **kwargs):
    del CTMM, res_space, window, grid, kwargs
    tracks = data if isinstance(data, list) else [data]
    interp = []
    for d in tracks:
        if hasattr(d, "data"):
            df = d.data
            tcol = d.time_col if hasattr(d, "time_col") else "t"
            xcol = d.x_col if hasattr(d, "x_col") else "x"
            ycol = d.y_col if hasattr(d, "y_col") else "y"
        else:
            df = d
            tcol, xcol, ycol = "t", "x", "y"
        ti, xi, yi = _interp_track(df, tcol=tcol, xcol=xcol, ycol=ycol, res_time=res_time)
        interp.append({"t": ti, "x": xi, "y": yi})

    n = len(interp)
    if n == 0:
        return {"type": "range", "variable": "utilization", "r": {}, "PDF": np.array([]), "weights": np.array([])}

    x_all = np.concatenate([u["x"] for u in interp])
    y_all = np.concatenate([u["y"] for u in interp])
    w_all = np.ones_like(x_all, dtype=float)
    w_all /= max(np.sum(w_all), 1.0)

    return {
        "type": "range",
        "variable": "utilization",
        "r": {"x": x_all, "y": y_all},
        "PDF": w_all,
        "weights": w_all,
        "tracks": interp,
    }


__all__ = ["corridor", "search"]
