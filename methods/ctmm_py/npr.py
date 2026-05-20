"""Partial parity translation of ctmm 1.3.0 ``R/npr.R``."""
from __future__ import annotations

import numpy as np


def _smooth_1d(x, y, xout=None):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if xout is None:
        xout = np.linspace(np.min(x), np.max(x), 128)
    xout = np.asarray(xout, dtype=float)
    if x.size <= 1:
        return {"x": xout, "fit": np.full_like(xout, y[0] if y.size else np.nan), "bandwidth": np.nan}
    h = max(1.06 * float(np.std(x, ddof=1)) * x.size ** (-1.0 / 5.0), np.finfo(float).eps)
    u = (xout[:, None] - x[None, :]) / h
    w = np.exp(-0.5 * u * u)
    fit = (w @ y) / np.maximum(np.sum(w, axis=1), np.finfo(float).eps)
    return {"x": xout, "fit": fit, "bandwidth": h}


def npr(data, UD=None, variable: str = "speed", normalize: bool = False, error: float = 0.001, **kwargs):
    del error, kwargs
    if UD is None:
        x, y = data
        return _smooth_1d(x, y)
    out = dict(UD)
    if not isinstance(UD, dict) or "PDF" not in UD:
        return out
    if variable == "revisitation":
        out["rate"] = float(np.nansum(np.asarray(UD["PDF"], dtype=float)) * np.prod(list(UD.get("dr", {"x": 1.0, "y": 1.0}).values())))
        if normalize:
            return out
    else:
        out[variable] = np.asarray(UD["PDF"], dtype=float)
        if normalize:
            pdf = np.asarray(out[variable], dtype=float)
            s = float(np.nansum(pdf))
            if s > 0:
                out["PDF"] = pdf / s
    return out


def revisitation(data, UD, error: float = 0.001, **kwargs):
    return npr(data, UD, variable="revisitation", normalize=True, error=error, **kwargs)


__all__ = ["npr", "revisitation"]
