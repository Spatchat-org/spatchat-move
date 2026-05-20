"""Partial parity translation of ctmm 1.3.0 ``R/diagnostic.R``."""

from __future__ import annotations

import numpy as np


def intensity(
    data,
    UD,
    RSF,
    R=None,
    variable=None,
    empirical: bool = False,
    level: float = 0.95,
    ticks: bool = True,
    smooth: bool = True,
    interpolate: bool = True,
    **kwargs,
):
    del level, ticks, smooth, interpolate, kwargs
    if hasattr(data, "data"):
        df = data.data.copy()
        xcol = data.x_col
        ycol = data.y_col
    else:
        df = data.copy()
        xcol = "x"
        ycol = "y"

    if variable is None:
        if R and len(R) == 1:
            variable = list(R.keys())[0]
        elif R:
            return {k: intensity(data, UD, RSF, R=R, variable=k, empirical=empirical) for k in R.keys()}
        else:
            variable = xcol

    x = np.asarray(df[xcol], dtype=float)
    y = np.asarray(df[ycol], dtype=float)
    mu = np.asarray(RSF.get("mu", [np.nanmean(x), np.nanmean(y)]), dtype=float)
    smaj = float(RSF.get("sigma", [[np.nanvar(x) + np.nanvar(y)]])[0][0]) if isinstance(RSF, dict) else float(np.nanvar(x) + np.nanvar(y))
    smaj = max(smaj, np.finfo(float).eps)
    log_p = -((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / (2.0 * smaj)

    if variable in df.columns:
        rv = np.asarray(df[variable], dtype=float)
    else:
        rv = np.linspace(np.nanmin(x), np.nanmax(x), len(x))

    order = np.argsort(rv)
    rv = rv[order]
    est = log_p[order] - np.nanmax(log_p)
    se = np.full_like(est, np.nanstd(est) / max(np.sqrt(len(est)), 1.0))
    emp = None
    if empirical:
        hist, edges = np.histogram(rv, bins=max(20, int(np.sqrt(len(rv)))), density=True)
        mids = (edges[:-1] + edges[1:]) / 2.0
        emp = {"x": mids, "log_pdf": np.log(np.maximum(hist, np.finfo(float).eps))}

    return {"variable": variable, "x": rv, "est": est, "se": se, "empirical": emp, "UD": UD, "RSF": RSF}


__all__ = ["intensity"]
