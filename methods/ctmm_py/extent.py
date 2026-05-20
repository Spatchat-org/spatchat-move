"""Parity-focused translation of ctmm 1.3.0 ``R/extent.R``."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .types import CTMMModel, Telemetry
from .stats import chisq_ci


def as_matrix_Extent(x, *args, **kwargs):
    del args, kwargs
    if isinstance(x, dict):
        return pd.DataFrame({k: v for k, v in x.items()}, index=["min", "max"])
    return pd.DataFrame(x, index=["min", "max"])


def _numeric_extent(df: pd.DataFrame, level: float = 1.0) -> pd.DataFrame:
    alpha = (1.0 - float(np.max(np.asarray(level, dtype=float)))) / 2.0
    probs = [alpha, 1.0 - alpha]
    out = {}
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            out[col] = np.nanquantile(vals.to_numpy(dtype=float), probs)
    return pd.DataFrame(out, index=["min", "max"])


def extent_telemetry(x, level: float = 1.0, *args, **kwargs):
    del args, kwargs
    df = x.data if isinstance(x, Telemetry) else x
    return _numeric_extent(pd.DataFrame(df), level=level)


def extent_matrix(x, level: float = 1.0, *args, **kwargs):
    del args, kwargs
    return _numeric_extent(pd.DataFrame(np.asarray(x, dtype=float)), level=level)


def extent_list(x, *args, **kwargs):
    ranges = [extent(item, *args, **kwargs) for item in x]
    ranges = [r for r in ranges if isinstance(r, pd.DataFrame) and not r.empty]
    if not ranges:
        return pd.DataFrame()
    cols = list(dict.fromkeys(c for r in ranges for c in r.columns))
    aligned = []
    for r in ranges:
        rr = r.copy()
        for c in cols:
            if c not in rr.columns:
                rr[c] = np.nan
        aligned.append(rr[cols])
    stack = pd.concat(aligned, axis=0)
    return _numeric_extent(stack, level=1.0)


def _sigma_matrix(model: CTMMModel) -> np.ndarray:
    sigma = model.params.get("sigma_matrix")
    if sigma is not None:
        return np.asarray(sigma, dtype=float)
    sig = model.params.get("sigma")
    if hasattr(sig, "sigma"):
        return np.asarray(sig.sigma, dtype=float)
    if sig is not None:
        return np.asarray(sig, dtype=float)
    return np.eye(2, dtype=float)


def extent_ctmm(x, level: float = 0.95, level_UD: float = 0.95, *args, **kwargs):
    del args, kwargs
    params = x.params if isinstance(x, CTMMModel) else x
    if params.get("mu") is None:
        raise ValueError("This model has no mean location. Try ctmm_guess.")
    mu = np.asarray(params.get("mu"), dtype=float).reshape(-1)
    axes = list(params.get("axes", ("x", "y")))
    sigma = _sigma_matrix(x if isinstance(x, CTMMModel) else CTMMModel("ctmm", params))
    dim = min(len(axes), mu.size, sigma.shape[0])
    alpha_ud = 1.0 - float(np.max(np.asarray(level_UD, dtype=float)))
    z = np.sqrt(-2.0 * np.log(alpha_ud)) if dim == 2 else norm.ppf(1.0 - alpha_ud / 2.0)
    const = 1.0
    dof = params.get("DOF", {})
    dof_area = dof.get("area") if isinstance(dof, dict) else params.get("DOF.area")
    if dof_area is not None and np.isfinite(float(dof_area)) and float(dof_area) > 0 and not np.isnan(level):
        df = 2.0 * float(dof_area)
        const = float(chi2.ppf(1.0 - (1.0 - float(level)) / 2.0, df) / df)
    buff = float(z) * np.sqrt(np.maximum(const * np.diag(sigma)[:dim], 0.0))
    out = {axes[i]: [mu[i] - buff[i], mu[i] + buff[i]] for i in range(dim)}
    return pd.DataFrame(out, index=["min", "max"])


def extent_UD(x, level: float = 0.95, level_UD: float = 0.95, complete: bool = False, *args, **kwargs):
    del complete, args, kwargs
    level = float(np.max(np.asarray(level, dtype=float)))
    level_UD = float(np.max(np.asarray(level_UD, dtype=float)))
    if level_UD == 1.0 or (not np.isnan(level) and level == 1.0 and x.get("DOF.area") is not None):
        return pd.DataFrame({"x": [-np.inf, np.inf], "y": [-np.inf, np.inf]}, index=["min", "max"])
    cdf = np.asarray(x["CDF"], dtype=float)
    if x.get("DOF.area") is None or np.isnan(level):
        p = level_UD
    else:
        ci = chisq_ci(level_UD, dof=2.0 * float(np.asarray(x["DOF.area"]).reshape(-1)[0]), level=level)
        p = float(ci[2])
    p = max(p, float(np.nanmin(cdf)))
    mask = cdf <= p
    gx = np.asarray(x["r"]["x"], dtype=float)
    gy = np.asarray(x["r"]["y"], dtype=float)
    ix = np.where(np.any(mask, axis=1))[0]
    iy = np.where(np.any(mask, axis=0))[0]
    if ix.size == 0 or iy.size == 0:
        return pd.DataFrame({"x": [np.nan, np.nan], "y": [np.nan, np.nan]}, index=["min", "max"])
    return pd.DataFrame({"x": [gx[ix[0]], gx[ix[-1]]], "y": [gy[iy[0]], gy[iy[-1]]]}, index=["min", "max"])


def extent_variogram(x, level: float = 0.95, threshold: float = 2.0, *args, **kwargs):
    del args, kwargs
    lag = np.asarray(x.get("lag", x.get("lags_s", [])), dtype=float)
    svf = np.asarray(x.get("SVF", x.get("gamma", [])), dtype=float)
    dof = np.asarray(x.get("DOF", np.ones_like(svf)), dtype=float)
    if lag.size == 0 or svf.size == 0:
        return pd.DataFrame({"x": [0.0, 0.0], "y": [0.0, 0.0]}, index=["min", "max"])
    alpha = 1.0 - float(level)
    upper = svf * np.array([chisq_ci(1.0, d, level=1.0 - alpha)[2] if np.isfinite(d) and d > 0 else 1.0 for d in dof[: svf.size]])
    ymax = min(float(np.nanmax(upper)), float(threshold * np.nanmax(svf)))
    return pd.DataFrame({"x": [0.0, float(np.nanmax(lag))], "y": [0.0, ymax]}, index=["min", "max"])


def min_extent(*args, na_rm: bool = False):
    if not args:
        return pd.DataFrame({"x": [-np.inf, np.inf], "y": [-np.inf, np.inf]}, index=["min", "max"])
    frames = [pd.DataFrame(a) for a in args]
    cols = list(dict.fromkeys(c for f in frames for c in f.columns))
    out = frames[0].copy()
    for c in cols:
        lows = [pd.to_numeric(f.get(c), errors="coerce").iloc[0] for f in frames if c in f]
        highs = [pd.to_numeric(f.get(c), errors="coerce").iloc[1] for f in frames if c in f]
        out.loc["min", c] = np.nanmax(lows) if na_rm else max(lows)
        out.loc["max", c] = np.nanmin(highs) if na_rm else min(highs)
        out[c] = np.sort(out[c].to_numpy(dtype=float))
    return out[cols]


def extent(x, level: float = 1.0, **kwargs):
    if isinstance(x, list):
        return extent_list(x, level=level, **kwargs)
    if isinstance(x, CTMMModel):
        return extent_ctmm(x, level=level, **kwargs)
    if isinstance(x, Telemetry) or isinstance(x, pd.DataFrame):
        return extent_telemetry(x, level=level, **kwargs)
    if isinstance(x, dict) and "CDF" in x and "r" in x:
        return extent_UD(x, level=level, **kwargs)
    if isinstance(x, dict) and ("SVF" in x or "gamma" in x):
        return extent_variogram(x, level=level, **kwargs)
    if isinstance(x, np.ndarray):
        return extent_matrix(x, level=level, **kwargs)
    raise TypeError("unsupported type for extent")


__all__ = [
    "as_matrix_Extent",
    "extent",
    "extent_list",
    "extent_telemetry",
    "extent_matrix",
    "extent_ctmm",
    "extent_UD",
    "extent_variogram",
    "min_extent",
]
