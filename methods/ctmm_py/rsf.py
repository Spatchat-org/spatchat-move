"""Parity-focused translation of ctmm 1.3.0 ``R/rsf.R`` helper surface."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def get_offset(formula):
    text = str(formula)
    return re.findall(r"offset\\(([^)]+)\\)", text)


def _design_from_formula(data: pd.DataFrame, formula=None):
    if formula is None:
        cols = [c for c in data.columns if c != "count" and pd.api.types.is_numeric_dtype(data[c])]
    else:
        rhs = str(formula).split("~", 1)[-1]
        cols = [c.strip() for c in re.split(r"\\+", rhs) if c.strip() and not c.strip().startswith("offset(")]
        cols = [c for c in cols if c in data.columns]
    if not cols:
        X = np.ones((len(data), 1), dtype=float)
        names = ["(Intercept)"]
    else:
        X = np.column_stack([np.ones(len(data)), *[pd.to_numeric(data[c], errors="coerce").fillna(0.0).to_numpy(dtype=float) for c in cols]])
        names = ["(Intercept)", *cols]
    return X, names


def nloglike(beta, X, y, offset=0):
    beta = np.asarray(beta, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    eta = X @ beta + offset
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta))


def nll(beta, X, y, offset=0):
    return nloglike(beta, X, y, offset=offset)


def rsf_loglike(beta, X, y, offset=0):
    return -nloglike(beta, X, y, offset=offset)


def is_stationary(model):
    if hasattr(model, "params"):
        return str(model.params.get("mean", "stationary")) == "stationary"
    if isinstance(model, dict):
        return str(model.get("mean", "stationary")) == "stationary"
    return True


def fn(*args, **kwargs):
    return rsf_fit(*args, **kwargs)


def evaluate(beta, X, offset=0):
    eta = np.asarray(X, dtype=float) @ np.asarray(beta, dtype=float) + offset
    return 1.0 / (1.0 + np.exp(-eta))


def expand_factors(R=None, formula=None, reference="auto", data=None, DVARS=None):
    del reference, DVARS
    if data is None:
        data = pd.DataFrame()
    data = data.copy()
    rasters = {} if R is None else dict(R)
    for name, values in list(rasters.items()):
        if isinstance(values, pd.Categorical) or getattr(values, "dtype", None) == object:
            dummies = pd.get_dummies(values, prefix=name, drop_first=True)
            for col in dummies:
                rasters[col] = dummies[col]
            rasters.pop(name, None)
    return {"data": data, "R": rasters, "formula": formula}


def R_prepare(R=None, data=None, **kwargs):
    del kwargs
    return {"R": {} if R is None else dict(R), "data": data}


def R_extract(R, data, interpolate=True, **kwargs):
    del interpolate, kwargs
    out = pd.DataFrame(index=np.arange(len(data)))
    for name, raster in (R or {}).items():
        if callable(raster):
            out[name] = raster(data)
        elif np.isscalar(raster):
            out[name] = float(raster)
        else:
            arr = np.asarray(raster).ravel()
            out[name] = np.resize(arr, len(data))
    return out


def R_grid(R, grid=None, **kwargs):
    del kwargs
    return {"R": R, "grid": grid}


def rsf_fit(data, UD=None, R=None, formula=None, integrated: bool = True, max_iter: int = 200, lr: float = 0.1, **kwargs):
    del UD, integrated, kwargs
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if R:
            df = pd.concat([df.reset_index(drop=True), R_extract(R, df).reset_index(drop=True)], axis=1)
        y = pd.to_numeric(df["count"], errors="coerce").fillna(1.0).to_numpy(dtype=float) if "count" in df else np.ones(len(df), dtype=float)
        X, names = _design_from_formula(df, formula=formula)
    else:
        X = np.asarray(data, dtype=float)
        y = np.asarray(UD, dtype=float).reshape(-1) if UD is not None else np.ones(X.shape[0], dtype=float)
        names = [f"b{i}" for i in range(X.shape[1])]
    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(int(max_iter)):
        p = evaluate(beta, X)
        g = X.T @ (p - y) / max(len(y), 1)
        beta -= float(lr) * g
    return {"beta": beta, "coef": dict(zip(names, beta)), "nloglike": nloglike(beta, X, y), "formula": formula}


__all__ = [
    "R_extract",
    "R_grid",
    "R_prepare",
    "evaluate",
    "expand_factors",
    "get_offset",
    "fn",
    "is_stationary",
    "nll",
    "nloglike",
    "rsf_loglike",
    "rsf_fit",
]
