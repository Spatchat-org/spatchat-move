"""Partial parity translation of ctmm 1.3.0 ``R/mean.ctmm.R``."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def _as_array(x, default=None):
    if x is None:
        return np.asarray(default if default is not None else [], dtype=float)
    return np.asarray(x, dtype=float)


def _weighted_average(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    sw = np.sum(w)
    if sw <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / sw
    return np.tensordot(w, v, axes=(0, 0))


def _extract_scalar_cov_feature(model: dict[str, Any], key: str) -> float:
    if key in model and np.isfinite(model[key]):
        return float(model[key])
    sigma = _as_array(model.get("sigma"))
    if sigma.ndim == 2 and sigma.shape[0] > 0:
        return float(np.trace(sigma) / sigma.shape[0])
    return float("nan")


def mean_features(x, debias: bool = True, weights=None, trace: bool = False, IC: str = "AICc", select: str = "all", formula=False, base=None, **kwargs):
    del debias, trace, IC, select, formula, kwargs
    if base is None:
        base = {}
    n = len(x)
    if n == 0:
        return {}
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    w = w / np.sum(w)

    out = copy.deepcopy(base if isinstance(base, dict) else {})
    major = []
    minor = []
    angle = []
    for m in x:
        major.append(_extract_scalar_cov_feature(m, "major"))
        minor.append(float(m.get("minor", major[-1])))
        angle.append(float(m.get("angle", 0.0)))
    out["major"] = float(np.nansum(w * np.asarray(major)))
    out["minor"] = float(np.nansum(w * np.asarray(minor)))
    out["angle"] = float(np.nansum(w * np.asarray(angle)))
    out["isotropic"] = bool(np.isclose(out["major"], out["minor"], equal_nan=False))
    return out


def cov_off(COV, names=None):
    arr = np.asarray(COV, dtype=float).copy()
    if arr.ndim != 2:
        return arr
    out = arr.copy()
    np.fill_diagonal(out, 0.0)
    if names is not None:
        return {"COV": out, "names": list(names)}
    return out


def make_names(*parts, sep: str = " "):
    vals = []
    for p in parts:
        if isinstance(p, (list, tuple, np.ndarray)):
            vals.extend([str(x) for x in p])
        elif p is not None:
            vals.append(str(p))
    return sep.join(vals)


def num_pars(CTMM) -> int:
    model = CTMM.params if hasattr(CTMM, "params") else CTMM
    if not isinstance(model, dict):
        return 0
    if "features" in model and model["features"] is not None:
        return len(model["features"])
    count = 0
    sigma = model.get("sigma")
    if hasattr(sigma, "par"):
        count += len(sigma.par)
    elif sigma is not None:
        arr = np.asarray(sigma, dtype=float)
        count += int(arr.shape[0] * (arr.shape[0] + 1) / 2) if arr.ndim == 2 else arr.size
    tau = model.get("tau", {})
    count += len(tau) if isinstance(tau, dict) else np.asarray(tau, dtype=float).size
    for key in ("omega", "circle", "error"):
        if key in model and model[key] not in (None, False):
            count += 1
    return int(count)


def mean_ctmm(x, formula=False, weights=None, sample: bool = True, debias: bool = True, IC: str = "AIC", trace: bool = True, **kwargs):
    del formula, debias, IC, trace, kwargs
    if len(x) == 0:
        return {}
    n = len(x)
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    w = w / np.max(w) if np.max(w) > 0 else np.ones(n, dtype=float)
    w = w / np.sum(w)

    mu_list = [_as_array(m.get("mu", [0.0, 0.0]), default=[0.0, 0.0]) for m in x]
    max_dim = max(m.size for m in mu_list)
    mu_stack = np.vstack([np.pad(m, (0, max_dim - m.size)) for m in mu_list])
    mu = _weighted_average(mu_stack, w)

    if sample:
        sigma_list = []
        for m in x:
            s = _as_array(m.get("sigma"))
            if s.ndim == 1:
                s = np.diag(np.pad(s, (0, max_dim - s.size)))
            if s.ndim == 0 or s.size == 0:
                s = np.eye(max_dim) * np.nan
            if s.shape != (max_dim, max_dim):
                pad = np.zeros((max_dim, max_dim), dtype=float)
                r = min(max_dim, s.shape[0])
                c = min(max_dim, s.shape[1])
                pad[:r, :c] = s[:r, :c]
                s = pad
            sigma_list.append(s)
        sigma = _weighted_average(np.stack(sigma_list, axis=0), w)
    else:
        m2 = np.zeros((max_dim, max_dim), dtype=float)
        for wi, m in zip(w, x):
            mu_i = _as_array(m.get("mu", np.zeros(max_dim)))
            mu_i = np.pad(mu_i, (0, max_dim - mu_i.size))
            s = _as_array(m.get("sigma", np.eye(max_dim)))
            if s.ndim == 1:
                s = np.diag(np.pad(s, (0, max_dim - s.size)))
            if s.shape != (max_dim, max_dim):
                pad = np.zeros((max_dim, max_dim), dtype=float)
                r = min(max_dim, s.shape[0])
                c = min(max_dim, s.shape[1])
                pad[:r, :c] = s[:r, :c]
                s = pad
            m2 += wi * (s + np.outer(mu_i, mu_i))
        sigma = m2 - np.outer(mu, mu)

    out = copy.deepcopy(x[0])
    out["mu"] = mu
    out["sigma"] = sigma
    out["weights"] = w
    out["isotropic"] = bool(np.isclose(np.trace(sigma), max_dim * sigma[0, 0], rtol=1e-6, atol=1e-12))
    return out


def mean_pop(CTMM):
    c = copy.deepcopy(CTMM)
    sigma = _as_array(c.get("sigma"))
    pov_mu = _as_array(c.get("POV.mu", np.zeros_like(sigma)))
    if sigma.ndim == 0:
        sigma = np.asarray([[float(sigma)]], dtype=float)
    if pov_mu.shape != sigma.shape:
        pov_mu = np.zeros_like(sigma)
    spread = sigma + pov_mu
    c["sigma"] = spread
    c["isotropic"] = bool(np.isclose(np.trace(spread), spread.shape[0] * spread[0, 0], rtol=1e-6, atol=1e-12))
    c["features"] = ["major"] if c["isotropic"] else ["major", "minor", "angle"]
    for k in ("POV", "COV.POV", "COV.POV.mu", "tau"):
        if k in c:
            c.pop(k, None)
    c["circle"] = False
    return c


__all__ = ["cov_off", "make_names", "mean_features", "mean_ctmm", "mean_pop", "num_pars"]
