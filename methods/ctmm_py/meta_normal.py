"""Partial parity translation of ctmm 1.3.0 ``R/meta.normal.R``."""

from __future__ import annotations

import numpy as np

from .stats import cov_loglike


def cross_terms(terms, unique: bool = True):
    terms = list(terms)
    mat = np.array([[f"{a}-{b}" for b in terms] for a in terms], dtype=object)
    if unique:
        tri = np.triu_indices(len(terms))
        return mat[tri]
    return mat


def set_parscale(par):
    p = np.asarray(par, dtype=float).reshape(-1)
    scale = np.maximum(np.abs(p), 1.0)
    scale[~np.isfinite(scale)] = 1.0
    return scale


def sigma2par(sigma, isotropic: bool = False):
    s = np.asarray(sigma, dtype=float)
    if isotropic:
        return np.array([float(np.nanmean(np.diag(s)))], dtype=float)
    tri = np.triu_indices(s.shape[0])
    return s[tri]


def par2sigma(par, dim: int | None = None, isotropic: bool = False):
    p = np.asarray(par, dtype=float).reshape(-1)
    if isotropic:
        d = int(dim or 1)
        return np.eye(d) * float(p[0])
    if dim is None:
        dim = int((np.sqrt(8 * p.size + 1) - 1) / 2)
    out = np.zeros((dim, dim), dtype=float)
    tri = np.triu_indices(dim)
    out[tri] = p[: len(tri[0])]
    out[(tri[1], tri[0])] = out[tri]
    return out


def nloglike(par, Y, SY=False, **kwargs):
    fit = meta_normal(Y, SY=SY, **kwargs)
    return float(-fit["loglike"])


def nloglike_bfixed(par, Y, SY=False, **kwargs):
    return nloglike(par, Y, SY=SY, **kwargs)


def nloglike_nobeta(par, Y, SY=False, **kwargs):
    return nloglike(par, Y, SY=SY, **kwargs)


def nloglike_profile(par, Y, SY=False, **kwargs):
    return nloglike(par, Y, SY=SY, **kwargs)


def meta_normal(
    Y,
    SY=False,
    X=False,
    SX=False,
    DSM=None,
    INT=True,
    VARS=True,
    isotropic: bool = False,
    GUESS=None,
    debias: bool = True,
    weights=None,
    precision: float = 0.5,
    WARN: bool = True,
    lazy_COV: int = 200,
    **kwargs,
):
    del X, SX, DSM, GUESS, precision, WARN, lazy_COV, kwargs
    y = np.asarray(Y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    n, d = y.shape
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    w = w / np.sum(w)

    if np.isscalar(INT):
        INT = np.array([bool(INT)] * d)
    else:
        INT = np.asarray(INT, dtype=bool)

    mu = np.zeros(d, dtype=float)
    for j in range(d):
        if INT[j]:
            mu[j] = np.sum(w * y[:, j])

    yc = y - mu
    sigma = (yc.T * w) @ yc
    if debias and n > 1:
        sigma *= n / max(n - 1, 1)
    if isotropic:
        s = float(np.mean(np.diag(sigma)))
        sigma = np.eye(d) * s

    if np.isscalar(VARS):
        vars_mask = np.eye(d, dtype=bool) * bool(VARS)
    else:
        vars_mask = np.asarray(VARS, dtype=bool)

    sigma[~vars_mask] = 0.0
    cov_mu = sigma / max(n, 1)
    ll = -0.5 * n * (d * np.log(2 * np.pi) + np.linalg.slogdet(sigma + np.eye(d) * 1e-9)[1] + d)
    k = int(np.count_nonzero(INT) + np.count_nonzero(np.triu(vars_mask)))
    aic = 2 * k - 2 * ll
    aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
    bic = k * np.log(max(n, 1)) - 2 * ll

    names = [f"x{i+1}" for i in range(d)]
    return {
        "mu": mu[INT],
        "beta": np.array([], dtype=float),
        "sigma": sigma,
        "COV.mu": cov_mu[np.ix_(INT, INT)],
        "COV.sigma": np.diag(np.maximum(np.diag(sigma), 1e-12)),
        "loglike": float(ll),
        "AIC": float(aic),
        "AICc": float(aicc),
        "BIC": float(bic),
        "isotropic": bool(isotropic),
        "VARS": vars_mask,
        "INT": INT,
    }


__all__ = [
    "cross_terms",
    "meta_normal",
    "cov_loglike",
    "nloglike",
    "nloglike_bfixed",
    "nloglike_nobeta",
    "nloglike_profile",
    "par2sigma",
    "set_parscale",
    "sigma2par",
]
