"""Partial parity translation of ctmm 1.3.0 ``R/cde.R``."""
from __future__ import annotations
import numpy as np

from .covm import covm
from .pd_matrix import pd_solve
from .types import CTMMModel
from .bandwidth import bandwidth
from .summary_ctmm import DOF_area

def cde(x, y=None, grid=None):
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if y is None:
        y = x
    y = np.asarray(y, dtype=float).reshape(-1)
    y = y[np.isfinite(y)]
    if grid is None:
        lo, hi = np.min(y), np.max(y)
        grid = np.linspace(lo, hi, 128)
    g = np.asarray(grid, dtype=float)
    h = max(bandwidth(y), np.finfo(float).eps)
    u = (g[:, None] - y[None, :]) / h
    d = np.exp(-0.5*u*u) / np.sqrt(2*np.pi)
    pdf = np.mean(d, axis=1) / h
    return {"x": g, "pdf": pdf}


def _model_params(model):
    return model.params if isinstance(model, CTMMModel) else model


def _sigma_matrix(model):
    p = _model_params(model)
    sigma = p.get("sigma")
    if hasattr(sigma, "sigma"):
        return np.asarray(sigma.sigma, dtype=float)
    if sigma is None:
        sigma = p.get("sigma_matrix", np.eye(2))
    arr = np.asarray(sigma, dtype=float)
    if arr.ndim == 1:
        return np.diag(arr)
    return arr


def _mu_vector(model, dim):
    p = _model_params(model)
    mu = np.asarray(p.get("mu", np.zeros(dim)), dtype=float).reshape(-1)
    if mu.size < dim:
        mu = np.pad(mu, (0, dim - mu.size))
    return mu[:dim]


def fn(CTMM, include=None, BIAS=None, bias=None, debias: bool = False):
    """Gaussian encounter moment function from ``cde.ctmm``.

    It combines pairwise products of Gaussian home-range approximations into a
    single mean/covariance parameter vector, matching the algebra of the R
    closure used inside ``gauss.comp``.
    """
    models = list(CTMM)
    n = len(models)
    if n == 0:
        return np.array([], dtype=float)
    dim = _sigma_matrix(models[0]).shape[0]
    include = np.ones((n, n), dtype=float) if include is None else np.asarray(include, dtype=float).copy()
    BIAS = np.ones(n, dtype=float) if BIAS is None else np.asarray(BIAS, dtype=float)
    bias = np.ones((n, n), dtype=float) if bias is None else np.asarray(bias, dtype=float)

    precisions = [pd_solve(_sigma_matrix(m)) for m in models]
    total = 0.0
    m1 = np.zeros(dim, dtype=float)
    m2 = np.zeros((dim, dim), dtype=float)

    for i in range(n - 1):
        for j in range(i + 1, n):
            pi = precisions[i].copy()
            pj = precisions[j].copy()
            if debias:
                pi *= BIAS[i]
                pj *= BIAS[j]
            pij = pi + pj
            sigma = pd_solve(pij)
            mui = _mu_vector(models[i], dim)
            muj = _mu_vector(models[j], dim)
            mu = sigma @ (pi @ mui + pj @ muj)
            weight = include[i, j] / np.sqrt(
                max(np.linalg.det(_sigma_matrix(models[i])), np.finfo(float).tiny)
                * max(np.linalg.det(_sigma_matrix(models[j])), np.finfo(float).tiny)
                * max(np.linalg.det(pij), np.finfo(float).tiny)
            )
            if debias:
                sigma = sigma / max(float(bias[i, j]), np.finfo(float).eps)
            total += weight
            m1 += weight * mu
            m2 += weight * (sigma + np.outer(mu, mu))

    if total <= 0:
        sigma = np.eye(dim, dtype=float)
        mu = np.zeros(dim, dtype=float)
    else:
        mu = m1 / total
        sigma = m2 / total - np.outer(mu, mu)
    pars = covm(sigma, axes=tuple(_model_params(models[0]).get("axes", ("x", "y"))), isotropic=False).par
    return np.r_[mu, [pars[k] for k in pars]]


def cde_ctmm(CTMM, include=None, exclude=None, debias: bool = False, **kwargs):
    del kwargs
    models = list(CTMM)
    n = len(models)
    if n == 0:
        raise ValueError("cde_ctmm requires at least one CTMM model")
    axes = tuple(_model_params(models[0]).get("axes", ("x", "y")))
    dim = len(axes)
    if include is None:
        if exclude is None:
            exclude = np.eye(n)
        include = 1.0 - np.asarray(exclude, dtype=float)
    else:
        include = np.asarray(include, dtype=float)

    dof = np.asarray([DOF_area(m) if isinstance(m, CTMMModel) else float(_model_params(m).get("DOF.area", np.inf)) for m in models], dtype=float)
    isotropic = np.asarray([bool(_model_params(m).get("isotropic", False)) for m in models], dtype=bool)
    wc = (1.0 + np.where(isotropic, 1.0, dim)) / np.where(isotropic, dim, 1.0)
    bias_ind = 1.0 / (1.0 + wc / np.clip(dof, 1.0, np.inf))

    precisions = [pd_solve(_sigma_matrix(m)) for m in models]
    pair_bias = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            pi = float(np.trace(precisions[i]))
            pj = float(np.trace(precisions[j]))
            pij = pi + pj
            denom = pi * pi / max(dof[i], np.finfo(float).eps) + pj * pj / max(dof[j], np.finfo(float).eps)
            pair_dof = pij * pij / max(denom, np.finfo(float).eps)
            pair_wc = (pi / pij) * wc[i] + (pj / pij) * wc[j] if pij else 0.0
            pair_bias[i, j] = 1.0 + pair_wc / np.clip(pair_dof, 2.0, np.inf)

    vec = fn(models, include=include, BIAS=bias_ind, bias=pair_bias, debias=debias)
    mu = vec[:dim]
    sigma_par = vec[dim:]
    sigma = covm(sigma_par, axes=axes, isotropic=bool(np.all(isotropic)))
    params = {
        "mu": mu,
        "sigma": sigma,
        "sigma_matrix": sigma.sigma,
        "axes": axes,
        "isotropic": bool(np.all(isotropic)),
        "DOF.area": float(np.nanmean(dof)) if dof.size else np.inf,
    }
    return {"CTMM": CTMMModel("ctmm", params), "BIAS": bias_ind, "bias": pair_bias}


__all__ = ["cde", "cde_ctmm", "fn"]
