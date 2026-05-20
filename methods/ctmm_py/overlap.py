"""Partial parity translation of ctmm 1.3.0 ``R/overlap.R``."""

from __future__ import annotations

import numpy as np

from .covm import Covm
from .pd_matrix import pd_logdet, pd_solve
from .types import CTMMModel
from .overlap_distance_ops import overlap_matrix as _overlap_matrix
from .overlap_distance_ops import overlap_pair as _overlap_pair
from .distance import BhattacharyyaD, EncounterD, EuclideanD, MahalanobisD, RateD
from .gaussian import gauss_comp
from .math import mpsigamma
from .stats import chi_bias, chi_var, chisq_ci
from .summary_ctmm import DOF_area


def DOF_wishart(CTMM) -> float:
    model = CTMM.params if isinstance(CTMM, CTMMModel) else CTMM
    sigma = model.get("sigma")
    if not isinstance(sigma, Covm):
        return 0.0
    axes = tuple(model.get("axes", sigma.axes))
    par = ["major"] if bool(model.get("isotropic", False)) else ["major", "minor"]
    est = np.asarray([float(sigma.par[p]) for p in par], dtype=float)
    cov = model.get("COV")
    if cov is None:
        return 0.0
    try:
        if hasattr(cov, "loc"):
            C = cov.loc[par, par].to_numpy(dtype=float)
        else:
            names = list(model.get("COV_rownames") or model.get("features") or [])
            idx = [names.index(p) for p in par]
            C = np.asarray(cov, dtype=float)[np.ix_(idx, idx)]
        if C.size != len(par) ** 2:
            return 0.0
        return float((2.0 / len(axes)) * (est @ pd_solve(C) @ est))
    except Exception:
        return 0.0


def soft_clamp(n: float, DIM: int) -> float:
    n = float(n)
    if n >= DIM + 2:
        return n
    A = -2 * DIM - 5 * DIM**2 - 4 * DIM**3 - DIM**4
    B = 4 + 16 * DIM + 25 * DIM**2 + 19 * DIM**3 + 7 * DIM**4 + DIM**5
    bias = 1 + (DIM + 1) / max(n, 1e-12) + A / max(n, 1e-12) ** 2 + B / max(n, 1e-12) ** 3
    return (DIM + 1) * bias / max(bias - 1, 1e-12)


def ElogW(s, n, DIM: int | None = None, add: bool = True):
    if DIM is None:
        arr = np.asarray(s.sigma if isinstance(s, Covm) else s, dtype=float)
        DIM = arr.shape[0] if arr.ndim == 2 else 1
    logdet = pd_logdet(s.sigma if isinstance(s, Covm) else np.asarray(s, dtype=float)) if add else 0.0
    n = np.asarray(n, dtype=float)
    return float(logdet + mpsigamma(n / 2.0, dim=int(DIM)) - int(DIM) * np.log(n / 2.0))


def _params(model):
    return model.params if isinstance(model, CTMMModel) else model


def _sigma(model):
    p = _params(model)
    s = p.get("sigma")
    if isinstance(s, Covm):
        return s.sigma
    if s is None:
        s = p.get("sigma_matrix", np.eye(len(p.get("axes", ("x", "y")))))
    return np.asarray(s, dtype=float)


def _mu(model):
    p = _params(model)
    dim = len(p.get("axes", ("x", "y")))
    mu = np.asarray(p.get("mu", np.zeros(dim)), dtype=float).reshape(-1)
    if mu.size < dim:
        mu = np.pad(mu, (0, dim - mu.size))
    return mu[:dim]


def _cov_mu(model):
    p = _params(model)
    c = p.get("COV.mu")
    dim = len(p.get("axes", ("x", "y")))
    if c is None:
        return np.zeros((dim, dim), dtype=float)
    arr = np.asarray(c, dtype=float)
    if arr.ndim == 4:
        arr = arr[:, 0, 0, :]
    return arr[:dim, :dim]


def _distance_fn(method: str):
    return {
        "bhattacharyya": BhattacharyyaD,
        "encounter": EncounterD,
        "mahalanobis": MahalanobisD,
        "euclidean": EuclideanD,
        "rate": RateD,
    }[method.lower()]


def overlap_ctmm(object, level: float = 0.95, debias: bool = True, COV: bool = True, method: str = "Bhattacharyya", distance: bool = False, sqrt: bool = False, **kwargs):
    del kwargs
    c1, c2 = object[0], object[1]
    dim = len(_params(c1).get("axes", ("x", "y")))
    stuff = gauss_comp(_distance_fn(method), [c1, c2], COV=COV)
    mle = float(np.ravel(stuff["MLE"])[0])
    var = float(np.ravel(stuff["COV"])[0]) if np.size(stuff["COV"]) else 0.0
    dof = float(2.0 * mle * mle / var) if var > 0 else np.inf

    mu = _mu(c1) - _mu(c2)
    cov_mu = _cov_mu(c1) + _cov_mu(c2)
    if method == "Euclidean":
        sigma = np.eye(dim, dtype=float)
    else:
        sigma = (_sigma(c1) + _sigma(c2)) / 2.0
    s0 = float(np.mean(np.diag(sigma)))
    s1 = float(np.mean(np.diag(_sigma(c1))))
    s2 = float(np.mean(np.diag(_sigma(c2))))
    n1 = max(DOF_area(c1), 1.0)
    n2 = max(DOF_area(c2), 1.0)
    n0 = 4.0 * s0 * s0 / max(s1 * s1 / n1 + s2 * s2 / n2, np.finfo(float).eps)
    n0 = max(n0, 2.0)
    n0 = soft_clamp(n0, dim)
    n1 = soft_clamp(n1, dim)
    n2 = soft_clamp(n2, dim)
    b = n0 / (n0 - dim - 1.0) if n0 > dim + 1.0 else 1.0
    if method == "Euclidean":
        b = 0.0
    elif method == "Rate":
        b -= 1.0
    bias = float(np.trace((b * np.outer(mu, mu) + cov_mu) @ pd_solve(sigma)))
    if method == "Bhattacharyya":
        bias = bias / 8.0 + max(ElogW(sigma, n0, dim) / 2.0 - ElogW(_sigma(c1), n1, dim) / 4.0 - ElogW(_sigma(c2), n2, dim) / 4.0, 0.0)
    elif method == "Encounter":
        bias = bias / 4.0 + max(ElogW(sigma, n0, dim) / 2.0 - ElogW(_sigma(c1), n1, dim) / 4.0 - ElogW(_sigma(c2), n2, dim) / 4.0, 0.0)
    elif method == "Rate":
        bias = bias / 4.0 + max(ElogW(sigma, n0, dim, add=False) / 2.0, ElogW(_sigma(c1), n1, dim, add=False) / 4.0 + ElogW(_sigma(c2), n2, dim, add=False) / 4.0)
    if method != "Rate":
        bias = bias / mle if mle != 0.0 else 1.0
    if distance:
        bias = 1.0 + bias
    if mle == 0.0 or not np.isfinite(bias):
        bias = 1.0

    if not level:
        return {"MLE": mle, "VAR": var, "DOF": dof, "BIAS": bias}
    if method != "Rate":
        if debias:
            mle = mle / bias
        dof = float(2.0 * mle * mle / var) if var > 0 else np.inf
        ci = chisq_ci(mle, dof=dof, alpha=1.0 - level)
        if distance:
            if sqrt:
                ci = np.sqrt(np.maximum(ci, 0.0))
                if debias:
                    ci = ci / chi_bias(np.array([max(dof, 1.0)]))[0]
            return ci
        ci = np.exp(-ci[::-1])
    else:
        if debias:
            mle = mle - bias
        mle = np.exp(-mle)
        var = mle * mle * var
        dof = float(2.0 * mle * mle / var) if var > 0 else np.inf
        ci = chisq_ci(mle, dof=dof, alpha=1.0 - level)
    return {"DOF": dof, "CI": ci}


def overlap_UD(object, level: float = 0.95, debias: bool = True, method: str = "Bhattacharyya", **kwargs):
    del level, debias, kwargs
    return _overlap_pair(object[0], object[1], method=method)


def overlap(object, method: str = "Bhattacharyya", level: float = 0.95, debias: bool = True, **kwargs):
    del level, debias, kwargs
    objs = object if isinstance(object, list) else [object]
    return _overlap_matrix(objs, method=method)


__all__ = ["DOF_wishart", "ElogW", "soft_clamp", "overlap_ctmm", "overlap_UD", "overlap"]
