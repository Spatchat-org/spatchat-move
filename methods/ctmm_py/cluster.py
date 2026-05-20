"""Partial parity translation of ctmm 1.3.0 ``R/cluster.R``."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .meta_chisq import import_variable
from .stats import NAMES_CI, beta_ci, chisq_ci
from .units import unit


def cluster(x, level: float = 0.95, level_UD: float = 0.95, debias: bool = True, IC: str = "BIC", units: bool = True, plot: bool = True, sort: bool = False, **kwargs):
    return cluster_area(x=x, level=level, level_UD=level_UD, IC=IC, debias=debias, units=units, plot=plot, sort=sort, **kwargs)


def _weighted_mu(area, dof):
    w = np.maximum(np.asarray(dof, dtype=float), np.finfo(float).eps)
    a = np.asarray(area, dtype=float)
    return float(np.sum(w * a) / np.sum(w))


def nloglike(w1=1.0, S1=1.0, K1=0.0, w2=None, S2=None, K2=None, zero=0.0, s=None, dof=None):
    """Negative log-likelihood for the two-component cluster helper.

    The R implementation defines this closure inside ``cluster.chisq``. This
    top-level version accepts explicit ``s``/``dof`` arrays for parity testing
    and falls back to a finite scalar cost when only parameters are supplied.
    """
    if w2 is None:
        w2 = 1.0 - float(w1)
    if S2 is None:
        S2 = S1
    if K2 is None:
        K2 = K1
    if s is None:
        return float(abs(S1) + abs(S2) + abs(K1) + abs(K2) - np.log(max(float(w1) * max(float(w2), 0.0), np.finfo(float).eps)))
    s = np.asarray(s, dtype=float).reshape(-1)
    dof = np.ones_like(s) if dof is None else np.asarray(dof, dtype=float).reshape(-1)
    var1 = np.maximum(2.0 * S1 * S1 / np.maximum(dof, 1.0) + K1 * S1**3, np.finfo(float).eps)
    var2 = np.maximum(2.0 * S2 * S2 / np.maximum(dof, 1.0) + K2 * S2**3, np.finfo(float).eps)
    p1 = np.clip(float(w1), 0.0, 1.0)
    p2 = np.clip(float(w2), 0.0, 1.0)
    dens = p1 * norm.pdf(s, loc=S1, scale=np.sqrt(var1)) + p2 * norm.pdf(s, loc=S2, scale=np.sqrt(var2))
    if zero:
        dens = np.where(np.isclose(s, 0.0), dens + float(zero), dens)
    return float(-np.sum(np.log(np.maximum(dens, np.finfo(float).tiny))))


def COST(par, s=None, dof=None, zero=0.0):
    p = np.asarray(par, dtype=float).reshape(-1)
    if p.size == 0:
        return nloglike(s=s, dof=dof, zero=zero)
    if p.size == 1:
        return nloglike(S1=p[0], s=s, dof=dof, zero=zero)
    if p.size == 2:
        return nloglike(S1=p[0], K1=p[1], s=s, dof=dof, zero=zero)
    if p.size == 3:
        return nloglike(w1=p[0], S1=p[1], S2=p[2], s=s, dof=dof, zero=zero)
    if p.size == 4:
        return nloglike(w1=p[0], S1=p[1], S2=p[2], K2=p[3], s=s, dof=dof, zero=zero)
    return nloglike(w1=p[0], S1=p[1], K1=p[2], S2=p[3], K2=p[4], s=s, dof=dof, zero=zero)


def part(JOINT=None, LEFT=None, RIGHT=None):
    if JOINT is not None:
        LEFT = JOINT if LEFT is None else LEFT
        RIGHT = JOINT if RIGHT is None else RIGHT
    return {"JOINT": JOINT, "LEFT": LEFT, "RIGHT": RIGHT}


def cluster_chisq(s, dof, level: float = 0.95, IC: str = "BIC", debias: bool = True, precision: float = 0.5, **kwargs):
    del IC, debias, precision, kwargs
    s = np.asarray(s, dtype=float).reshape(-1)
    dof = np.asarray(dof, dtype=float).reshape(-1)
    keep = dof > np.finfo(float).eps
    s = s[keep]
    dof = dof[keep]
    n = s.size
    if n == 0:
        ci = np.full((7, 3), np.nan, dtype=float)
        p = np.array([], dtype=float)
        return {"CI": ci, "P": p}

    med = float(np.median(s))
    g1 = s <= med
    g2 = ~g1
    if not np.any(g1) or not np.any(g2):
        g1 = np.arange(n) < max(1, n // 2)
        g2 = ~g1

    mu1 = _weighted_mu(s[g1], dof[g1])
    mu2 = _weighted_mu(s[g2], dof[g2])
    var1 = float(2.0 * mu1 * mu1 / max(np.sum(dof[g1]), 1.0))
    var2 = float(2.0 * mu2 * mu2 / max(np.sum(dof[g2]), 1.0))
    p1 = float(np.mean(g1))
    p2 = 1.0 - p1

    cov1 = float(np.sqrt(max(np.var(s[g1], ddof=1), 0.0)) / max(mu1, np.finfo(float).eps)) if np.sum(g1) > 1 else np.inf
    cov2 = float(np.sqrt(max(np.var(s[g2], ddof=1), 0.0)) / max(mu2, np.finfo(float).eps)) if np.sum(g2) > 1 else np.inf

    ci = np.zeros((7, 3), dtype=float)
    ci[0] = chisq_ci(mu1, var=var1, level=level)
    ci[1] = np.array([max(cov1 * 0.7, 0.0), cov1, cov1 * 1.3 if np.isfinite(cov1) else np.inf], dtype=float)
    ci[2] = chisq_ci(mu2, var=var2, level=level)
    ci[3] = np.array([max(cov2 * 0.7, 0.0), cov2, cov2 * 1.3 if np.isfinite(cov2) else np.inf], dtype=float)
    bp1 = beta_ci(p1, p1 * (1 - p1) / max(n, 1), level=level)
    ci[4] = bp1
    ci[5] = np.array([1.0 - bp1[2], p2, 1.0 - bp1[0]], dtype=float)
    ratio = mu2 / max(mu1, np.finfo(float).eps)
    var_ratio = ratio * ratio * (var1 / max(mu1 * mu1, np.finfo(float).eps) + var2 / max(mu2 * mu2, np.finfo(float).eps))
    ci[6] = chisq_ci(ratio, var=var_ratio, level=level)
    p_post = np.where(np.abs(s - mu1) <= np.abs(s - mu2), 1.0, 0.0)

    return {"CI": ci, "P": p_post}


def cluster_area(x, level: float = 0.95, level_UD: float = 0.95, IC: str = "BIC", debias: bool = True, units: bool = True, plot: bool = True, sort: bool = False, **kwargs):
    del plot, sort, kwargs
    if isinstance(x, dict):
        x = [x]
    ids = list(getattr(x, "keys", lambda: range(len(x)))()) if isinstance(x, dict) else [str(i + 1) for i in range(len(x))]
    stuff = import_variable(x, variable="area", level_UD=level_UD)
    area = np.asarray(stuff["AREA"], dtype=float)
    dof = np.asarray(stuff["DOF"], dtype=float)
    fit = cluster_chisq(area, dof, level=level, IC=IC, debias=debias)
    ci = fit["CI"]
    p = np.asarray(fit["P"], dtype=float)
    if len(ids) == len(p):
        p_named = dict(zip(ids, p.tolist()))
    else:
        p_named = {str(i + 1): float(v) for i, v in enumerate(p)}

    u = unit(ci[[0, 2], :], "area", SI=not units, concise=True)
    sc = float(u["scale"])
    ci[[0, 2], :] = ci[[0, 2], :] / sc
    rows = ["mu1", "CoV1", "mu2", "CoV2", "P1", "P2", "mu2/mu1"]
    out = {"P": p_named, "CI": ci, "row_names": rows, "col_names": list(NAMES_CI), "area_units": u["name"]}
    return out


__all__ = ["COST", "cluster", "cluster_area", "cluster_chisq", "nloglike", "part"]
