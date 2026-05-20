"""Partial parity translation of ctmm 1.3.0 ``R/meta.chisq.R``."""

from __future__ import annotations

import numpy as np

from .meta_normal import meta_normal
from .stats import NAMES_CI, beta_ci, chisq_ci


def summary_meta(object, IC: str = "AICc", **kwargs):
    del kwargs
    if isinstance(object, list):
        return summary_meta_list(object, IC=IC)
    return summary_meta_single(object)


def summary_meta_list(object, IC: str = "AICc", **kwargs):
    del kwargs
    out = {"n": len(object), "IC": IC, "models": []}
    for item in object:
        out["models"].append(summary_meta_single(item))
    return out


def summary_meta_single(object, **kwargs):
    del kwargs
    if isinstance(object, dict):
        return {
            "CI": object.get("CI"),
            "VAR": object.get("VAR"),
            "dIC": object.get("dIC"),
        }
    return {"CI": None, "VAR": None, "dIC": None}


def _inverse_mean(mu: float, vm2: float, debias: bool = True) -> float:
    if not debias:
        return 1.0 / max(mu, np.finfo(float).eps)
    dof = np.inf if vm2 <= 0 else 2.0 / vm2
    if np.isinf(dof) or dof >= 3.0:
        return 1.0 / max(mu, np.finfo(float).eps) * (1.0 - 2.0 / dof if np.isfinite(dof) else 1.0)
    return 1.0 / max(mu, np.finfo(float).eps) * (dof * dof / 27.0)


def inverse_mean(mu: float, vm2: float, debias: bool = True) -> float:
    return _inverse_mean(mu, vm2, debias=debias)


def k_std(k, var=None):
    k = np.asarray(k, dtype=float)
    if var is None:
        return np.sqrt(np.maximum(k, 0.0))
    return np.sqrt(np.maximum(var, 0.0))


def shrink_chisq(s, dof, debias: bool = True):
    s = np.asarray(s, dtype=float)
    dof = np.asarray(dof, dtype=float)
    if not debias:
        return s
    return s * np.maximum(dof - 2.0, 0.0) / np.maximum(dof, np.finfo(float).eps)


def fit_blue(s, dof, **kwargs):
    return meta_chisq(s, dof, method="blue", **kwargs)


def fit_mle(s, dof, **kwargs):
    return meta_chisq(s, dof, method="mle", **kwargs)


def _ci_matrix(level: float) -> np.ndarray:
    ci = np.array([[0.0, 0.0, np.inf]] * 4, dtype=float)
    return ci


def meta_chisq(
    s,
    dof,
    level: float = 0.95,
    level_pop: float = 0.95,
    IC: str = "AICc",
    method: str = "mle",
    boot: bool = False,
    iterate: bool = False,
    error: float = 0.01,
    debias: bool = True,
    precision: float = 0.5,
    CI_FN: str = "chisq",
    **kwargs,
):
    del level_pop, boot, iterate, error, precision, kwargs
    s = np.asarray(s, dtype=float).reshape(-1)
    dof = np.asarray(dof, dtype=float).reshape(-1)
    keep = dof > np.finfo(float).eps
    s = s[keep]
    dof = dof[keep]
    if s.size == 0:
        ci = _ci_matrix(level)
        return {"CI": ci, "VAR": np.array([np.inf, np.inf, np.inf, np.inf]), "dIC": np.array([[0.0], [0.0]])}

    n = s.size
    mu = float(np.average(s, weights=np.maximum(dof, np.finfo(float).eps)))
    vm = float(2.0 * mu * mu / max(np.sum(dof), np.finfo(float).eps))
    k = float(max(np.var(s, ddof=1) - np.mean(2.0 * s * s / np.maximum(dof, 1.0)), 0.0) / max(mu**3, np.finfo(float).eps)) if n > 1 else 0.0

    if method.lower() == "blue":
        var_obs = 2.0 * s * s / np.maximum(dof, 1.0)
        fit = meta_normal(s[:, None], np.diag(var_obs), debias=debias)
        mu = float(np.ravel(fit["mu"])[0])
        vm = float(np.ravel(fit["COV.mu"])[0]) if np.size(fit["COV.mu"]) else vm
        k = float(max(np.ravel(fit["sigma"])[0], 0.0) / max(mu**3, np.finfo(float).eps))

    ci = _ci_matrix(level)
    v = np.zeros(4, dtype=float)
    ci_mean = chisq_ci(mu, var=vm, level=level)
    ci[0, :] = ci_mean
    v[0] = vm

    inv_mu = _inverse_mean(mu, vm / max(mu * mu, np.finfo(float).eps), debias=debias)
    inv_var = vm / max(mu**4, np.finfo(float).eps)
    ci_inv = 1.0 / np.array([ci_mean[2], ci_mean[1], ci_mean[0]], dtype=float)
    ci_inv[1] = inv_mu
    ci[1, :] = ci_inv
    v[1] = inv_var

    cov2 = mu * k
    var_cov2 = max(k * k * vm + mu * mu * max(k, 0.0) / max(n, 1), 0.0)
    ci[2, :] = chisq_ci(cov2, var=var_cov2, level=level)
    v[2] = var_cov2

    ci[3, :] = np.sqrt(np.maximum(ci[2, :], 0.0))
    v[3] = v[2] / max(4.0 * ci[2, 1], np.finfo(float).eps)

    if CI_FN.lower() == "beta":
        for i in (0, 2, 3):
            bci = beta_ci(float(np.clip(ci[i, 1], 0.0, 1.0)), float(max(v[i], 0.0)), level=level)
            ci[i, :] = 100.0 * np.sqrt(np.maximum(bci, 0.0))

    d0 = float(np.sum((s - mu) ** 2 / np.maximum(2.0 * s * s / np.maximum(dof, 1.0), np.finfo(float).eps)))
    d1 = float(np.sum((s - mu) ** 2 / np.maximum(2.0 * s * s / np.maximum(dof, 1.0) + mu**3 * k, np.finfo(float).eps)))
    dIC = np.array([[d0], [d1]], dtype=float)
    dIC -= np.min(dIC)

    return {"CI": ci, "VAR": v, "dIC": dIC, "IC": IC}


def import_variable(x, variable: str = "area", level_UD: float = 0.95, chi: bool = False):
    del level_UD, chi
    if isinstance(x, dict):
        x = [x]
    ids = []
    area = []
    dof = []
    for i, item in enumerate(x):
        if isinstance(item, dict):
            ids.append(str(item.get("id", item.get("identity", f"id{i+1}"))))
            if variable == "distance":
                area.append(float(item.get("distance", item.get("est", np.nan))))
            else:
                area.append(float(item.get(variable, item.get("est", np.nan))))
            dof.append(float(item.get("dof", item.get("DOF", 0.0))))
        else:
            ids.append(f"id{i+1}")
            area.append(float(item))
            dof.append(0.0)
    return {"ID": ids, "AREA": np.asarray(area, dtype=float), "DOF": np.asarray(dof, dtype=float), "variable": variable}


def meta_uni(
    x,
    variable: str = "area",
    level: float = 0.95,
    level_UD: float = 0.95,
    level_pop: float = 0.95,
    method: str = "mle",
    IC: str = "AICc",
    boot: bool = False,
    error: float = 0.01,
    debias: bool = True,
    verbose: bool = False,
    units: bool = True,
    plot: bool = True,
    sort: bool = False,
    mean: bool = True,
    col: str = "black",
    **kwargs,
):
    del verbose, units, plot, sort, mean, col, kwargs
    stuff = import_variable(x, variable=variable, level_UD=level_UD)
    fit = meta_chisq(
        stuff["AREA"],
        stuff["DOF"],
        level=level,
        level_pop=level_pop,
        IC=IC,
        method=method,
        boot=boot,
        error=error,
        debias=debias,
        CI_FN="beta" if variable in ("periodicity", "cyclicity") else "chisq",
    )
    ci = fit["CI"][[0, 2, 3], :].copy()
    return ci


def meta(
    x,
    variable: str = "area",
    level: float = 0.95,
    level_UD: float = 0.95,
    method: str = "MLE",
    IC: str = "AICc",
    boot: bool = False,
    error: float = 0.01,
    debias: bool = True,
    verbose: bool = False,
    units: bool = True,
    plot: bool = True,
    sort: bool = False,
    mean: bool = True,
    col: str = "black",
    **kwargs,
):
    return meta_uni(
        x=x,
        variable=str(variable).lower(),
        level=level,
        level_UD=level_UD,
        method=str(method).lower(),
        IC=IC,
        boot=boot,
        error=error,
        debias=debias,
        verbose=verbose,
        units=units,
        plot=plot,
        sort=sort,
        mean=mean,
        col=col,
        **kwargs,
    )


__all__ = [
    "summary_meta",
    "summary_meta_list",
    "summary_meta_single",
    "meta_chisq",
    "meta",
    "import_variable",
    "inverse_mean",
    "k_std",
    "fit_blue",
    "fit_mle",
    "meta_uni",
    "shrink_chisq",
    "NAMES_CI",
]
