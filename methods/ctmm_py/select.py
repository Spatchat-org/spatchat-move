"""Partial parity translation of ctmm 1.3.0 ``R/select.R``."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .covm import Covm, covm as covm_factory, scale_covm
from .models import ctmm_select as _ctmm_select
from .types import CTMMModel
from .time import timelink_complexify, timelink_simplify, timelink_name


def alpha_ctmm(CTMM, alpha: float):
    aic = float(CTMM.params.get("AIC", np.nan))
    aicc = float(CTMM.params.get("AICc", np.nan))
    z = float(norm.ppf(alpha))
    z = np.sqrt(z * z + max(aicc - aic, 0.0))
    out = 1.0 - float(norm.cdf(z))
    return out if np.isfinite(out) else 0.0


def get_MSPE(CTMM, MSPE: str = "position"):
    mspe = CTMM.params.get("MSPE", {})
    if isinstance(mspe, dict):
        return float(mspe.get(MSPE, np.inf))
    return np.inf


def get_IC(CTMM, IC: str = "AICc"):
    return float(CTMM.params.get(IC, np.inf))


def name_ctmm(CTMM, whole: bool = True):
    if CTMM is None:
        return None
    params = CTMM.params if isinstance(CTMM, CTMMModel) else CTMM
    tau = params.get("tau", {})
    tau_list = list(params.get("tau_list", []) or [])
    if isinstance(tau, dict) and not tau_list:
        order = ["position", "velocity", "acceleration"]
        tau_list = [float(tau[k]) for k in order if k in tau]
    range_ = bool(params.get("range", True))
    omega = float(params.get("omega", 0.0) or 0.0)
    if len(tau_list) >= 2:
        if not range_ or np.isposinf(float(tau_list[0])):
            base = "IOU"
        elif float(tau_list[0]) > float(tau_list[1]):
            base = "OUF"
        elif omega:
            base = "OUOmega"
        else:
            base = "OUf"
    elif len(tau_list) == 1:
        base = "OU" if range_ and np.isfinite(float(tau_list[0])) else "BM"
    else:
        base = "IID"
    parts = [base]
    isotropic = params.get("isotropic", True)
    if isinstance(isotropic, (list, tuple, np.ndarray)):
        iso0 = bool(np.asarray(isotropic).reshape(-1)[0])
    else:
        iso0 = bool(isotropic)
    axes = params.get("axes", ("x", "y"))
    if len(axes) > 1 and not iso0:
        parts.append("anisotropic")
    if float(params.get("circle", 0.0) or 0.0):
        parts.append("circulation")
    if bool(params.get("error", False)):
        parts.append("error")
    tname = timelink_name(CTMM)
    if tname:
        parts.append(str(tname))
    mean = str(params.get("mean", "stationary") or "stationary")
    if whole and mean not in {"stationary", "None"}:
        parts.append(mean)
    if not whole:
        return [parts[0], mean if mean else "stationary"]
    return " ".join(parts)


def _copy_model(M, **updates):
    params = dict(M.params if isinstance(M, CTMMModel) else M)
    params.update(updates)
    tau_list = list(params.get("tau_list", []) or [])
    if "tau" in updates and isinstance(updates["tau"], dict):
        order = ["position", "velocity", "acceleration"]
        tau_list = [float(updates["tau"][k]) for k in order if k in updates["tau"]]
        params["tau_list"] = tau_list
    range_ = bool(params.get("range", True))
    omega = float(params.get("omega", 0.0) or 0.0)
    if not range_:
        model = "IOU" if len(tau_list) > 1 else "BM"
    elif len(tau_list) == 0:
        model = "IID"
    elif len(tau_list) == 1:
        model = "OU"
    elif omega:
        model = "OUOmega"
    elif np.isclose(float(tau_list[0]), float(tau_list[1]), rtol=1e-10, atol=1e-12):
        model = "OUf"
    else:
        model = "OUF"
    return CTMMModel(model=model, params=params)


def _tau_dict(vals):
    names = ("position", "velocity", "acceleration")
    return {names[i]: float(vals[i]) for i in range(min(len(vals), len(names)))}


def simplify_ctmm(M, par):
    params = dict(M.params if isinstance(M, CTMMModel) else M)
    pars = [par] if isinstance(par, str) else list(par)
    tau_list = list(params.get("tau_list", []) or [])
    updates = {}
    if "minor" in pars:
        updates["isotropic"] = True
        sig = params.get("sigma")
        if isinstance(sig, Covm):
            updates["sigma"] = covm_factory(sig, isotropic=True, axes=sig.axes)
    if "major" in pars:
        updates["isotropic"] = True
        updates["tau"] = {}
        updates["tau_list"] = []
        updates["sigma_matrix"] = np.zeros_like(np.asarray(params.get("sigma_matrix", np.eye(2)), dtype=float))
    if "circle" in pars:
        updates["circle"] = 0.0
    if "range" in pars and tau_list:
        first = float(tau_list[0])
        new_tau = list(tau_list)
        new_tau[0] = float("inf")
        updates["range"] = False
        updates["tau_list"] = new_tau
        updates["tau"] = _tau_dict(new_tau)
        sig = params.get("sigma")
        if isinstance(sig, Covm) and np.isfinite(first) and first > 0:
            updates["sigma"] = scale_covm(sig, 1.0 / first)
    if "diff.tau" in pars and len(tau_list) >= 2:
        tau = float(np.mean(tau_list[:2]))
        updates["tau_list"] = [tau, tau]
        updates["tau"] = _tau_dict([tau, tau])
        updates["omega"] = 0.0
    if "tau velocity" in pars and tau_list:
        new_tau = [float(tau_list[0])]
        updates["tau_list"] = new_tau
        updates["tau"] = _tau_dict(new_tau)
    if any(p in pars for p in ("tau position", "tau")):
        updates["tau_list"] = []
        updates["tau"] = {}
    if "omega" in pars:
        updates["omega"] = 0.0
    error_pars = [p for p in pars if str(p).startswith("error")]
    if error_pars:
        err = params.get("error", False)
        if isinstance(err, dict):
            err = dict(err)
            for ep in error_pars:
                cls = str(ep).removeprefix("error").strip()
                if cls:
                    err[cls] = False
            updates["error"] = err
        else:
            updates["error"] = False
    if "timelink" in pars:
        return timelink_simplify(M)
    return _copy_model(M, **updates)


def complexify_ctmm(M, par, TARGET=None):
    params = dict(M.params if isinstance(M, CTMMModel) else M)
    target = TARGET.params if isinstance(TARGET, CTMMModel) else (TARGET or params)
    pars = [par] if isinstance(par, str) else list(par)
    tau_list = list(params.get("tau_list", []) or [])
    target_tau = list(target.get("tau_list", []) or tau_list)
    updates = {}
    if "minor" in pars and bool(params.get("isotropic", True)):
        updates["isotropic"] = False
    if "circle" in pars and not float(params.get("circle", 0.0) or 0.0):
        sign = np.sign(float(target.get("circle", 1.0) or 1.0))
        updates["circle"] = float(2.0 * np.finfo(float).eps * sign)
    if "range" in pars and not bool(params.get("range", True)) and target_tau:
        new_tau = list(tau_list) if tau_list else [float("inf")]
        new_tau[0] = float(target_tau[0])
        updates["range"] = True
        updates["tau_list"] = new_tau
        updates["tau"] = _tau_dict(new_tau)
    if "tau velocity" in pars:
        if len(target_tau) > 1:
            new_tau = list(tau_list[:1] or target_tau[:1]) + [float(target_tau[1])]
        elif tau_list:
            new_tau = [float(tau_list[0]), max(float(tau_list[0]) / 10.0, np.finfo(float).eps)]
        else:
            new_tau = [1.0, 0.1]
        updates["tau_list"] = new_tau
        updates["tau"] = _tau_dict(new_tau)
    if "omega" in pars:
        om = float(target.get("omega", 0.0) or np.sqrt(np.finfo(float).eps))
        updates["omega"] = om
        if len(tau_list) < 2:
            tau = float(tau_list[0]) if tau_list else 1.0
            updates["tau_list"] = [tau, tau]
            updates["tau"] = _tau_dict([tau, tau])
    if "timelink" in pars:
        return timelink_complexify(M)
    return _copy_model(M, **updates)


def get_mle(FIT):
    return FIT.params.get("MLE", FIT) if isinstance(FIT, CTMMModel) else FIT.get("MLE", FIT)


def fix_vars(models):
    return models


def update_fix(CTMM):
    return CTMM


def iterate(data, CTMM, *args, **kwargs):
    return ctmm_iterate(data, CTMM, *args, **kwargs)


def ctmm_iterate(data, CTMM, verbose: bool = False, level: float = 1.0, IC: str = "AICc", MSPE: str = "position", trace: bool = False, cores: int = 1, recurse: bool = False, TRYS=None, **kwargs):
    del recurse, TRYS
    return ctmm_select(data, CTMM, verbose=verbose, level=level, IC=IC, MSPE=MSPE, trace=trace, cores=cores, **kwargs)


def summary_ctmm_list(object, IC: str | None = None, MSPE: str | None = None, units: bool = True, **kwargs):
    del units, kwargs
    models = sort_ctmm(object, IC=IC or "AICc", MSPE=MSPE or "position")
    rows = []
    for m in models:
        rows.append(
            {
                "name": name_ctmm(m),
                "AIC": get_IC(m, "AIC"),
                "AICc": get_IC(m, "AICc"),
                "BIC": get_IC(m, "BIC"),
                "MSPE.position": get_MSPE(m, "position"),
                "DOF.area": float(m.params.get("DOF", {}).get("area", np.nan)) if isinstance(m.params.get("DOF"), dict) else np.nan,
            }
        )
    return rows


def sort_ctmm(x, decreasing: bool = False, IC: str = "AICc", MSPE: str = "position", flatten: bool = True, INF: bool = False):
    del flatten, INF
    models = list(x)
    if IC is None:
        key = lambda m: get_MSPE(m, MSPE)
    elif MSPE is None:
        key = lambda m: get_IC(m, IC)
    else:
        key = lambda m: (get_IC(m, IC), get_MSPE(m, MSPE))
    return sorted(models, key=key, reverse=decreasing)


def min_ctmm(x, IC: str = "AICc", MSPE: str = "position"):
    s = sort_ctmm(x, IC=IC, MSPE=MSPE)
    return s[0] if s else None


def ctmm_select(data, CTMM, verbose: bool = False, level: float = 1.0, IC: str = "AICc", MSPE: str = "position", trace: bool = False, cores: int = 1, **kwargs):
    del trace, cores
    models = CTMM if isinstance(CTMM, list) else [CTMM]
    out = _ctmm_select(data, models, verbose=verbose, level=level, IC=IC, MSPE=MSPE, **kwargs)
    return out if not verbose else out


__all__ = [
    "alpha_ctmm",
    "get_MSPE",
    "get_IC",
    "simplify_ctmm",
    "complexify_ctmm",
    "get_mle",
    "fix_vars",
    "update_fix",
    "iterate",
    "ctmm_iterate",
    "name_ctmm",
    "sort_ctmm",
    "min_ctmm",
    "summary_ctmm_list",
    "ctmm_select",
]
