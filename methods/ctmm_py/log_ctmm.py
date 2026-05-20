"""Parity-focused translation of ctmm 1.3.0 ``R/log.ctmm.R``."""

from __future__ import annotations

import numpy as np

from .covm import Covm, covm, exp_covm, log_covm
from .types import CTMMModel


def log_ctmm(CTMM: CTMMModel | list[CTMMModel], debias: bool = False, **kwargs):
    del debias, kwargs
    if isinstance(CTMM, list):
        return [log_ctmm(m) for m in CTMM]
    p = CTMM.params
    par = {}
    tau = p.get("tau", {}) or {}
    for name, value in tau.items():
        par[f"tau {name}"] = np.log(float(value))
    if p.get("omega", 0):
        par["omega"] = np.log(float(p["omega"]))
    sigma = p.get("sigma")
    if isinstance(sigma, Covm):
        ls = log_covm(sigma)
        par.update(ls.par)
    cov = p.get("COV")
    return {"par": par, "COV": cov, "isotropic": bool(p.get("isotropic", False))}


def exp_ctmm(CTMM, debias: bool = False, variance: bool = True, base=None):
    del debias, variance
    base_params = {} if base is None else dict(getattr(base, "params", base if isinstance(base, dict) else {}))
    par = dict(CTMM.get("par", {})) if isinstance(CTMM, dict) else {}
    tau = {}
    sig_par = {}
    for name, value in par.items():
        if name.startswith("tau "):
            tau[name.split(" ", 1)[1]] = float(np.exp(value))
        elif name in {"major", "minor", "angle"}:
            sig_par[name] = float(value if name == "angle" else np.exp(value))
        else:
            base_params[name] = float(np.exp(value))
    if tau:
        base_params["tau"] = dict(sorted(tau.items(), key=lambda kv: float(kv[1]), reverse=True))
        base_params["tau_list"] = [float(v) for v in base_params["tau"].values()]
    if sig_par:
        if "minor" not in sig_par and "major" in sig_par:
            sig_par["minor"] = sig_par["major"]
        sig_par.setdefault("angle", 0.0)
        base_params["sigma"] = covm(sig_par, isotropic=bool(CTMM.get("isotropic", False)))
        base_params["sigma_matrix"] = base_params["sigma"].sigma
    return CTMMModel(getattr(base, "model", "ctmm") if base is not None else "ctmm", base_params)


__all__ = ["exp_ctmm", "log_ctmm"]
