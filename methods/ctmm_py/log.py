"""Parity-focused translation of ctmm 1.3.0 ``R/log.R``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .log_ops import Exp, Log, exp_log
from .stats import chi_dof


def _as_list(x):
    return x if isinstance(x, list) else [x]


def _extract_est_dof(obj, variable: str = "area"):
    if isinstance(obj, dict):
        if "AREA" in obj and "DOF" in obj:
            return obj["AREA"], obj["DOF"]
        ci = obj.get("CI")
        dof = obj.get("DOF")
        if isinstance(ci, dict) and variable in ci:
            return ci[variable][1], dof.get(variable, 0.0) if isinstance(dof, dict) else dof
        if hasattr(ci, "loc"):
            row = [r for r in ci.index if variable in str(r)]
            if row:
                return float(ci.loc[row[0], "est"]), dof.get(variable, 0.0) if isinstance(dof, dict) else dof
    return np.nan, 0.0


def log_ctmms(x, variable: str = "area", debias: bool = True, level_UD: float = 0.95, **kwargs):
    del level_UD, kwargs
    xs = _as_list(x)
    est = []
    dof = []
    for obj in xs:
        e, d = _extract_est_dof(obj, variable=variable)
        est.append(e)
        dof.append(d)
    res = Log(est, dof, variable=variable, debias=debias)
    return res


def mlog_ctmms(x, variable=("area",), debias: bool = True, level_UD: float = 0.95, **kwargs):
    return {v: log_ctmms(x, variable=v, debias=debias, level_UD=level_UD, **kwargs) for v in variable}


def log_area(x, variable: str = "area", debias: bool = True, **kwargs):
    return log_ctmms(x, variable=variable, debias=debias, **kwargs)


def log_UD(x, variable: str = "area", debias: bool = True, level_UD: float = 0.95, **kwargs):
    return log_area(x, variable=variable, debias=debias, level_UD=level_UD, **kwargs)


def log_speed(x, variable: str = "speed", debias: bool = True, **kwargs):
    return log_ctmms(x, variable=variable, debias=debias, **kwargs)


def log_dataframe(CTMM, EST=None, variable: str = "area", debias: bool = True, **kwargs):
    del EST
    out = log_ctmms(CTMM, variable=variable, debias=debias, **kwargs)
    return pd.DataFrame({"log": np.asarray(out["log"]).ravel(), "VAR.log": np.asarray(out["VAR.log"]).ravel()})


def Exp_ci(est, VAR_est=0, VAR=0, VAR_VAR=0, variable: str = "area", debias: bool = True, level: float = 0.95, units: bool = True, **kwargs):
    del level, units, kwargs
    return Exp(est, var_est=VAR_est, var=VAR, var_var=VAR_VAR, variable=variable, debias=debias)


def chi_dof_from_log(est, var):
    est = np.asarray(est, dtype=float)
    var = np.asarray(var, dtype=float)
    return np.vectorize(chi_dof)(est, est * est + var)


def fn(*args, **kwargs):
    return log_ctmms(*args, **kwargs)


__all__ = [
    "Exp",
    "Exp_ci",
    "Log",
    "chi_dof_from_log",
    "exp_log",
    "fn",
    "log_area",
    "log_ctmms",
    "log_dataframe",
    "log_speed",
    "log_UD",
    "mlog_ctmms",
]
