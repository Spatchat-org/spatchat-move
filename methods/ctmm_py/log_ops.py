from __future__ import annotations

import numpy as np
from scipy import special

from .generic_utils import nant
from .series_utils import log_chi2_bias


def _as_arr(x):
    return np.asarray(x, dtype=float)


def exp_log(est, var_est=0.0, var=0.0, var_var=0.0):
    est = _as_arr(est)
    var_est = _as_arr(var_est)
    var = _as_arr(var)
    var_var = _as_arr(var_var)
    mu = np.exp(est + var / 2.0)
    var_mu = (mu**2) * var_est + ((mu / 2.0) ** 2) * var_var
    return {"mu": mu, "VAR": var_mu}


def Log(values, dof, variable: str = "area", debias: bool = True):
    values = _as_arr(values)
    dof = _as_arr(dof)
    out_log = np.log(values)
    if variable == "speed":
        var_log = 0.25 * (special.polygamma(1, dof / 2.0))
        if debias:
            bias = 0.5 * (special.digamma(dof / 2.0) - np.log(dof / 2.0))
            out_log = out_log - nant(bias, 0.0)
        else:
            var_log = 0.25 * (2.0 / dof)
    else:
        var_log = special.polygamma(1, dof / 2.0) if debias else (2.0 / dof)
        if debias:
            out_log = out_log - log_chi2_bias(dof)
    return {"log": out_log, "VAR.log": np.asarray(var_log, dtype=float)}


def Exp(est, var_est=0.0, var=0.0, var_var=0.0, variable: str = "area", debias: bool = True):
    res = exp_log(est=est, var_est=var_est, var=var, var_var=var_var)
    mu = _as_arr(res["mu"])
    var_mu = _as_arr(res["VAR"])
    if variable == "speed":
        dof = 2.0 * (mu**2) / np.maximum((mu**2 + var_mu) - mu**2, np.finfo(float).eps)
    else:
        dof = 2.0 * mu**2 / np.maximum(var_mu, np.finfo(float).eps)
    if debias:
        bias = special.digamma(dof / 2.0) - np.log(dof / 2.0)
        if variable == "speed":
            bias = bias / 2.0
        mu = mu + nant(bias, 0.0)
    return {"est": mu, "VAR": var_mu, "DOF": dof}

