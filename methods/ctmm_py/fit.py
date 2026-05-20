"""Partial parity translation of ctmm 1.3.0 ``R/fit.R``."""

from __future__ import annotations

from typing import Any

import numpy as np

from .generic_utils import epoch_seconds
from .models import ctmm_fit as _ctmm_fit
from .models import ctmm_guess as _ctmm_guess
from .models import ctmm_loglike as _ctmm_loglike
from .variogram import variogram as _variogram


def telemetry_mins(data, axes=("x", "y")):
    if hasattr(data, "time_col"):
        t = epoch_seconds(data.data[data.time_col])
        dt = np.diff(t)
    else:
        dt = np.array([1.0])
    dt = dt[dt > 0]
    dt_med = float(np.median(dt)) if dt.size else 1.0
    df = float(2.0 * np.pi / max((t[-1] - t[0]) if "t" in locals() and len(t) > 1 else 1.0, 1e-12))

    if hasattr(data, "x_col") and hasattr(data, "y_col"):
        arr = data.data[[data.x_col, data.y_col]].to_numpy(dtype=float)
        d = np.diff(arr, axis=0)
        dz = np.sqrt(np.sum(d * d, axis=1))
        dz = dz[dz > 0]
        dz_min = float(np.min(dz)) if dz.size else 1.0
    else:
        dz_min = 1.0
    return {"dt": dt_med, "df": df, "dz": dz_min}


def get_loglike(data):
    del data
    return _ctmm_loglike


def ic_ctmm(CTMM, n: int):
    q = len(CTMM.params.get("axes", ("x", "y")))
    k_mean = len(np.asarray(CTMM.params.get("mu", []), dtype=float).reshape(-1))
    features = CTMM.params.get("features", [])
    nu = len(features)
    k = nu + k_mean
    ll = float(CTMM.params.get("loglike", np.nan))
    aic = 2 * k - 2 * ll
    bic = np.log(max(n, 1)) * k - 2 * ll
    denom = max(q * n - k - max(nu, 1), 1e-12)
    aicc = -2 * ll + q * n * (2 * k / denom)
    CTMM.params["AIC"] = float(aic)
    CTMM.params["AICc"] = float(aicc) if np.isfinite(aicc) else float("inf")
    CTMM.params["BIC"] = float(bic)
    return CTMM


def ctmm_guess(data, CTMM=None, variogram=None, name: str = "GUESS", interactive: bool = True):
    del name, interactive
    if variogram is None:
        try:
            variogram = _variogram(data)
        except Exception:
            variogram = {"lags_s": np.array([1.0, 2.0]), "gamma": np.array([1.0, 1.0])}
    return _ctmm_guess(variogram, model=CTMM)


def ctmm_fit(data, CTMM, method: str = "pHREML", COV: bool = True, control: dict[str, Any] | None = None, trace: bool = False):
    return _ctmm_fit(data, CTMM, method=method, COV=COV, control=control, trace=trace)


def setup_parameters(CTMM, *args, **kwargs):
    del args, kwargs
    params = CTMM.params if hasattr(CTMM, "params") else CTMM
    names = list(params.get("features", []))
    return {"NAMES": names, "pars": np.asarray([], dtype=float), "parscale": np.asarray([], dtype=float), "lower": np.asarray([], dtype=float), "upper": np.asarray([], dtype=float)}


def store_pars(CTMM, pars=None, *args, **kwargs):
    del args, kwargs
    if pars is not None:
        if hasattr(CTMM, "params"):
            CTMM.params["pars"] = pars
        else:
            CTMM["pars"] = pars
    return CTMM


def unscale_ctmm(CTMM, *args, **kwargs):
    del args, kwargs
    return CTMM


def reset(CTMM, *args, **kwargs):
    del args, kwargs
    if hasattr(CTMM, "params"):
        CTMM.params.pop("COV", None)
        CTMM.params.pop("COV.mu", None)
    return CTMM


def mspe(data, CTMM, *args, **kwargs):
    del args, kwargs
    try:
        fit = ctmm_fit(data, CTMM)
        return fit.params.get("MSPE", {})
    except Exception:
        return {}


def fn(p, data=None, CTMM=None, **kwargs):
    del p
    if data is None or CTMM is None:
        return np.inf
    return -float(get_loglike(data)(data, CTMM, **kwargs))


__all__ = [
    "telemetry_mins",
    "get_loglike",
    "ic_ctmm",
    "ctmm_guess",
    "ctmm_fit",
    "setup_parameters",
    "store_pars",
    "unscale_ctmm",
    "reset",
    "mspe",
    "fn",
]
