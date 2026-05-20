"""Parity-focused translation of ctmm 1.3.0 ``R/emulate.R``."""

from __future__ import annotations

import copy

import numpy as np

from .misc_ops import ctmm_boot as _ctmm_boot
from .types import CTMMModel, Telemetry
from .krige import simulate_ctmm
from .parameters import get_parameters, set_parameters


POSITIVE_PARAMETERS = {"major", "minor", "tau", "tau position", "tau velocity", "omega", "error", "period"}


def Transform(p, inverse: bool = False):
    del inverse
    return p


def ctmm_reduce(CTMM: CTMMModel):
    out = CTMMModel(CTMM.model, dict(CTMM.params))
    features = list(out.params.get("features", []) or [])
    if features:
        params = get_parameters(out, features)
        keep = [f for f in features if not (f in POSITIVE_PARAMETERS and float(params.get(f, 1.0) or 0.0) == 0.0)]
        out.params["features"] = keep
    return out


def emulate_ctmm(object: CTMMModel, data: Telemetry | None = None, fast: bool = False, seed=None, **kwargs):
    del kwargs
    if not fast and data is not None:
        return emulate_telemetry(data, object, fast=fast, seed=seed)
    out = CTMMModel(object.model, copy.deepcopy(object.params))
    cov = out.params.get("COV")
    names = out.params.get("COV_rownames") or out.params.get("features")
    if cov is not None and names:
        par = get_parameters(out, names)
        mu = np.array([float(par[n]) for n in names], dtype=float)
        C = np.asarray(cov, dtype=float)[: len(names), : len(names)]
        try:
            draw = np.random.default_rng(seed).multivariate_normal(mu, C)
            updates = {n: draw[i] for i, n in enumerate(names)}
            for n in names:
                if n in POSITIVE_PARAMETERS or n.startswith("tau "):
                    updates[n] = max(float(updates[n]), np.finfo(float).tiny)
            out = set_parameters(out, updates)
        except Exception:
            pass
    return out


def emulate_telemetry(object: Telemetry, CTMM: CTMMModel, fast: bool = False, seed=None, **kwargs):
    if fast:
        return emulate_ctmm(CTMM, data=object, fast=True, seed=seed, **kwargs)
    sim = simulate_ctmm(CTMM, data=object, seed=seed)
    from .models import ctmm_fit

    return ctmm_fit(sim, CTMM, method=CTMM.params.get("method", "pHREML"), COV=False, **kwargs)


def emulate(object, data=None, fast: bool = False, **kwargs):
    if isinstance(object, CTMMModel):
        return emulate_ctmm(object, data=data, fast=fast, **kwargs)
    if isinstance(object, Telemetry):
        if data is None or not isinstance(data, CTMMModel):
            raise TypeError("emulate.telemetry requires CTMM model")
        return emulate_telemetry(object, data, fast=fast, **kwargs)
    return copy.deepcopy(object)


def ctmm_boot(data: Telemetry, CTMM: CTMMModel, **kwargs):
    return _ctmm_boot(data, CTMM, **kwargs)


def Replicate(i=0, DATA=None, CTMM=None, **kwargs):
    del i
    if CTMM is None:
        return DATA
    return emulate(CTMM, data=DATA, **kwargs)


def rolling_update(state, value, alpha: float = 1.0):
    if state is None:
        return value
    return (1.0 - alpha) * state + alpha * value


def estimator(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(np.nanmean(arr)), "sd": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0}


__all__ = [
    "POSITIVE_PARAMETERS",
    "Replicate",
    "Transform",
    "ctmm_boot",
    "ctmm_reduce",
    "emulate",
    "emulate_ctmm",
    "emulate_telemetry",
    "estimator",
    "rolling_update",
]
