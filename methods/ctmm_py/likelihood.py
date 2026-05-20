"""Partial parity translation of ctmm 1.3.0 ``R/likelihood.R``."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .models import ctmm_loglike as _ctmm_loglike


def get_link(CTMM) -> dict[str, Any]:
    link_name = CTMM.params.get("link", "identity")
    if link_name == "identity":
        fn = lambda x: x
        grad = lambda x: np.ones_like(np.asarray(x, dtype=float))
    elif link_name == "log":
        fn = np.log
        grad = lambda x: 1.0 / np.asarray(x, dtype=float)
    else:
        fn = lambda x: x
        grad = lambda x: np.ones_like(np.asarray(x, dtype=float))
    return {"name": link_name, "fn": fn, "grad": grad}


def ctmm_circulate(CTMM, t):
    t = np.asarray(t, dtype=float)
    circle = float(CTMM.params.get("circle", 0.0) or 0.0)
    if t.size == 0:
        return np.array([], dtype=float)
    return circle * (t - t[0])


def ctmm_apply(CTMM, fn: Callable = lambda x: x, states=None):
    del states
    return fn(CTMM)


def sigma_apply(CTMM, fn: Callable = lambda x: x, states=None):
    del states
    p = dict(CTMM.params)
    if "sigma" in p:
        p["sigma"] = fn(p["sigma"])
    CTMM.params = p
    return CTMM


def ctmm_loglike(data, CTMM, REML: bool = False, profile: bool = True, zero: float = 0.0, verbose: bool = False, compute: bool = True, **kwargs):
    del REML, profile, zero, verbose, compute, kwargs
    return _ctmm_loglike(data, CTMM)


def COVM(data, CTMM, **kwargs):
    del kwargs
    return np.asarray(CTMM.params.get("sigma_matrix", np.eye(2)), dtype=float)


def scale_vars(data, CTMM=None, **kwargs):
    del CTMM, kwargs
    return data


def fn(data, CTMM, **kwargs):
    return ctmm_loglike(data, CTMM, **kwargs)


__all__ = ["COVM", "fn", "get_link", "ctmm_circulate", "ctmm_apply", "sigma_apply", "scale_vars", "ctmm_loglike"]
