"""Partial parity translation of ctmm 1.3.0 ``R/ctmm.R``."""

from __future__ import annotations

from .ctmm_dynamics import continuity, get_states, get_taus
from .models import ctmm, ctmm_fit, ctmm_guess, ctmm_loglike, ctmm_select


def ctmm_ctmm(CTMM=None, **kwargs):
    if CTMM is None:
        return ctmm(**kwargs)
    if hasattr(CTMM, "params"):
        params = dict(CTMM.params)
        params.update(kwargs)
        return ctmm(**params)
    if isinstance(CTMM, dict):
        params = dict(CTMM)
        params.update(kwargs)
        return ctmm(**params)
    return ctmm(**kwargs)


def pars_tauv(tau):
    return {"tau": tau}


def ctmm_prepare(model, *args, **kwargs):
    del args, kwargs
    return model


def ctmm_repair(model, *args, **kwargs):
    del args, kwargs
    return model


__all__ = [
    "continuity",
    "ctmm",
    "ctmm_ctmm",
    "ctmm_fit",
    "ctmm_guess",
    "ctmm_loglike",
    "ctmm_prepare",
    "ctmm_repair",
    "ctmm_select",
    "get_states",
    "get_taus",
    "pars_tauv",
]
