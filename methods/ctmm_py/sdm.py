"""Partial parity translation of ctmm 1.3.0 ``R/sdm.R``."""
from __future__ import annotations
import numpy as np
from .rsf import rsf_fit
from .rsf_select import rsf_select

def sdm_fit(X, y):
    return rsf_fit(X, y)

def sdm_select(models, X, y):
    return rsf_select(models, X, y)

def sdm_integrate(pred, weights=None):
    p = np.asarray(pred, dtype=float).reshape(-1)
    if weights is None:
        return float(np.mean(p))
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = w/np.sum(w)
    return float(np.sum(w*p))


def copy_PRE(x):
    return None if x is None else np.array(x, copy=True) if not isinstance(x, dict) else dict(x)


def copy_beta(x):
    return None if x is None else np.array(x, copy=True) if not isinstance(x, dict) else dict(x)


def sdm_UD(UD, model=None, **kwargs):
    del kwargs
    out = dict(UD) if isinstance(UD, dict) else {"UD": UD}
    if model is not None:
        out["SDM"] = model
    return out


__all__ = ["copy_PRE", "copy_beta", "sdm_UD", "sdm_fit", "sdm_select", "sdm_integrate"]
