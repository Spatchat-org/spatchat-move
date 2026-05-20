"""Partial parity translation of ctmm 1.3.0 ``R/rsf.select.R``."""
from __future__ import annotations
import numpy as np
from .rsf import rsf_fit


def get_terms(formula):
    if formula is None:
        return []
    if isinstance(formula, str):
        rhs = formula.split("~", 1)[-1]
        return [t.strip() for t in rhs.replace("*", "+").replace(":", "+").split("+") if t.strip() and t.strip() != "1"]
    return list(formula)


def terms2formula(terms, response: str = "y"):
    terms = list(terms)
    rhs = " + ".join(terms) if terms else "1"
    return f"{response} ~ {rhs}"


def rsf_select(models, X, y):
    best = None
    best_aic = np.inf
    scores = []
    for m in models:
        cols = np.asarray(m, dtype=int)
        fit = rsf_fit(X[:, cols], y)
        b = fit['beta']
        z = X[:, cols] @ b
        p = 1/(1+np.exp(-z))
        ll = np.sum(y*np.log(np.maximum(p,1e-12))+(1-y)*np.log(np.maximum(1-p,1e-12)))
        k = len(b)
        aic = 2*k - 2*ll
        scores.append(aic)
        if aic < best_aic:
            best_aic = aic
            best = {"cols": cols, "fit": fit, "AIC": aic}
    return {"best": best, "AIC": np.asarray(scores, dtype=float)}

__all__ = ["get_terms", "rsf_select", "terms2formula"]
