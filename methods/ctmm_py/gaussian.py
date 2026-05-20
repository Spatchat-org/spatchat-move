"""Parity-focused translation of ctmm 1.3.0 ``R/gaussian.R``."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .covm import Covm, covm, eigenvalues_covm
from .types import CTMMModel


def _params(model):
    return model.params if isinstance(model, CTMMModel) else model


def _copy_model(model, params):
    if isinstance(model, CTMMModel):
        return CTMMModel(model.model, params)
    return params


def _sigma(model):
    p = _params(model)
    s = p.get("sigma")
    if isinstance(s, Covm):
        return s
    if s is None:
        s = p.get("sigma_matrix", np.eye(len(p.get("axes", ("x", "y")))))
    return covm(s, isotropic=bool(p.get("isotropic", False)), axes=tuple(p.get("axes", ("x", "y"))))


def _cov_lookup(COV, names, row, col):
    if COV is None:
        return 0.0
    if hasattr(COV, "loc"):
        try:
            return float(COV.loc[row, col])
        except Exception:
            return 0.0
    arr = np.asarray(COV, dtype=float)
    if names and row in names and col in names:
        return float(arr[names.index(row), names.index(col)])
    return 0.0


def _pack(models):
    par = []
    parscale = []
    lower = []
    upper = []
    blocks = []
    for model in models:
        p = _params(model)
        axes = tuple(p.get("axes", ("x", "y")))
        ax = len(axes)
        sig = _sigma(model)
        max_ev = float(np.nanmax(eigenvalues_covm(sig))) if sig is not None else 1.0
        max_ev = max(max_ev, np.finfo(float).eps)
        mu = np.asarray(p.get("mu", np.zeros(ax)), dtype=float).reshape(-1)
        if mu.size < ax:
            mu = np.pad(mu, (0, ax - mu.size))

        block = {"axes": axes, "mu": [], "sigma": [], "isotropic": bool(p.get("isotropic", False))}
        for k in range(ax):
            par.append(float(mu[k]))
            parscale.append(float(np.sqrt(max_ev)))
            lower.append(-np.inf)
            upper.append(np.inf)
            block["mu"].append(len(par) - 1)

        if block["isotropic"]:
            par.append(float(sig.par.get("major", max_ev)))
            parscale.append(max_ev)
            lower.append(0.0)
            upper.append(np.inf)
            block["sigma"] = [("major", len(par) - 1)]
        else:
            for name, scale, lo, hi in (
                ("major", max_ev, 0.0, np.inf),
                ("minor", max_ev, 0.0, np.inf),
                ("angle", np.pi / 2.0, -np.inf, np.inf),
            ):
                par.append(float(sig.par.get(name, 0.0)))
                parscale.append(float(scale))
                lower.append(float(lo))
                upper.append(float(hi))
                block["sigma"].append((name, len(par) - 1))
        blocks.append(block)
    return np.asarray(par), np.asarray(parscale), np.asarray(lower), np.asarray(upper), blocks


def _unpack(models, blocks, PAR):
    out = []
    for model, block in zip(models, blocks):
        p = deepcopy(_params(model))
        axes = block["axes"]
        p["mu"] = np.asarray([PAR[i] for i in block["mu"]], dtype=float)
        if block["isotropic"]:
            sig_par = [PAR[block["sigma"][0][1]]]
        else:
            sig_par = [PAR[i] for _, i in block["sigma"]]
        sig = covm(sig_par, isotropic=block["isotropic"], axes=axes)
        p["sigma"] = sig
        p["sigma_matrix"] = sig.sigma
        out.append(_copy_model(model, p))
    return out


def _gradient(fn, par, parscale, lower, upper):
    f0 = np.asarray(fn(par), dtype=float).reshape(-1)
    grad = np.zeros((f0.size, par.size), dtype=float)
    eps = np.sqrt(np.finfo(float).eps)
    for j in range(par.size):
        step = eps * max(abs(float(par[j])), abs(float(parscale[j])), 1.0)
        lo_ok = par[j] - step >= lower[j]
        hi_ok = par[j] + step <= upper[j]
        if lo_ok and hi_ok:
            p_hi = par.copy()
            p_lo = par.copy()
            p_hi[j] += step
            p_lo[j] -= step
            grad[:, j] = (np.asarray(fn(p_hi), dtype=float).reshape(-1) - np.asarray(fn(p_lo), dtype=float).reshape(-1)) / (2.0 * step)
        elif hi_ok:
            p_hi = par.copy()
            p_hi[j] += step
            grad[:, j] = (np.asarray(fn(p_hi), dtype=float).reshape(-1) - f0) / step
        elif lo_ok:
            p_lo = par.copy()
            p_lo[j] -= step
            grad[:, j] = (f0 - np.asarray(fn(p_lo), dtype=float).reshape(-1)) / step
    return grad


def gauss_comp(fn, CTMM, COV: bool = True, **kwargs):
    del kwargs
    models = CTMM if isinstance(CTMM, list) else [CTMM]
    mle = np.asarray(fn(models), dtype=float).reshape(-1)
    if not COV:
        return {"MLE": mle, "COV": np.zeros((mle.size, mle.size), dtype=float)}

    par, parscale, lower, upper, blocks = _pack(models)

    def wrapped(PAR):
        return np.asarray(fn(_unpack(models, blocks, PAR)), dtype=float).reshape(-1)

    grad = _gradient(wrapped, par, parscale, lower, upper)
    cov_par = np.zeros((par.size, par.size), dtype=float)
    for model, block in zip(models, blocks):
        p = _params(model)
        cov_mu = p.get("COV.mu")
        if cov_mu is not None:
            c = np.asarray(cov_mu, dtype=float)
            if c.ndim == 4:
                c = c[:, 0, 0, :]
            idx = block["mu"]
            r = min(len(idx), c.shape[0], c.shape[1])
            cov_par[np.ix_(idx[:r], idx[:r])] = c[:r, :r]
        cov = p.get("COV")
        names = p.get("COV_rownames") or p.get("features")
        names = list(names) if names is not None else None
        for row_name, row_idx in block["sigma"]:
            for col_name, col_idx in block["sigma"]:
                cov_par[row_idx, col_idx] = _cov_lookup(cov, names, row_name, col_name)
    return {"MLE": mle, "COV": grad @ cov_par @ grad.T}


def FN(PAR):
    return PAR


__all__ = ["FN", "gauss_comp"]
