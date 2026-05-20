"""Parity-focused translation of ctmm 1.3.0 ``R/ctpm.R``."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .models import CTMMModel


def PHYsolve(M, v, CND: bool = False, method: str = "eigen"):
    del CND
    m = np.asarray(M, dtype=float)
    vv = np.asarray(v, dtype=float)
    if vv.ndim == 1:
        vv = vv[:, None]
    n = m.shape[0]
    fail = {"Q": np.diag(np.full(vv.shape[1], np.inf)), "log_det": float("inf")}
    if m.shape != (n, n):
        return fail

    if method == "eigen":
        try:
            vals, vecs = np.linalg.eigh((m + m.T) / 2.0)
        except Exception:
            return fail
        if np.any(~np.isfinite(vals)) or float(np.min(vals)) <= np.finfo(float).eps * n:
            return fail
        vals = np.abs(vals)
        log_det = float(np.sum(np.log(vals)))
        vt = vecs.T @ vv
        inv_vals = 1.0 / vals
        Q = vt.T @ (inv_vals[:, None] * vt)
        return {"Q": Q, "log_det": log_det}

    if method == "svd":
        try:
            u, s, vh = np.linalg.svd(m, full_matrices=False)
        except Exception:
            return fail
        if np.any(~np.isfinite(s)) or float(np.min(s)) <= np.finfo(float).eps * n:
            return fail
        log_det = float(np.sum(np.log(np.abs(s))))
        U = np.real(u + vh.T) / 2.0
        vt = U.T @ vv
        inv_s = 1.0 / s
        Q = vt.T @ (inv_s[:, None] * vt)
        return {"Q": Q, "log_det": log_det}

    return fail


def ctpm_loglike(data, CTMM: CTMMModel, REML: bool = False, profile: bool = True, zero: float = 0.0, verbose: bool = False):
    # Data expectations: dict-like with lag matrix/vector and trait vector.
    if isinstance(data, dict):
        lag = np.asarray(data.get("lag"), dtype=float)
        trait = np.asarray(data.get("trait"), dtype=float).reshape(-1)
    else:
        raise TypeError("ctpm_loglike currently expects data as dict with keys: lag, trait")

    n = int(trait.size)
    tau = CTMM.params.get("tau", {})
    tau_vals = list(tau.values()) if isinstance(tau, dict) else []
    K = len(tau_vals)
    range_ = bool(CTMM.params.get("range", True))
    if not range_:
        REML = True
    elif K and np.isinf(float(tau_vals[0])):
        return (CTMM if profile else float("-inf")) if verbose else float("-inf")

    N = n if range_ else n - 1
    DOF = n - 1 if REML else N
    VAR_MULT = N / max(DOF, 1)

    mu = float(np.mean(trait))
    z = trait - mu

    if K == 0:
        Q = float(np.sum(z * z))
        log_det = 0.0
        COV_mu = 1.0 / max(n, 1)
    else:
        if lag.ndim == 1:
            L = np.abs(lag[:, None] - lag[None, :])
        else:
            L = np.abs(lag)
        tau_pos = float(tau.get("position", tau_vals[0] if tau_vals else 1.0))
        tau_pos = max(tau_pos, 1e-9)
        acf = np.exp(-L / tau_pos)
        COR = acf if range_ else -acf
        S = np.column_stack([z, np.ones(n, dtype=float)])
        slv = PHYsolve(COR, S, CND=not range_)
        QQ = np.asarray(slv["Q"], dtype=float)
        log_det = float(slv["log_det"])
        mu = mu + float(QQ[0, 1] / max(QQ[1, 1], 1e-18))
        COV_mu = 1.0 / max(float(QQ[1, 1]), 1e-18)
        Q = float(QQ[0, 0] - QQ[0, 1] * QQ[1, 0] / max(QQ[1, 1], 1e-18))

    if profile:
        sigma = Q / max(DOF, 1)
        Qn = 0.0
    else:
        sig = CTMM.params.get("sigma")
        if hasattr(sig, "par"):
            sigma = float(sig.par.get("major", 1.0))
        else:
            sigma = 1.0
        Qn = Q / max(N, 1) / max(sigma, 1e-18) - 1.0 / VAR_MULT
    COV_mu = sigma * COV_mu

    ll_const = -1.0 / 2.0 / VAR_MULT - 0.5 * math.log(2.0 * math.pi)
    loglike = -0.5 * (log_det / max(N, 1) + math.log(max(sigma, 1e-18)) + Qn)
    loglike = N * (loglike + (ll_const - zero / max(N, 1)))
    if REML or not range_:
        loglike += 0.5 * math.log(max(COV_mu, 1e-18))
    if not np.isfinite(loglike):
        loglike = float("-inf")

    if verbose and profile:
        out = CTMMModel(model=CTMM.model, params=dict(CTMM.params))
        out.params["mu"] = mu
        out.params["COV.mu"] = COV_mu
        out.params["loglike"] = float(loglike)
        return out
    return float(loglike)


__all__ = ["PHYsolve", "ctpm_loglike"]
