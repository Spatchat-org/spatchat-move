from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from .covm import Covm, covm as covm_factory
from .core_math import dexp1, dexp2
from .ctmm_dynamics import get_taus
from .generic_utils import epoch_seconds
from .r_math import sinc, sinch
from .types import CTMMModel, Telemetry

_TAU_NAMES = ("position", "velocity", "acceleration")


def _get_param(params: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in params:
            return params[name]
        dot = name.replace("_", ".")
        if dot in params:
            return params[dot]
        under = name.replace(".", "_")
        if under in params:
            return params[under]
    return default


def _as_1d_float(value: Any, default: list[float] | tuple[float, ...] | None = None) -> np.ndarray:
    if value is None:
        value = [] if default is None else default
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _timelink_par(model: CTMMModel) -> np.ndarray:
    return _as_1d_float(_get_param(model.params, "timelink.par", "timelink_par", default=[]))


def _model_times(telem: Telemetry, model: CTMMModel) -> np.ndarray:
    """Return clock time after the ctmm timelink transform.

    R applies ``linktime`` before preparing time lags. When sundial columns are
    absent, the transform cannot be evaluated and the clock time is left intact.
    """
    t = epoch_seconds(telem.data[telem.time_col])
    link = str(model.params.get("timelink", "identity") or "identity")
    par = _timelink_par(model)
    if link == "identity" or par.size == 0:
        return t
    df = telem.data
    if link == "switch" and {"light.time", "dark.time"}.issubset(df.columns):
        p = float(np.clip(par[0], -1.0, 1.0))
        light = pd.to_numeric(df["light.time"], errors="coerce").to_numpy(dtype=float)
        dark = pd.to_numeric(df["dark.time"], errors="coerce").to_numpy(dtype=float)
        linked = light * (1.0 + p) + dark * (1.0 - p)
        if np.all(np.isfinite(linked)):
            return linked
    return t


def _periodic_omega(params: dict[str, Any]) -> np.ndarray:
    period = _as_1d_float(_get_param(params, "period", default=[86400.0]))
    harmonic = _as_1d_float(_get_param(params, "harmonic", default=np.zeros(period.size, dtype=float)))
    if harmonic.size == 0:
        return np.array([], dtype=float)
    if period.size == 1 and harmonic.size > 1:
        period = np.repeat(period, harmonic.size)
    if harmonic.size == 1 and period.size > 1:
        harmonic = np.repeat(harmonic, period.size)
    out: list[float] = []
    for p, h in zip(period, harmonic):
        if not np.isfinite(p) or p <= 0:
            continue
        hk = int(max(math.floor(float(h)), 0))
        out.extend((2.0 * math.pi / float(p)) * k for k in range(1, hk + 1))
    return np.asarray(out, dtype=float)


def _drift_design(model: CTMMModel, t: np.ndarray) -> np.ndarray:
    """Linear mean design from R ``drift.mean`` for supported mean families."""
    t = np.asarray(t, dtype=float)
    mean = str(model.params.get("mean", "stationary") or "stationary")
    if mean not in {"stationary", "zero", "periodic"}:
        try:
            from .mean import drift_mean

            design = np.asarray(drift_mean(model, t), dtype=float)
            if design.ndim == 1:
                design = design.reshape(-1, 1)
            if design.shape[0] == t.size:
                return design
        except Exception:
            pass
    if mean == "zero":
        return np.zeros((t.size, 0), dtype=float)
    if mean == "periodic":
        cols = [np.ones(t.size, dtype=float)]
        for omega in _periodic_omega(model.params):
            theta = t * float(omega)
            cols.append(np.sin(theta))
            cols.append(np.cos(theta))
        return np.column_stack(cols) if cols else np.zeros((t.size, 0), dtype=float)
    return np.ones((t.size, 1), dtype=float)


def _profile_mean_cov(z: np.ndarray, basis: np.ndarray | None, q: np.ndarray, reml: bool, isotropic: bool):
    z = np.asarray(z, dtype=float)
    n, dim = z.shape
    if basis is None:
        basis = np.ones((n, 1), dtype=float)
    B = np.asarray(basis, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    if B.shape[0] != n:
        B = np.ones((n, 1), dtype=float)
    q = np.maximum(np.asarray(q, dtype=float).reshape(-1), np.finfo(float).tiny)
    if q.size != n:
        q = np.ones(n, dtype=float)
    if B.shape[1] == 0:
        beta = np.zeros((0, dim), dtype=float)
        r = z
        logdet_basis = 0.0
        rank = 0
    else:
        w = 1.0 / q
        Bw = B * w[:, None]
        gram = B.T @ Bw
        rhs = Bw.T @ z
        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram) @ rhs
        r = z - B @ beta
        sign, logdet_basis = np.linalg.slogdet(gram)
        rank = int(np.linalg.matrix_rank(gram))
        if sign <= 0 or not np.isfinite(logdet_basis):
            logdet_basis = 0.0
    dof = max(n - rank, 1) if reml else n
    A = (r.T / q) @ r
    if isotropic:
        s2 = float(np.trace(A) / max(dof * dim, 1))
        if not np.isfinite(s2) or s2 <= 0:
            return None
        sigma = np.eye(dim, dtype=float) * s2
    else:
        sigma = A / dof
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0 or not np.isfinite(logdet):
        return None
    value = dof * logdet + dim * float(np.sum(np.log(q)))
    if reml and rank > 0:
        value += dim * float(logdet_basis)
    return float(value), beta, sigma, rank


def _cov_loglike(hess: np.ndarray, grad: np.ndarray | None = None, tol: float = np.finfo(float).eps) -> np.ndarray:
    """Translation of R ``cov.loglike`` for pREML parameter corrections."""
    hess = np.nan_to_num(np.asarray(hess, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if grad is None:
        grad = np.zeros(hess.shape[0], dtype=float)
    grad = np.nan_to_num(np.asarray(grad, dtype=float), nan=np.inf, posinf=np.inf, neginf=-np.inf)

    if hess.ndim != 2 or hess.shape[0] != hess.shape[1]:
        return np.zeros((0, 0), dtype=float)
    if grad.size != hess.shape[0]:
        grad = np.resize(grad, hess.shape[0])

    if np.all(np.diag(hess) > 0):
        try:
            cov = np.linalg.inv(hess)
            if np.all(np.diag(cov) > 0):
                return cov
        except np.linalg.LinAlgError:
            pass

    scale = np.sqrt(np.abs(np.diag(hess)))
    scale = np.maximum(scale, np.abs(grad))
    scale[scale <= tol] = 1.0
    outer = np.outer(scale, scale)

    g = np.nan_to_num(grad / scale, nan=1.0, posinf=1.0, neginf=-1.0)
    h = np.nan_to_num(hess / outer, nan=1.0, posinf=1.0, neginf=-1.0)
    max_off = np.outer(np.sqrt(np.abs(np.diag(h))), np.sqrt(np.abs(np.diag(h))))
    h = np.minimum(np.maximum(h, -max_off), max_off)

    try:
        values, vectors = np.linalg.eigh(h)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(hess)
    values = np.maximum(values, 0.0)
    g = vectors.T @ g
    out_values = np.zeros_like(values, dtype=float)
    infinite = np.zeros_like(values, dtype=bool)
    for i, (val, gi) in enumerate(zip(values, g)):
        det = val + gi * gi
        if val == 0.0:
            if gi == 0.0:
                infinite[i] = True
            else:
                out_values[i] = 1.0 / (2.0 * gi) ** 2
        elif det >= 0.0:
            out_values[i] = ((math.sqrt(det) - gi) / val) ** 2
        elif gi == 0.0:
            infinite[i] = True
        else:
            out_values[i] = 1.0 / (2.0 * gi) ** 2

    cov = np.zeros_like(h)
    for i, val in enumerate(out_values):
        vv = np.outer(vectors[:, i], vectors[:, i])
        if infinite[i] or not np.isfinite(val):
            d = np.diag(np.diag(vv))
            d[d > 0] = np.inf
            cov += d
        else:
            cov += val * vv
    return cov / outer


def _covm_par_from_matrix(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.asarray(sigma, dtype=float)
    vals, vecs = np.linalg.eigh((sigma + sigma.T) / 2.0)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], np.finfo(float).tiny)
    vecs = vecs[:, order]
    angle = math.atan2(float(vecs[1, 0]), float(vecs[0, 0]))
    angle = ((angle / math.pi + 0.5) % 1.0 - 0.5) * math.pi
    return np.array([float(vals[0]), float(vals[1]), float(angle)], dtype=float), vecs


def _matrix_from_covm_par(par: np.ndarray) -> np.ndarray:
    major, minor, angle = float(par[0]), float(par[1]), float(par[2])
    c = math.cos(angle)
    s = math.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return rot @ np.diag([major, minor]) @ rot.T


def _linear_cov_jacobian(par: np.ndarray) -> np.ndarray:
    """Jacobian from ctmm covm parameters to linear xx, yy, xy parameters."""
    p = np.asarray(par, dtype=float)
    jac = np.zeros((3, 3), dtype=float)

    def lin(v: np.ndarray) -> np.ndarray:
        s = _matrix_from_covm_par(v)
        return np.array([s[0, 0], s[1, 1], s[0, 1]], dtype=float)

    for i in range(3):
        h = max(abs(float(p[i])) * 1e-6, 1e-6)
        e = np.zeros(3, dtype=float)
        e[i] = h
        jac[:, i] = (lin(p + e) - lin(p - e)) / (2.0 * h)
    return jac


def _ou_value_for_sigma(
    z: np.ndarray,
    t: np.ndarray,
    tau: float,
    sigma: np.ndarray,
    *,
    reml: bool = False,
    design: np.ndarray | None = None,
) -> float:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n, dim = z.shape
    if n < 2 or tau <= 0 or sigma.shape != (dim, dim):
        return float("inf")
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0 or not np.isfinite(logdet):
        return float("inf")
    dt = np.diff(t)
    if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
        return float("inf")
    phi = np.exp(-dt / float(tau))
    q = np.r_[1.0, 1.0 - phi * phi]
    q = np.maximum(q, np.finfo(float).tiny)
    y = np.empty_like(z)
    y[0] = z[0]
    y[1:] = z[1:] - phi[:, None] * z[:-1]
    if design is None:
        basis = np.empty((n, 1), dtype=float)
        basis[0, 0] = 1.0
        basis[1:, 0] = 1.0 - phi
    else:
        d = np.asarray(design, dtype=float)
        if d.ndim == 1:
            d = d.reshape(-1, 1)
        if d.shape[0] != n:
            return float("inf")
        basis = np.empty_like(d, dtype=float)
        basis[0] = d[0]
        basis[1:] = d[1:] - phi[:, None] * d[:-1]
    if basis.shape[1] == 0:
        resid = y
        rank = 0
        logdet_basis = 0.0
    else:
        w = 1.0 / q
        bw = basis * w[:, None]
        gram = basis.T @ bw
        rhs = bw.T @ y
        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram) @ rhs
        resid = y - basis @ beta
        sign_b, logdet_basis = np.linalg.slogdet(gram)
        rank = int(np.linalg.matrix_rank(gram))
        if sign_b <= 0 or not np.isfinite(logdet_basis):
            logdet_basis = 0.0
    try:
        inv_sigma = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        return float("inf")
    quad = float(np.sum(np.einsum("ij,jk,ik->i", resid, inv_sigma, resid) / q))
    value = n * float(logdet) + dim * float(np.sum(np.log(q))) + quad
    if reml and rank > 0:
        value = (n - rank) * float(logdet) + dim * float(np.sum(np.log(q))) + quad + dim * float(logdet_basis)
    return float(value)


def _ou_preml_correction(
    z: np.ndarray,
    t: np.ndarray,
    tau_ml: float,
    sigma_ml: np.ndarray,
    *,
    design: np.ndarray | None = None,
    isotropic: bool = False,
) -> tuple[float, np.ndarray, float] | None:
    """R-style OU pREML correction in standardized fit units."""
    if isotropic:
        return None
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma_ml = np.asarray(sigma_ml, dtype=float)
    if z.ndim != 2 or z.shape[1] != 2 or z.shape[0] < 3 or sigma_ml.shape != (2, 2):
        return None
    if design is not None:
        d0 = np.asarray(design, dtype=float)
        if d0.ndim == 1:
            d0 = d0.reshape(-1, 1)
        # The stationary-mean path is the common ctmm.select/AKDE path; more
        # complex drift terms require the full R parameter block.
        if d0.shape[0] != z.shape[0] or d0.shape[1] != 1:
            return None
    dt = np.diff(t)
    pos_dt = dt[np.isfinite(dt) & (dt > 0)]
    if pos_dt.size == 0 or not np.isfinite(tau_ml) or tau_ml <= 0:
        return None

    z0 = z - np.nanmean(z, axis=0)
    length_scale = float(math.sqrt(max(float(np.nanmean(z0 * z0)), np.finfo(float).tiny)))
    time_scale = float(np.nanmedian(pos_dt))
    if not np.isfinite(length_scale) or length_scale <= 0:
        length_scale = 1.0
    if not np.isfinite(time_scale) or time_scale <= 0:
        time_scale = 1.0
    zs = z0 / length_scale
    ts = (t - float(t[0])) / time_scale
    tau_s = float(tau_ml) / time_scale
    sigma_s = sigma_ml / (length_scale * length_scale)

    cov_par, _ = _covm_par_from_matrix(sigma_s)
    par = np.r_[cov_par, tau_s]
    linear_par = np.array([sigma_s[0, 0], sigma_s[1, 1], sigma_s[0, 1], tau_s], dtype=float)
    if not np.all(np.isfinite(par)) or np.any(par[[0, 1, 3]] <= 0):
        return None

    def sigma_from_par(p: np.ndarray) -> np.ndarray | None:
        if p[0] <= 0 or p[1] <= 0 or p[3] <= 0:
            return None
        s = _matrix_from_covm_par(p[:3])
        if np.linalg.det(s) <= 0:
            return None
        return s

    def objective(p: np.ndarray, *, reml: bool) -> float:
        p = np.asarray(p, dtype=float)
        if p.size != 4 or not np.all(np.isfinite(p)):
            return 1e100
        s = sigma_from_par(p)
        if s is None:
            return 1e100
        val = _ou_value_for_sigma(zs, ts, float(p[3]), s, reml=reml, design=design)
        return 1e100 if not np.isfinite(val) else float(val)

    step = np.maximum(np.abs(par) * 1e-4, 1e-4)
    m = par.size
    hess = np.zeros((m, m), dtype=float)
    grad_ml = np.zeros(m, dtype=float)
    grad_reml = np.zeros(m, dtype=float)
    f0 = objective(par, reml=False)
    if not np.isfinite(f0) or f0 >= 1e99:
        return None
    for i in range(m):
        ei = np.zeros(m, dtype=float)
        ei[i] = step[i]
        fp = objective(par + ei, reml=False)
        fm = objective(par - ei, reml=False)
        if not np.isfinite(fp) or fp >= 1e99 or not np.isfinite(fm) or fm >= 1e99:
            return None
        hess[i, i] = (fp - 2.0 * f0 + fm) / (step[i] * step[i])
        grad_ml[i] = (fp - fm) / (2.0 * step[i])
        rp = objective(par + ei, reml=True)
        rm = objective(par - ei, reml=True)
        if not np.isfinite(rp) or rp >= 1e99 or not np.isfinite(rm) or rm >= 1e99:
            return None
        grad_reml[i] = (rp - rm) / (2.0 * step[i])
        for j in range(i + 1, m):
            ej = np.zeros(m, dtype=float)
            ej[j] = step[j]
            fpp = objective(par + ei + ej, reml=False)
            fpm = objective(par + ei - ej, reml=False)
            fmp = objective(par - ei + ej, reml=False)
            fmm = objective(par - ei - ej, reml=False)
            if not all(np.isfinite(v) and v < 1e99 for v in (fpp, fpm, fmp, fmm)):
                return None
            hess[i, j] = hess[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * step[i] * step[j])

    cov = _cov_loglike(hess, grad_ml)
    if cov.shape != (m, m) or not np.all(np.isfinite(cov)):
        return None
    try:
        eig_val, eig_vec = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    # R clamps the pREML inverse-Hessian contribution to one DOF after
    # switching to linear covariance parameters.
    eig_val = np.minimum(eig_val, 2.0 * (eig_vec.T @ linear_par) ** 2)
    cov = eig_vec @ np.diag(eig_val) @ eig_vec.T
    jac = np.eye(4, dtype=float)
    jac[:3, :3] = _linear_cov_jacobian(par[:3])
    delta = -(jac @ cov @ grad_reml)
    if not np.all(np.isfinite(delta)):
        return None
    cand = linear_par + delta
    for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
        cand = linear_par + scale * delta
        tau_c = float(cand[3])
        sigma_c = np.array([[cand[0], cand[2]], [cand[2], cand[1]]], dtype=float)
        if tau_c <= 0 or np.linalg.det(sigma_c) <= 0:
            continue
        val = _ou_value_for_sigma(zs, ts, tau_c, sigma_c, reml=False, design=design)
        if np.isfinite(val):
            return tau_c * time_scale, sigma_c * (length_scale * length_scale), float(val)
    return None


def _gls_mean_value(y: np.ndarray, basis: np.ndarray, covs: np.ndarray, reml: bool):
    y = np.asarray(y, dtype=float)
    B = np.asarray(basis, dtype=float)
    covs = np.asarray(covs, dtype=float)
    n, dim = y.shape
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    p = B.shape[1]
    if p == 0:
        beta = np.zeros((0, dim), dtype=float)
        resid = y
        normal = np.zeros((0, 0), dtype=float)
    else:
        normal = np.zeros((p * dim, p * dim), dtype=float)
        rhs = np.zeros(p * dim, dtype=float)
        for i in range(n):
            cov = covs[i]
            sign, _ = np.linalg.slogdet(cov)
            if sign <= 0 or not np.all(np.isfinite(cov)):
                return None
            inv = np.linalg.inv(cov)
            X = np.kron(B[i : i + 1, :], np.eye(dim, dtype=float))
            normal += X.T @ inv @ X
            rhs += X.T @ inv @ y[i]
        try:
            beta_flat = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            beta_flat = np.linalg.pinv(normal) @ rhs
        beta = beta_flat.reshape(p, dim)
        resid = y - B @ beta
    value = 0.0
    for i in range(n):
        cov = covs[i]
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0 or not np.isfinite(logdet):
            return None
        try:
            q = float(resid[i] @ np.linalg.inv(cov) @ resid[i])
        except np.linalg.LinAlgError:
            return None
        value += logdet + q
    if reml and p:
        sign, logdet = np.linalg.slogdet(normal)
        if sign > 0 and np.isfinite(logdet):
            value += float(logdet)
    return float(value), beta


def _pack_mu(beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    if beta.ndim == 2 and beta.shape[0] == 1:
        return beta[0]
    return beta


def _mean_at(model: CTMMModel, t: np.ndarray, dim: int = 2) -> np.ndarray:
    U = _drift_design(model, t)
    beta = np.asarray(model.params.get("mu", []), dtype=float)
    if U.shape[1] == 0:
        return np.zeros((U.shape[0], dim), dtype=float)
    if beta.ndim == 1:
        if beta.size == 0:
            return np.zeros((U.shape[0], dim), dtype=float)
        if beta.size == dim and U.shape[1] == 1:
            beta = beta.reshape(1, dim)
        else:
            beta = np.resize(beta, (U.shape[1], dim))
    if beta.shape[0] != U.shape[1]:
        beta = np.resize(beta, (U.shape[1], dim))
    return U @ beta[:, :dim]


def _fit_ou_profile(z: np.ndarray, t: np.ndarray, method: str = "pHREML", isotropic: bool = False, design: np.ndarray | None = None, error_var: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray, float] | None:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.all(np.isfinite(z), axis=1) & np.isfinite(t)
    z = z[ok]
    t = t[ok]
    if design is not None:
        D = np.asarray(design, dtype=float)
        design = D[ok] if D.shape[0] == ok.shape[0] else None
    if error_var is not None:
        ev = np.asarray(error_var, dtype=float).reshape(-1)
        error_var = ev[ok] if ev.size == ok.size else None
    if z.shape[0] < 3:
        return None
    dt = np.diff(t)
    pos_dt = dt[np.isfinite(dt) & (dt > 0)]
    if pos_dt.size == 0:
        return None
    n, dim = z.shape
    lo = max(float(np.min(pos_dt)) / 100.0, 1e-9)
    hi = max(float(t[-1] - t[0]) * 10.0, lo * 10.0)
    err_active = error_var is not None and np.any(np.asarray(error_var, dtype=float) > 0)

    if err_active:
        reml = str(method) in ("pREML", "pHREML", "HREML", "REML")
        base = _fit_ou_profile(z, t, method=method, isotropic=isotropic, design=design, error_var=None)
        if base is None:
            s0 = np.cov(z.T)
            tau0 = max(float(np.median(pos_dt)), lo * 10.0)
        else:
            tau0, _, s0, _ = base
        if np.asarray(s0).shape != (dim, dim) or not np.all(np.isfinite(s0)):
            s0 = np.eye(dim, dtype=float) * max(float(np.nanvar(z)), np.finfo(float).tiny)
        s0 = np.asarray(s0, dtype=float)

        def unpack(par: np.ndarray):
            tau = float(np.exp(par[0]))
            if isotropic:
                sigma = np.eye(dim, dtype=float) * float(np.exp(par[1]))
            elif dim == 2:
                L = np.array([[math.exp(float(par[1])), 0.0], [float(par[2]), math.exp(float(par[3]))]], dtype=float)
                sigma = L @ L.T
            else:
                sigma = np.eye(dim, dtype=float) * float(np.exp(par[1]))
            return tau, sigma

        def objective(par: np.ndarray) -> float:
            tau, sigma = unpack(np.asarray(par, dtype=float))
            if tau <= lo or tau >= hi:
                return 1e100
            phi = np.exp(-dt / tau)
            v = np.r_[1.0, 1.0 - phi * phi]
            y = np.empty_like(z)
            y[0] = z[0]
            y[1:] = z[1:] - phi[:, None] * z[:-1]
            if design is None:
                B = np.empty((n, 1), dtype=float)
                B[0, 0] = 1.0
                B[1:, 0] = 1.0 - phi
            else:
                B = np.empty_like(design, dtype=float)
                B[0] = design[0]
                B[1:] = design[1:] - phi[:, None] * design[:-1]
            e = np.asarray(error_var, dtype=float)
            e_y = np.r_[e[0], e[1:] + phi * phi * e[:-1]]
            eye = np.eye(dim, dtype=float)
            covs = np.empty((n, dim, dim), dtype=float)
            for i in range(n):
                covs[i] = v[i] * sigma + e_y[i] * eye
            out = _gls_mean_value(y, B, covs, reml=reml)
            return 1e100 if out is None else float(out[0])

        try:
            L0 = np.linalg.cholesky(s0 + np.eye(dim) * np.finfo(float).eps)
        except np.linalg.LinAlgError:
            L0 = np.linalg.cholesky(np.eye(dim) * max(float(np.trace(s0)) / max(dim, 1), np.finfo(float).tiny))
        if isotropic:
            p0 = np.array([math.log(max(float(tau0), lo * 1.01)), math.log(max(float(np.trace(s0)) / max(dim, 1), np.finfo(float).tiny))], dtype=float)
            bounds = [(math.log(lo), math.log(hi)), (math.log(np.finfo(float).tiny), None)]
        elif dim == 2:
            p0 = np.array([math.log(max(float(tau0), lo * 1.01)), math.log(max(L0[0, 0], np.finfo(float).tiny)), float(L0[1, 0]), math.log(max(L0[1, 1], np.finfo(float).tiny))], dtype=float)
            bounds = [(math.log(lo), math.log(hi)), (math.log(np.finfo(float).tiny), None), (None, None), (math.log(np.finfo(float).tiny), None)]
        else:
            p0 = np.array([math.log(max(float(tau0), lo * 1.01)), math.log(max(float(s0[0, 0]), np.finfo(float).tiny))], dtype=float)
            bounds = [(math.log(lo), math.log(hi)), (math.log(np.finfo(float).tiny), None)]
        opt = minimize(objective, p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400, "ftol": 1e-9, "gtol": 1e-6, "maxls": 40})
        if not np.isfinite(opt.fun) or opt.fun >= 1e99:
            opt = minimize(objective, p0, method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-7, "fatol": 1e-5})
        if not np.isfinite(opt.fun) or opt.fun >= 1e99:
            return None
        tau_hat, sigma_hat = unpack(np.asarray(opt.x, dtype=float))
        phi = np.exp(-dt / tau_hat)
        v = np.r_[1.0, 1.0 - phi * phi]
        y = np.empty_like(z)
        y[0] = z[0]
        y[1:] = z[1:] - phi[:, None] * z[:-1]
        if design is None:
            B = np.empty((n, 1), dtype=float)
            B[0, 0] = 1.0
            B[1:, 0] = 1.0 - phi
        else:
            B = np.empty_like(design, dtype=float)
            B[0] = design[0]
            B[1:] = design[1:] - phi[:, None] * design[:-1]
        e = np.asarray(error_var, dtype=float)
        e_y = np.r_[e[0], e[1:] + phi * phi * e[:-1]]
        eye = np.eye(dim, dtype=float)
        covs = np.empty((n, dim, dim), dtype=float)
        for i in range(n):
            covs[i] = v[i] * sigma_hat + e_y[i] * eye
        final = _gls_mean_value(y, B, covs, reml=reml)
        if final is None:
            return None
        fit_value, beta = final
        return float(tau_hat), _pack_mu(beta), sigma_hat, float(fit_value)

    def profile_parts(tau: float, reml_sigma: bool = False):
        tau = float(tau)
        phi = np.exp(-dt / tau)
        v = np.r_[1.0, 1.0 - phi * phi]
        v = np.maximum(v, np.finfo(float).tiny)
        y = np.empty_like(z)
        y[0] = z[0]
        y[1:] = z[1:] - phi[:, None] * z[:-1]
        if design is None:
            B = np.empty((n, 1), dtype=float)
            B[0, 0] = 1.0
            B[1:, 0] = 1.0 - phi
        else:
            B = np.empty_like(design, dtype=float)
            B[0] = design[0]
            B[1:] = design[1:] - phi[:, None] * design[:-1]
        prof = _profile_mean_cov(y, B, v, reml=reml_sigma, isotropic=isotropic)
        if prof is None:
            return None
        value, beta, sigma, rank = prof
        return v, value, beta, sigma, rank

    def profile_value(tau: float, reml: bool = False) -> float:
        parts = profile_parts(tau, reml_sigma=reml)
        if parts is None:
            return float("inf")
        v, value, _, sigma, _ = parts
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0 or not np.isfinite(logdet):
            return float("inf")
        return float(value)

    opt = minimize_scalar(lambda log_tau: profile_value(float(np.exp(log_tau)), reml=False), bounds=(math.log(lo), math.log(hi)), method="bounded", options={"xatol": 1e-10})
    if not opt.success:
        return None
    tau_ml = float(np.exp(opt.x))
    method = str(method)
    use_preml = method in ("pREML", "pHREML")
    use_reml_sigma = method in ("pREML", "pHREML", "HREML", "REML")
    tau_hat = tau_ml
    sigma_preml = None
    ml_parts = profile_parts(tau_ml, reml_sigma=False)
    if use_preml:
        if ml_parts is not None:
            _, _, _, sigma_ml, _ = ml_parts
            corr = _ou_preml_correction(z, t, tau_ml, sigma_ml, design=design, isotropic=isotropic)
            if corr is not None:
                tau_corr, sigma_corr, _ = corr
                if np.isfinite(tau_corr) and tau_corr >= lo:
                    tau_hat = float(tau_corr)
                    sigma_preml = np.asarray(sigma_corr, dtype=float)
        if sigma_preml is None:
            h = max(abs(tau_ml) * 1e-4, 1.0)
            ml0 = profile_value(tau_ml, reml=False)
            mlp = profile_value(tau_ml + h, reml=False)
            mlm = profile_value(max(tau_ml - h, lo), reml=False)
            if tau_ml - h <= lo:
                hess_ml = (profile_value(tau_ml + 2.0 * h, reml=False) - 2.0 * mlp + ml0) / (h * h)
            else:
                hess_ml = (mlp - 2.0 * ml0 + mlm) / (h * h)
            rp = profile_value(tau_ml + h, reml=True)
            rm = profile_value(max(tau_ml - h, lo), reml=True)
            if tau_ml - h <= lo:
                grad_reml = (rp - profile_value(tau_ml, reml=True)) / h
            else:
                grad_reml = (rp - rm) / (2.0 * h)
            if np.isfinite(hess_ml) and hess_ml > 0 and np.isfinite(grad_reml):
                tau_hat = max(tau_ml - grad_reml / hess_ml, lo)
    parts = profile_parts(tau_hat, reml_sigma=use_reml_sigma)
    if parts is None:
        return None
    _, _, mu, sigma, _ = parts
    if sigma_preml is not None and sigma_preml.shape == sigma.shape:
        sigma = sigma_preml
        fit_value = float(profile_value(tau_hat, reml=use_reml_sigma))
    else:
        fit_value = float(profile_value(tau_hat, reml=use_reml_sigma))
    return tau_hat, _pack_mu(mu), sigma, fit_value


def _langevin2_base(dt: float, tau: list[float]) -> tuple[np.ndarray, np.ndarray]:
    tauv = np.asarray(tau[:2], dtype=float)
    omega2 = float(np.prod(1.0 / tauv))
    green = np.zeros((2, 2), dtype=float)
    sigma = np.array([[1.0, 0.0], [0.0, omega2]], dtype=float)
    if not np.isfinite(dt):
        return green, sigma
    f = float(np.mean(1.0 / tauv))
    nu = float((1.0 / tauv[1] - 1.0 / tauv[0]) / 2.0)
    tt = float(2.0 * f / omega2)
    fdt = f * dt
    nudt = nu * dt
    use_exp = bool(tauv[0] > tauv[1] and nudt > 0.8813736)
    if use_exp:
        dtau = dt / tauv
        dift = float(np.diff(tauv)[0])
        exp0 = np.exp(-dtau)
        expv = exp0 / dift
        c0 = float(np.diff(expv * tauv)[0])
        c1 = float(-np.diff(expv)[0])
        c2 = float(np.diff(expv / tauv)[0])
    else:
        expv = float(np.exp(-fdt))
        if tauv[0] > tauv[1]:
            sin0 = float(np.sinh(nudt))
            sinc0 = float(sinch(nudt, sin0))
            cos0 = float(np.cosh(nudt))
        else:
            sin0 = float(np.sin(nudt))
            sinc0 = float(sinc(nudt, sin0))
            cos0 = float(np.cos(nudt))
        since = sinc0 * expv
        cose = cos0 * expv
        c0 = cose + fdt * since
        c1 = -(omega2 * dt) * since
        c2 = -omega2 * (cose - fdt * since)
    green[0, 0] = c0
    green[1, 0] = c1
    green[0, 1] = -c1 / omega2
    green[1, 1] = -c2 / omega2
    if use_exp:
        dift = float(np.diff(tauv)[0])
        dift2 = dift * dift
        t2 = tauv * tauv
        dtau = dt / tauv
        exp0 = np.exp(-dtau)
        s1 = float(dexp2(dtau[0], exp0[0]))
        s2 = float(dexp2(dtau[1], exp0[1]))
        s12 = float(2.0 * tauv[0] * tauv[1] * dexp1(fdt, exp0[0] * exp0[1]))
        sigma[0, 0] = (t2[0] * s1 - s12 + t2[1] * s2) / dift2
        sigma[1, 1] = (t2[1] * s1 - s12 + t2[0] * s2) / dift2 * omega2
    else:
        cross = fdt * sinc0 * expv
        outer = cos0 * cos0 * float(dexp2(fdt, expv)) - cross * cross
        cross = 2.0 * cos0 * expv * cross
        sin2 = sin0 * sin0
        if tauv[0] > tauv[1]:
            sigma[0, 0] = outer - sin2 - cross
            sigma[1, 1] = (outer - sin2 + cross) * omega2
        else:
            sigma[0, 0] = outer + sin2 - cross
            sigma[1, 1] = (outer + sin2 + cross) * omega2
    c12 = c1 * c1
    sigma[0, 0] -= c12 / omega2
    sigma[0, 1] = sigma[1, 0] = tt * c12
    sigma[1, 1] -= c12
    return green, sigma


def _langevin2_components(dt, tau: list[float]):
    dt = np.asarray(dt, dtype=float)
    tauv = np.asarray(tau[:2], dtype=float)
    omega2 = float(np.prod(1.0 / tauv))
    f = float(np.mean(1.0 / tauv))
    nu = float((1.0 / tauv[1] - 1.0 / tauv[0]) / 2.0)
    tt = float(2.0 * f / omega2)

    c0 = np.zeros_like(dt, dtype=float)
    c1 = np.zeros_like(dt, dtype=float)
    c2 = np.zeros_like(dt, dtype=float)
    s00 = np.zeros_like(dt, dtype=float)
    s01 = np.zeros_like(dt, dtype=float)
    s11 = np.zeros_like(dt, dtype=float)

    fdt = f * dt
    nudt = nu * dt
    use_exp = (tauv[0] > tauv[1]) & (nudt > 0.8813736)

    if np.any(use_exp):
        dte = dt[use_exp]
        dtau = dte[:, None] / tauv[None, :]
        dift = float(np.diff(tauv)[0])
        exp0 = np.exp(-dtau)
        expv = exp0 / dift
        c0e = np.diff(expv * tauv[None, :], axis=1).ravel()
        c1e = -np.diff(expv, axis=1).ravel()
        c2e = np.diff(expv / tauv[None, :], axis=1).ravel()
        t2 = tauv * tauv
        dift2 = dift * dift
        s1 = dexp2(dtau[:, 0], exp0[:, 0])
        s2 = dexp2(dtau[:, 1], exp0[:, 1])
        s12 = 2.0 * tauv[0] * tauv[1] * dexp1(f * dte, exp0[:, 0] * exp0[:, 1])
        s00e = (t2[0] * s1 - s12 + t2[1] * s2) / dift2
        s11e = (t2[1] * s1 - s12 + t2[0] * s2) / dift2 * omega2
        c12e = c1e * c1e
        s00e = s00e - c12e / omega2
        s01e = tt * c12e
        s11e = s11e - c12e
        c0[use_exp] = c0e
        c1[use_exp] = c1e
        c2[use_exp] = c2e
        s00[use_exp] = s00e
        s01[use_exp] = s01e
        s11[use_exp] = s11e

    use_direct = ~use_exp
    if np.any(use_direct):
        dtd = dt[use_direct]
        fdtd = fdt[use_direct]
        nudtd = nudt[use_direct]
        expv = np.exp(-fdtd)
        if tauv[0] > tauv[1]:
            sin0 = np.sinh(nudtd)
            sinc0 = sinch(nudtd, sin0)
            cos0 = np.cosh(nudtd)
        else:
            sin0 = np.sin(nudtd)
            sinc0 = sinc(nudtd, sin0)
            cos0 = np.cos(nudtd)
        since = sinc0 * expv
        cose = cos0 * expv
        c0d = cose + fdtd * since
        c1d = -(omega2 * dtd) * since
        c2d = -omega2 * (cose - fdtd * since)
        cross = fdtd * sinc0 * expv
        outer = cos0 * cos0 * dexp2(fdtd, expv) - cross * cross
        cross = 2.0 * cos0 * expv * cross
        sin2 = sin0 * sin0
        if tauv[0] > tauv[1]:
            s00d = outer - sin2 - cross
            s11d = (outer - sin2 + cross) * omega2
        else:
            s00d = outer + sin2 - cross
            s11d = (outer + sin2 + cross) * omega2
        c12d = c1d * c1d
        s00d = s00d - c12d / omega2
        s01d = tt * c12d
        s11d = s11d - c12d
        c0[use_direct] = c0d
        c1[use_direct] = c1d
        c2[use_direct] = c2d
        s00[use_direct] = s00d
        s01[use_direct] = s01d
        s11[use_direct] = s11d

    g00 = c0
    g10 = c1
    g01 = -c1 / omega2
    g11 = -c2 / omega2
    return g00, g01, g10, g11, s00, s01, s11, omega2


def _innovations_ouf_scalar(x: np.ndarray, t: np.ndarray, tau: list[float], error_var: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    n = x.size
    if error_var is None:
        errv = np.zeros(n, dtype=float)
    else:
        errv = np.asarray(error_var, dtype=float).reshape(-1)
        if errv.size != n:
            errv = np.zeros(n, dtype=float)
        errv = np.maximum(np.nan_to_num(errv, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    dt = np.diff(t)
    g00, g01, g10, g11, s00, s01, s11, omega2 = _langevin2_components(dt, tau)
    state0 = 0.0
    state1 = 0.0
    cov00 = 1.0
    cov01 = 0.0
    cov11 = omega2
    err = np.empty(n, dtype=float)
    var = np.empty(n, dtype=float)
    tiny = np.finfo(float).tiny
    for i in range(n):
        qi = float(cov00 + errv[i])
        if qi < tiny:
            qi = tiny
        ei = float(x[i] - state0)
        gain0 = cov00 / qi
        gain1 = cov01 / qi
        state0 = state0 + gain0 * ei
        state1 = state1 + gain1 * ei
        cov00_old = cov00
        cov01_old = cov01
        cov11_old = cov11
        cov00 = cov00_old - gain0 * cov00_old
        cov01 = cov01_old - gain0 * cov01_old
        cov11 = cov11_old - gain1 * cov01_old
        err[i] = ei
        var[i] = qi
        if i < n - 1:
            state0, state1 = (
                g00[i] * state0 + g01[i] * state1,
                g10[i] * state0 + g11[i] * state1,
            )
            cov00, cov01, cov11 = (
                g00[i] * g00[i] * cov00 + 2.0 * g00[i] * g01[i] * cov01 + g01[i] * g01[i] * cov11 + s00[i],
                g00[i] * g10[i] * cov00 + (g00[i] * g11[i] + g01[i] * g10[i]) * cov01 + g01[i] * g11[i] * cov11 + s01[i],
                g10[i] * g10[i] * cov00 + 2.0 * g10[i] * g11[i] * cov01 + g11[i] * g11[i] * cov11 + s11[i],
            )
    return err, var


def _fit_ouf_profile(z: np.ndarray, t: np.ndarray, start_tau: list[float], method: str = "pHREML", isotropic: bool = False, design: np.ndarray | None = None, error_var: np.ndarray | None = None) -> tuple[list[float], np.ndarray, np.ndarray, float] | None:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.all(np.isfinite(z), axis=1) & np.isfinite(t)
    z = z[ok]
    t = t[ok]
    if design is not None:
        D = np.asarray(design, dtype=float)
        design = D[ok] if D.shape[0] == ok.shape[0] else None
    if error_var is not None:
        ev = np.asarray(error_var, dtype=float).reshape(-1)
        error_var = ev[ok] if ev.size == ok.size else None
    if z.shape[0] < 4:
        return None
    dt = np.diff(t)
    pos_dt = dt[np.isfinite(dt) & (dt > 0)]
    if pos_dt.size == 0:
        return None
    n, dim = z.shape
    span = max(float(t[-1] - t[0]), float(np.max(pos_dt)))
    min_dt = float(np.min(pos_dt))
    med_dt = float(np.median(pos_dt))

    def finish_for_tau(tau: list[float], reml: bool = False):
        if design is None:
            B, q = _innovations_ouf_scalar(np.ones(n, dtype=float), t, tau, error_var=error_var)
            B = B.reshape(-1, 1)
        else:
            cols = []
            q = None
            for k in range(design.shape[1]):
                b, qk = _innovations_ouf_scalar(design[:, k], t, tau, error_var=error_var)
                cols.append(b)
                if q is None:
                    q = qk
            B = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)
            if q is None:
                _, q = _innovations_ouf_scalar(np.ones(n, dtype=float), t, tau, error_var=error_var)
        q = np.maximum(np.asarray(q, dtype=float), np.finfo(float).tiny)
        residuals = []
        for j in range(dim):
            e, _ = _innovations_ouf_scalar(z[:, j], t, tau, error_var=error_var)
            residuals.append(e)
        r = np.column_stack(residuals)
        prof = _profile_mean_cov(r, B, q, reml=reml, isotropic=isotropic)
        if prof is None:
            return None
        value, beta, sigma, _ = prof
        return value, beta, sigma

    def objective(par: np.ndarray, reml: bool = False) -> float:
        tau1 = float(np.exp(par[0]))
        tau2 = float(np.exp(par[1]))
        if tau2 <= 0 or tau1 <= tau2 or tau1 > 10.0 * span or tau2 < min_dt / 100.0:
            return 1e100
        out = finish_for_tau([tau1, tau2], reml=reml)
        return 1e100 if out is None else float(out[0])

    starts = [
        [max(float(start_tau[0]), min_dt * 10.0), max(float(start_tau[1]), min_dt / 2.0)],
        [max(float(start_tau[0]), min_dt * 10.0), max(med_dt, min_dt / 2.0)],
        [10.0 * 86400.0, max(min_dt, 60.0)],
        [10.0 * 86400.0, max(med_dt, min_dt / 2.0)],
        [20.0 * 86400.0, max(min_dt, 60.0)],
        [20.0 * 86400.0, max(med_dt, min_dt / 2.0)],
    ]
    bounds = [
        (math.log(max(min_dt * 1.01, min_dt / 100.0)), math.log(max(10.0 * span, min_dt * 10.0))),
        (math.log(max(min_dt / 100.0, np.finfo(float).tiny)), math.log(max(span, min_dt * 10.0))),
    ]
    best = None
    for st in starts:
        st[0] = max(st[0], st[1] * 1.01)
        x0 = np.log(st)
        opt = minimize(
            lambda p: objective(p, reml=False),
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 120, "ftol": 1e-10, "gtol": 1e-6, "maxls": 30},
        )
        if np.isfinite(opt.fun) and opt.fun < 1e99 and (best is None or opt.fun < best.fun):
            best = opt
    if best is None:
        for st in starts[:1]:
            st[0] = max(st[0], st[1] * 1.01)
            opt = minimize(lambda p: objective(p, reml=False), np.log(st), method="Nelder-Mead", options={"maxiter": 120, "xatol": 1e-5, "fatol": 1e-4})
            if opt.success and (best is None or opt.fun < best.fun):
                best = opt
    if best is None:
        return None
    par = np.asarray(best.x, dtype=float)
    method = str(method)
    use_reml = method in ("pREML", "pHREML", "HREML", "REML")
    if method in ("pREML", "pHREML"):
        tau_ml = np.exp(par)
        step = np.maximum(np.abs(tau_ml) * 3e-4, 1.0)

        def objective_tau(tau_vec, reml: bool = False) -> float:
            tau_vec = np.asarray(tau_vec, dtype=float)
            if tau_vec.size != 2 or tau_vec[1] <= 0 or tau_vec[0] <= tau_vec[1] or tau_vec[0] > 10.0 * span or tau_vec[1] < min_dt / 100.0:
                return 1e100
            out = finish_for_tau([float(tau_vec[0]), float(tau_vec[1])], reml=reml)
            return 1e100 if out is None else float(out[0])

        def finite_grad(fn, p):
            g = np.zeros_like(p, dtype=float)
            for i in range(p.size):
                dp = np.zeros_like(p, dtype=float)
                dp[i] = step[i]
                g[i] = (fn(p + dp) - fn(p - dp)) / (2.0 * step[i])
            return g

        def finite_hess(fn, p):
            hess = np.zeros((p.size, p.size), dtype=float)
            f0 = fn(p)
            for i in range(p.size):
                dpi = np.zeros_like(p, dtype=float)
                dpi[i] = step[i]
                fp = fn(p + dpi)
                fm = fn(p - dpi)
                hess[i, i] = (fp - 2.0 * f0 + fm) / (step[i] * step[i])
                for j in range(i + 1, p.size):
                    dpj = np.zeros_like(p, dtype=float)
                    dpj[j] = step[j]
                    fpp = fn(p + dpi + dpj)
                    fpm = fn(p + dpi - dpj)
                    fmp = fn(p - dpi + dpj)
                    fmm = fn(p - dpi - dpj)
                    hess[i, j] = hess[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * step[i] * step[j])
            return hess

        hess_ml = finite_hess(lambda p: objective_tau(p, reml=False), tau_ml)
        grad_reml = finite_grad(lambda p: objective_tau(p, reml=True), tau_ml)
        if np.all(np.isfinite(hess_ml)) and np.all(np.isfinite(grad_reml)):
            try:
                delta = -np.linalg.solve(hess_ml, grad_reml)
            except np.linalg.LinAlgError:
                delta = -np.linalg.pinv(hess_ml) @ grad_reml
            cand_tau = np.sort(tau_ml + delta)[::-1]
            if np.all(np.isfinite(cand_tau)) and objective_tau(cand_tau, reml=True) < 1e99:
                par = np.log(cand_tau)
    elif method == "REML":
        best_reml = None
        reml_starts = [
            par,
            np.log([max(float(start_tau[0]), min_dt * 10.0), max(float(start_tau[1]), min_dt / 2.0)]),
        ]
        for st in reml_starts:
            opt = minimize(lambda p: objective(p, reml=True), np.asarray(st, dtype=float), method="Nelder-Mead", options={"maxiter": 350, "xatol": 1e-5, "fatol": 1e-4})
            if opt.success and (best_reml is None or opt.fun < best_reml.fun):
                best_reml = opt
        if best_reml is not None:
            par = np.asarray(best_reml.x, dtype=float)
    tau = [float(np.exp(par[0])), float(np.exp(par[1]))]
    final = finish_for_tau(tau, reml=use_reml)
    if final is None:
        return None
    fit_value, mu, sigma = final
    return tau, _pack_mu(mu), sigma, float(fit_value)


def _fit_ouf_tied_profile(z: np.ndarray, t: np.ndarray, start_tau: float, method: str = "pHREML", isotropic: bool = False, design: np.ndarray | None = None, error_var: np.ndarray | None = None) -> tuple[list[float], np.ndarray, np.ndarray, float] | None:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.all(np.isfinite(z), axis=1) & np.isfinite(t)
    z = z[ok]
    t = t[ok]
    if design is not None:
        D = np.asarray(design, dtype=float)
        design = D[ok] if D.shape[0] == ok.shape[0] else None
    if error_var is not None:
        ev = np.asarray(error_var, dtype=float).reshape(-1)
        error_var = ev[ok] if ev.size == ok.size else None
    if z.shape[0] < 4:
        return None
    dt = np.diff(t)
    pos_dt = dt[np.isfinite(dt) & (dt > 0)]
    if pos_dt.size == 0:
        return None
    n, dim = z.shape
    span = max(float(t[-1] - t[0]), float(np.max(pos_dt)))
    min_dt = float(np.min(pos_dt))
    lo = max(min_dt / 100.0, np.finfo(float).tiny)
    hi = max(10.0 * span, min_dt * 10.0)

    def finish_for_tau(tau_scalar: float, reml: bool = False):
        tau_pair = [float(tau_scalar), float(tau_scalar)]
        if design is None:
            B, q = _innovations_ouf_scalar(np.ones(n, dtype=float), t, tau_pair, error_var=error_var)
            B = B.reshape(-1, 1)
        else:
            cols = []
            q = None
            for k in range(design.shape[1]):
                b, qk = _innovations_ouf_scalar(design[:, k], t, tau_pair, error_var=error_var)
                cols.append(b)
                if q is None:
                    q = qk
            B = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)
            if q is None:
                _, q = _innovations_ouf_scalar(np.ones(n, dtype=float), t, tau_pair, error_var=error_var)
        q = np.maximum(np.asarray(q, dtype=float), np.finfo(float).tiny)
        residuals = []
        for j in range(dim):
            e, _ = _innovations_ouf_scalar(z[:, j], t, tau_pair, error_var=error_var)
            residuals.append(e)
        r = np.column_stack(residuals)
        prof = _profile_mean_cov(r, B, q, reml=reml, isotropic=isotropic)
        if prof is None:
            return None
        value, beta, sigma, _ = prof
        return value, beta, sigma

    def profile_value(tau_scalar: float, reml: bool = False) -> float:
        if tau_scalar <= lo or tau_scalar >= hi:
            return 1e100
        out = finish_for_tau(float(tau_scalar), reml=reml)
        return 1e100 if out is None else float(out[0])

    opt = minimize_scalar(lambda log_tau: profile_value(float(np.exp(log_tau)), reml=False), bounds=(math.log(lo), math.log(hi)), method="bounded", options={"xatol": 1e-10})
    if not opt.success or not np.isfinite(opt.fun) or opt.fun >= 1e99:
        tau0 = min(max(float(start_tau), lo * 1.01), hi / 1.01)
        opt2 = minimize(lambda p: profile_value(float(np.exp(p[0])), reml=False), np.array([math.log(tau0)], dtype=float), method="Nelder-Mead", options={"maxiter": 160, "xatol": 1e-6, "fatol": 1e-5})
        if not opt2.success or not np.isfinite(opt2.fun) or opt2.fun >= 1e99:
            return None
        tau_ml = float(np.exp(opt2.x[0]))
    else:
        tau_ml = float(np.exp(opt.x))

    method = str(method)
    use_reml = method in ("pREML", "pHREML", "HREML", "REML")
    tau_hat = tau_ml
    if method in ("pREML", "pHREML"):
        h = max(abs(tau_ml) * 3e-4, 1.0)
        ml0 = profile_value(tau_ml, reml=False)
        mlp = profile_value(min(tau_ml + h, hi / 1.01), reml=False)
        mlm = profile_value(max(tau_ml - h, lo * 1.01), reml=False)
        hess_ml = (mlp - 2.0 * ml0 + mlm) / (h * h)
        rp = profile_value(min(tau_ml + h, hi / 1.01), reml=True)
        rm = profile_value(max(tau_ml - h, lo * 1.01), reml=True)
        grad_reml = (rp - rm) / (2.0 * h)
        if np.isfinite(hess_ml) and hess_ml > 0 and np.isfinite(grad_reml):
            cand = tau_ml - grad_reml / hess_ml
            if lo < cand < hi and profile_value(cand, reml=use_reml) < 1e99:
                tau_hat = float(cand)
    elif method == "REML":
        opt_reml = minimize_scalar(lambda log_tau: profile_value(float(np.exp(log_tau)), reml=True), bounds=(math.log(lo), math.log(hi)), method="bounded", options={"xatol": 1e-10})
        if opt_reml.success and np.isfinite(opt_reml.fun) and opt_reml.fun < 1e99:
            tau_hat = float(np.exp(opt_reml.x))

    final = finish_for_tau(tau_hat, reml=use_reml)
    if final is None:
        return None
    fit_value, mu, sigma = final
    return [float(tau_hat), float(tau_hat)], _pack_mu(mu), sigma, float(fit_value)


def _ouomega_components(dt, tau: float, omega: float):
    dt = np.asarray(dt, dtype=float)
    tau = float(tau)
    omega = float(omega)
    f = 1.0 / max(tau, np.finfo(float).tiny)
    nu = max(omega, np.finfo(float).tiny)
    omega2 = f * f + nu * nu
    edt = np.exp(-f * dt)
    c = np.cos(nu * dt)
    s = np.sin(nu * dt)
    g00 = edt * (c + (f / nu) * s)
    g01 = edt * (s / nu)
    g10 = -edt * (omega2 / nu) * s
    g11 = edt * (c - (f / nu) * s)
    s00 = 1.0 - (g00 * g00 + omega2 * g01 * g01)
    s01 = -(g00 * g10 + omega2 * g01 * g11)
    s11 = omega2 - (g10 * g10 + omega2 * g11 * g11)
    tiny = np.finfo(float).tiny
    s00 = np.maximum(s00, tiny)
    s11 = np.maximum(s11, tiny)
    return g00, g01, g10, g11, s00, s01, s11, omega2


def _innovations_ouomega_scalar(x: np.ndarray, t: np.ndarray, tau: float, omega: float, error_var: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    n = x.size
    if error_var is None:
        errv = np.zeros(n, dtype=float)
    else:
        errv = np.asarray(error_var, dtype=float).reshape(-1)
        if errv.size != n:
            errv = np.zeros(n, dtype=float)
        errv = np.maximum(np.nan_to_num(errv, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    dt = np.diff(t)
    g00, g01, g10, g11, s00, s01, s11, omega2 = _ouomega_components(dt, tau, omega)
    state0 = 0.0
    state1 = 0.0
    cov00 = 1.0
    cov01 = 0.0
    cov11 = omega2
    err = np.empty(n, dtype=float)
    var = np.empty(n, dtype=float)
    for i in range(n):
        qi = max(float(cov00 + errv[i]), np.finfo(float).tiny)
        ei = float(x[i] - state0)
        gain0 = cov00 / qi
        gain1 = cov01 / qi
        state0 = state0 + gain0 * ei
        state1 = state1 + gain1 * ei
        cov00_old = cov00
        cov01_old = cov01
        cov11_old = cov11
        cov00 = cov00_old - gain0 * cov00_old
        cov01 = cov01_old - gain0 * cov01_old
        cov11 = cov11_old - gain1 * cov01_old
        err[i] = ei
        var[i] = qi
        if i < n - 1:
            state0, state1 = (
                g00[i] * state0 + g01[i] * state1,
                g10[i] * state0 + g11[i] * state1,
            )
            cov00, cov01, cov11 = (
                g00[i] * g00[i] * cov00 + 2.0 * g00[i] * g01[i] * cov01 + g01[i] * g01[i] * cov11 + s00[i],
                g00[i] * g10[i] * cov00 + (g00[i] * g11[i] + g01[i] * g10[i]) * cov01 + g01[i] * g11[i] * cov11 + s01[i],
                g10[i] * g10[i] * cov00 + 2.0 * g10[i] * g11[i] * cov01 + g11[i] * g11[i] * cov11 + s11[i],
            )
    return err, var


def _fit_ouomega_profile(z: np.ndarray, t: np.ndarray, start_tau: float, start_omega: float, method: str = "pHREML", isotropic: bool = False, design: np.ndarray | None = None, error_var: np.ndarray | None = None) -> tuple[float, float, np.ndarray, np.ndarray, float] | None:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.all(np.isfinite(z), axis=1) & np.isfinite(t)
    z = z[ok]
    t = t[ok]
    if design is not None:
        D = np.asarray(design, dtype=float)
        design = D[ok] if D.shape[0] == ok.shape[0] else None
    if error_var is not None:
        ev = np.asarray(error_var, dtype=float).reshape(-1)
        error_var = ev[ok] if ev.size == ok.size else None
    if z.shape[0] < 4:
        return None
    dt = np.diff(t)
    pos_dt = dt[np.isfinite(dt) & (dt > 0)]
    if pos_dt.size == 0:
        return None
    n, dim = z.shape
    span = max(float(t[-1] - t[0]), float(np.max(pos_dt)))
    min_dt = float(np.min(pos_dt))

    def finish_for_par(tau: float, omega: float, reml: bool = False):
        if design is None:
            B, q = _innovations_ouomega_scalar(np.ones(n, dtype=float), t, tau, omega, error_var=error_var)
            B = B.reshape(-1, 1)
        else:
            cols = []
            q = None
            for k in range(design.shape[1]):
                b, qk = _innovations_ouomega_scalar(design[:, k], t, tau, omega, error_var=error_var)
                cols.append(b)
                if q is None:
                    q = qk
            B = np.column_stack(cols) if cols else np.zeros((n, 0), dtype=float)
            if q is None:
                _, q = _innovations_ouomega_scalar(np.ones(n, dtype=float), t, tau, omega, error_var=error_var)
        q = np.maximum(np.asarray(q, dtype=float), np.finfo(float).tiny)
        residuals = []
        for j in range(dim):
            e, _ = _innovations_ouomega_scalar(z[:, j], t, tau, omega, error_var=error_var)
            residuals.append(e)
        r = np.column_stack(residuals)
        prof = _profile_mean_cov(r, B, q, reml=reml, isotropic=isotropic)
        if prof is None:
            return None
        value, beta, sigma, _ = prof
        return value, beta, sigma

    tau_lo = max(min_dt / 100.0, np.finfo(float).tiny)
    tau_hi = max(10.0 * span, min_dt * 10.0)
    omega_lo = max(1.0 / max(100.0 * span, min_dt), np.finfo(float).tiny)
    omega_hi = math.pi / max(min_dt, np.finfo(float).tiny)

    def objective(par: np.ndarray, reml: bool = False) -> float:
        tau = float(np.exp(par[0]))
        omega = float(np.exp(par[1]))
        if tau <= tau_lo or tau >= tau_hi or omega <= omega_lo or omega >= omega_hi:
            return 1e100
        out = finish_for_par(tau, omega, reml=reml)
        return 1e100 if out is None else float(out[0])

    starts = [
        [max(float(start_tau), min_dt), max(float(start_omega), omega_lo * 10.0)],
        [max(span / 4.0, min_dt), max(2.0 * math.pi / max(span / 2.0, min_dt), omega_lo * 10.0)],
        [max(span / 2.0, min_dt), max(2.0 * math.pi / max(span, min_dt), omega_lo * 10.0)],
    ]
    bounds = [(math.log(tau_lo), math.log(tau_hi)), (math.log(omega_lo), math.log(omega_hi))]
    best = None
    for st in starts:
        st[0] = min(max(st[0], tau_lo * 1.01), tau_hi / 1.01)
        st[1] = min(max(st[1], omega_lo * 1.01), omega_hi / 1.01)
        opt = minimize(
            lambda p: objective(p, reml=False),
            np.log(st),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 140, "ftol": 1e-10, "gtol": 1e-6, "maxls": 30},
        )
        if np.isfinite(opt.fun) and opt.fun < 1e99 and (best is None or opt.fun < best.fun):
            best = opt
    if best is None:
        return None
    par = np.asarray(best.x, dtype=float)
    use_reml = str(method) in ("pREML", "pHREML", "HREML", "REML")
    if str(method) == "REML":
        opt = minimize(lambda p: objective(p, reml=True), par, method="L-BFGS-B", bounds=bounds, options={"maxiter": 140, "ftol": 1e-10})
        if np.isfinite(opt.fun) and opt.fun < 1e99:
            par = np.asarray(opt.x, dtype=float)
    tau = float(np.exp(par[0]))
    omega = float(np.exp(par[1]))
    final = finish_for_par(tau, omega, reml=use_reml)
    if final is None:
        return None
    fit_value, mu, sigma = final
    return tau, omega, _pack_mu(mu), sigma, float(fit_value)


def _finite_hessian(fn, par: np.ndarray, rel_step: float = 1e-3) -> np.ndarray:
    p = np.asarray(par, dtype=float)
    step = np.maximum(np.abs(p) * rel_step, 1.0)
    hess = np.zeros((p.size, p.size), dtype=float)
    f0 = fn(p)
    for i in range(p.size):
        dpi = np.zeros_like(p)
        dpi[i] = step[i]
        fp = fn(p + dpi)
        fm = fn(p - dpi)
        hess[i, i] = (fp - 2.0 * f0 + fm) / (step[i] * step[i])
        for j in range(i + 1, p.size):
            dpj = np.zeros_like(p)
            dpj[j] = step[j]
            fpp = fn(p + dpi + dpj)
            fpm = fn(p + dpi - dpj)
            fmp = fn(p - dpi + dpj)
            fmm = fn(p - dpi - dpj)
            hess[i, j] = hess[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * step[i] * step[j])
    return hess


def _dof_area_from_hessian(hess: np.ndarray, par: np.ndarray, sigma_start: int) -> float | None:
    h = np.asarray(hess, dtype=float)
    if not np.all(np.isfinite(h)):
        return None
    try:
        cov = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(h)
    if not np.all(np.isfinite(cov)):
        return None
    p = np.asarray(par, dtype=float)
    s11, s22, s12 = p[sigma_start], p[sigma_start + 1], p[sigma_start + 2]
    det = s11 * s22 - s12 * s12
    if det <= 0 or not np.isfinite(det):
        return None
    area = math.sqrt(det)
    grad = np.zeros_like(p)
    grad[sigma_start] = 0.5 * s22 / area
    grad[sigma_start + 1] = 0.5 * s11 / area
    grad[sigma_start + 2] = -s12 / area
    var = float(grad @ cov @ grad)
    if not np.isfinite(var) or var <= 0:
        return None
    dof = float(area * area / var)
    return dof if np.isfinite(dof) and dof > 0 else None


def _dof_area_isotropic_from_hessian(hess: np.ndarray, par: np.ndarray, sigma_index: int) -> float | None:
    h = np.asarray(hess, dtype=float)
    if not np.all(np.isfinite(h)):
        return None
    try:
        cov = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(h)
    if not np.all(np.isfinite(cov)):
        return None
    area = float(np.asarray(par, dtype=float)[sigma_index])
    if not np.isfinite(area) or area <= 0:
        return None
    grad = np.zeros_like(np.asarray(par, dtype=float))
    grad[sigma_index] = 1.0
    var = float(grad @ cov @ grad)
    if not np.isfinite(var) or var <= 0:
        return None
    dof = float(area * area / var)
    return dof if np.isfinite(dof) and dof > 0 else None


def _fit_iid_profile(z: np.ndarray, method: str = "pHREML", isotropic: bool = False, design: np.ndarray | None = None, error_var: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    z = np.asarray(z, dtype=float)
    ok = np.all(np.isfinite(z), axis=1)
    z = z[ok]
    if design is not None:
        D = np.asarray(design, dtype=float)
        design = D[ok] if D.shape[0] == ok.shape[0] else None
    if error_var is not None:
        ev = np.asarray(error_var, dtype=float).reshape(-1)
        error_var = ev[ok] if ev.size == ok.size else None
    if z.shape[0] < 2:
        return None
    n, dim = z.shape
    reml = str(method) in ("pREML", "pHREML", "HREML", "REML")
    B = np.ones((n, 1), dtype=float) if design is None else design
    if error_var is not None and np.any(np.asarray(error_var, dtype=float) > 0):
        base = _fit_iid_profile(z, method=method, isotropic=isotropic, design=B, error_var=None)
        s0 = base[1] if base is not None else np.cov(z.T)
        if np.asarray(s0).shape != (dim, dim) or not np.all(np.isfinite(s0)):
            s0 = np.eye(dim, dtype=float) * max(float(np.nanvar(z)), np.finfo(float).tiny)
        s0 = np.asarray(s0, dtype=float)
        eye = np.eye(dim, dtype=float)

        def unpack(par: np.ndarray) -> np.ndarray:
            if isotropic:
                return eye * float(np.exp(par[0]))
            if dim == 2:
                L = np.array([[math.exp(float(par[0])), 0.0], [float(par[1]), math.exp(float(par[2]))]], dtype=float)
                return L @ L.T
            return eye * float(np.exp(par[0]))

        def objective(par: np.ndarray) -> float:
            sigma = unpack(np.asarray(par, dtype=float))
            covs = np.asarray([sigma + float(ei) * eye for ei in error_var], dtype=float)
            out = _gls_mean_value(z, B, covs, reml=reml)
            return 1e100 if out is None else float(out[0])

        if isotropic:
            p0 = np.array([math.log(max(float(np.trace(s0)) / max(dim, 1), np.finfo(float).tiny))], dtype=float)
            bounds = [(math.log(np.finfo(float).tiny), None)]
        elif dim == 2:
            try:
                L0 = np.linalg.cholesky(s0 + eye * np.finfo(float).eps)
            except np.linalg.LinAlgError:
                L0 = np.linalg.cholesky(eye * max(float(np.trace(s0)) / max(dim, 1), np.finfo(float).tiny))
            p0 = np.array([math.log(max(L0[0, 0], np.finfo(float).tiny)), float(L0[1, 0]), math.log(max(L0[1, 1], np.finfo(float).tiny))], dtype=float)
            bounds = [(math.log(np.finfo(float).tiny), None), (None, None), (math.log(np.finfo(float).tiny), None)]
        else:
            p0 = np.array([math.log(max(float(s0[0, 0]), np.finfo(float).tiny))], dtype=float)
            bounds = [(math.log(np.finfo(float).tiny), None)]
        opt = minimize(objective, p0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400, "ftol": 1e-9, "gtol": 1e-6})
        if not np.isfinite(opt.fun) or opt.fun >= 1e99:
            opt = minimize(objective, p0, method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-6})
        if not np.isfinite(opt.fun) or opt.fun >= 1e99:
            return None
        sigma = unpack(np.asarray(opt.x, dtype=float))
        covs = np.asarray([sigma + float(ei) * eye for ei in error_var], dtype=float)
        final = _gls_mean_value(z, B, covs, reml=reml)
        if final is None:
            return None
        value, beta = final
        rank = int(B.shape[1])
        dof_area = float(max(n - rank if reml else n, 1))
        return _pack_mu(beta), sigma, float(value), dof_area
    prof = _profile_mean_cov(z, B, np.ones(n, dtype=float), reml=reml, isotropic=isotropic)
    if prof is None:
        return None
    value, beta, sigma, rank = prof
    dof_area = float(max(n - rank if reml else n, 1))
    return _pack_mu(beta), sigma, value, dof_area


def _fit_bm_profile(z: np.ndarray, t: np.ndarray, method: str = "pHREML", isotropic: bool = False, error_var: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float] | None:
    del method
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.all(np.isfinite(z), axis=1) & np.isfinite(t)
    z = z[ok]
    t = t[ok]
    if error_var is not None:
        ev = np.asarray(error_var, dtype=float).reshape(-1)
        error_var = ev[ok] if ev.size == ok.size else None
    if z.shape[0] < 3:
        return None
    dt = np.diff(t)
    dz = np.diff(z, axis=0)
    keep = np.isfinite(dt) & (dt > 0) & np.all(np.isfinite(dz), axis=1)
    dt = dt[keep]
    dz = dz[keep]
    if error_var is not None:
        e_inc = (error_var[1:] + error_var[:-1])[keep]
        e_inc = np.maximum(np.nan_to_num(e_inc, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    else:
        e_inc = None
    if dt.size < 2:
        return None
    n, dim = dz.shape
    if e_inc is not None and np.any(e_inc > 0):
        def nll_from_diffusion(diffusion: np.ndarray) -> float:
            sign_d, _ = np.linalg.slogdet(diffusion)
            if sign_d <= 0 or not np.all(np.isfinite(diffusion)):
                return 1e100
            val = 0.0
            eye = np.eye(dim, dtype=float)
            for dti, dzi, ei in zip(dt, dz, e_inc):
                cov = 2.0 * float(dti) * diffusion + float(ei) * eye
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0 or not np.isfinite(logdet):
                    return 1e100
                try:
                    q = float(dzi @ np.linalg.inv(cov) @ dzi)
                except np.linalg.LinAlgError:
                    return 1e100
                val += logdet + q
            return float(val)

        scaled = dz / np.sqrt(dt[:, None])
        A0 = scaled.T @ scaled / max(2.0 * n, 1.0)
        A0 = np.asarray(A0, dtype=float)
        if isotropic:
            start = max(float(np.trace(A0) / max(dim, 1)), np.finfo(float).tiny)
            lo = math.log(max(start * 1e-6, np.finfo(float).tiny))
            hi = math.log(max(start * 1e6, np.finfo(float).tiny * 10.0))
            opt = minimize_scalar(lambda p: nll_from_diffusion(np.eye(dim, dtype=float) * float(np.exp(p))), bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
            if not opt.success or not np.isfinite(opt.fun) or opt.fun >= 1e99:
                return None
            diffusion = np.eye(dim, dtype=float) * float(np.exp(opt.x))
            value = float(opt.fun)
        else:
            try:
                L0 = np.linalg.cholesky(A0 + np.eye(dim) * np.finfo(float).eps)
            except np.linalg.LinAlgError:
                L0 = np.linalg.cholesky(np.eye(dim) * max(float(np.trace(A0)) / max(dim, 1), np.finfo(float).tiny))

            def unpack(p: np.ndarray) -> np.ndarray:
                if dim == 2:
                    L = np.array([[math.exp(float(p[0])), 0.0], [float(p[1]), math.exp(float(p[2]))]], dtype=float)
                    return L @ L.T
                return np.eye(dim, dtype=float) * math.exp(float(p[0]))

            if dim == 2:
                p0 = np.array([math.log(max(L0[0, 0], np.finfo(float).tiny)), float(L0[1, 0]), math.log(max(L0[1, 1], np.finfo(float).tiny))], dtype=float)
            else:
                p0 = np.array([math.log(max(float(A0[0, 0]), np.finfo(float).tiny))], dtype=float)
            opt = minimize(lambda p: nll_from_diffusion(unpack(p)), p0, method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-6})
            if not opt.success or not np.isfinite(opt.fun) or opt.fun >= 1e99:
                opt = minimize(lambda p: nll_from_diffusion(unpack(p)), p0, method="BFGS", options={"maxiter": 500, "gtol": 1e-5})
            if not np.isfinite(opt.fun) or opt.fun >= 1e99:
                return None
            diffusion = unpack(np.asarray(opt.x, dtype=float))
            value = float(opt.fun)
        mu = z[0].astype(float)
        return mu, diffusion, value

    scaled = dz / np.sqrt(dt[:, None])
    A = scaled.T @ scaled
    if isotropic:
        d2 = float(np.trace(A) / max(2.0 * n * dim, 1.0))
        if not np.isfinite(d2) or d2 <= 0:
            return None
        diffusion = np.eye(dim, dtype=float) * d2
    else:
        diffusion = A / max(2.0 * n, 1.0)
    sign, logdet = np.linalg.slogdet(2.0 * diffusion)
    if sign <= 0 or not np.isfinite(logdet):
        return None
    inv_cov = np.linalg.inv(2.0 * diffusion)
    quad = float(np.sum(np.einsum("ij,jk,ik->i", dz, inv_cov, dz) / dt))
    value = float(n * logdet + dim * np.sum(np.log(dt)) + quad)
    mu = z[0].astype(float)
    return mu, diffusion, value


def _iou_components(dt, tau: float):
    dt = np.asarray(dt, dtype=float)
    tau = max(float(tau), np.finfo(float).tiny)
    phi = np.exp(-dt / tau)
    f01 = tau * (1.0 - phi)
    q11 = np.maximum(2.0 * tau * dt - tau * tau * (3.0 - 4.0 * phi + phi * phi), np.finfo(float).tiny)
    q12 = tau * (1.0 - phi) ** 2
    q22 = np.maximum(1.0 - phi * phi, np.finfo(float).tiny)
    return phi, f01, q11, q12, q22


def _innovations_iou_scalar(x: np.ndarray, t: np.ndarray, tau: float, error_var: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    n = x.size
    dt = np.diff(t)
    phi, f01, q11, q12, q22 = _iou_components(dt, tau)
    if error_var is None:
        errv = np.zeros(n, dtype=float)
    else:
        errv = np.asarray(error_var, dtype=float).reshape(-1)
        if errv.size != n:
            errv = np.zeros(n, dtype=float)
        errv = np.maximum(np.nan_to_num(errv, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    state0 = float(x[0])
    state1 = 0.0
    cov00 = max(float(errv[0]), 0.0)
    cov01 = 0.0
    cov11 = 1.0
    err = np.empty(n, dtype=float)
    var = np.empty(n, dtype=float)
    err[0] = 0.0
    var[0] = np.inf
    for i in range(1, n):
        state0, state1 = state0 + f01[i - 1] * state1, phi[i - 1] * state1
        cov00, cov01, cov11 = (
            cov00 + 2.0 * f01[i - 1] * cov01 + f01[i - 1] * f01[i - 1] * cov11 + q11[i - 1],
            phi[i - 1] * (cov01 + f01[i - 1] * cov11) + q12[i - 1],
            phi[i - 1] * phi[i - 1] * cov11 + q22[i - 1],
        )
        qi = max(float(cov00 + errv[i]), np.finfo(float).tiny)
        ei = float(x[i] - state0)
        gain0 = cov00 / qi
        gain1 = cov01 / qi
        state0 = state0 + gain0 * ei
        state1 = state1 + gain1 * ei
        cov00_old = cov00
        cov01_old = cov01
        cov11_old = cov11
        cov00 = cov00_old - gain0 * cov00_old
        cov01 = cov01_old - gain0 * cov01_old
        cov11 = cov11_old - gain1 * cov01_old
        err[i] = ei
        var[i] = qi
    return err[1:], var[1:]


def _fit_iou_profile(z: np.ndarray, t: np.ndarray, start_tau: float, method: str = "pHREML", isotropic: bool = False, error_var: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray, float] | None:
    del method
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.all(np.isfinite(z), axis=1) & np.isfinite(t)
    z = z[ok]
    t = t[ok]
    if error_var is not None:
        ev = np.asarray(error_var, dtype=float).reshape(-1)
        error_var = ev[ok] if ev.size == ok.size else None
    if z.shape[0] < 4:
        return None
    dt = np.diff(t)
    pos_dt = dt[np.isfinite(dt) & (dt > 0)]
    if pos_dt.size == 0:
        return None
    n, dim = z.shape
    min_dt = float(np.min(pos_dt))
    span = max(float(t[-1] - t[0]), min_dt)
    lo = max(min_dt / 100.0, np.finfo(float).tiny)
    hi = max(100.0 * span, min_dt * 10.0, float(start_tau) * 100.0)

    def finish_for_tau(tau: float):
        residuals = []
        q_ref = None
        for j in range(dim):
            e, q = _innovations_iou_scalar(z[:, j], t, tau, error_var=error_var)
            residuals.append(e)
            if q_ref is None:
                q_ref = q
        if q_ref is None:
            return None
        q_ref = np.maximum(q_ref, np.finfo(float).tiny)
        r = np.column_stack(residuals)
        A = (r.T / q_ref) @ r
        dof = max(n - 1, 1)
        if isotropic:
            s2 = float(np.trace(A) / max(dof * dim, 1))
            if not np.isfinite(s2) or s2 <= 0:
                return None
            sigma = np.eye(dim, dtype=float) * s2
        else:
            sigma = A / dof
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0 or not np.isfinite(logdet):
            return None
        value = dof * logdet + dim * float(np.sum(np.log(q_ref)))
        return value, z[0].astype(float), sigma

    def objective(log_tau):
        tau = float(np.exp(log_tau))
        if tau <= lo or tau >= hi:
            return 1e100
        out = finish_for_tau(tau)
        return 1e100 if out is None else float(out[0])

    starts = [float(start_tau), span / 4.0, span / 2.0]
    best = None
    bounds = (math.log(lo), math.log(hi))
    for st in starts:
        x0 = min(max(st, lo * 1.01), hi / 1.01)
        opt = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-10})
        if opt.success and np.isfinite(opt.fun) and opt.fun < 1e99 and (best is None or opt.fun < best.fun):
            best = opt
    if best is None:
        return None
    tau_hat = float(np.exp(best.x))
    final = finish_for_tau(tau_hat)
    if final is None:
        return None
    fit_value, mu, sigma = final
    return tau_hat, mu, sigma, float(fit_value)


def _estimate_dof_area_iid(z: np.ndarray, sigma: np.ndarray, method: str = "pHREML", isotropic: bool = False) -> float | None:
    z = np.asarray(z, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    ok = np.all(np.isfinite(z), axis=1)
    z = z[ok]
    if z.shape[0] < 3:
        return None
    n, dim = z.shape
    reml = str(method) in ("pREML", "pHREML", "HREML", "REML")

    if isotropic:
        def nll_iso(p: np.ndarray) -> float:
            s2 = float(p[0])
            if s2 <= 0 or not np.isfinite(s2):
                return 1e100
            mu = np.nanmean(z, axis=0)
            r = z - mu
            A = r.T @ r
            dof = n - 1 if reml else n
            value = 0.5 * (dof * dim * math.log(s2) + float(np.trace(A)) / s2)
            if reml:
                value += 0.5 * dim * math.log(max(n, np.finfo(float).tiny))
            return float(value)

        par_iso = np.array([float(np.mean(np.diag(sigma)))], dtype=float)
        return _dof_area_isotropic_from_hessian(_finite_hessian(nll_iso, par_iso), par_iso, sigma_index=0)

    def nll(p: np.ndarray) -> float:
        s11, s22, s12 = [float(v) for v in p]
        s = np.array([[s11, s12], [s12, s22]], dtype=float)
        sign, logdet = np.linalg.slogdet(s)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e100
        mu = np.nanmean(z, axis=0)
        r = z - mu
        A = r.T @ r
        try:
            inv_s = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return 1e100
        dof = n - 1 if reml else n
        value = 0.5 * (dof * logdet + float(np.trace(inv_s @ A)))
        if reml:
            value += 0.5 * dim * math.log(max(n, np.finfo(float).tiny))
        return float(value)

    par = np.array([float(sigma[0, 0]), float(sigma[1, 1]), float(sigma[0, 1])], dtype=float)
    return _dof_area_from_hessian(_finite_hessian(nll, par), par, sigma_start=0)


def _estimate_dof_area_ou(z: np.ndarray, t: np.ndarray, tau: float, sigma: np.ndarray, method: str = "pHREML", isotropic: bool = False) -> float | None:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    ok = np.all(np.isfinite(z), axis=1) & np.isfinite(t)
    z = z[ok]
    t = t[ok]
    if z.shape[0] < 4:
        return None
    dt = np.diff(t)
    n, dim = z.shape
    reml = str(method) in ("pREML", "pHREML", "HREML", "REML")

    def nll_iso(p: np.ndarray) -> float:
        tau_i, s2 = [float(v) for v in p]
        if tau_i <= 0 or s2 <= 0 or not np.isfinite(s2):
            return 1e100
        phi = np.exp(-dt / tau_i)
        v = np.r_[1.0, 1.0 - phi * phi]
        v = np.maximum(v, np.finfo(float).tiny)
        y = np.empty_like(z)
        c = np.empty(n, dtype=float)
        y[0] = z[0]
        c[0] = 1.0
        y[1:] = z[1:] - phi[:, None] * z[:-1]
        c[1:] = 1.0 - phi
        w = 1.0 / v
        W = float(np.sum(w * c * c))
        if W <= 0 or not np.isfinite(W):
            return 1e100
        mu = np.sum(w[:, None] * c[:, None] * y, axis=0) / W
        r = y - c[:, None] * mu
        A = (r.T * w) @ r
        dof = n - 1 if reml else n
        value = 0.5 * (dof * dim * math.log(s2) + dim * float(np.sum(np.log(v))) + float(np.trace(A)) / s2)
        if reml:
            value += 0.5 * dim * math.log(max(W, np.finfo(float).tiny))
        return float(value)

    if isotropic:
        par_iso = np.array([float(tau), float(np.mean(np.diag(sigma)))], dtype=float)
        return _dof_area_isotropic_from_hessian(_finite_hessian(nll_iso, par_iso), par_iso, sigma_index=1)

    def nll(p: np.ndarray) -> float:
        tau_i, s11, s22, s12 = [float(v) for v in p]
        if tau_i <= 0:
            return 1e100
        s = np.array([[s11, s12], [s12, s22]], dtype=float)
        sign, logdet = np.linalg.slogdet(s)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e100
        phi = np.exp(-dt / tau_i)
        v = np.r_[1.0, 1.0 - phi * phi]
        v = np.maximum(v, np.finfo(float).tiny)
        y = np.empty_like(z)
        c = np.empty(n, dtype=float)
        y[0] = z[0]
        c[0] = 1.0
        y[1:] = z[1:] - phi[:, None] * z[:-1]
        c[1:] = 1.0 - phi
        w = 1.0 / v
        W = float(np.sum(w * c * c))
        if W <= 0 or not np.isfinite(W):
            return 1e100
        mu = np.sum(w[:, None] * c[:, None] * y, axis=0) / W
        r = y - c[:, None] * mu
        A = (r.T * w) @ r
        try:
            inv_s = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return 1e100
        dof = n - 1 if reml else n
        value = 0.5 * (dof * logdet + dim * float(np.sum(np.log(v))) + float(np.trace(inv_s @ A)))
        if reml:
            value += 0.5 * dim * math.log(max(W, np.finfo(float).tiny))
        return float(value)

    par = np.array([float(tau), float(sigma[0, 0]), float(sigma[1, 1]), float(sigma[0, 1])], dtype=float)
    return _dof_area_from_hessian(_finite_hessian(nll, par), par, sigma_start=1)


def _estimate_dof_area_ouf(z: np.ndarray, t: np.ndarray, tau: list[float], sigma: np.ndarray, method: str = "pHREML", isotropic: bool = False) -> float | None:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    ok = np.all(np.isfinite(z), axis=1) & np.isfinite(t)
    z = z[ok]
    t = t[ok]
    if z.shape[0] < 5:
        return None
    n, dim = z.shape
    min_dt = float(np.min(np.diff(t)[np.diff(t) > 0]))
    span = max(float(t[-1] - t[0]), min_dt)
    reml = str(method) in ("pREML", "pHREML", "HREML", "REML")

    if isotropic:
        def nll_iso(p: np.ndarray) -> float:
            tau1, tau2, s2 = [float(v) for v in p]
            if tau2 <= 0 or tau1 <= tau2 or tau1 > 10.0 * span or tau2 < min_dt / 100.0 or s2 <= 0:
                return 1e100
            basis, q = _innovations_ouf_scalar(np.ones(n, dtype=float), t, [tau1, tau2])
            q = np.maximum(q, np.finfo(float).tiny)
            W = float(np.sum(basis * basis / q))
            if W <= 0 or not np.isfinite(W):
                return 1e100
            residuals = []
            for j in range(dim):
                e, _ = _innovations_ouf_scalar(z[:, j], t, [tau1, tau2])
                mu_j = float(np.sum(basis * e / q) / W)
                residuals.append(e - basis * mu_j)
            r = np.column_stack(residuals)
            A = (r.T / q) @ r
            dof = n - 1 if reml else n
            value = 0.5 * (dof * dim * math.log(s2) + dim * float(np.sum(np.log(q))) + float(np.trace(A)) / s2)
            if reml:
                value += 0.5 * dim * math.log(max(W, np.finfo(float).tiny))
            return float(value)

        par_iso = np.array([float(tau[0]), float(tau[1]), float(np.mean(np.diag(sigma)))], dtype=float)
        return _dof_area_isotropic_from_hessian(_finite_hessian(nll_iso, par_iso), par_iso, sigma_index=2)

    def nll(p: np.ndarray) -> float:
        tau1, tau2, s11, s22, s12 = [float(v) for v in p]
        if tau2 <= 0 or tau1 <= tau2 or tau1 > 10.0 * span or tau2 < min_dt / 100.0:
            return 1e100
        s = np.array([[s11, s12], [s12, s22]], dtype=float)
        sign, logdet = np.linalg.slogdet(s)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e100
        basis, q = _innovations_ouf_scalar(np.ones(n, dtype=float), t, [tau1, tau2])
        q = np.maximum(q, np.finfo(float).tiny)
        W = float(np.sum(basis * basis / q))
        if W <= 0 or not np.isfinite(W):
            return 1e100
        residuals = []
        for j in range(dim):
            e, _ = _innovations_ouf_scalar(z[:, j], t, [tau1, tau2])
            mu_j = float(np.sum(basis * e / q) / W)
            residuals.append(e - basis * mu_j)
        r = np.column_stack(residuals)
        A = (r.T / q) @ r
        try:
            inv_s = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return 1e100
        dof = n - 1 if reml else n
        value = 0.5 * (dof * logdet + dim * float(np.sum(np.log(q))) + float(np.trace(inv_s @ A)))
        if reml:
            value += 0.5 * dim * math.log(max(W, np.finfo(float).tiny))
        return float(value)

    par = np.array([float(tau[0]), float(tau[1]), float(sigma[0, 0]), float(sigma[1, 1]), float(sigma[0, 1])], dtype=float)
    return _dof_area_from_hessian(_finite_hessian(nll, par), par, sigma_start=2)


def _estimate_dof_area_ouf_tied(z: np.ndarray, t: np.ndarray, tau: float, sigma: np.ndarray, method: str = "pHREML", isotropic: bool = False) -> float | None:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    ok = np.all(np.isfinite(z), axis=1) & np.isfinite(t)
    z = z[ok]
    t = t[ok]
    if z.shape[0] < 5:
        return None
    n, dim = z.shape
    min_dt = float(np.min(np.diff(t)[np.diff(t) > 0]))
    span = max(float(t[-1] - t[0]), min_dt)
    reml = str(method) in ("pREML", "pHREML", "HREML", "REML")

    if isotropic:
        def nll_iso(p: np.ndarray) -> float:
            tau_i, s2 = [float(v) for v in p]
            if tau_i <= min_dt / 100.0 or tau_i > 10.0 * span or s2 <= 0:
                return 1e100
            basis, q = _innovations_ouf_scalar(np.ones(n, dtype=float), t, [tau_i, tau_i])
            q = np.maximum(q, np.finfo(float).tiny)
            W = float(np.sum(basis * basis / q))
            if W <= 0 or not np.isfinite(W):
                return 1e100
            residuals = []
            for j in range(dim):
                e, _ = _innovations_ouf_scalar(z[:, j], t, [tau_i, tau_i])
                mu_j = float(np.sum(basis * e / q) / W)
                residuals.append(e - basis * mu_j)
            r = np.column_stack(residuals)
            A = (r.T / q) @ r
            dof = n - 1 if reml else n
            value = 0.5 * (dof * dim * math.log(s2) + dim * float(np.sum(np.log(q))) + float(np.trace(A)) / s2)
            if reml:
                value += 0.5 * dim * math.log(max(W, np.finfo(float).tiny))
            return float(value)

        par_iso = np.array([float(tau), float(np.mean(np.diag(sigma)))], dtype=float)
        return _dof_area_isotropic_from_hessian(_finite_hessian(nll_iso, par_iso), par_iso, sigma_index=1)

    def nll(p: np.ndarray) -> float:
        tau_i, s11, s22, s12 = [float(v) for v in p]
        if tau_i <= min_dt / 100.0 or tau_i > 10.0 * span:
            return 1e100
        s = np.array([[s11, s12], [s12, s22]], dtype=float)
        sign, logdet = np.linalg.slogdet(s)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e100
        basis, q = _innovations_ouf_scalar(np.ones(n, dtype=float), t, [tau_i, tau_i])
        q = np.maximum(q, np.finfo(float).tiny)
        W = float(np.sum(basis * basis / q))
        if W <= 0 or not np.isfinite(W):
            return 1e100
        residuals = []
        for j in range(dim):
            e, _ = _innovations_ouf_scalar(z[:, j], t, [tau_i, tau_i])
            mu_j = float(np.sum(basis * e / q) / W)
            residuals.append(e - basis * mu_j)
        r = np.column_stack(residuals)
        A = (r.T / q) @ r
        try:
            inv_s = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return 1e100
        dof = n - 1 if reml else n
        value = 0.5 * (dof * logdet + dim * float(np.sum(np.log(q))) + float(np.trace(inv_s @ A)))
        if reml:
            value += 0.5 * dim * math.log(max(W, np.finfo(float).tiny))
        return float(value)

    par = np.array([float(tau), float(sigma[0, 0]), float(sigma[1, 1]), float(sigma[0, 1])], dtype=float)
    return _dof_area_from_hessian(_finite_hessian(nll, par), par, sigma_start=1)


def _estimate_dof_area_ouomega(z: np.ndarray, t: np.ndarray, tau: float, omega: float, sigma: np.ndarray, method: str = "pHREML", isotropic: bool = False) -> float | None:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    ok = np.all(np.isfinite(z), axis=1) & np.isfinite(t)
    z = z[ok]
    t = t[ok]
    if z.shape[0] < 5:
        return None
    n, dim = z.shape
    dt = np.diff(t)
    pos_dt = dt[np.isfinite(dt) & (dt > 0)]
    if pos_dt.size == 0:
        return None
    min_dt = float(np.min(pos_dt))
    span = max(float(t[-1] - t[0]), min_dt)
    reml = str(method) in ("pREML", "pHREML", "HREML", "REML")

    if isotropic:
        def nll_iso(p: np.ndarray) -> float:
            tau_i, omega_i, s2 = [float(v) for v in p]
            if tau_i <= min_dt / 100.0 or tau_i > 10.0 * span or omega_i <= 0 or s2 <= 0:
                return 1e100
            basis, q = _innovations_ouomega_scalar(np.ones(n, dtype=float), t, tau_i, omega_i)
            q = np.maximum(q, np.finfo(float).tiny)
            W = float(np.sum(basis * basis / q))
            if W <= 0 or not np.isfinite(W):
                return 1e100
            residuals = []
            for j in range(dim):
                e, _ = _innovations_ouomega_scalar(z[:, j], t, tau_i, omega_i)
                mu_j = float(np.sum(basis * e / q) / W)
                residuals.append(e - basis * mu_j)
            r = np.column_stack(residuals)
            A = (r.T / q) @ r
            dof = n - 1 if reml else n
            value = 0.5 * (dof * dim * math.log(s2) + dim * float(np.sum(np.log(q))) + float(np.trace(A)) / s2)
            if reml:
                value += 0.5 * dim * math.log(max(W, np.finfo(float).tiny))
            return float(value)

        par_iso = np.array([float(tau), float(omega), float(np.mean(np.diag(sigma)))], dtype=float)
        return _dof_area_isotropic_from_hessian(_finite_hessian(nll_iso, par_iso), par_iso, sigma_index=2)

    def nll(p: np.ndarray) -> float:
        tau_i, omega_i, s11, s22, s12 = [float(v) for v in p]
        if tau_i <= min_dt / 100.0 or tau_i > 10.0 * span or omega_i <= 0:
            return 1e100
        s = np.array([[s11, s12], [s12, s22]], dtype=float)
        sign, logdet = np.linalg.slogdet(s)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e100
        basis, q = _innovations_ouomega_scalar(np.ones(n, dtype=float), t, tau_i, omega_i)
        q = np.maximum(q, np.finfo(float).tiny)
        W = float(np.sum(basis * basis / q))
        if W <= 0 or not np.isfinite(W):
            return 1e100
        residuals = []
        for j in range(dim):
            e, _ = _innovations_ouomega_scalar(z[:, j], t, tau_i, omega_i)
            mu_j = float(np.sum(basis * e / q) / W)
            residuals.append(e - basis * mu_j)
        r = np.column_stack(residuals)
        A = (r.T / q) @ r
        try:
            inv_s = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return 1e100
        dof = n - 1 if reml else n
        value = 0.5 * (dof * logdet + dim * float(np.sum(np.log(q))) + float(np.trace(inv_s @ A)))
        if reml:
            value += 0.5 * dim * math.log(max(W, np.finfo(float).tiny))
        return float(value)

    par = np.array([float(tau), float(omega), float(sigma[0, 0]), float(sigma[1, 1]), float(sigma[0, 1])], dtype=float)
    return _dof_area_from_hessian(_finite_hessian(nll, par), par, sigma_start=2)


def _canonical_model_name(tau: list[float], range_: bool, omega: float) -> str:
    if not range_:
        if len(tau) <= 1:
            return "BM"
        return "IOU"
    if len(tau) == 0:
        return "IID"
    if len(tau) == 1:
        return "OU"
    if len(tau) >= 2 and np.isclose(float(tau[0]), float(tau[1]), rtol=1e-10, atol=1e-12) and not omega:
        return "OUf"
    if omega and omega > 0:
        return "OUOmega"
    return "OUF"


def _tau_list_to_dict(tau_vals: list[float]) -> dict[str, float]:
    return {_TAU_NAMES[i]: tau_vals[i] for i in range(len(tau_vals))}


def ctmm(
    tau: list[float] | None = None,
    omega: float | bool = False,
    isotropic: bool = False,
    range: bool = True,
    circle: bool = False,
    error: float | bool = False,
    axes: tuple[str, str] | list[str] = ("x", "y"),
    **params,
) -> CTMMModel:
    """
    Python analog of ``ctmm::ctmm()`` — builds a :class:`CTMMModel` plus an internal
    ``ctmm_internal`` dict (``get.taus`` output) for Kalman / likelihood code.
    """
    extra = dict(params)
    sigma_in = extra.pop("sigma", None)
    mean = extra.pop("mean", "stationary")
    dynamics = extra.pop("dynamics", "stationary")
    link = extra.pop("link", "identity")
    timelink = extra.pop("timelink", "identity")
    COV = extra.pop("COV", None)
    COV_rownames = extra.pop("COV_rownames", None)

    tau_vals: list[float] = [] if tau is None else sorted([float(t) for t in tau], reverse=True)

    omega_active = omega is not False and bool(omega)
    omega_val = float(omega) if omega_active else 0.0

    if omega_active and len(tau_vals) >= 2:
        inv_mean = float(np.mean([1.0 / t for t in tau_vals]))
        tau_vals = [1.0 / inv_mean, 1.0 / inv_mean]

    if tau_vals and np.isposinf(tau_vals[0]):
        range = False
    elif tau_vals:
        range = True

    if not range and len(tau_vals) > 1 and tau_vals[0] == tau_vals[1]:
        raise ValueError("Ballistic motion not yet supported.")
    if not range and circle:
        raise ValueError("Inconsistent model options: range=False, circle=True.")
    if not range and len(tau_vals) == 0:
        tau_vals = [float("inf")]
    circle_val = float(circle) if circle is not False and bool(circle) else 0.0

    ax = tuple(axes)
    if len(ax) == 1:
        isotropic = True

    if sigma_in is not None:
        sig = covm_factory(sigma_in, isotropic=isotropic, axes=ax)
    elif len(ax) == 2:
        sig = covm_factory(1.0, isotropic=isotropic, axes=ax)
    else:
        sig = covm_factory([1.0], isotropic=True, axes=ax)

    tau_dict = _tau_list_to_dict(tau_vals) if tau_vals else {}

    if isinstance(error, bool):
        errors_flag = bool(error)
    else:
        errors_flag = bool(np.any(np.asarray(error, dtype=float).ravel() > 0))

    stm: dict[str, Any] = {
        "tau": tau_dict,
        "omega": omega_val,
        "isotropic": isotropic,
        "range": bool(range),
        "circle": circle_val,
        "error": error,
        "axes": list(ax),
        "sigma": sig,
        "mean": mean,
        "dynamics": dynamics,
        "link": link,
        "timelink": timelink,
        "COV": COV,
        "COV_rownames": COV_rownames,
        "errors": errors_flag,
    }
    get_taus(stm, zeroes=False, simplify=False)

    model_name = _canonical_model_name(tau_vals, bool(range), omega_val)
    p: dict[str, Any] = dict(extra)
    p.update(
        {
            "tau": tau_dict,
            "tau_list": tau_vals,
            "omega": omega_val,
            "isotropic": bool(isotropic),
            "range": bool(range),
            "circle": circle_val,
            "error": error,
            "axes": ax,
            "sigma": sig,
            "mean": mean,
            "dynamics": dynamics,
            "link": link,
            "timelink": timelink,
            "ctmm_internal": stm,
        }
    )
    return CTMMModel(model=model_name, params=p)


def ctmm_guess(variogram_obj: dict[str, Any], model: CTMMModel | None = None) -> CTMMModel:
    lags = np.asarray(variogram_obj.get("lags_s", []), dtype=float)
    gamma = np.asarray(variogram_obj.get("gamma", []), dtype=float)
    if lags.size < 2 or gamma.size < 2:
        return model if model is not None else ctmm(tau=[1.0], range=True)

    valid = np.isfinite(lags) & np.isfinite(gamma) & (lags > 0)
    lags = lags[valid]
    gamma = gamma[valid]
    if lags.size < 2:
        return model if model is not None else ctmm(tau=[1.0], range=True)

    sigma = float(np.nanmean(gamma))
    slope_like = np.divide(gamma, lags, out=np.zeros_like(gamma), where=lags > 0)
    i = int(np.argmax(slope_like))
    D = float(max(slope_like[i], 1e-12))
    tau_pos = float(max(sigma / D, 1e-9))

    curv_like = np.divide(gamma, lags * lags, out=np.zeros_like(gamma), where=lags > 0)
    v2 = float(max(2.0 * np.nanmax(curv_like), 1e-12))
    tau_vel = float(max(D / v2, 1e-9))

    base = model if model is not None else ctmm()
    bsig = base.params.get("sigma")
    if isinstance(bsig, Covm):
        s0 = float(bsig.par["major"])
    else:
        s0 = sigma
    return ctmm(
        tau=sorted([tau_pos, tau_vel], reverse=True),
        sigma=s0,
        isotropic=bool(base.params.get("isotropic", False)),
    )


def _profile_ic(value: float, n: int, k: int) -> dict[str, float]:
    v = float(value)
    kk = int(max(k, 1))
    nn = int(max(n, kk + 2))
    aic = v + 2.0 * kk
    aicc = aic + (2.0 * kk * (kk + 1.0)) / max(nn - kk - 1.0, 1.0)
    bic = v + math.log(max(nn, 1)) * kk
    return {"AIC": float(aic), "AICc": float(aicc), "BIC": float(bic), "value": v, "k": float(kk)}


def _model_error_rms(model: CTMMModel, telem: Telemetry) -> float:
    err = model.params.get("error", False)
    if isinstance(err, bool):
        if not err:
            return 0.0
        u = telem.metadata.get("UERE", {})
        if isinstance(u, dict):
            val = u.get("UERE", {}).get("horizontal") if isinstance(u.get("UERE"), dict) else None
            if val is not None and np.isfinite(float(val)):
                return float(val)
        return 1.0
    if isinstance(err, dict):
        if "class" in telem.data.columns:
            classes = telem.data["class"].astype(str)
            vals = [float(err.get(c, err.get("all", 0.0)) or 0.0) for c in classes]
            vals = [v for v in vals if np.isfinite(v) and v > 0]
            return float(np.nanmedian(vals)) if vals else 0.0
        vals = [float(v) for v in err.values() if np.isfinite(float(v)) and float(v) > 0]
        return float(np.nanmedian(vals)) if vals else 0.0
    try:
        return max(float(err), 0.0)
    except Exception:
        return 0.0


def _measurement_error_var(telem: Telemetry, model: CTMMModel, ok: np.ndarray | None = None) -> np.ndarray | None:
    rms = _model_error_rms(model, telem)
    rms_vec = None
    err = model.params.get("error", False)
    if "class" in telem.data.columns:
        classes = telem.data["class"].astype(str).to_numpy()
        if err is True:
            u = telem.metadata.get("UERE", {})
            class_map = None
            if isinstance(u, dict) and isinstance(u.get("UERE"), dict):
                class_map = u["UERE"].get("class")
            if isinstance(class_map, dict):
                rms_vec = np.asarray([float(class_map.get(c, class_map.get("all", rms)) or rms) for c in classes], dtype=float)
        elif isinstance(err, dict):
            rms_vec = np.asarray([float(err.get(c, err.get("all", rms)) or rms) for c in classes], dtype=float)
    df = telem.data
    n = len(df)
    if rms <= 0 and not any(c in df.columns for c in ("VAR.xy", "COV.x.x", "HDOP")):
        return None
    if "VAR.xy" in df.columns:
        var = pd.to_numeric(df["VAR.xy"], errors="coerce").to_numpy(dtype=float)
    elif "COV.x.x" in df.columns and "COV.y.y" in df.columns:
        vx = pd.to_numeric(df["COV.x.x"], errors="coerce").to_numpy(dtype=float)
        vy = pd.to_numeric(df["COV.y.y"], errors="coerce").to_numpy(dtype=float)
        var = (vx + vy) / 2.0
    else:
        if "HDOP" in df.columns:
            dop = pd.to_numeric(df["HDOP"], errors="coerce").to_numpy(dtype=float)
            dop = np.where(np.isfinite(dop), dop, 1.0)
        else:
            dop = np.ones(n, dtype=float)
        rms_use = rms if rms_vec is None else np.where(np.isfinite(rms_vec), rms_vec, rms)
        var = (rms_use * dop) ** 2 / 2.0
    var = np.maximum(np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    if ok is not None and ok.shape[0] == var.shape[0]:
        var = var[ok]
    return var


def _rotate_track(z: np.ndarray, t: np.ndarray, circle: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    if z.shape[1] != 2 or not circle:
        return z
    theta = -float(circle) * (t - t[0])
    c = np.cos(theta)
    s = np.sin(theta)
    out = np.empty_like(z, dtype=float)
    out[:, 0] = c * z[:, 0] - s * z[:, 1]
    out[:, 1] = s * z[:, 0] + c * z[:, 1]
    return out


def _fit_ou_circle_profile(z: np.ndarray, t: np.ndarray, start_tau: float, method: str = "pHREML", isotropic: bool = False, design: np.ndarray | None = None, error_var: np.ndarray | None = None) -> tuple[float, float, np.ndarray, np.ndarray, float] | None:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.all(np.isfinite(z), axis=1) & np.isfinite(t)
    z = z[ok]
    t = t[ok]
    if design is not None:
        D = np.asarray(design, dtype=float)
        design = D[ok] if D.shape[0] == ok.shape[0] else None
    if error_var is not None:
        ev = np.asarray(error_var, dtype=float).reshape(-1)
        error_var = ev[ok] if ev.size == ok.size else None
    if z.shape[0] < 5 or z.shape[1] != 2:
        return None
    dt = np.diff(t)
    pos_dt = dt[np.isfinite(dt) & (dt > 0)]
    if pos_dt.size == 0:
        return None
    span = max(float(t[-1] - t[0]), float(np.max(pos_dt)))
    min_dt = float(np.min(pos_dt))
    circ_hi = math.pi / max(min_dt, np.finfo(float).tiny)

    def finish(circle: float):
        zr = _rotate_track(z, t, circle)
        out = _fit_ou_profile(zr, t, method=method, isotropic=isotropic, design=design, error_var=error_var)
        if out is None:
            return None
        tau, mu, sigma, value = out
        return float(value), float(tau), mu, sigma

    def objective(par):
        circle = float(par[0])
        if not np.isfinite(circle) or abs(circle) >= circ_hi:
            return 1e100
        out = finish(circle)
        return 1e100 if out is None else float(out[0])

    starts = [0.0, 2.0 * math.pi / max(span, min_dt), -2.0 * math.pi / max(span, min_dt)]
    best = None
    for st in starts:
        opt = minimize(
            lambda p: objective(p),
            np.array([st], dtype=float),
            method="L-BFGS-B",
            bounds=[(-circ_hi, circ_hi)],
            options={"maxiter": 80, "ftol": 1e-8, "gtol": 1e-6},
        )
        if np.isfinite(opt.fun) and opt.fun < 1e99 and (best is None or opt.fun < best.fun):
            best = opt
    if best is None:
        return None
    circle = float(best.x[0])
    final = finish(circle)
    if final is None:
        return None
    fit_value, tau, mu, sigma = final
    return tau, circle, mu, sigma, float(fit_value)


def _fit_range_circle_profile(
    z: np.ndarray,
    t: np.ndarray,
    tau0: list[float],
    start_circle: float,
    method: str = "pHREML",
    isotropic: bool = False,
    omega: float = 0.0,
    tied: bool = False,
    design: np.ndarray | None = None,
    error_var: np.ndarray | None = None,
) -> tuple[str, list[float], float, float, np.ndarray, np.ndarray, float] | None:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.all(np.isfinite(z), axis=1) & np.isfinite(t)
    z = z[ok]
    t = t[ok]
    if design is not None:
        D = np.asarray(design, dtype=float)
        design = D[ok] if D.shape[0] == ok.shape[0] else None
    if error_var is not None:
        ev = np.asarray(error_var, dtype=float).reshape(-1)
        error_var = ev[ok] if ev.size == ok.size else None
    if z.shape[0] < 5 or z.shape[1] != 2:
        return None
    dt = np.diff(t)
    pos_dt = dt[np.isfinite(dt) & (dt > 0)]
    if pos_dt.size == 0:
        return None
    span = max(float(t[-1] - t[0]), float(np.max(pos_dt)))
    min_dt = float(np.min(pos_dt))
    circ_hi = math.pi / max(min_dt, np.finfo(float).tiny)
    tau_start = [float(tau0[0]), float(tau0[1])]

    def finish(circle: float):
        zr = _rotate_track(z, t, circle)
        if omega > 0:
            out = _fit_ouomega_profile(zr, t, float(np.mean(tau_start)), omega, method=method, isotropic=isotropic, design=design, error_var=error_var)
            if out is None:
                return None
            tau_hat, omega_hat, mu, sigma, value = out
            return "OUOmega-circulation", [float(tau_hat), float(tau_hat)], float(omega_hat), mu, sigma, float(value)
        if tied:
            out = _fit_ouf_tied_profile(zr, t, float(np.mean(tau_start)), method=method, isotropic=isotropic, design=design, error_var=error_var)
            if out is None:
                return None
            tau_hat, mu, sigma, value = out
            return "OUf-circulation", [float(tau_hat[0]), float(tau_hat[1])], 0.0, mu, sigma, float(value)
        out = _fit_ouf_profile(zr, t, tau_start, method=method, isotropic=isotropic, design=design, error_var=error_var)
        if out is None:
            return None
        tau_hat, mu, sigma, value = out
        return "OUF-circulation", [float(tau_hat[0]), float(tau_hat[1])], 0.0, mu, sigma, float(value)

    def objective(par):
        circle = float(par[0])
        if not np.isfinite(circle) or abs(circle) >= circ_hi:
            return 1e100
        out = finish(circle)
        return 1e100 if out is None else float(out[-1])

    base = 2.0 * math.pi / max(span, min_dt)
    starts = [float(start_circle), 0.0, base, -base]
    best = None
    for st in starts:
        st = min(max(st, -0.95 * circ_hi), 0.95 * circ_hi)
        opt = minimize(
            lambda p: objective(p),
            np.array([st], dtype=float),
            method="L-BFGS-B",
            bounds=[(-circ_hi, circ_hi)],
            options={"maxiter": 100, "ftol": 1e-8, "gtol": 1e-6},
        )
        if np.isfinite(opt.fun) and opt.fun < 1e99 and (best is None or opt.fun < best.fun):
            best = opt
    if best is None:
        return None
    circle = float(best.x[0])
    final = finish(circle)
    if final is None:
        return None
    name, tau_hat, omega_hat, mu, sigma, value = final
    return name, tau_hat, float(omega_hat), circle, mu, sigma, float(value)


def ctmm_fit(telem: Telemetry, model: CTMMModel, **kwargs):
    method = str(kwargs.pop("method", "pHREML"))
    COV = bool(kwargs.pop("COV", True))
    control = kwargs.pop("control", None)
    trace = bool(kwargs.pop("trace", False))
    del COV, control, trace, kwargs
    out = CTMMModel(model=model.model, params=dict(model.params))
    # Populate covariance scale from observed projected locations.
    xy = telem.data[[telem.x_col, telem.y_col]].to_numpy(dtype=float)
    ok = np.all(np.isfinite(xy), axis=1)
    xy = xy[ok]
    error_var = _measurement_error_var(telem, out, ok=ok)
    tsec_all = _model_times(telem, out)
    t_fit_all = tsec_all[ok] if ok.shape[0] == tsec_all.shape[0] else tsec_all
    design_fit_all = _drift_design(out, t_fit_all)
    if xy.shape[0] >= 2:
        s = np.cov(xy.T)
        if s.shape == (2, 2) and np.all(np.isfinite(s)):
            out.params["sigma_matrix"] = np.asarray(s, dtype=float)
            try:
                out.params["sigma"] = covm_factory(s, isotropic=bool(out.params.get("isotropic", False)), axes=tuple(out.params.get("axes", ("x", "y"))))
            except Exception:
                pass
    tau0 = out.params.get("tau_list", [])
    if not bool(out.params.get("range", True)):
        t_fit = t_fit_all
        isotropic = bool(out.params.get("isotropic", False))
        if len(tau0) <= 1:
            bm_fit = _fit_bm_profile(xy, t_fit, method=method, isotropic=isotropic, error_var=error_var)
            if bm_fit is not None:
                mu_hat, diffusion_hat, fit_value = bm_fit
                out.model = "BM"
                out.params["range"] = False
                out.params["tau"] = {"position": float("inf")}
                out.params["tau_list"] = [float("inf")]
                out.params["mu"] = np.asarray(mu_hat, dtype=float)
                out.params["sigma_matrix"] = np.asarray(diffusion_hat, dtype=float)
                out.params["_profile_IC"] = _profile_ic(float(fit_value), max(xy.shape[0] - 1, 1), 1 if isotropic else 3)
                out.params["DOF.area.fit"] = float("nan")
                try:
                    out.params["sigma"] = covm_factory(diffusion_hat, isotropic=isotropic, axes=tuple(out.params.get("axes", ("x", "y"))))
                except Exception:
                    pass
        else:
            start_tau = float(tau0[1]) if len(tau0) > 1 and np.isfinite(float(tau0[1])) else max(float(np.nanmedian(np.diff(t_fit))), 1.0)
            iou_fit = _fit_iou_profile(xy, t_fit, start_tau, method=method, isotropic=isotropic, error_var=error_var)
            if iou_fit is not None:
                tau_hat, mu_hat, sigma_hat, fit_value = iou_fit
                out.model = "IOU"
                out.params["range"] = False
                out.params["tau"] = {"position": float("inf"), "velocity": float(tau_hat)}
                out.params["tau_list"] = [float("inf"), float(tau_hat)]
                out.params["mu"] = np.asarray(mu_hat, dtype=float)
                out.params["sigma_matrix"] = np.asarray(sigma_hat, dtype=float)
                out.params["_profile_IC"] = _profile_ic(float(fit_value), max(xy.shape[0] - 1, 1), 2 if isotropic else 4)
                out.params["DOF.area.fit"] = float("nan")
                try:
                    out.params["sigma"] = covm_factory(sigma_hat, isotropic=isotropic, axes=tuple(out.params.get("axes", ("x", "y"))))
                except Exception:
                    pass
    elif len(tau0) == 0 and bool(out.params.get("range", True)):
        isotropic = bool(out.params.get("isotropic", False))
        iid_fit = _fit_iid_profile(xy, method=method, isotropic=isotropic, design=design_fit_all, error_var=error_var)
        if iid_fit is not None:
            mu_hat, sigma_hat, fit_value, dof_area = iid_fit
            out.params["tau"] = {}
            out.params["tau_list"] = []
            out.params["mu"] = np.asarray(mu_hat, dtype=float)
            out.params["sigma_matrix"] = np.asarray(sigma_hat, dtype=float)
            out.params["_profile_IC"] = _profile_ic(float(fit_value), xy.shape[0], 1 if isotropic else 3)
            dof_est = _estimate_dof_area_iid(xy, np.asarray(sigma_hat, dtype=float), method=method, isotropic=isotropic)
            if dof_est is not None:
                dof_area = float(dof_est)
            out.params["DOF.area.fit"] = float(dof_area)
            try:
                out.params["sigma"] = covm_factory(sigma_hat, isotropic=isotropic, axes=tuple(out.params.get("axes", ("x", "y"))))
            except Exception:
                pass
    elif len(tau0) == 1 and bool(out.params.get("range", True)):
        t_fit = t_fit_all
        isotropic = bool(out.params.get("isotropic", False))
        if bool(out.params.get("circle", False)):
            circ_fit = _fit_ou_circle_profile(xy, t_fit, float(tau0[0]), method=method, isotropic=isotropic, design=design_fit_all, error_var=error_var)
            ou_fit = None
        else:
            circ_fit = None
            ou_fit = _fit_ou_profile(xy, t_fit, method=method, isotropic=isotropic, design=design_fit_all, error_var=error_var)
        if circ_fit is not None:
            tau_hat, circle_hat, mu_hat, sigma_hat, fit_value = circ_fit
            out.model = "OU-circulation"
            out.params["tau"] = {"position": float(tau_hat)}
            out.params["tau_list"] = [float(tau_hat)]
            out.params["circle"] = float(circle_hat)
            out.params["mu"] = np.asarray(mu_hat, dtype=float)
            out.params["sigma_matrix"] = np.asarray(sigma_hat, dtype=float)
            out.params["_profile_IC"] = _profile_ic(float(fit_value), xy.shape[0], 3 if isotropic else 5)
            xy_dof = _rotate_track(xy, t_fit, float(circle_hat))
            dof_est = _estimate_dof_area_ou(xy_dof, t_fit, float(tau_hat), np.asarray(sigma_hat, dtype=float), method=method, isotropic=isotropic)
            if dof_est is not None:
                out.params["DOF.area.fit"] = float(dof_est)
            try:
                out.params["sigma"] = covm_factory(sigma_hat, isotropic=isotropic, axes=tuple(out.params.get("axes", ("x", "y"))))
            except Exception:
                pass
        if ou_fit is not None:
            tau_hat, mu_hat, sigma_hat, fit_value = ou_fit
            out.params["tau"] = {"position": float(tau_hat)}
            out.params["tau_list"] = [float(tau_hat)]
            out.params["mu"] = np.asarray(mu_hat, dtype=float)
            out.params["sigma_matrix"] = np.asarray(sigma_hat, dtype=float)
            out.params["_profile_IC"] = _profile_ic(float(fit_value), xy.shape[0], 2 if isotropic else 4)
            dof_est = _estimate_dof_area_ou(xy, t_fit, float(tau_hat), np.asarray(sigma_hat, dtype=float), method=method, isotropic=isotropic)
            if dof_est is not None:
                out.params["DOF.area.fit"] = float(dof_est)
            try:
                out.params["sigma"] = covm_factory(sigma_hat, isotropic=bool(out.params.get("isotropic", False)), axes=tuple(out.params.get("axes", ("x", "y"))))
            except Exception:
                pass
    elif len(tau0) >= 2 and bool(out.params.get("range", True)):
        t_fit = t_fit_all
        isotropic = bool(out.params.get("isotropic", False))
        omega = float(out.params.get("omega", 0.0) or 0.0)
        tied = (not omega) and (bool(out.params.get("tau_tied", False)) or np.isclose(float(tau0[0]), float(tau0[1]), rtol=1e-10, atol=1e-12))
        oscillatory = bool(omega > 0)
        range_circle_fit = None
        if bool(out.params.get("circle", False)):
            range_circle_fit = _fit_range_circle_profile(
                xy,
                t_fit,
                [float(tau0[0]), float(tau0[1])],
                float(out.params.get("circle", 0.0) or 0.0),
                method=method,
                isotropic=isotropic,
                omega=omega,
                tied=tied,
                design=design_fit_all,
                error_var=error_var,
            )
            ouomega_fit = None
            ouf_fit = None
        elif oscillatory:
            start_tau = float(np.mean([float(tau0[0]), float(tau0[1])]))
            ouomega_fit = _fit_ouomega_profile(xy, t_fit, start_tau, omega, method=method, isotropic=isotropic, design=design_fit_all, error_var=error_var)
            ouf_fit = None
        elif tied:
            ouf_fit = _fit_ouf_tied_profile(xy, t_fit, float(np.mean([float(tau0[0]), float(tau0[1])])), method=method, isotropic=isotropic, design=design_fit_all, error_var=error_var)
        else:
            ouf_fit = _fit_ouf_profile(xy, t_fit, [float(tau0[0]), float(tau0[1])], method=method, isotropic=isotropic, design=design_fit_all, error_var=error_var)
        if range_circle_fit is not None:
            name_hat, tau_hat, omega_hat, circle_hat, mu_hat, sigma_hat, fit_value = range_circle_fit
            out.model = name_hat
            out.params["tau"] = {"position": float(tau_hat[0]), "velocity": float(tau_hat[1])}
            out.params["tau_list"] = [float(tau_hat[0]), float(tau_hat[1])]
            out.params["omega"] = float(omega_hat)
            out.params["circle"] = float(circle_hat)
            out.params["mu"] = np.asarray(mu_hat, dtype=float)
            out.params["sigma_matrix"] = np.asarray(sigma_hat, dtype=float)
            fit_k = (4 if isotropic else 6) if ("OUOmega" in name_hat or "OUF" in name_hat) else (3 if isotropic else 5)
            out.params["_profile_IC"] = _profile_ic(float(fit_value), xy.shape[0], fit_k)
            xy_dof = _rotate_track(xy, t_fit, float(circle_hat))
            if "OUOmega" in name_hat:
                dof_est = _estimate_dof_area_ouomega(xy_dof, t_fit, float(tau_hat[0]), float(omega_hat), np.asarray(sigma_hat, dtype=float), method=method, isotropic=isotropic)
            elif "OUf" in name_hat:
                dof_est = _estimate_dof_area_ouf_tied(xy_dof, t_fit, float(tau_hat[0]), np.asarray(sigma_hat, dtype=float), method=method, isotropic=isotropic)
            else:
                dof_est = _estimate_dof_area_ouf(xy_dof, t_fit, [float(tau_hat[0]), float(tau_hat[1])], np.asarray(sigma_hat, dtype=float), method=method, isotropic=isotropic)
            if dof_est is not None:
                out.params["DOF.area.fit"] = float(dof_est)
            try:
                out.params["sigma"] = covm_factory(sigma_hat, isotropic=isotropic, axes=tuple(out.params.get("axes", ("x", "y"))))
            except Exception:
                pass
        if oscillatory and ouomega_fit is not None:
            tau_hat, omega_hat, mu_hat, sigma_hat, fit_value = ouomega_fit
            out.model = "OUOmega"
            out.params["tau"] = {"position": float(tau_hat), "velocity": float(tau_hat)}
            out.params["tau_list"] = [float(tau_hat), float(tau_hat)]
            out.params["omega"] = float(omega_hat)
            out.params["mu"] = np.asarray(mu_hat, dtype=float)
            out.params["sigma_matrix"] = np.asarray(sigma_hat, dtype=float)
            out.params["_profile_IC"] = _profile_ic(float(fit_value), xy.shape[0], 3 if isotropic else 5)
            dof_est = _estimate_dof_area_ouomega(xy, t_fit, float(tau_hat), float(omega_hat), np.asarray(sigma_hat, dtype=float), method=method, isotropic=isotropic)
            if dof_est is not None:
                out.params["DOF.area.fit"] = float(dof_est)
            try:
                out.params["sigma"] = covm_factory(sigma_hat, isotropic=isotropic, axes=tuple(out.params.get("axes", ("x", "y"))))
            except Exception:
                pass
        if ouf_fit is not None:
            tau_hat, mu_hat, sigma_hat, fit_value = ouf_fit
            out.params["tau"] = {"position": float(tau_hat[0]), "velocity": float(tau_hat[1])}
            out.params["tau_list"] = [float(tau_hat[0]), float(tau_hat[1])]
            out.params["mu"] = np.asarray(mu_hat, dtype=float)
            out.params["sigma_matrix"] = np.asarray(sigma_hat, dtype=float)
            out.model = "OUf" if tied else "OUF"
            out.params["_profile_IC"] = _profile_ic(float(fit_value), xy.shape[0], (2 if isotropic else 4) if tied else (3 if isotropic else 5))
            if tied:
                dof_est = _estimate_dof_area_ouf_tied(xy, t_fit, float(tau_hat[0]), np.asarray(sigma_hat, dtype=float), method=method, isotropic=isotropic)
            else:
                dof_est = _estimate_dof_area_ouf(xy, t_fit, [float(tau_hat[0]), float(tau_hat[1])], np.asarray(sigma_hat, dtype=float), method=method, isotropic=isotropic)
            if dof_est is not None:
                out.params["DOF.area.fit"] = float(dof_est)
            try:
                out.params["sigma"] = covm_factory(sigma_hat, isotropic=isotropic, axes=tuple(out.params.get("axes", ("x", "y"))))
            except Exception:
                pass
    ll = ctmm_loglike(telem, out)
    out.params["loglike"] = float(ll)
    n = max(len(telem.data), 1)
    q = len(out.params.get("axes", ("x", "y")))
    k_mean = len(np.asarray(out.params.get("mu", []), dtype=float).reshape(-1))
    features = out.params.get("features", [])
    nu = len(features)
    k = nu + k_mean
    aic = 2.0 * k - 2.0 * ll
    bic = math.log(max(n, 1)) * k - 2.0 * ll
    denom = max(q * n - k - max(nu, 1), 1e-12)
    aicc = -2.0 * ll + q * n * (2.0 * k / denom)
    out.params["AIC"] = float(aic)
    out.params["AICc"] = float(aicc) if np.isfinite(aicc) else float("inf")
    out.params["BIC"] = float(bic)
    profile_ic = out.params.get("_profile_IC")
    if isinstance(profile_ic, dict) and np.isfinite(ll):
        out.params["AIC"] = float(profile_ic.get("AIC", out.params["AIC"]))
        out.params["AICc"] = float(profile_ic.get("AICc", out.params["AICc"]))
        out.params["BIC"] = float(profile_ic.get("BIC", out.params["BIC"]))
    # lightweight effective sample-size proxies consumed by AKDE summaries.
    tau = out.params.get("tau", {})
    tau_pos = float(tau.get("position", 1.0)) if isinstance(tau, dict) and tau else 1.0
    t = _model_times(telem, out)
    if t.size > 1:
        dof_area = max(float((t[-1] - t[0]) / max(tau_pos, 1e-9)), 1.0)
    else:
        dof_area = 1.0
    dof_fit = out.params.get("DOF.area.fit")
    if dof_fit is not None and np.isfinite(float(dof_fit)) and float(dof_fit) > 0:
        dof_area = float(dof_fit)
    elif not bool(out.params.get("range", True)):
        dof_area = float("nan")
    out.params["DOF"] = {"area": dof_area, "mean": dof_area, "speed": dof_area}
    return out


def _finite_positive(values: list[float]) -> list[float]:
    return [float(v) for v in values if np.isfinite(float(v)) and float(v) > 0]


def _data_time_scale(telem: Telemetry) -> tuple[float, float]:
    t = epoch_seconds(telem.data[telem.time_col])
    t = t[np.isfinite(t)]
    if t.size < 2:
        return 86400.0, 3600.0
    dt = np.diff(np.sort(t))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 86400.0, 3600.0
    span = max(float(np.max(t) - np.min(t)), float(np.max(dt)))
    step = float(np.median(dt))
    return span, step


def _ctmm_structure_key(model: CTMMModel) -> tuple[Any, ...]:
    tau = model.params.get("tau_list", [])
    tau_len = len(tau) if isinstance(tau, list) else len(list(tau or []))
    timelink_par = _timelink_par(model)
    period = tuple(np.asarray(_get_param(model.params, "period", default=[]), dtype=float).reshape(-1).tolist())
    harmonic = tuple(np.asarray(_get_param(model.params, "harmonic", default=[]), dtype=float).reshape(-1).tolist())
    return (
        tau_len,
        bool(model.params.get("range", True)),
        bool(model.params.get("isotropic", False)),
        bool(model.params.get("tau_tied", False)),
        bool(model.params.get("circle", False)),
        bool(float(model.params.get("omega", 0.0) or 0.0)),
        bool(model.params.get("error", False)),
        str(model.params.get("mean", "stationary")),
        period,
        harmonic,
        str(model.params.get("timelink", "identity")),
        int(timelink_par.size),
    )


def _copy_model_with(model: CTMMModel, **updates) -> CTMMModel:
    params = dict(model.params)
    params.update(updates)
    return CTMMModel(model=model.model, params=params)


def _trend_link_variants(model: CTMMModel) -> list[CTMMModel]:
    params = model.params
    variants: list[CTMMModel] = []
    if str(params.get("mean", "stationary") or "stationary") == "periodic":
        period = np.asarray(_get_param(params, "period", default=[86400.0]), dtype=float).reshape(-1)
        harmonic = np.asarray(_get_param(params, "harmonic", default=np.zeros(period.size)), dtype=float).reshape(-1)
        if harmonic.size == 0:
            harmonic = np.zeros(period.size, dtype=float)
        if period.size == 1 and harmonic.size > 1:
            period = np.repeat(period, harmonic.size)
        if harmonic.size == 1 and period.size > 1:
            harmonic = np.repeat(harmonic, period.size)
        for i in range(harmonic.size):
            h = harmonic.copy()
            h[i] = max(h[i] - 1.0, 0.0)
            variants.append(_copy_model_with(model, period=period.copy(), harmonic=h))
            h = harmonic.copy()
            h[i] = h[i] + 1.0
            variants.append(_copy_model_with(model, period=period.copy(), harmonic=h))
    link = str(params.get("timelink", "identity") or "identity")
    par = _timelink_par(model)
    if link != "identity":
        if par.size:
            variants.append(_copy_model_with(model, **{"timelink.par": np.asarray([], dtype=float), "timelink_par": np.asarray([], dtype=float)}))
        if link == "switch" and par.size == 0:
            variants.append(_copy_model_with(model, **{"timelink.par": np.asarray([0.0]), "timelink_par": np.asarray([0.0])}))
        elif link == "fourier":
            variants.append(_copy_model_with(model, **{"timelink.par": np.r_[par, 0.0, 0.0], "timelink_par": np.r_[par, 0.0, 0.0]}))
            if par.size > 2:
                variants.append(_copy_model_with(model, **{"timelink.par": par[:-2], "timelink_par": par[:-2]}))
        elif link in {"cosine", "spline"}:
            variants.append(_copy_model_with(model, **{"timelink.par": np.r_[par, 0.0], "timelink_par": np.r_[par, 0.0]}))
            if par.size > 1:
                variants.append(_copy_model_with(model, **{"timelink.par": par[:-1], "timelink_par": par[:-1]}))
    return variants


def _ctmm_select_candidates(telem: Telemetry, models: list[CTMMModel], include_range_free: bool = False) -> list[CTMMModel]:
    """R-like stationary selector candidate expansion for app-level AKDE auto.

    R's ``ctmm.select`` performs iterative simplify/complexify passes. The Python
    port currently has exact fit support for stationary IID, OU, and OUF, so this
    expands the initial guess across those supported simplify/complexify branches
    instead of using the estimator-layer three-model shortcut.
    """
    span, step = _data_time_scale(telem)
    tau_pos_values: list[float] = []
    tau_vel_values: list[float] = []
    base_isotropic = False
    target_circle = False
    target_error = False
    target_extra: dict[str, Any] = {}
    error_cols = {"HDOP", "VAR.xy", "COV.x.x", "COV.y.y", "COV.major", "COV.minor"}
    target_error = target_error or any(c in telem.data.columns for c in error_cols)

    for idx, m in enumerate(models):
        if idx == 0:
            for key in ("mean", "dynamics", "link", "timelink", "period", "harmonic", "timelink.par", "timelink_par", "error"):
                if key in m.params:
                    target_extra[key] = m.params[key]
        base_isotropic = base_isotropic or bool(m.params.get("isotropic", False))
        target_circle = target_circle or bool(float(m.params.get("circle", 0.0) or 0.0))
        target_error = target_error or bool(m.params.get("error", False))
        vals = _finite_positive(list(m.params.get("tau_list", []) or []))
        if vals:
            tau_pos_values.append(vals[0])
        if len(vals) > 1:
            tau_vel_values.append(vals[1])

    try:
        from .variogram import variogram

        guess = ctmm_guess(variogram(telem), models[0] if models else None)
        vals = _finite_positive(list(guess.params.get("tau_list", []) or []))
        if vals:
            tau_pos_values.append(vals[0])
        if len(vals) > 1:
            tau_vel_values.append(vals[1])
    except Exception:
        pass

    tau_pos = float(np.nanmedian(tau_pos_values)) if tau_pos_values else max(span / 2.0, step)
    tau_vel = float(np.nanmedian(tau_vel_values)) if tau_vel_values else max(step, tau_pos / 10.0)
    tau_pos = max(tau_pos, step * 1.01)
    tau_vel = max(min(tau_vel, tau_pos / 1.01), step / 100.0)
    if tau_pos <= tau_vel:
        tau_pos = tau_vel * 1.01

    if "error" in target_extra:
        target_error = target_error or bool(target_extra.get("error", False))
        target_extra.pop("error", None)

    candidates: list[CTMMModel] = []
    def make_candidate(*, tau, range=True, isotropic=False, **extra):
        opts = dict(target_extra)
        opts.update(extra)
        return ctmm(tau=tau, range=range, isotropic=isotropic, **opts)

    isotropy_options = [False, True]
    if base_isotropic:
        isotropy_options = [True, False]
    error_options = [False]
    if target_error:
        error_options = [True, False]
    for err in error_options:
        for iso in isotropy_options:
            candidates.append(make_candidate(tau=None, range=True, isotropic=iso, error=err))
            candidates.append(make_candidate(tau=[tau_pos], range=True, isotropic=iso, error=err))
            if target_circle:
                circle0 = 2.0 * math.pi / max(span, step)
                candidates.append(make_candidate(tau=[tau_pos], range=True, isotropic=iso, circle=circle0, error=err))
            tau_crit = float(np.sqrt(max(tau_pos * tau_vel, np.finfo(float).tiny)))
            candidates.append(make_candidate(tau=[tau_crit, tau_crit], range=True, isotropic=iso, tau_tied=True, error=err))
            omega0 = max(2.0 * math.pi / max(span, step), 1.0 / max(100.0 * span, step))
            candidates.append(make_candidate(tau=[tau_crit, tau_crit], omega=omega0, range=True, isotropic=iso, error=err))
            candidates.append(make_candidate(tau=[tau_pos, tau_vel], range=True, isotropic=iso, error=err))
            if target_circle:
                candidates.append(make_candidate(tau=[tau_crit, tau_crit], range=True, isotropic=iso, circle=circle0, tau_tied=True, error=err))
                candidates.append(make_candidate(tau=[tau_crit, tau_crit], omega=omega0, range=True, isotropic=iso, circle=circle0, error=err))
                candidates.append(make_candidate(tau=[tau_pos, tau_vel], range=True, isotropic=iso, circle=circle0, error=err))
            if include_range_free:
                candidates.append(make_candidate(tau=[float("inf")], range=False, isotropic=iso, error=err))
                candidates.append(make_candidate(tau=[float("inf"), tau_vel], range=False, isotropic=iso, error=err))

    if target_extra.get("mean") == "periodic" or str(target_extra.get("timelink", "identity")) != "identity":
        expanded: list[CTMMModel] = []
        for cand in candidates:
            expanded.append(cand)
            expanded.extend(_trend_link_variants(cand))
        candidates = expanded

    unique: list[CTMMModel] = []
    seen: set[tuple[Any, ...]] = set()
    for cand in candidates:
        key = _ctmm_structure_key(cand)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def _ctmm_sort_key(model: CTMMModel, IC: str, MSPE: str | None = "position") -> tuple[float, float]:
    key = IC if IC in ("AIC", "AICc", "BIC", "LOOCV", "HSCV") else "AICc"
    ic = float(model.params.get(key, np.inf))
    if MSPE is None or str(MSPE).lower() in {"none", "na", "nan"}:
        return ic, 0.0
    mspe = model.params.get("MSPE", {})
    if isinstance(mspe, dict):
        return ic, float(mspe.get(str(MSPE), np.inf))
    return ic, np.inf


def _fit_select_candidate(payload):
    idx, telem, cand, ic_upper = payload
    try:
        fit = ctmm_fit(telem, cand)
        if ic_upper in {"LOOCV", "HSCV"}:
            try:
                from .cv import HSCV, LOOCV

                fit.params[ic_upper] = float(LOOCV(telem, fit) if ic_upper == "LOOCV" else HSCV(telem, fit))
            except Exception:
                fit.params[ic_upper] = float("inf")
        fit.params["_select_candidate"] = {
            "model": fit.model,
            "isotropic": bool(fit.params.get("isotropic", False)),
            "circle": float(fit.params.get("circle", 0.0) or 0.0),
            "error": bool(fit.params.get("error", False)),
            "AIC": float(fit.params.get("AIC", np.inf)),
            "AICc": float(fit.params.get("AICc", np.inf)),
            "BIC": float(fit.params.get("BIC", np.inf)),
            "LOOCV": float(fit.params.get("LOOCV", np.inf)),
            "HSCV": float(fit.params.get("HSCV", np.inf)),
        }
        return idx, fit, None
    except Exception as exc:
        return idx, None, {"model": cand.model, "error": f"{type(exc).__name__}: {exc}"}


def _auto_select_cores(cores: int, n_candidates: int) -> int:
    if n_candidates <= 1:
        return 1
    if cores <= 0:
        # Windows process pools require the launched app module to be safely
        # importable under spawn. Keep auto mode sequential there; Linux/HF can
        # use forked workers without changing model-selection logic.
        if os.name == "nt":
            return 1
        cores = min(4, os.cpu_count() or 1)
    return max(1, min(int(cores), int(n_candidates)))


def ctmm_select(telem: Telemetry, models: list[CTMMModel], **kwargs):
    IC = str(kwargs.pop("IC", "AICc"))
    MSPE = kwargs.pop("MSPE", "position")
    level = float(kwargs.pop("level", 1.0))
    verbose = bool(kwargs.pop("verbose", False))
    trace = bool(kwargs.pop("trace", False))
    cores = int(kwargs.pop("cores", 0))
    iterate = bool(kwargs.pop("iterate", True))
    del level, trace, kwargs
    if isinstance(models, CTMMModel):
        models = [models]
    if not models:
        return None
    ic_upper = str(IC).upper()
    include_range_free = ic_upper in {"LOOCV", "HSCV", "NA", "NAN", "NONE"}
    candidates = _ctmm_select_candidates(telem, models, include_range_free=include_range_free) if iterate else list(models)
    cores_use = _auto_select_cores(int(cores), len(candidates))
    fitted: list[CTMMModel] = []
    errors: list[dict[str, str]] = []
    payloads = [(idx, telem, cand, ic_upper) for idx, cand in enumerate(candidates)]
    results = []
    if cores_use > 1:
        old_skip_launch = os.environ.get("SPATCHAT_SKIP_LAUNCH")
        os.environ["SPATCHAT_SKIP_LAUNCH"] = "1"
        try:
            with ProcessPoolExecutor(max_workers=cores_use) as executor:
                futures = [executor.submit(_fit_select_candidate, payload) for payload in payloads]
                for future in as_completed(futures):
                    results.append(future.result())
        except Exception as exc:
            errors.append({"model": "parallel", "error": f"{type(exc).__name__}: {exc}; falling back to sequential selection"})
            results = [_fit_select_candidate(payload) for payload in payloads]
        finally:
            if old_skip_launch is None:
                os.environ.pop("SPATCHAT_SKIP_LAUNCH", None)
            else:
                os.environ["SPATCHAT_SKIP_LAUNCH"] = old_skip_launch
    else:
        results = [_fit_select_candidate(payload) for payload in payloads]
    for _, fit, err in sorted(results, key=lambda item: item[0]):
        if fit is not None:
            fitted.append(fit)
        elif err is not None:
            errors.append(err)
    if not fitted:
        if errors:
            raise RuntimeError(f"ctmm.select could not fit any candidate: {errors[0]['error']}")
        return None
    fitted = sorted(fitted, key=lambda m: _ctmm_sort_key(m, IC, MSPE))
    best = fitted[0]
    best.params["_select_candidates"] = [m.params.get("_select_candidate", {}) for m in fitted]
    if errors:
        best.params["_select_errors"] = errors
    if verbose:
        return fitted
    return best


def ctmm_loglike(telem: Telemetry, model: CTMMModel, **kwargs) -> float:
    del kwargs
    df = telem.data
    x = df[telem.x_col].to_numpy(dtype=float)
    y = df[telem.y_col].to_numpy(dtype=float)
    t = _model_times(telem, model)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    x = x[ok]
    y = y[ok]
    t = t[ok]
    error_var = _measurement_error_var(telem, model, ok=ok)
    if x.size < 3:
        return float("-inf")

    sig = model.params.get("sigma")
    if isinstance(sig, Covm):
        sigma_matrix = np.asarray(sig.sigma, dtype=float)
    else:
        s = model.params.get("sigma_matrix")
        if s is not None:
            sigma_matrix = np.asarray(s, dtype=float)
        else:
            sigma_matrix = np.cov(np.column_stack([x, y]).T)
    if sigma_matrix.shape != (2, 2) or not np.all(np.isfinite(sigma_matrix)):
        return float("-inf")

    tau = model.params.get("tau", {})
    has_tau = isinstance(tau, dict) and bool(tau)
    range_ = bool(model.params.get("range", True))

    if range_ and not has_tau:
        z = np.column_stack([x, y])
        mu = model.params.get("mu")
        if mu is None:
            mean = np.broadcast_to(np.nanmean(z, axis=0), z.shape)
        else:
            mean = _mean_at(model, t, dim=2)
        r = z - mean
        sign, logdet = np.linalg.slogdet(sigma_matrix)
        if sign <= 0 or not np.isfinite(logdet):
            return float("-inf")
        if error_var is not None:
            ll = 0.0
            eye = np.eye(2, dtype=float)
            for ri, zi in zip(error_var, r):
                s_i = sigma_matrix + float(ri) * eye
                sign_i, logdet_i = np.linalg.slogdet(s_i)
                if sign_i <= 0 or not np.isfinite(logdet_i):
                    return float("-inf")
                q_i = float(zi @ np.linalg.inv(s_i) @ zi)
                ll += -0.5 * (2.0 * math.log(2.0 * math.pi) + logdet_i + q_i)
            return float(ll)
        try:
            inv_s = np.linalg.inv(sigma_matrix)
        except np.linalg.LinAlgError:
            return float("-inf")
        q = np.einsum("ij,jk,ik->i", r, inv_s, r)
        return float(-0.5 * np.sum(2.0 * math.log(2.0 * math.pi) + logdet + q))

    dt = np.diff(t)
    if range_ and has_tau and np.any(dt <= np.finfo(float).eps) and error_var is None:
        return float("-inf")
    dx = np.diff(x)
    dy = np.diff(y)
    keep = np.isfinite(dt) & (dt > 0) & np.isfinite(dx) & np.isfinite(dy)
    dt = dt[keep]
    dx = dx[keep]
    dy = dy[keep]
    if dx.size == 0 or dt.size == 0:
        return float("-inf")

    if not range_:
        if isinstance(tau, dict) and "velocity" in tau:
            tau_vel = float(tau.get("velocity"))
            z = np.column_stack([x, y])
            residuals = []
            q_ref = None
            for j in range(2):
                e, q = _innovations_iou_scalar(z[:, j], t, tau_vel, error_var=error_var)
                residuals.append(e)
                if q_ref is None:
                    q_ref = q
            r = np.column_stack(residuals)
            q_ref = np.maximum(q_ref, np.finfo(float).tiny)
            sign, logdet = np.linalg.slogdet(sigma_matrix)
            if sign <= 0 or not np.isfinite(logdet):
                return float("-inf")
            try:
                inv_s = np.linalg.inv(sigma_matrix)
            except np.linalg.LinAlgError:
                return float("-inf")
            quad = np.einsum("ij,jk,ik->i", r, inv_s, r) / q_ref
            return float(-0.5 * (r.shape[0] * (2.0 * math.log(2.0 * math.pi) + logdet) + 2.0 * np.sum(np.log(q_ref)) + np.sum(quad)))
        if sigma_matrix.shape != (2, 2):
            return float("-inf")
        cov_base = 2.0 * sigma_matrix
        sign, logdet_base = np.linalg.slogdet(cov_base)
        if sign <= 0 or not np.isfinite(logdet_base):
            return float("-inf")
        try:
            inv_base = np.linalg.inv(cov_base)
        except np.linalg.LinAlgError:
            return float("-inf")
        dz = np.column_stack([dx, dy])
        if error_var is not None and error_var.size == x.size:
            e_inc = error_var[1:] + error_var[:-1]
            e_inc = e_inc[keep]
            ll = 0.0
            eye = np.eye(2, dtype=float)
            for dti, dzi, ei in zip(dt, dz, e_inc):
                cov = cov_base * float(dti) + float(ei) * eye
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0 or not np.isfinite(logdet):
                    return float("-inf")
                try:
                    q_i = float(dzi @ np.linalg.inv(cov) @ dzi)
                except np.linalg.LinAlgError:
                    return float("-inf")
                ll += -0.5 * (2.0 * math.log(2.0 * math.pi) + logdet + q_i)
            return float(ll)
        q = np.einsum("ij,jk,ik->i", dz, inv_base, dz) / dt
        return float(-0.5 * np.sum(2.0 * math.log(2.0 * math.pi) + logdet_base + 2.0 * np.log(dt) + q))

    if isinstance(tau, dict) and "velocity" in tau:
        tau_list = [float(tau.get("position")), float(tau.get("velocity"))]
        z = np.column_stack([x, y])
        if float(model.params.get("circle", 0.0) or 0.0):
            z = _rotate_track(z, t, float(model.params.get("circle", 0.0) or 0.0))
        design = _drift_design(model, t)
        omega = float(model.params.get("omega", 0.0) or 0.0)
        if omega > 0:
            if design.shape[1]:
                basis_cols = []
                q = None
                for k in range(design.shape[1]):
                    b, qk = _innovations_ouomega_scalar(design[:, k], t, float(tau_list[0]), omega, error_var=error_var)
                    basis_cols.append(b)
                    if q is None:
                        q = qk
                basis = np.column_stack(basis_cols)
            else:
                _, q = _innovations_ouomega_scalar(np.ones(z.shape[0], dtype=float), t, float(tau_list[0]), omega, error_var=error_var)
                basis = np.zeros((z.shape[0], 0), dtype=float)
            residuals = []
            for j in range(2):
                e, _ = _innovations_ouomega_scalar(z[:, j], t, float(tau_list[0]), omega, error_var=error_var)
                residuals.append(e)
        else:
            if design.shape[1]:
                basis_cols = []
                q = None
                for k in range(design.shape[1]):
                    b, qk = _innovations_ouf_scalar(design[:, k], t, tau_list, error_var=error_var)
                    basis_cols.append(b)
                    if q is None:
                        q = qk
                basis = np.column_stack(basis_cols)
            else:
                _, q = _innovations_ouf_scalar(np.ones(z.shape[0], dtype=float), t, tau_list, error_var=error_var)
                basis = np.zeros((z.shape[0], 0), dtype=float)
            residuals = []
            for j in range(2):
                e, _ = _innovations_ouf_scalar(z[:, j], t, tau_list, error_var=error_var)
                residuals.append(e)
        beta = np.asarray(model.params.get("mu", []), dtype=float)
        if beta.ndim == 1:
            if beta.size == 2 and basis.shape[1] == 1:
                beta = beta.reshape(1, 2)
            elif basis.shape[1] == 0:
                beta = np.zeros((0, 2), dtype=float)
            else:
                beta = np.resize(beta, (basis.shape[1], 2))
        r = np.column_stack(residuals) - (basis @ beta[:, :2] if basis.shape[1] else 0.0)
        q = np.maximum(q, np.finfo(float).tiny)
        sign, logdet = np.linalg.slogdet(sigma_matrix)
        if sign <= 0 or not np.isfinite(logdet):
            return float("-inf")
        try:
            inv_s = np.linalg.inv(sigma_matrix)
        except np.linalg.LinAlgError:
            return float("-inf")
        quad = np.einsum("ij,jk,ik->i", r, inv_s, r) / q
        return float(-0.5 * (z.shape[0] * (2.0 * math.log(2.0 * math.pi) + logdet) + 2.0 * np.sum(np.log(q)) + np.sum(quad)))

    tau_pos = float(tau.get("position", 1.0)) if isinstance(tau, dict) and tau else 1.0
    tau_pos = max(tau_pos, 1e-9)
    z = np.column_stack([x, y])
    if float(model.params.get("circle", 0.0) or 0.0):
        z = _rotate_track(z, t, float(model.params.get("circle", 0.0) or 0.0))
    mean = _mean_at(model, t, dim=2)
    r = z - mean
    eye = np.eye(2, dtype=float)
    ll = 0.0

    def add_normal(e: np.ndarray, cov: np.ndarray) -> bool:
        nonlocal ll
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0 or not np.isfinite(logdet):
            return False
        try:
            qv = float(e @ np.linalg.inv(cov) @ e)
        except np.linalg.LinAlgError:
            return False
        ll += -0.5 * (2.0 * math.log(2.0 * math.pi) + logdet + qv)
        return True

    e0 = 0.0 if error_var is None or error_var.size != r.shape[0] else float(error_var[0])
    if not add_normal(r[0], sigma_matrix + e0 * eye):
        return float("-inf")
    phi_all = np.exp(-np.diff(t) / tau_pos)
    for i in range(1, r.shape[0]):
        dti = float(t[i] - t[i - 1])
        if not np.isfinite(dti) or dti <= 0:
            continue
        phi_i = float(phi_all[i - 1])
        innov = r[i] - phi_i * r[i - 1]
        cov = (1.0 - phi_i * phi_i) * sigma_matrix
        if error_var is not None and error_var.size == r.shape[0]:
            cov = cov + (float(error_var[i]) + phi_i * phi_i * float(error_var[i - 1])) * eye
        if not add_normal(innov, cov):
            return float("-inf")
    return float(ll)


def variogram_fit(variogram_obj: dict[str, Any], model: CTMMModel | None = None) -> dict[str, Any]:
    guessed = ctmm_guess(variogram_obj, model=model)
    return {
        "model": guessed,
        "fraction": 0.5,
        "interactive": False,
    }
