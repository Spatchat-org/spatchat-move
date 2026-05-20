"""Partial parity translation of ctmm 1.3.0 ``R/summary.ctmm.R``."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2, norm

from .covm import Covm, area_covm, covm, sqrtm_covm, var_covm
from .pd_matrix import pd_solve
from .types import CTMMModel
from .covm import axes2var
from .diffusion import diffusion
from .speed import speed


def _chisq_ci(mean: float, dof: float | None = None, var: float | None = None, alpha: float = 0.05) -> np.ndarray:
    m = float(mean)
    if not np.isfinite(m) or m < 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    if dof is None:
        if var is None or not np.isfinite(var) or var <= 0:
            return np.array([m, m, m], dtype=float)
        dof = max(2.0 * m * m / float(var), 1e-12)

    dof = float(max(dof, 1e-12))
    lo_q = chi2.ppf(alpha / 2.0, dof)
    hi_q = chi2.ppf(1.0 - alpha / 2.0, dof)
    lo = m * (lo_q / dof) if np.isfinite(lo_q) else np.nan
    hi = m * (hi_q / dof) if np.isfinite(hi_q) else np.nan
    return np.array([lo, m, hi], dtype=float)


def ci_tau(tau: float, cov: float, alpha: float = 0.05, min_: float = 0.0, max_: float = np.inf) -> np.ndarray:
    """Translation of ``ci.tau`` helper."""
    tau = float(tau)
    cov = float(cov)
    if np.isnan(cov):
        cov = np.inf

    ci1 = _chisq_ci(tau, var=cov, alpha=alpha)
    inv_tau = 1.0 / max(tau, 1e-12)
    inv_var = cov / max(tau**4, 1e-24)
    ci2_inv = _chisq_ci(inv_tau, var=inv_var, alpha=alpha)
    ci2 = np.array([1.0 / ci2_inv[2], 1.0 / ci2_inv[1], 1.0 / ci2_inv[0]], dtype=float)

    lo = float(np.nanmin([ci1[0], ci1[2], ci2[0], ci2[2]]))
    hi = float(np.nanmax([ci1[0], ci1[2], ci2[0], ci2[2]]))
    lo = max(lo, float(min_))
    hi = min(hi, float(max_))
    return np.array([lo, tau, hi], dtype=float)


def DOF_area(model: CTMMModel) -> float:
    p = model.params
    dof = p.get("DOF")
    if isinstance(dof, dict):
        v = float(dof.get("area", np.nan))
        if np.isfinite(v) and v >= 0:
            return v
    fit = p.get("DOF.area.fit")
    if fit is not None and np.isfinite(float(fit)) and float(fit) >= 0:
        return float(fit)
    if not p.get("range", True):
        return 0.0
    sigma = p.get("sigma")
    cov = p.get("COV")
    if sigma is None:
        sigma = p.get("sigma_matrix")
    if sigma is None or cov is None:
        return 0.0
    try:
        sig = sigma if isinstance(sigma, Covm) else covm(sigma, isotropic=bool(p.get("isotropic", False)), axes=tuple(p.get("axes", ("x", "y"))))
        area = float(area_covm(sig))
        names = list(p.get("COV_rownames") or p.get("features") or [])
        if hasattr(cov, "loc"):
            if "major" not in cov.index:
                var = np.inf
            elif bool(p.get("isotropic", False)):
                var = float(cov.loc["major", "major"])
            else:
                P = ["major", "minor"]
                grad = np.asarray([area / float(sig.par[k]) / 2.0 for k in P], dtype=float)
                var = float(grad @ cov.loc[P, P].to_numpy(dtype=float) @ grad)
        else:
            arr = np.asarray(cov, dtype=float)
            if "major" not in names:
                var = np.inf
            elif bool(p.get("isotropic", False)):
                i = names.index("major")
                var = float(arr[i, i])
            else:
                if "minor" not in names:
                    return 0.0
                idx = [names.index("major"), names.index("minor")]
                P = ["major", "minor"]
                grad = np.asarray([area / float(sig.par[k]) / 2.0 for k in P], dtype=float)
                var = float(grad @ arr[np.ix_(idx, idx)] @ grad)
        return float(area * area / abs(var)) if np.isfinite(var) and var != 0 else 0.0
    except Exception:
        return 0.0


def _as_matrix(value):
    if value is None:
        return None
    if isinstance(value, Covm):
        return np.asarray(value.sigma, dtype=float)
    return np.asarray(value, dtype=float)


def _sqrtm_psd(value):
    mat = _as_matrix(value)
    vals, vecs = np.linalg.eigh((mat + mat.T) / 2.0)
    vals = np.clip(vals, 0.0, np.inf)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def DOF_mean(model: CTMMModel) -> float:
    p = model.params
    dof = p.get("DOF")
    if isinstance(dof, dict):
        v = float(dof.get("mean", np.nan))
        if np.isfinite(v) and v >= 0:
            return v
    if not p.get("range", True) or "COV.mu" not in p or "mu" not in p:
        return 0.0
    try:
        COV = np.asarray(p["COV.mu"], dtype=float)
        if COV.ndim == 4:
            COV = COV[:, 0, 0, :]
        if np.any(np.isnan(COV)):
            return 0.0
        if "POV.mu" in p:
            sigma = np.asarray(p["POV.mu"], dtype=float)
            if sigma.ndim == 4:
                sigma = sigma[:, 0, 0, :]
            sigma = _sqrtm_psd(sigma)
        else:
            sig = p.get("sigma", p.get("sigma_matrix"))
            sig = sig if isinstance(sig, Covm) else covm(sig, isotropic=bool(p.get("isotropic", False)), axes=tuple(p.get("axes", ("x", "y"))))
            sigma = sqrtm_covm(sig).sigma
        dof_m = sigma @ pd_solve(COV) @ sigma
        return float(np.mean(np.diag(dof_m)))
    except Exception:
        return 0.0


def DOF_speed(model: CTMMModel) -> float:
    """
    Translation hook for ``DOF.speed``.
    Uses summary output if available, otherwise returns 0.
    """
    d = model.params.get("DOF")
    if isinstance(d, dict):
        v = float(d.get("speed", np.nan))
        if np.isfinite(v) and v >= 0:
            return v
    s = summary_ctmm(model)
    dof = s.get("DOF", {})
    return float(dof.get("speed", 0.0))


def DOF_var(model: CTMMModel) -> float:
    d = model.params.get("DOF")
    if isinstance(d, dict):
        v = float(d.get("var", d.get("variance", np.nan)))
        if np.isfinite(v) and v >= 0:
            return v
    sigma = model.params.get("sigma", model.params.get("sigma_matrix"))
    if sigma is None:
        return 0.0
    try:
        sig = sigma if isinstance(sigma, Covm) else covm(sigma, isotropic=bool(model.params.get("isotropic", False)), axes=tuple(model.params.get("axes", ("x", "y"))))
        msd = float(var_covm(sig, ave=False))
        cov_ran = axes2var(model, MEAN=False)
        if hasattr(cov_ran, "loc"):
            var = float(cov_ran.loc["variance", "variance"]) if "variance" in cov_ran.index else float(np.asarray(cov_ran, dtype=float).reshape(-1)[0])
        else:
            var = float(np.asarray(cov_ran, dtype=float).reshape(-1)[0])
        return float(2.0 * msd * msd / var) if var > 0 else 0.0
    except Exception:
        return 0.0


def J_zero(POV=None):
    if POV is None:
        return np.array([], dtype=float)
    if isinstance(POV, int):
        return np.zeros(int(POV), dtype=float)
    arr = np.asarray(POV, dtype=float)
    return np.zeros_like(arr, dtype=float)


def rand_speed(model: CTMMModel, n: int = 1, seed=None):
    rng = np.random.default_rng(seed)
    s = speed(model)
    est = float(s.get("CI").iloc[0, 1]) if hasattr(s.get("CI"), "iloc") else 0.0
    dof = max(float(s.get("DOF", {}).get("speed", 0.0)), 1.0)
    return rng.chisquare(dof, size=int(n)) * est / dof


def confint_ctmm(model: CTMMModel, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """
    Partial translation of ``confint.ctmm`` focusing on area/tau intervals.
    """
    out: dict[str, np.ndarray] = {}
    dof_a = DOF_area(model)
    sigma = model.params.get("sigma")
    if sigma is not None:
        if hasattr(sigma, "par"):
            major = float(sigma.par.get("major", np.nan))
            minor = float(sigma.par.get("minor", major))
            area = float(np.pi * np.sqrt(max(major, 0.0) * max(minor, 0.0)))
            out["area"] = _chisq_ci(area, dof=max(2.0 * dof_a, 1e-12), alpha=alpha)
        else:
            sm = np.asarray(sigma, dtype=float)
            if sm.ndim == 2 and sm.shape[0] == sm.shape[1] and sm.shape[0] >= 2:
                major = float(np.max(np.diag(sm)))
                minor = float(np.min(np.diag(sm)))
                area = float(np.pi * np.sqrt(max(major, 0.0) * max(minor, 0.0)))
                out["area"] = _chisq_ci(area, dof=max(2.0 * dof_a, 1e-12), alpha=alpha)

    tau_map = model.params.get("tau", {}) or {}
    cov = model.params.get("COV")
    cov_diag = None
    if cov is not None:
        try:
            cov_diag = np.diag(np.asarray(cov, dtype=float))
        except Exception:
            cov_diag = None

    for i, (name, tau_val) in enumerate(tau_map.items()):
        var = float(cov_diag[i]) if cov_diag is not None and i < len(cov_diag) else np.inf
        out[f"tau[{name}]"] = ci_tau(float(tau_val), var, alpha=alpha)
    return out


def summary_ctmm_single(object: CTMMModel, level: float = 0.95, level_UD: float = 0.95, units: bool = True, **kwargs: Any):
    """
    Partial translation of ``summary.ctmm.single``.
    Returns a compact summary dict using currently available model fields.
    """
    del level_UD, units, kwargs
    alpha = 1.0 - float(level)
    ci = confint_ctmm(object, alpha=alpha)
    diffusion_dof = 0.0
    speed_dof = 0.0
    try:
        diff_info = diffusion(object, level=level, finish=False)
        diffusion_dof = float(diff_info.get("DOF", 0.0))
        ci["diffusion"] = diffusion(object, level=level, finish=True)
    except Exception:
        pass
    try:
        speed_info = speed(object, level=level, prior=True, fast=True)
        speed_dof = float(speed_info.get("DOF", {}).get("speed", 0.0))
        speed_ci = speed_info.get("CI")
        if hasattr(speed_ci, "iloc"):
            ci["speed"] = speed_ci.iloc[0].to_numpy(dtype=float)
    except Exception:
        pass
    return {
        "name": object.model,
        "DOF": {
            "mean": DOF_mean(object),
            "area": DOF_area(object),
            "diffusion": diffusion_dof,
            "speed": speed_dof,
            "var": DOF_var(object),
        },
        "CI": ci,
    }


def summary_ctmm_list(object: list[CTMMModel], level: float = 0.95, level_UD: float = 0.95, units: bool = True, **kwargs: Any):
    """Translation of ``summary.ctmm.list`` dispatch behavior."""
    return [summary_ctmm_single(m, level=level, level_UD=level_UD, units=units, **kwargs) for m in object]


def summary_ctmm(object: CTMMModel | list[CTMMModel], level: float = 0.95, level_UD: float = 0.95, units: bool = True, **kwargs: Any):
    """Translation entrypoint for ``summary.ctmm`` generic behavior."""
    if isinstance(object, list):
        return summary_ctmm_list(object, level=level, level_UD=level_UD, units=units, **kwargs)
    return summary_ctmm_single(object, level=level, level_UD=level_UD, units=units, **kwargs)


__all__ = [
    "summary_ctmm",
    "summary_ctmm_single",
    "summary_ctmm_list",
    "confint_ctmm",
    "ci_tau",
    "DOF_area",
    "DOF_mean",
    "DOF_speed",
    "DOF_var",
    "J_zero",
    "rand_speed",
]
