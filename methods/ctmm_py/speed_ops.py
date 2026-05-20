from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import special

from .covm import Covm, covm
from .types import CTMMModel, Telemetry
from .stats import NAMES_CI, chi_bias, chi_dof, chisq_ci


def _empirical_speeds(telem: Telemetry) -> pd.DataFrame:
    df = telem.data.sort_values([telem.id_col, telem.time_col]).copy()
    out = []
    for aid, grp in df.groupby(telem.id_col, sort=False):
        t = grp[telem.time_col].astype("int64").to_numpy(dtype=float) / 1e9
        x = grp[telem.x_col].to_numpy(dtype=float)
        y = grp[telem.y_col].to_numpy(dtype=float)
        if len(grp) < 2:
            continue
        dt = np.diff(t)
        dx = np.diff(x)
        dy = np.diff(y)
        good = np.isfinite(dt) & (dt > 0) & np.isfinite(dx) & np.isfinite(dy)
        if not np.any(good):
            continue
        sp = np.sqrt(dx[good] ** 2 + dy[good] ** 2) / dt[good]
        t_mid = (t[:-1][good] + t[1:][good]) / 2.0
        out.append(
            pd.DataFrame(
                {
                    "id": aid,
                    "t": t_mid,
                    "speed": sp,
                }
            )
        )
    if not out:
        return pd.DataFrame(columns=["id", "t", "speed"])
    return pd.concat(out, axis=0, ignore_index=True)


def _empirical_speed(telem: Telemetry) -> float:
    sp = _empirical_speeds(telem)
    if sp.empty:
        return float("nan")
    return float(np.nanmean(sp["speed"].to_numpy(dtype=float)))


def _tau_values(model: CTMMModel) -> list[float]:
    tau = model.params.get("tau")
    if isinstance(tau, dict) and tau:
        return [float(v) for _, v in sorted(tau.items(), key=lambda kv: float(kv[1]), reverse=True)]
    return [float(v) for v in (model.params.get("tau_list", []) or [])]


def _sigma_matrix(model: CTMMModel, sigma: Any = None) -> np.ndarray:
    if sigma is None:
        sigma = model.params.get("sigma")
    if isinstance(sigma, Covm):
        return np.asarray(sigma.sigma, dtype=float)
    if sigma is not None:
        try:
            s = np.asarray(sigma, dtype=float)
            if s.ndim == 2:
                return s
            return covm(s, isotropic=bool(model.params.get("isotropic", False)), axes=tuple(model.params.get("axes", ("x", "y")))).sigma
        except Exception:
            pass
    s = model.params.get("sigma_matrix")
    if s is None:
        raise ValueError("CTMM model has no sigma estimate")
    return np.asarray(s, dtype=float)


def _speed_ci_frame(ci: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([np.asarray(ci, dtype=float)], index=["speed (meters/second)"], columns=list(NAMES_CI))


def _bad_speed_result() -> dict[str, Any]:
    return {"DOF": {"speed": 0.0}, "CI": _speed_ci_frame(np.array([0.0, float("inf"), float("inf")]))}


def speed_deterministic(CTMM: CTMMModel, sigma: Any = None) -> float:
    """``speed_deterministic`` for stationary OUF/IOU movement models."""
    tau = _tau_values(CTMM)
    sig = _sigma_matrix(CTMM, sigma=sigma).astype(float)
    if sig.shape[0] < 2 or sig.shape[1] < 2:
        sig = np.diag(np.repeat(float(sig.reshape(-1)[0]), 2))

    if bool(CTMM.params.get("range", True)):
        if len(tau) < 2:
            return float("inf")
        denom = float(np.prod(tau[:2]))
    else:
        if len(tau) < 2:
            return float("inf")
        denom = float(tau[1])
    if not np.isfinite(denom) or denom <= 0.0:
        return 0.0 if np.isposinf(denom) else float("inf")
    sig = sig / denom

    vals = np.linalg.eigvalsh(sig)
    vals = np.sort(np.real(vals))[::-1]
    vals = np.clip(vals, 0.0, np.inf)
    if vals[0] <= 0.0:
        return 0.0
    if bool(CTMM.params.get("isotropic", False)) or np.isclose(vals[0], vals[1], rtol=1e-12, atol=1e-15):
        return float(math.sqrt(vals[0] * math.pi / 2.0))
    m = 1.0 - float(np.clip(vals[1] / vals[0], 0.0, 1.0))
    return float(math.sqrt(2.0 / math.pi) * math.sqrt(vals[0]) * special.ellipe(m))


def _parameter_names(model: CTMMModel) -> list[str]:
    cov = model.params.get("COV")
    rownames = model.params.get("COV_rownames")
    if rownames is None and hasattr(cov, "index"):
        rownames = list(cov.index)
    if rownames is None:
        rownames = model.params.get("features")
    return [str(v) for v in rownames] if rownames else []


def _copy_model_with_param(model: CTMMModel, name: str, value: float) -> CTMMModel:
    params = dict(model.params)
    tau = dict(params.get("tau", {}) or {})
    tau_list = list(params.get("tau_list", []) or [])
    if name.startswith("tau "):
        key = name.split(" ", 1)[1]
        tau[key] = float(value)
        ordered = sorted(tau.items(), key=lambda kv: float(kv[1]), reverse=True)
        params["tau"] = dict(ordered)
        params["tau_list"] = [float(v) for _, v in ordered]
    elif name == "omega":
        params["omega"] = float(value)
    elif name in {"major", "minor", "angle", "variance"}:
        sig = model.params.get("sigma")
        if isinstance(sig, Covm):
            par = dict(sig.par)
            if name == "variance":
                par["major"] = par["minor"] = float(value)
            else:
                par[name] = float(value)
            sig2 = covm(par, isotropic=sig.isotropic, axes=sig.axes)
            params["sigma"] = sig2
            params["sigma_matrix"] = sig2.sigma
    return CTMMModel(model.model, params)


def speed_variance(object: CTMMModel, MEAN: float | None = None) -> dict[str, Any]:
    """Finite-difference port of ``speed_variance`` over available COV rows."""
    names = _parameter_names(object)
    cov = object.params.get("COV")
    if not names or cov is None:
        return {"MEAN": speed_deterministic(object) if MEAN is None else MEAN, "VAR": float("inf"), "J": np.array([], dtype=float)}
    try:
        cov_arr = np.asarray(cov, dtype=float)[: len(names), : len(names)]
    except Exception:
        return {"MEAN": speed_deterministic(object) if MEAN is None else MEAN, "VAR": float("inf"), "J": np.array([], dtype=float)}

    def get_value(name: str) -> float | None:
        if name.startswith("tau "):
            return float((object.params.get("tau", {}) or {}).get(name.split(" ", 1)[1], np.nan))
        if name == "omega":
            return float(object.params.get("omega", np.nan))
        sig = object.params.get("sigma")
        if isinstance(sig, Covm) and name in sig.par:
            return float(sig.par[name])
        if name == "variance" and isinstance(sig, Covm):
            return float(np.mean(np.diag(sig.sigma)))
        return None

    grad = np.zeros(len(names), dtype=float)
    for i, name in enumerate(names):
        value = get_value(name)
        if value is None or not np.isfinite(value):
            continue
        step = math.sqrt(np.finfo(float).eps) * max(abs(value), 1.0)
        lo = value - step
        hi = value + step
        if name.startswith("tau ") or name in {"major", "minor", "variance", "omega"}:
            lo = max(lo, np.finfo(float).tiny)
        try:
            f_hi = speed_deterministic(_copy_model_with_param(object, name, hi))
            f_lo = speed_deterministic(_copy_model_with_param(object, name, lo))
            grad[i] = (f_hi - f_lo) / (hi - lo)
        except Exception:
            grad[i] = 0.0
    var = float(grad @ cov_arr @ grad) if grad.size else float("inf")
    if not np.isfinite(var):
        var = float("inf")
    return {"MEAN": speed_deterministic(object) if MEAN is None else MEAN, "VAR": var, "J": grad}


def _speed_ctmm(
    object: CTMMModel,
    data: Telemetry | None = None,
    t=None,
    level: float = 0.95,
    robust: bool = False,
    units: bool = True,
    prior: bool = True,
    fast: bool = True,
    **kwargs,
) -> dict[str, Any]:
    del t, robust, units, kwargs
    tau = _tau_values(object)
    if len(tau) < 2 or tau[1] <= np.finfo(float).eps:
        return _bad_speed_result()
    if tau[1] == float("inf"):
        return {"DOF": {"speed": 0.0}, "CI": _speed_ci_frame(np.array([0.0, 0.0, 0.0]))}
    if data is not None or not fast:
        # Full simulation-based speed.R branches depend on simulate/emulate velocity paths.
        # The deterministic stationary branch is the exact analytic branch used by ctmm
        # when no data simulation is requested.
        prior = False

    mean = speed_deterministic(object)
    if prior and fast:
        stuff = speed_variance(object, MEAN=mean)
        var = float(stuff["VAR"])
        m2 = var + mean * mean
        dof = chi_dof(mean, m2)
        ci = np.sqrt(chisq_ci(m2, dof=dof, alpha=1.0 - level))
        bias = np.asarray(chi_bias(np.array([dof], dtype=float)), dtype=float)[0]
        if bias > 0.0 and np.isfinite(bias):
            ci = ci / bias
        ci[0] = 0.0 if not np.isfinite(ci[0]) else ci[0]
        ci[1] = mean
    else:
        dof = 0.0
        ci = np.array([mean, mean, mean], dtype=float)
    return {"DOF": {"speed": float(dof) / 2.0}, "CI": _speed_ci_frame(ci)}


def abs_bivar(mu, Sigma, return_VAR: bool = False, return_var: bool | None = None):
    """``abs_bivar``: approximate E[|X|] for a bivariate normal vector."""
    if return_var is not None:
        return_VAR = bool(return_var)
    mu_arr = np.asarray(mu, dtype=float).reshape(-1)
    Sigma_arr = np.asarray(Sigma, dtype=float)
    sigma0 = float(np.mean(np.diag(Sigma_arr)))
    mu2 = float(mu_arr @ mu_arr)
    mu_norm = math.sqrt(max(mu2, 0.0))
    vals = np.sort(np.linalg.eigvalsh(Sigma_arr))[::-1]
    vals = np.clip(vals, 0.0, np.inf)
    barg = mu2 / (4.0 * sigma0) if sigma0 != 0.0 else float("inf")
    if sigma0 == 0.0 or barg >= 2**16:
        m1 = mu_norm
    else:
        b0 = special.ive(0, barg)
        b1 = special.ive(1, barg)
        sqrtpi2 = math.sqrt(math.pi / 2.0)
        bv = sqrtpi2 * math.sqrt(max(barg, 0.0)) * (b0 + b1) * mu_norm
        ratio = 0.0 if vals[0] <= 0.0 else float(np.clip(vals[1] / vals[0], 0.0, 1.0))
        bs = b0 / sqrtpi2 * math.sqrt(vals[0]) * special.ellipe(1.0 - ratio)
        m1 = float(bv + bs)
    if return_VAR:
        m2 = mu2 + 2.0 * sigma0
        return np.array([m1, max(0.0, m2 - m1 * m1)], dtype=float)
    return float(m1)


def abs_data(data: pd.DataFrame, axes=("x", "y")) -> dict[str, Any]:
    """``abs_data`` subset for dataframes with velocity columns."""
    vx, vy = ("vx", "vy") if "vx" in data.columns and "vy" in data.columns else tuple(axes)
    v = data[[vx, vy]].to_numpy(dtype=float)
    n = v.shape[0]
    var = np.zeros((n, 2, 2), dtype=float)
    if {"vx_var", "vy_var"}.issubset(data.columns):
        var[:, 0, 0] = pd.to_numeric(data["vx_var"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        var[:, 1, 1] = pd.to_numeric(data["vy_var"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    m2 = np.sum(v * v, axis=1) + np.trace(var, axis1=1, axis2=2)
    m1 = np.array([abs_bivar(v[i], var[i]) for i in range(n)], dtype=float)
    dof = np.array([chi_dof(m1[i], m2[i]) for i in range(n)], dtype=float)
    variance = np.maximum(0.0, m2 - m1 * m1)
    return {"r": v, "M1": m1, "M2": m2, "VAR": variance, "DOF": dof}


def speeds_fast(data: Telemetry | pd.DataFrame, CTMM: CTMMModel | None = None, t=None, level: float | None = 0.95, robust: bool = False, append: bool = False, **kwargs):
    """Fast ``speeds`` branch for observed/predicted velocity columns."""
    del CTMM, t, robust, kwargs
    df = data.data if isinstance(data, Telemetry) else data
    if "vx" not in df.columns or "vy" not in df.columns:
        if isinstance(data, Telemetry):
            return _empirical_speeds(data)
        raise ValueError("speeds_fast requires vx/vy columns or a Telemetry object")
    stuff = abs_data(df, axes=("vx", "vy"))
    if append:
        out = df.copy()
        out["speed"] = stuff["M1"]
        return out
    if level is None:
        return pd.DataFrame({"speed": stuff["M1"], "DOF": stuff["DOF"], "VAR": stuff["VAR"]})
    ci = np.vstack([np.sqrt(chisq_ci(stuff["M2"][i], dof=stuff["DOF"][i], level=level)) for i in range(len(stuff["M1"]))])
    ci[:, 1] = stuff["M1"]
    return pd.DataFrame(ci, columns=list(NAMES_CI))


def speeds(object: Telemetry | CTMMModel, CTMM: CTMMModel | None = None, **kwargs) -> pd.DataFrame:
    if isinstance(object, CTMMModel):
        data = kwargs.pop("data", None)
        if data is None:
            raise TypeError("speeds.ctmm requires data")
        return speeds(data, CTMM=object, **kwargs)
    if not isinstance(object, Telemetry):
        raise TypeError("speeds expects a Telemetry or CTMMModel object")
    if CTMM is None:
        if kwargs:
            raise TypeError(f"speeds got unexpected keyword arguments without CTMM: {sorted(kwargs)}")
        return _empirical_speeds(object)
    fast = bool(kwargs.pop("fast", True))
    prior = bool(kwargs.pop("prior", False))
    if not prior and fast:
        return speeds_fast(object, CTMM=CTMM, **kwargs)
    return speeds_fast(object, CTMM=CTMM, **kwargs)


def speed(object: Telemetry | CTMMModel, CTMM: CTMMModel | None = None, **kwargs):
    if isinstance(object, CTMMModel):
        return _speed_ctmm(object, **kwargs)
    if not isinstance(object, Telemetry):
        raise TypeError("speed expects a Telemetry or CTMMModel object")
    if CTMM is None:
        if kwargs:
            raise TypeError(f"speed got unexpected keyword arguments without CTMM: {sorted(kwargs)}")
        return _empirical_speed(object)
    return _speed_ctmm(CTMM, data=object, **kwargs)


__all__ = [
    "abs_bivar",
    "abs_data",
    "speed",
    "speed_deterministic",
    "speed_variance",
    "speeds",
    "speeds_fast",
]
