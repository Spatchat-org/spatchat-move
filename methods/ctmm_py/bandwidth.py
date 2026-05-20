"""Partial parity translation of ctmm 1.3.0 ``R/bandwidth.R``."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from .covm import Covm
from .generic_utils import epoch_seconds
from .types import CTMMModel, Telemetry
from .kde import akde_bias
from .plot_variogram import svf_func


def _silverman_scale(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if x.size == 1:
        return 0.0
    s = float(np.std(x, ddof=1))
    iqr = float(np.subtract(*np.quantile(x, [0.75, 0.25])))
    a = min(s, iqr / 1.34) if iqr > 0 else s
    return float(0.9 * a * (x.size ** (-1.0 / 5.0)))


def lag_DOF(data: Telemetry, dt: float | None = None, weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    t = epoch_seconds(data.data[data.time_col])
    if t.size < 2:
        return {"DOF": np.array([1.0]), "lag": np.array([0.0])}
    if weights is None:
        weights = np.ones(t.size, dtype=float) / t.size
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        weights = weights / max(np.sum(weights), np.finfo(float).eps)
    if dt is None:
        d = np.diff(t)
        d = d[d > 0]
        dt = float(np.median(d)) if d.size else 1.0
    t0 = float(t[0])
    idx = np.rint((t - t0) / dt).astype(int)
    n = int(np.max(idx)) + 1
    grid_w = np.zeros(n, dtype=float)
    np.add.at(grid_w, idx, weights)
    nfft = 1 << int(np.ceil(np.log2(max(2 * n, 1))))
    fw = np.fft.fft(grid_w, n=nfft)
    ac = np.fft.ifft(np.abs(fw) ** 2).real[:n]
    lag = np.arange(n, dtype=float) * dt
    dof = np.maximum(ac, 0.0)
    if dof.size:
        dof[0] = float(np.sum(weights**2))
        if dof.size > 1:
            tail = np.sum(dof[1:])
            if tail > 0:
                dof[1:] *= (1.0 - dof[0]) / tail
    return {"DOF": dof, "lag": lag}


def _acf_from_model(model: CTMMModel, lag: np.ndarray) -> np.ndarray:
    return np.asarray(svf_func(model, moment=False)["ACF"](lag), dtype=float)


def _svf_from_model(model: CTMMModel, lag: np.ndarray) -> np.ndarray:
    # R/bandwidth.R standardizes CTMM$sigma to identity before svf.func().
    return 1.0 - _acf_from_model(model, lag)


def _sigma_matrix(model: CTMMModel) -> np.ndarray:
    s = model.params.get("sigma_matrix")
    if s is not None:
        m = np.asarray(s, dtype=float)
        if m.shape == (2, 2):
            return m
    s = model.params.get("sigma")
    if isinstance(s, Covm):
        return np.asarray(s.sigma, dtype=float)
    if s is not None:
        m = np.asarray(s, dtype=float)
        if m.shape == (2, 2):
            return m
    return np.eye(2, dtype=float)


def bandwidth(
    data: Telemetry,
    CTMM: CTMMModel,
    VMM: CTMMModel | None = None,
    weights: bool | np.ndarray = False,
    fast: bool | None = None,
    dt: float | None = None,
    PC: str = "Markov",
    error: float = 0.01,
    precision: float = 0.5,
    verbose: bool = False,
    trace: bool = False,
    dt_plot: bool = True,
    **kwargs: Any,
):
    del VMM, fast, PC, error, precision, verbose, trace, dt_plot, kwargs
    x = data.data[data.x_col].to_numpy(dtype=float)
    y = data.data[data.y_col].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    n = max(x.size, 1)
    if isinstance(weights, bool):
        w = np.ones(n, dtype=float) / n
        weights_opt = bool(weights)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != n:
            raise ValueError("weights length must match telemetry length")
        w = w / max(np.sum(w), np.finfo(float).eps)
        weights_opt = False

    sigma = _sigma_matrix(CTMM)
    try:
        L = np.linalg.cholesky(sigma)
        Li = np.linalg.inv(L)
        XY = np.column_stack([x, y])
        Z = (Li @ XY.T).T
        sx = _silverman_scale(Z[:, 0])
        sy = _silverman_scale(Z[:, 1])
        h0 = float(np.nanmean([sx, sy]))
    except Exception:
        sx = _silverman_scale(x)
        sy = _silverman_scale(y)
        h0 = float(np.nanmean([sx, sy]))
    if not np.isfinite(h0) or h0 <= 0:
        h0 = 1.0

    dt_eff = dt
    if dt_eff is None:
        t_all = epoch_seconds(data.data[data.time_col])
        dtt = np.diff(t_all)
        dtt = dtt[np.isfinite(dtt) & (dtt > 0)]
        if dtt.size:
            DT = float(np.median(dtt))
            dt_min = float(np.min(dtt))
            div = max(int(np.ceil(DT / dt_min)), 1)
            dt_eff = DT / div
    lag_info = lag_DOF(data, dt=dt_eff, weights=w)
    lag = lag_info["lag"]
    DOF = lag_info["DOF"]
    tau = CTMM.params.get("tau", {})
    tau_empty = not tau if isinstance(tau, dict) else False
    w2d = float(np.sum(w**2))
    w2o = max(1.0 - w2d, 0.0)

    if tau_empty:
        # IID branch (R/bandwidth.R, DIM==2).
        def mise(h: float) -> float:
            if h <= 0 or not np.isfinite(h):
                return float("inf")
            return w2d / (2.0 * h * h) + w2o / (2.0 + 2.0 * h * h) - 2.0 / (2.0 + h * h) + 0.5
    else:
        # Autocorrelated branch with fixed weights (R/bandwidth.R, DIM==2).
        G = np.clip(_svf_from_model(CTMM, lag), 0.0, np.inf)

        def mise(h: float) -> float:
            if h <= 0 or not np.isfinite(h):
                return float("inf")
            return float(np.sum(DOF / (G + h * h)) / 2.0 - 2.0 / (2.0 + h * h) + 0.5)

    lo = max(h0 / 16.0, 1e-9)
    hi = max(16.0 * h0, 16.0)
    opt = minimize_scalar(mise, bounds=(lo, hi), method="bounded")
    h = float(opt.x if opt.success else h0)
    H = (h * h) * sigma
    # R DOF.H formula for DIM==2.
    dof_h = (1.0 / (2.0 * h * h) ** 2 - 1.0 / (2.0 + 2.0 * h * h) ** 2) / (
        1.0 / (2.0 + h * h) ** 2 - 1.0 / (2.0 + 2.0 * h * h) ** 2
    )

    bias_obj = akde_bias(CTMM=CTMM, H=H, lag=lag, DOF=DOF, weights=w)
    out = {
        "h": np.array([h, h], dtype=float),
        "H": H,
        "weights": w,
        "DOF": DOF,
        "lag": lag,
        "MISE": float(opt.fun if opt.success else mise(h)),
        "dt": None if dt_eff is None else float(dt_eff),
        "bias": np.asarray(bias_obj["bias"], dtype=float),
        "COV": np.asarray(bias_obj["COV"], dtype=float),
        "DOF.H": float(dof_h),
        "weights_opt": weights_opt,
    }
    return out


def MISE(h, *args, **kwargs):
    del args, kwargs
    h = np.asarray(h, dtype=float)
    return float(np.sum(h * h))


def bandwidth_pop(*args, **kwargs):
    return bandwidth(*args, **kwargs)


__all__ = ["MISE", "lag_DOF", "bandwidth", "bandwidth_pop"]
