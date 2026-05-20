"""Parity-focused translation of ctmm 1.3.0 ``R/krige.R`` surfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import CTMMModel, Telemetry


def _epoch_seconds(col):
    s = pd.to_datetime(col, errors="coerce", utc=True)
    return s.astype("int64").to_numpy(dtype=float) / 1e9


def COVM(*args, isotropic: bool = False, axes=("x", "y"), **kwargs):
    from .covm import covm

    del kwargs
    return covm(*args, isotropic=isotropic, axes=axes)


def fill_data(data: Telemetry, t=None, dt=None, **kwargs):
    del kwargs
    if not isinstance(data, Telemetry):
        raise TypeError("fill_data expects Telemetry")
    df = data.data.sort_values(data.time_col).reset_index(drop=True)
    if t is None:
        if dt is None:
            return data
        sec = _epoch_seconds(df[data.time_col])
        t = np.arange(sec[0], sec[-1] + float(dt) / 2.0, float(dt))
    t = np.asarray(t, dtype=float)
    src_t = _epoch_seconds(df[data.time_col])
    out = pd.DataFrame({data.time_col: pd.to_datetime(t, unit="s", utc=True)})
    out[data.x_col] = np.interp(t, src_t, df[data.x_col].to_numpy(dtype=float))
    out[data.y_col] = np.interp(t, src_t, df[data.y_col].to_numpy(dtype=float))
    out[data.id_col] = df[data.id_col].iloc[0] if data.id_col in df else "animal"
    out["record"] = np.isin(t, src_t)
    return Telemetry(out, id_col=data.id_col, time_col=data.time_col, x_col=data.x_col, y_col=data.y_col, crs=data.crs, metadata=dict(data.metadata))


def smoother(data: Telemetry, CTMM: CTMMModel, precompute=False, sample: bool = False, residual: bool = False, **kwargs):
    del precompute, sample, kwargs
    if residual:
        df = data.data.sort_values(data.time_col)
        arr = df[[data.x_col, data.y_col]].to_numpy(dtype=float)
        return arr - np.nanmean(arr, axis=0)
    return predict_telemetry(data, CTMM=CTMM)


def _sigma_matrix(model: CTMMModel):
    sig = model.params.get("sigma")
    if hasattr(sig, "sigma"):
        return np.asarray(sig.sigma, dtype=float)
    s = model.params.get("sigma_matrix")
    if s is not None:
        return np.asarray(s, dtype=float)
    return np.eye(2)


def simulate_ctmm(object: CTMMModel, data: Telemetry | None = None, t=None, seed=None, **kwargs):
    del kwargs
    rng = np.random.default_rng(seed)
    if t is None:
        if data is not None:
            t = _epoch_seconds(data.data[data.time_col])
        else:
            t = np.arange(0.0, 100.0)
    t = np.asarray(t, dtype=float)
    sigma = _sigma_matrix(object)
    tau = object.params.get("tau_list", []) or []
    tau0 = float(tau[0]) if tau and np.isfinite(tau[0]) and tau[0] > 0 else np.inf
    z = np.zeros((t.size, 2), dtype=float)
    if t.size:
        mu = np.asarray(object.params.get("mu", [0.0, 0.0]), dtype=float).reshape(-1)[:2]
        if mu.size < 2:
            mu = np.pad(mu, (0, 2 - mu.size))
        z[0] = mu
        for i in range(1, t.size):
            dt = max(float(t[i] - t[i - 1]), 0.0)
            if np.isfinite(tau0):
                phi = np.exp(-dt / tau0)
                cov = sigma * max(1.0 - phi * phi, 0.0)
                z[i] = mu + phi * (z[i - 1] - mu) + rng.multivariate_normal(np.zeros(2), cov)
            else:
                z[i] = z[i - 1] + rng.multivariate_normal(np.zeros(2), sigma * max(dt, 0.0))
    df = pd.DataFrame({"t": t, "x": z[:, 0], "y": z[:, 1]})
    if data is not None:
        df[data.time_col] = pd.to_datetime(t, unit="s", utc=True)
        df[data.id_col] = data.data[data.id_col].iloc[0] if data.id_col in data.data else "animal"
        return Telemetry(df, id_col=data.id_col, time_col=data.time_col, x_col="x", y_col="y", crs=data.crs, metadata=dict(data.metadata))
    return df


def simulate_telemetry(object: Telemetry, CTMM: CTMMModel, **kwargs):
    return simulate_ctmm(CTMM, data=object, **kwargs)


def predict_telemetry(data: Telemetry, CTMM: CTMMModel | None = None, t=None, complete: bool = False, **kwargs):
    del CTMM, complete, kwargs
    df = data.data.sort_values(data.time_col).reset_index(drop=True)
    src_t = _epoch_seconds(df[data.time_col])
    if t is None:
        return data
    t = np.asarray(t, dtype=float)
    out = pd.DataFrame({data.time_col: pd.to_datetime(t, unit="s", utc=True)})
    out[data.x_col] = np.interp(t, src_t, df[data.x_col].to_numpy(dtype=float))
    out[data.y_col] = np.interp(t, src_t, df[data.y_col].to_numpy(dtype=float))
    out[data.id_col] = df[data.id_col].iloc[0] if data.id_col in df else "animal"
    return Telemetry(out, id_col=data.id_col, time_col=data.time_col, x_col=data.x_col, y_col=data.y_col, crs=data.crs, metadata=dict(data.metadata))


def predict_ctmm(object: CTMMModel, data=None, t=None, **kwargs):
    if isinstance(data, Telemetry):
        return predict_telemetry(data, CTMM=object, t=t, **kwargs)
    if data is None:
        return simulate_ctmm(object, t=t, **kwargs)
    return predict(data, t=t, **kwargs)


def predict(data, t=None, x=None, CTMM: CTMMModel | None = None, **kwargs):
    if isinstance(data, CTMMModel):
        return predict_ctmm(data, t=t, **kwargs)
    if isinstance(data, Telemetry):
        return predict_telemetry(data, CTMM=CTMM, t=t, **kwargs)
    arr = np.asarray(data, dtype=float)
    if t is None:
        return arr
    t = np.asarray(t, dtype=float)
    if x is None:
        x = np.arange(arr.shape[0], dtype=float)
    x = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return np.interp(t, x, arr)
    return np.column_stack([np.interp(t, x, arr[:, k]) for k in range(arr.shape[1])])


__all__ = [
    "COVM",
    "fill_data",
    "predict",
    "predict_ctmm",
    "predict_telemetry",
    "simulate_ctmm",
    "simulate_telemetry",
    "smoother",
]
