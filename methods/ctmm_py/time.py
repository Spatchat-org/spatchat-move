"""Partial parity translation of ctmm 1.3.0 ``R/time.R``."""
from __future__ import annotations

import builtins
from copy import deepcopy
from typing import Any, Callable

import numpy as np
import pandas as pd

from .types import CTMMModel, Telemetry
from .transition import transition


def _params(obj: Any) -> dict[str, Any]:
    return obj.params if isinstance(obj, CTMMModel) else obj


def _copy_with_params(obj: Any, params: dict[str, Any]):
    if isinstance(obj, CTMMModel):
        return CTMMModel(model=obj.model, params=params)
    return params


def _get(params: dict[str, Any], key: str, default=None):
    if key in params:
        return params[key]
    return params.get(key.replace(".", "_"), default)


def _set(params: dict[str, Any], key: str, value):
    params[key] = value
    params[key.replace(".", "_")] = value


def _timelink_par(CTMM) -> np.ndarray:
    params = _params(CTMM)
    value = _get(params, "timelink.par", [])
    arr = np.asarray([] if value is None else value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _frame(data) -> pd.DataFrame:
    if isinstance(data, Telemetry):
        return data.data
    return data


def lag_seconds(timestamps):
    t = pd.to_datetime(timestamps, utc=True, errors="coerce").astype("int64").to_numpy(dtype=float) / 1e9
    d = np.diff(t)
    return d[np.isfinite(d)]


def linktime(data, CTMM):
    params = _params(CTMM)
    timelink = params.get("timelink", "identity")
    par = _timelink_par(CTMM)
    df = _frame(data)
    if timelink in (None, "identity") or par.size == 0:
        if isinstance(data, Telemetry):
            return pd.to_datetime(df[data.time_col], utc=True, errors="coerce").astype("int64").to_numpy(dtype=float) / 1e9
        if "t" in df:
            return np.asarray(df["t"], dtype=float)
        return pd.to_datetime(df["timestamp"], utc=True, errors="coerce").astype("int64").to_numpy(dtype=float) / 1e9
    if timelink == "switch":
        if not {"light.time", "dark.time"}.issubset(df.columns):
            return linktime(data, _copy_with_params(CTMM, {**params, "timelink": "identity", "timelink.par": []}))
        p = float(np.clip(par[0], -1.0, 1.0))
        light = pd.to_numeric(df["light.time"], errors="coerce").to_numpy(dtype=float)
        dark = pd.to_numeric(df["dark.time"], errors="coerce").to_numpy(dtype=float)
        return light * (1.0 + p) + dark * (1.0 - p)
    return linktime(data, _copy_with_params(CTMM, {**params, "timelink": "identity", "timelink.par": []}))


def timelink_parinfo(CTMM):
    name = _params(CTMM).get("timelink", "identity")
    par = _timelink_par(CTMM)
    if name in (None, "identity") or par.size == 0:
        return {}
    return globals()[f"{name}_timelink_parinfo"](CTMM)


def timelink_clean(par, timelink: str = "identity"):
    arr = np.asarray(par, dtype=float)
    if timelink in (None, "identity") or arr.size == 0:
        return arr
    return globals()[f"{timelink}_timelink_clean"](arr)


def timelink_rate(data, CTMM):
    params = _params(CTMM)
    name = params.get("timelink", "identity")
    par = _timelink_par(CTMM)
    df = _frame(data)
    if name in (None, "identity") or par.size == 0:
        return np.ones(len(df), dtype=float)
    if name == "switch":
        return switch_timelink_rate(data, CTMM)
    R = timelink_fn(CTMM)
    if "sundial" not in df:
        return np.ones(len(df), dtype=float)
    return R["fn"](pd.to_numeric(df["sundial"], errors="coerce").to_numpy(dtype=float))


def timelink_fn(CTMM) -> dict[str, Callable]:
    params = _params(CTMM)
    name = params.get("timelink", "identity")
    par = _timelink_par(CTMM)
    if name in (None, "identity") or par.size == 0:
        return {"fn": lambda angle: np.ones_like(np.asarray(angle, dtype=float)), "grad": lambda angle: np.zeros_like(np.asarray(angle, dtype=float))}
    return globals()[f"{name}_timelink_fn"](CTMM)


def timelink_complexify(CTMM):
    params = deepcopy(_params(CTMM))
    name = params.get("timelink", "identity")
    if name in (None, "identity"):
        return _copy_with_params(CTMM, params)
    return globals()[f"{name}_timelink_complexify"](_copy_with_params(CTMM, params))


def timelink_simplify(CTMM):
    params = deepcopy(_params(CTMM))
    name = params.get("timelink", "identity")
    par = _timelink_par(CTMM)
    if name in (None, "identity") or par.size == 0:
        return _copy_with_params(CTMM, params)
    return globals()[f"{name}_timelink_simplify"](_copy_with_params(CTMM, params))


def timelink_name(CTMM):
    params = _params(CTMM)
    name = params.get("timelink", "identity")
    par = _timelink_par(CTMM)
    if name in (None, "identity") or par.size == 0:
        return None
    return globals()[f"{name}_timelink_name"](CTMM)


def switch_timelink_parinfo(CTMM):
    return {"lower": -1.0, "upper": 1.0, "parscale": 1.0}


def switch_timelink_clean(par):
    return np.clip(np.asarray(par, dtype=float), -1.0, 1.0)


def switch_timelink_rate(data, CTMM):
    p = float(np.clip(_timelink_par(CTMM)[0], -1.0, 1.0))
    df = _frame(data)
    if "light" not in df:
        return np.ones(len(df), dtype=float)
    light = np.asarray(df["light"], dtype=bool)
    return np.where(light, 1.0 + p, 1.0 - p)


def switch_timelink_fn(CTMM):
    p = float(np.clip(_timelink_par(CTMM)[0], -1.0, 1.0))
    return {
        "fn": lambda angle: np.where((0 <= np.asarray(angle)) & (np.asarray(angle) < np.pi), 1.0 + p, 1.0 - p),
        "grad": lambda angle: np.where((0 <= np.asarray(angle)) & (np.asarray(angle) < np.pi), 1.0, -1.0),
    }


def switch_timelink_complexify(CTMM):
    params = deepcopy(_params(CTMM))
    _set(params, "timelink.par", np.asarray([0.0]))
    return _copy_with_params(CTMM, params)


def switch_timelink_simplify(CTMM):
    params = deepcopy(_params(CTMM))
    _set(params, "timelink.par", np.asarray([], dtype=float))
    return _copy_with_params(CTMM, params)


def switch_timelink_name(CTMM):
    return "diel-switch"


def spline_timelink_fn(CTMM, even: bool = False, half: bool = False, fast: bool = False):
    del even, half, fast
    y0 = _timelink_par(CTMM)
    p0 = y0.size
    y = np.r_[p0 + 1.0 - np.sum(y0), y0]
    p = y.size
    h = 2.0 * np.pi / p
    yn = np.r_[y[1:], y[0]]
    yp = np.r_[y[-1], y[:-1]]
    M = np.zeros((3 * p, 3 * p), dtype=float)
    b = np.r_[yn - y, np.zeros(2 * p, dtype=float)]
    for i in range(p):
        M[i, 3 * i : 3 * i + 3] = [h, h * h, h**3]
        row = p + i
        M[row, 3 * i : 3 * i + 3] += [1.0, 2.0 * h, 3.0 * h * h]
        M[row, 3 * ((i + 1) % p)] -= 1.0
        row = 2 * p + i
        M[row, 3 * i : 3 * i + 3] += [0.0, 2.0, 6.0 * h]
        M[row, 3 * ((i + 1) % p) + 1] -= 2.0
    coeff = np.linalg.solve(M, b).reshape(p, 3)
    Q = np.column_stack([y, coeff])

    def _parts(angle):
        a = (np.asarray(angle, dtype=float) - np.pi / 2.0) % (2.0 * np.pi)
        da = a % h
        idx = np.floor((a - da) / h + 0.5).astype(int) % p
        return idx, da

    def fn(angle):
        idx, da = _parts(angle)
        return np.sum(Q[idx] * np.column_stack([np.ones_like(da), da, da**2, da**3]), axis=1)

    return {"fn": fn, "grad": lambda angle: np.zeros((p0, np.asarray(angle).size), dtype=float)}


def spline_timelink_parinfo(CTMM):
    p0 = _timelink_par(CTMM).size
    return {"lower": 0.0, "upper": float(p0 + 1), "parscale": 1.0}


def spline_timelink_clean(par):
    arr = np.maximum(np.asarray(par, dtype=float), 0.0)
    total = np.sum(arr)
    if total > arr.size + 1:
        arr = arr * (arr.size + 1.0) / total
    return arr


def spline_timelink_complexify(CTMM):
    params = deepcopy(_params(CTMM))
    par = _timelink_par(CTMM)
    p = par.size + 2
    theta = np.linspace(np.pi / 2.0, 5.0 * np.pi / 2.0, p + 2)[: p + 1]
    fn = spline_timelink_fn(CTMM)["fn"]
    new = fn(theta)
    new = new / np.mean(new)
    _set(params, "timelink.par", new[1:])
    return _copy_with_params(CTMM, params)


def spline_timelink_simplify(CTMM):
    params = deepcopy(_params(CTMM))
    par = _timelink_par(CTMM)
    if par.size <= 1:
        _set(params, "timelink.par", np.asarray([], dtype=float))
        return _copy_with_params(CTMM, params)
    p = par.size
    theta = np.linspace(np.pi / 2.0, 5.0 * np.pi / 2.0, p + 1)[:p]
    fn = spline_timelink_fn(CTMM)["fn"]
    new = fn(theta)
    new = new / np.mean(new)
    _set(params, "timelink.par", new[1:])
    return _copy_with_params(CTMM, params)


def spline_timelink_name(CTMM):
    n = _timelink_par(CTMM).size
    return f"spline-timelink {n}" if n else None


def fourier_timelink_fn(CTMM):
    par = _timelink_par(CTMM)
    p = par.size // 2

    def fn(sundial):
        s = np.asarray(sundial, dtype=float)
        r = np.ones_like(s, dtype=float)
        for i in range(1, p + 1):
            if i % 2:
                r = r + par[2 * i - 2] * np.sin(i * s) + par[2 * i - 1] * np.cos(i * s)
            else:
                r = r + par[2 * i - 2] * np.cos(i * s) + par[2 * i - 1] * np.sin(i * s)
        return r

    def grad(sundial):
        s = np.asarray(sundial, dtype=float)
        cols = []
        for i in range(1, p + 1):
            cols.extend([np.sin(i * s), np.cos(i * s)] if i % 2 else [np.cos(i * s), np.sin(i * s)])
        return np.vstack(cols) if cols else np.zeros((0, s.size), dtype=float)

    return {"fn": fn, "grad": grad}


def fourier_timelink_complexify(CTMM):
    params = deepcopy(_params(CTMM))
    _set(params, "timelink.par", np.r_[_timelink_par(CTMM), 0.0, 0.0])
    return _copy_with_params(CTMM, params)


def fourier_timelink_simplify(CTMM):
    params = deepcopy(_params(CTMM))
    par = _timelink_par(CTMM)
    _set(params, "timelink.par", par[:-2] if par.size > 2 else np.asarray([], dtype=float))
    return _copy_with_params(CTMM, params)


def fourier_timelink_name(CTMM):
    return f"fourier-timelink {_timelink_par(CTMM).size // 2}"


def cosine_timelink_fn(CTMM):
    par = _timelink_par(CTMM)

    def fn(sundial):
        s = np.asarray(sundial, dtype=float)
        r = np.ones_like(s, dtype=float)
        for i, p in enumerate(par, start=1):
            r = r + p * (np.sin(i * s) if i % 2 else np.cos(i * s))
        return r

    def grad(sundial):
        s = np.asarray(sundial, dtype=float)
        cols = [np.sin(i * s) if i % 2 else np.cos(i * s) for i in range(1, par.size + 1)]
        return np.vstack(cols) if cols else np.zeros((0, s.size), dtype=float)

    return {"fn": fn, "grad": grad}


def cosine_timelink_complexify(CTMM):
    params = deepcopy(_params(CTMM))
    _set(params, "timelink.par", np.r_[_timelink_par(CTMM), 0.0])
    return _copy_with_params(CTMM, params)


def cosine_timelink_simplify(CTMM):
    return switch_timelink_simplify(CTMM)


def cosine_timelink_name(CTMM):
    return f"cosine-timelink {_timelink_par(CTMM).size}"


def switch_timelink_summary(object, level: float = 0.95):
    par = float(_timelink_par(object)[0]) if _timelink_par(object).size else 0.0
    pct = 100.0 * (1.0 + abs(par)) / 2.0
    name = "% diurnal" if par > 0 else ("% nocturnal" if par < 0 else "% cathemeral")
    return {"rowname": name, "CI": np.asarray([[max(0.0, pct), pct, min(100.0, pct)]], dtype=float), "colnames": ("low", "est", "high")}


def spline_timelink_summary(object, level: float = 0.95):
    del object, level
    return None


def fourier_timelink_summary(object, level: float = 0.95):
    del object, level
    return None


def cosine_timelink_summary(object, level: float = 0.95):
    del object, level
    return None


def timelink_summary(CTMM, level: float = 0.95):
    params = _params(CTMM)
    name = params.get("timelink", "identity")
    par = _timelink_par(CTMM)
    if name in (None, "identity") or par.size == 0:
        return None
    fn = globals().get(f"{name}_timelink_summary")
    return fn(CTMM, level=level) if fn is not None else None


def get_sundial(object, CTMM=None, twilight: str = "civil", dt_max: float = 6 * 3600.0):
    del CTMM, twilight, dt_max
    df = _frame(object).copy()
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    elif hasattr(object, "time_col"):
        ts = pd.to_datetime(df[object.time_col], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce")
    seconds = ts.dt.hour.to_numpy(dtype=float) * 3600.0 + ts.dt.minute.to_numpy(dtype=float) * 60.0 + ts.dt.second.to_numpy(dtype=float)
    angle = (seconds / 86400.0 * 2.0 * np.pi) % (2.0 * np.pi)
    light = (angle > 0.0) & (angle <= np.pi)
    light_dt = np.zeros(angle.size, dtype=float)
    dark_dt = np.zeros(angle.size, dtype=float)
    if angle.size > 1:
        if "t" in df.columns:
            tt = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
        else:
            tt = ts.astype("int64").to_numpy(dtype=float) / 1e9
        dt = np.r_[0.0, np.maximum(np.diff(tt), 0.0)]
        light_dt = np.cumsum(np.where(light, dt, 0.0))
        dark_dt = np.cumsum(np.where(~light, dt, 0.0))
    return pd.DataFrame({"light.time": light_dt, "dark.time": dark_dt, "light": light, "sundial": angle, "suntime": np.full(angle.size, 12 * 3600.0)})


def circadian(CTMM, level: float = 0.95, n: int = 100, **kwargs):
    del level, kwargs
    theta = np.linspace(0.0, 2.0 * np.pi, builtins.int(n), dtype=float)
    rate = timelink_fn(CTMM)["fn"](theta)
    return {"theta": theta, "rate": np.asarray(rate, dtype=float)}


def int(angle, CTMM=None):
    if CTMM is None:
        return np.asarray(angle, dtype=float)
    R = timelink_fn(CTMM)
    a = np.asarray(angle, dtype=float)
    grid = np.linspace(0.0, 2.0 * np.pi, 720, dtype=float)
    vals = np.asarray(R["fn"](grid), dtype=float)
    integ = np.r_[0.0, np.cumsum((vals[1:] + vals[:-1]) / 2.0 * np.diff(grid))]
    return np.interp(a % (2.0 * np.pi), grid, integ)


__all__ = [
    "transition",
    "lag_seconds",
    "get_sundial",
    "linktime",
    "timelink_parinfo",
    "timelink_clean",
    "timelink_rate",
    "timelink_fn",
    "timelink_complexify",
    "timelink_simplify",
    "timelink_name",
    "timelink_summary",
    "switch_timelink_summary",
    "spline_timelink_summary",
    "fourier_timelink_summary",
    "cosine_timelink_summary",
    "circadian",
    "int",
]
