"""Partial parity translation of ctmm 1.3.0 ``R/mean.R``."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .types import CTMMModel
from .median import median


def mean(x, axis=0):
    return np.nanmean(np.asarray(x, dtype=float), axis=axis)


def EST(t=None):
    arr = np.asarray(0.0 if t is None else t, dtype=float)
    return np.zeros_like(arr, dtype=float)


def VAR(t=None):
    arr = np.asarray(0.0 if t is None else t, dtype=float)
    return np.zeros_like(arr, dtype=float)


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


def _call(fn: str, CTMM, *args, **kwargs):
    name = str(_params(CTMM).get("mean", "stationary") or "stationary").replace(".", "_").replace("-", "_")
    func = globals().get(f"{name}_{fn}", globals()[f"stationary_{fn}"])
    return func(CTMM, *args, **kwargs)


def drift_fn(fn: str, CTMM, *args, **kwargs):
    return _call(fn.replace(".", "_"), CTMM, *args, **kwargs)


def drift_name(CTMM, *args, **kwargs):
    return _call("name", CTMM, *args, **kwargs)


def drift_pars(CTMM, *args, **kwargs):
    return _call("pars", CTMM, *args, **kwargs)


def drift_assign(CTMM, *args, **kwargs):
    return _call("assign", CTMM, *args, **kwargs)


def drift_mean(CTMM, *args, **kwargs):
    return _call("mean", CTMM, *args, **kwargs)


def drift_velocity(CTMM, *args, **kwargs):
    return _call("velocity", CTMM, *args, **kwargs)


def drift_energy(CTMM, *args, **kwargs):
    return _call("energy", CTMM, *args, **kwargs)


def drift_init(CTMM, *args, **kwargs):
    return _call("init", CTMM, *args, **kwargs)


def drift_shift(CTMM, *args, **kwargs):
    return _call("shift", CTMM, *args, **kwargs)


def drift_svf(CTMM, *args, **kwargs):
    return _call("svf", CTMM, *args, **kwargs)


def drift_complexify(CTMM, *args, **kwargs):
    return _call("complexify", CTMM, *args, **kwargs)


def drift_simplify(CTMM, *args, **kwargs):
    return _call("simplify", CTMM, *args, **kwargs)


def drift_is_stationary(CTMM, *args, **kwargs):
    return _call("is_stationary", CTMM, *args, **kwargs)


def drift_scale(CTMM, *args, **kwargs):
    return _call("scale", CTMM, *args, **kwargs)


def drift_speed(CTMM, *args, **kwargs):
    return _call("speed", CTMM, *args, **kwargs)


def drift_summary(CTMM, *args, **kwargs):
    return _call("summary", CTMM, *args, **kwargs)


def drift_is_finite(CTMM, *args, **kwargs):
    return _call("is_finite", CTMM, *args, **kwargs)


def stationary_name(CTMM, *args, **kwargs):
    return None


def zero_name(CTMM, *args, **kwargs):
    return "mean-zero"


def stationary_is_stationary(CTMM, *args, **kwargs):
    return True


def stationary_init(CTMM, data=None, *args, **kwargs):
    del args, kwargs
    if data is None:
        return CTMM
    params = deepcopy(_params(CTMM))
    if params.get("mu") is not None:
        return CTMM
    if hasattr(data, "data"):
        df = data.data
        cols = [getattr(data, "x_col", "x"), getattr(data, "y_col", "y")]
        cols = [c for c in cols if c in df.columns]
        if cols:
            params["mu"] = np.nanmean(df[cols].to_numpy(dtype=float), axis=0).reshape(1, -1)
    return _copy_with_params(CTMM, params)


def stationary_mean(CTMM, t, *args, **kwargs):
    t = np.asarray(t, dtype=float)
    return np.ones((t.size, 1), dtype=float)


def zero_mean(CTMM, t, *args, **kwargs):
    t = np.asarray(t, dtype=float)
    return np.zeros((t.size, 0), dtype=float)


def stationary_velocity(CTMM, t, *args, **kwargs):
    t = np.asarray(t, dtype=float)
    return np.zeros((t.size, 1), dtype=float)


def zero_velocity(CTMM, t, *args, **kwargs):
    t = np.asarray(t, dtype=float)
    return np.zeros((t.size, 0), dtype=float)


def stationary_assign(CTMM, value):
    return CTMM


def stationary_pars(CTMM, all: bool = False, **kwargs):
    return {"NAMES": [], "pars": np.array([]), "parscale": np.array([]), "lower": np.array([]), "upper": np.array([])} if all else None


def stationary_simplify(CTMM, *args, **kwargs):
    return []


def stationary_complexify(CTMM, *args, **kwargs):
    return []


def stationary_scale(CTMM, time, *args, **kwargs):
    return CTMM


def stationary_shift(CTMM, dmu, *args, **kwargs):
    del args, kwargs
    params = deepcopy(_params(CTMM))
    mu = np.asarray(params.get("mu", []), dtype=float)
    if mu.size:
        if mu.ndim == 1:
            mu = mu.reshape(1, -1)
        d = np.asarray(dmu, dtype=float).reshape(-1)
        n = min(mu.shape[1], d.size)
        mu[0, :n] += d[:n]
        params["mu"] = mu
    return _copy_with_params(CTMM, params)


def uniform_shift(CTMM, dmu, *args, **kwargs):
    del args, kwargs
    params = deepcopy(_params(CTMM))
    mu = np.asarray(params.get("mu", []), dtype=float)
    if mu.size:
        if mu.ndim == 1:
            mu = mu.reshape(1, -1)
        d = np.asarray(dmu, dtype=float).reshape(-1)
        if d.size < mu.shape[1]:
            d = np.pad(d, (0, mu.shape[1] - d.size))
        params["mu"] = mu + d[: mu.shape[1]][None, :]
    return _copy_with_params(CTMM, params)


def stationary_summary(CTMM, level=0.95, level_UD=0.95, *args, **kwargs):
    del CTMM, level, level_UD, args, kwargs
    return None


def stationary_svf(CTMM, *args, **kwargs):
    del CTMM, args, kwargs
    return {
        "EST": lambda t: np.zeros_like(np.asarray(t, dtype=float)),
        "VAR": lambda t: np.zeros_like(np.asarray(t, dtype=float)),
    }


def stationary_speed(CTMM, *args, **kwargs):
    del CTMM, args, kwargs
    return {"EST": 0.0, "VAR": 0.0}


def stationary_energy(CTMM, *args, **kwargs):
    del CTMM, args, kwargs
    return {"UU": np.asarray([[1.0]], dtype=float), "VV": np.asarray([[0.0]], dtype=float)}


def zero_energy(CTMM, *args, **kwargs):
    del CTMM, args, kwargs
    return {"UU": np.zeros((0, 0), dtype=float), "VV": np.zeros((0, 0), dtype=float)}


def stationary_is_finite(CTMM, *args, **kwargs):
    del CTMM, args, kwargs
    return True


def _periodic_arrays(params: dict[str, Any]):
    period = np.asarray(_get(params, "period", [86400.0]), dtype=float).reshape(-1)
    harmonic = np.asarray(_get(params, "harmonic", np.zeros(period.size)), dtype=float).reshape(-1)
    if harmonic.size == 0:
        harmonic = np.zeros(period.size, dtype=float)
    if period.size == 1 and harmonic.size > 1:
        period = np.repeat(period, harmonic.size)
    if harmonic.size == 1 and period.size > 1:
        harmonic = np.repeat(harmonic, period.size)
    return period, np.floor(np.maximum(harmonic, 0)).astype(int)


def periodic_name(CTMM, *args, **kwargs):
    _, harmonic = _periodic_arrays(_params(CTMM))
    return "harmonic " + " ".join(str(int(h)) for h in harmonic)


def periodic_is_stationary(CTMM, *args, **kwargs):
    _, harmonic = _periodic_arrays(_params(CTMM))
    return bool(np.sum(harmonic) == 0)


def periodic_omega(CTMM):
    period, harmonic = _periodic_arrays(_params(CTMM))
    out: list[float] = []
    for p, h in zip(period, harmonic):
        if np.isfinite(p) and p > 0:
            out.extend((2.0 * np.pi / p) * k for k in range(1, int(h) + 1))
    return np.asarray(out, dtype=float)


def periodic_namer(CTMM):
    n = int(np.sum(_periodic_arrays(_params(CTMM))[1]))
    names = ["1"]
    for i in range(1, n + 1):
        names.extend([f"Sin.{i}", f"Cos.{i}"])
    return names


def periodic_mean(CTMM, t, verbose: bool = True, *args, **kwargs):
    t = np.asarray(t, dtype=float)
    cols = [np.ones(t.size, dtype=float)]
    for omega in periodic_omega(CTMM):
        theta = t * omega
        cols.extend([np.sin(theta), np.cos(theta)])
    return np.column_stack(cols)


def periodic_velocity(CTMM, t, *args, **kwargs):
    t = np.asarray(t, dtype=float)
    cols = [np.zeros(t.size, dtype=float)]
    for omega in periodic_omega(CTMM):
        theta = t * omega
        cols.extend([omega * np.cos(theta), -omega * np.sin(theta)])
    return np.column_stack(cols)


def periodic_assign(CTMM, value):
    params = deepcopy(_params(CTMM))
    val = np.asarray(value, dtype=float)
    params["period"] = 2.0 * np.pi / val[val != 0]
    return _copy_with_params(CTMM, params)


def periodic_pars(CTMM, all: bool = False, fit: bool = False, **kwargs):
    params = _params(CTMM)
    if fit and periodic_is_stationary(CTMM):
        return stationary_pars(CTMM, all=all, **kwargs)
    period = np.sort(np.asarray(_get(params, "period", []), dtype=float).reshape(-1))
    freq = 2.0 * np.pi / period if period.size else np.asarray([], dtype=float)
    names = [f"period.{i}" for i in range(1, freq.size + 1)]
    if all:
        return {"NAMES": names, "pars": freq, "parscale": freq, "lower": np.zeros(freq.size), "upper": np.full(freq.size, np.inf)}
    return freq


def periodic_simplify(CTMM, *args, **kwargs):
    params = _params(CTMM)
    period, harmonic = _periodic_arrays(params)
    guesses = []
    for i, h in enumerate(harmonic):
        if h > 0:
            p = deepcopy(params)
            hh = harmonic.copy()
            hh[i] -= 1
            p["period"] = period.copy()
            p["harmonic"] = hh
            guesses.append(_copy_with_params(CTMM, p))
    return guesses


def periodic_complexify(CTMM, *args, **kwargs):
    params = _params(CTMM)
    period, harmonic = _periodic_arrays(params)
    guesses = []
    for i in range(period.size):
        p = deepcopy(params)
        hh = harmonic.copy()
        hh[i] += 1
        p["period"] = period.copy()
        p["harmonic"] = hh
        guesses.append(_copy_with_params(CTMM, p))
    return guesses


def periodic_scale(CTMM, time, *args, **kwargs):
    params = deepcopy(_params(CTMM))
    params["period"] = np.asarray(_get(params, "period", []), dtype=float) / float(time)
    return _copy_with_params(CTMM, params)


def periodic_init(CTMM, data=None, *args, **kwargs):
    del args, kwargs
    params = deepcopy(_params(CTMM))
    params.setdefault("period", np.asarray([86400.0], dtype=float))
    period, harmonic = _periodic_arrays(params)
    params["period"] = period
    params["harmonic"] = harmonic
    if params.get("mu") is None:
        return stationary_init(_copy_with_params(CTMM, params), data=data)
    return _copy_with_params(CTMM, params)


def periodic_energy(CTMM):
    omega = periodic_omega(CTMM)
    k = omega.size
    if k == 0:
        return stationary_energy(CTMM)
    uu = np.diag(np.r_[1.0, np.repeat(0.5, 2 * k)])
    vv = np.diag(np.r_[0.0, np.ravel(np.column_stack([omega * omega / 2.0, omega * omega / 2.0]))])
    return {"UU": uu, "VV": vv}


def periodic_stuff(CTMM):
    params = _params(CTMM)
    omega = periodic_omega(CTMM)
    axes = len(params.get("axes", ("x", "y")))
    mu = np.asarray(params.get("mu", np.zeros((1 + 2 * omega.size, axes))), dtype=float)
    if mu.ndim == 1:
        mu = np.resize(mu, (max(1, int(np.ceil(mu.size / max(axes, 1)))), max(axes, 1)))
    a = mu[1:, :] if mu.shape[0] > 1 else np.zeros((0, mu.shape[1]), dtype=float)
    amp = a.reshape(-1)
    om = np.repeat(np.ravel(np.column_stack([omega, omega])), mu.shape[1])[: amp.size]
    cov = np.asarray(params.get("COV.mu", params.get("COV_mu", np.zeros((amp.size, amp.size)))), dtype=float)
    if cov.ndim < 2 or cov.shape != (amp.size, amp.size):
        cov = np.zeros((amp.size, amp.size), dtype=float)
    return {"A": amp, "COV": cov, "omega": om}


def periodic_speed(CTMM, *args, **kwargs):
    del args, kwargs
    if periodic_omega(CTMM).size == 0:
        return stationary_speed(CTMM)
    stuff = periodic_stuff(CTMM)
    omega = stuff["omega"]
    a = stuff["A"]
    cov = stuff["COV"]
    est = float(np.sum((omega * a) ** 2) / 2.0)
    grad = omega * omega * a
    var = float(grad @ cov @ grad) if cov.size else 0.0
    return {"EST": est, "VAR": var}


def periodic_svf(CTMM, *args, **kwargs):
    del args, kwargs
    if periodic_omega(CTMM).size == 0:
        return stationary_svf(CTMM)
    stuff = periodic_stuff(CTMM)
    omega = stuff["omega"]
    a = stuff["A"]
    cov = stuff["COV"]

    def est(t):
        tt = np.asarray(t, dtype=float).reshape(-1)
        return np.array([float(np.sum(0.25 * a * a * (1.0 - np.cos(omega * ti)))) for ti in tt], dtype=float)

    def var(t):
        tt = np.asarray(t, dtype=float).reshape(-1)
        out = []
        for ti in tt:
            grad = 0.5 * a * (1.0 - np.cos(omega * ti))
            out.append(float(grad @ cov @ grad) if cov.size else 0.0)
        return np.asarray(out, dtype=float)

    return {"EST": est, "VAR": var}


def periodic_variances(CTMM, *args, **kwargs):
    del args, kwargs
    stuff = periodic_stuff(CTMM)
    a = stuff["A"]
    cov = stuff["COV"]
    rot = float(np.sum(a * a) / 2.0)
    sig = np.asarray(_params(CTMM).get("sigma_matrix", np.eye(2)), dtype=float)
    ran = float(np.trace(sig)) if sig.ndim == 2 else float(np.nan)
    mle = rot / (rot + ran) if np.isfinite(ran) and (rot + ran) > 0 else 0.0
    cov_rot = float(a @ cov @ a) if cov.size else 0.0
    return {"R": {"MLE": mle, "VAR": cov_rot}, "V": {"MLE": 0.0, "VAR": np.inf}}


def periodic_summary(CTMM, level=0.95, level_UD=0.95, units=True, *args, **kwargs):
    del CTMM, level, level_UD, units, args, kwargs
    return None


def periodic_is_finite(CTMM, data=None, *args, **kwargs):
    del args, kwargs
    period = np.asarray(_get(_params(CTMM), "period", []), dtype=float).reshape(-1)
    if period.size == 0 or data is None:
        return True
    t = np.asarray(data.data.get("t", []), dtype=float) if hasattr(data, "data") else np.asarray(getattr(data, "t", []), dtype=float)
    if t.size < 2:
        return True
    return bool(float(np.nanmax(period)) <= 2.0 * np.pi * (float(np.nanmax(t)) - float(np.nanmin(t))) * 10.0)


def _change_points(params: dict[str, Any]):
    cp = _get(params, "change.point.mu", None)
    if cp is None:
        cp = _get(params, "change.point", None)
    if cp is None:
        return None
    try:
        import pandas as pd

        df = cp if hasattr(cp, "columns") else pd.DataFrame(cp)
        if "state" in df.columns:
            return df.copy()
    except Exception:
        return None
    return None


def change_point_name(CTMM, *args, **kwargs):
    del args, kwargs
    params = _params(CTMM)
    cp = _change_points(params)
    if cp is None:
        return None
    names = []
    for state in list(dict.fromkeys(cp["state"].astype(str))):
        sub = params.get(state)
        if sub is not None:
            name = drift_name(sub)
            if name:
                names.append(str(name))
    return "-".join(names) if names else None


def change_point_is_stationary(CTMM, *args, **kwargs):
    del args, kwargs
    cp = _change_points(_params(CTMM))
    return True if cp is None else int(cp["state"].astype(str).nunique()) <= 1


def change_point_init(CTMM, data=None, *args, **kwargs):
    del data, args, kwargs
    return CTMM


def change_point_mean(CTMM, t, velocity: bool = False, *args, **kwargs):
    del args, kwargs
    params = _params(CTMM)
    cp = _change_points(params)
    t = np.asarray(t, dtype=float)
    if cp is None or t.size == 0:
        return stationary_velocity(CTMM, t) if velocity else stationary_mean(CTMM, t)
    states = list(dict.fromkeys(cp["state"].astype(str)))
    widths = []
    mats = {}
    for state in states:
        sub = params.get(state, {"mean": "stationary"})
        mat = drift_velocity(sub, t) if velocity else drift_mean(sub, t)
        mats[state] = mat
        widths.append(mat.shape[1])
    out = np.zeros((t.size, int(np.sum(widths))), dtype=float)
    offsets = np.cumsum([0] + widths)
    for _, row in cp.iterrows():
        state = str(row["state"])
        start = float(row.get("start", -np.inf))
        stop = float(row.get("stop", np.inf))
        mask = (t >= start) & (t < stop)
        i = states.index(state)
        out[mask, offsets[i] : offsets[i + 1]] = mats[state][mask]
    return out


def change_point_velocity(CTMM, t, *args, **kwargs):
    return change_point_mean(CTMM, t, velocity=True, *args, **kwargs)


def change_point_complexify(CTMM, simplify: bool = False):
    params = _params(CTMM)
    cp = _change_points(params)
    if cp is None:
        return []
    guesses = []
    for state in list(dict.fromkeys(cp["state"].astype(str))):
        sub = params.get(state)
        if sub is not None:
            guesses.extend(drift_simplify(sub) if simplify else drift_complexify(sub))
    return guesses


def change_point_simplify(CTMM):
    return change_point_complexify(CTMM, simplify=True)


def change_point_scale(CTMM, time, *args, **kwargs):
    del args, kwargs
    params = deepcopy(_params(CTMM))
    cp = _change_points(params)
    if cp is not None:
        for state in list(dict.fromkeys(cp["state"].astype(str))):
            if state in params:
                params[state] = drift_scale(params[state], time)
    return _copy_with_params(CTMM, params)


def change_point_summary(CTMM, level=0.95, level_UD=0.95, *args, **kwargs):
    del CTMM, level, level_UD, args, kwargs
    return None


def change_point_svf(CTMM, speed: bool = False, *args, **kwargs):
    del args, kwargs
    params = _params(CTMM)
    cp = _change_points(params)
    if cp is None:
        return stationary_speed(CTMM) if speed else stationary_svf(CTMM)
    states = list(dict.fromkeys(cp["state"].astype(str)))
    duration = {s: 0.0 for s in states}
    for _, row in cp.iterrows():
        duration[str(row["state"])] += max(float(row.get("stop", 0.0)) - float(row.get("start", 0.0)), 0.0)
    total = sum(duration.values()) or 1.0
    fns = {s: (drift_speed(params.get(s, {"mean": "stationary"})) if speed else drift_svf(params.get(s, {"mean": "stationary"}))) for s in states}

    def est(t):
        tt = np.asarray(t, dtype=float)
        out = np.zeros_like(tt, dtype=float)
        for state in states:
            fn = fns[state]["EST"]
            out += (duration[state] / total) * np.asarray(fn(tt) if callable(fn) else fn, dtype=float)
        return out

    def var(t):
        tt = np.asarray(t, dtype=float)
        out = np.zeros_like(tt, dtype=float)
        for state in states:
            fn = fns[state]["VAR"]
            out += (duration[state] / total) ** 2 * np.asarray(fn(tt) if callable(fn) else fn, dtype=float)
        return out

    return {"EST": est, "VAR": var}


def change_point_speed(CTMM, *args, **kwargs):
    return change_point_svf(CTMM, speed=True, *args, **kwargs)


def change_point_energy(CTMM, *args, **kwargs):
    del args, kwargs
    params = _params(CTMM)
    cp = _change_points(params)
    if cp is None:
        return stationary_energy(CTMM)
    blocks_u = []
    blocks_v = []
    for state in list(dict.fromkeys(cp["state"].astype(str))):
        e = drift_energy(params.get(state, {"mean": "stationary"}))
        blocks_u.append(np.asarray(e["UU"], dtype=float))
        blocks_v.append(np.asarray(e["VV"], dtype=float))
    n = int(sum(b.shape[0] for b in blocks_u))
    uu = np.zeros((n, n), dtype=float)
    vv = np.zeros((n, n), dtype=float)
    off = 0
    for bu, bv in zip(blocks_u, blocks_v):
        k = bu.shape[0]
        uu[off : off + k, off : off + k] = bu
        vv[off : off + k, off : off + k] = bv
        off += k
    return {"UU": uu, "VV": vv}


def uspline_name(CTMM, *args, **kwargs):
    del args, kwargs
    p = _params(CTMM)
    return f"degree {_get(p, 'degree', 1)} knot {_get(p, 'knot', 1)}"


def uspline_is_stationary(CTMM, *args, **kwargs):
    del args, kwargs
    return bool(np.sum(np.asarray(_get(_params(CTMM), "knot", [1]), dtype=float)) == 0)


def uspline_init(CTMM, data=None, *args, **kwargs):
    del args, kwargs
    params = deepcopy(_params(CTMM))
    params.setdefault("degree", 1)
    params.setdefault("knot", 1)
    if "domain" not in params and data is not None and hasattr(data, "data") and "t" in data.data:
        tt = np.asarray(data.data["t"], dtype=float)
        params["domain"] = np.asarray([np.nanmin(tt), np.nanmax(tt)], dtype=float)
    return stationary_init(_copy_with_params(CTMM, params), data=data)


def uspline_stuff(CTMM, *args, **kwargs):
    del args, kwargs
    p = _params(CTMM)
    knot = int(np.asarray(_get(p, "knot", 1), dtype=float).reshape(-1)[0])
    domain = np.asarray(_get(p, "domain", [0.0, 1.0]), dtype=float).reshape(-1)
    if domain.size < 2:
        domain = np.asarray([0.0, 1.0], dtype=float)
    tknot = np.linspace(float(domain[0]), float(domain[-1]), max(knot, 1), dtype=float)
    dt = float((tknot[-1] - tknot[0]) / max(knot - 1, 1))
    return {"tknot": tknot, "dt": dt}


def uspline_mean(CTMM, t, *args, **kwargs):
    del args, kwargs
    p = _params(CTMM)
    degree = int(np.asarray(_get(p, "degree", 1), dtype=float).reshape(-1)[0])
    knot = int(np.asarray(_get(p, "knot", 1), dtype=float).reshape(-1)[0])
    t = np.asarray(t, dtype=float)
    stuff = uspline_stuff(CTMM)
    tknot = stuff["tknot"]
    dt = stuff["dt"] if stuff["dt"] > 0 else 1.0
    if knot <= 1:
        u = stationary_mean(CTMM, t)
        return u if degree == 1 else np.column_stack([u[:, 0], t - tknot[0]])
    u = np.zeros((t.size, knot, degree), dtype=float)
    for i, ti in enumerate(t):
        k = int(np.searchsorted(tknot, ti, side="right") - 1)
        k = max(0, min(k, knot - 2))
        s = (ti - tknot[k]) / dt
        if degree == 1:
            u[i, k, 0] = 1.0 - s
            u[i, k + 1, 0] = s
        else:
            u[i, k, 0] = 2 * s**3 - 3 * s**2 + 1
            u[i, k, 1] = (s**3 - 2 * s**2 + s) * dt
            u[i, k + 1, 0] = -2 * s**3 + 3 * s**2
            u[i, k + 1, 1] = (s**3 - s**2) * dt
    return u.reshape(t.size, knot * degree)


def uspline_complexify(CTMM):
    params = deepcopy(_params(CTMM))
    params["knot"] = int(np.asarray(_get(params, "knot", 1), dtype=float).reshape(-1)[0]) + 1
    return [_copy_with_params(CTMM, params)]


def uspline_scale(CTMM, time, *args, **kwargs):
    del args, kwargs
    params = deepcopy(_params(CTMM))
    params["domain"] = np.asarray(_get(params, "domain", [0.0, 1.0]), dtype=float) / float(time)
    return _copy_with_params(CTMM, params)


def uspline_speed(CTMM, *args, **kwargs):
    del CTMM, args, kwargs
    return {"EST": 0.0, "VAR": np.inf}


def pwstationary_is_stationary(CTMM):
    breaks = np.asarray(_get(_params(CTMM), "breaks", []), dtype=float)
    return breaks.size == 0


def pwstationary_mean(CTMM, t, *args, **kwargs):
    del args, kwargs
    t = np.asarray(t, dtype=float)
    breaks = np.asarray(_get(_params(CTMM), "breaks", []), dtype=float).reshape(-1)
    cols = [np.ones(t.size, dtype=float)]
    for b in breaks:
        cols.append((t >= b).astype(float))
    return np.column_stack(cols)


def pwstationary_scale(CTMM, time, *args, **kwargs):
    del args, kwargs
    params = deepcopy(_params(CTMM))
    params["breaks"] = np.asarray(_get(params, "breaks", []), dtype=float) / float(time)
    return _copy_with_params(CTMM, params)


__all__ = [
    "mean",
    "median",
    "EST",
    "VAR",
    "drift_fn",
    "drift_name",
    "drift_pars",
    "drift_assign",
    "drift_mean",
    "drift_velocity",
    "drift_energy",
    "drift_init",
    "drift_shift",
    "drift_svf",
    "drift_complexify",
    "drift_simplify",
    "drift_is_stationary",
    "drift_scale",
    "drift_speed",
    "drift_summary",
    "drift_is_finite",
    "periodic_omega",
    "periodic_namer",
    "periodic_speed",
    "periodic_energy",
    "periodic_stuff",
    "periodic_variances",
    "change_point_mean",
    "uspline_mean",
    "pwstationary_mean",
]
