from __future__ import annotations

import numpy as np

from .types import Telemetry
from .core_math import composite, clamp, pad, rpad, FFT, IFFT


def _epoch_seconds(col) -> np.ndarray:
    v = col.astype("int64").to_numpy()
    m = float(np.nanmedian(np.abs(v))) if v.size else 0.0
    if m > 1e17:   # ns
        return v / 1e9
    if m > 1e14:   # us
        return v / 1e6
    if m > 1e11:   # ms
        return v / 1e3
    return v.astype(float)


def variogram(
    telem: Telemetry,
    *,
    dt: float | None = None,
    max_lag_s: float | None = None,
    bins: int = 25,
) -> dict:
    """ctmm-style FFT variogram (Markov CI path, no calibration-error path)."""
    df = telem.data
    t = _epoch_seconds(df[telem.time_col])
    x = df[telem.x_col].to_numpy(dtype=float)
    y = df[telem.y_col].to_numpy(dtype=float)
    z = np.column_stack([x, y]).astype(float)
    col = z.shape[1]
    nobs = len(t)
    if nobs < 3:
        return {"lags_s": np.array([]), "gamma": np.array([]), "counts": np.array([])}

    # merge duplicate timestamps by averaging positions, mirroring variogram.dt preprocessing.
    order = np.argsort(t)
    t = t[order]
    z = z[order]
    uniq_t, inv = np.unique(t, return_inverse=True)
    z_u = np.zeros((len(uniq_t), col), dtype=float)
    c_u = np.zeros(len(uniq_t), dtype=float)
    for i, g in enumerate(inv):
        z_u[g] += z[i]
        c_u[g] += 1.0
    z_u /= np.maximum(c_u[:, None], 1.0)
    t = uniq_t
    z = z_u

    dti = np.diff(t)
    dti = dti[np.isfinite(dti) & (dti > 0)]
    dt0 = float(np.median(dti)) if dti.size else 1.0
    if dt is not None and float(dt) > 0:
        dt0 = float(dt)
    dt0 = max(dt0, 1e-9)

    # ---- grid.init / pregridder / gridder ----
    w_inner = clamp(np.diff(t) / dt0, 0.0, 1.0)
    w = (np.r_[1.0, w_inner] + np.r_[w_inner, 1.0]) / 2.0
    theta = (2.0 * np.pi / dt0) * t
    sinv = float(np.dot(w, np.sin(theta)))
    cosv = float(np.dot(w, np.cos(theta)))
    q = np.divide(sinv, cosv)
    t0 = -dt0 / (2.0 * np.pi) * np.arctan(q)
    t0 = -np.round((t0 - t[0]) / dt0) * dt0
    tt = t - t0
    index = tt / dt0
    index = index - np.round(index[0])
    while index[0] < 1:
        index += 1
    while index[0] >= 2:
        index -= 1
    floor_idx = np.floor(index).astype(int)
    p = 1.0 - (index - floor_idx)
    ngrid = int(np.ceil(index[-1])) + 1
    lag_grid = np.arange(ngrid, dtype=float) * dt0

    Wg = np.zeros(ngrid, dtype=float)
    Zg = np.zeros((ngrid, col), dtype=float)
    for i in range(len(t)):
        j1 = floor_idx[i]
        j2 = j1 + 1
        if 0 <= j1 < ngrid:
            w1 = p[i] * w[i]
            Wg[j1] += w1
            Zg[j1] += w1 * z[i]
        if 0 <= j2 < ngrid:
            w2 = (1.0 - p[i]) * w[i]
            Wg[j2] += w2
            Zg[j2] += w2 * z[i]
    pos = Wg > 0
    Zg[pos] /= Wg[pos, None]
    Wg = clamp(Wg, 0.0, 1.0)

    # ---- variogram.fast FFT core ----
    n = len(lag_grid)
    N = composite(2 * n)
    dfw = FFT(pad(Wg, size=N))
    ind = np.sign(Wg)
    wfft = np.conj(FFT(pad(ind, size=N)))
    zpad = rpad(Zg, size=N)
    zfft = FFT(zpad)
    zzfft = FFT(zpad * zpad)

    dof_ind = np.rint(np.real(IFFT(np.abs(wfft) ** 2)[:n])).astype(float)
    svf_raw = np.real(
        IFFT(
            np.real(wfft * np.sum(zzfft, axis=1)) - np.sum(np.abs(zfft) ** 2, axis=1)
        )[:n]
    )
    dof0 = col * dof_ind
    svf = np.zeros_like(svf_raw)
    good = dof0 > 0
    svf[good] = svf_raw[good] / dof0[good]
    svf[~good] = 0.0
    if svf.size:
        svf[0] = 0.0

    dof = col * np.real(IFFT(np.abs(dfw) ** 2)[:n])

    # Markov CI effective DOF cap
    dof_cap = np.zeros_like(dof)
    dof_cap[0] = col * len(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        dof_cap[1:] = col * (t[-1] - t[0]) / np.maximum(lag_grid[1:], 1e-12)
    dof = np.minimum(dof, dof_cap)

    if max_lag_s is not None and max_lag_s > 0:
        keep = lag_grid <= float(max_lag_s)
        lag_grid = lag_grid[keep]
        svf = svf[keep]
        dof = dof[keep]

    return {"lags_s": lag_grid, "gamma": svf, "counts": np.rint(dof).astype(int)}


def correlogram(telem: Telemetry, *, max_lag_s: float | None = None, bins: int = 25) -> dict:
    """Empirical position correlogram derived from variogram and sample variance."""
    vg = variogram(telem, max_lag_s=max_lag_s, bins=bins)
    if len(vg["lags_s"]) == 0:
        return {"lags_s": np.array([]), "acf": np.array([]), "counts": np.array([])}

    df = telem.data
    x = df[telem.x_col].to_numpy(dtype=float)
    y = df[telem.y_col].to_numpy(dtype=float)
    var_tot = float(np.var(x) + np.var(y))
    if not np.isfinite(var_tot) or var_tot <= 0:
        acf = np.zeros_like(vg["gamma"], dtype=float)
    else:
        # gamma(h)=0.5*E[(X(t+h)-X(t))^2], so ACF≈1-gamma/var
        acf = 1.0 - (np.asarray(vg["gamma"], dtype=float) / var_tot)
    return {"lags_s": vg["lags_s"], "acf": np.clip(acf, -1.0, 1.0), "counts": vg["counts"]}


def subset_variogram(x, *args, **kwargs):
    del kwargs
    if hasattr(x, "iloc"):
        return x.iloc[args[0]] if args else x
    return x

def variogram_dt(data):
    vg = variogram(data)
    lags = np.asarray(vg.get("lags_s", []), dtype=float)
    if lags.size < 2:
        return float("nan")
    d = np.diff(lags)
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.min(d)) if d.size else float("nan")


def grid_init(*args, **kwargs):
    return {"args": args, "kwargs": kwargs}


def pregridder(*args, **kwargs):
    return grid_init(*args, **kwargs)


def gridder(*args, **kwargs):
    return grid_init(*args, **kwargs)


def variogram_fast(data, *args, **kwargs):
    return variogram(data, *args, **kwargs)


def variogram_slow(data, *args, **kwargs):
    return variogram(data, *args, **kwargs)


def accumulate(values, weights=None):
    v = np.asarray(values, dtype=float)
    if weights is None:
        return np.cumsum(v)
    return np.cumsum(v * np.asarray(weights, dtype=float))


def variogram_ci(vg, level: float = 0.95):
    del level
    return vg


def mean_variogram(x, *args, **kwargs):
    del args, kwargs
    if not x:
        return {}
    keys = set().union(*(d.keys() for d in x if isinstance(d, dict)))
    out = {}
    for k in keys:
        vals = [np.asarray(d[k], dtype=float) for d in x if isinstance(d, dict) and k in d]
        if vals and all(v.shape == vals[0].shape for v in vals):
            out[k] = np.nanmean(np.stack(vals, axis=0), axis=0)
    return out


def mean_info(x):
    return {"n": len(x) if hasattr(x, "__len__") else 1}


__all__ = [
    "accumulate",
    "correlogram",
    "grid_init",
    "gridder",
    "mean_info",
    "mean_variogram",
    "pregridder",
    "subset_variogram",
    "variogram",
    "variogram_ci",
    "variogram_dt",
    "variogram_fast",
    "variogram_slow",
]
