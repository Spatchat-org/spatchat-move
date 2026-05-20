from __future__ import annotations

import numpy as np

from .types import Telemetry
from .core_math import pad, rpad, FFT


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


def periodogram(telem: Telemetry, *, detrend: bool = True) -> dict:
    """ctmm periodogram wrapper parity: default slow path when n < 1e4."""
    df = telem.data
    t = _epoch_seconds(df[telem.time_col])
    x = df[telem.x_col].to_numpy(dtype=float)
    y = df[telem.y_col].to_numpy(dtype=float)
    if len(t) < 4:
        return {"freq_hz": np.array([]), "power": np.array([])}

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return {"freq_hz": np.array([]), "power": np.array([])}

    sample_dt = float(np.median(dt))
    T = float(t[-1] - t[0])
    n_grid = int(round(T / sample_dt)) + 1
    n_grid = max(n_grid, 3)
    if detrend:
        x = x - np.mean(x)
        y = y - np.mean(y)
    z = np.column_stack([x, y]).astype(float)
    dfreq = 1.0 / (2.0 * n_grid * sample_dt)
    freq = np.arange(1, n_grid, dtype=float) * dfreq
    # ctmm default: n<1e4 -> slow periodogram path.
    if n_grid < 10**4:
        col = z.shape[1]
        theta2 = (4.0 * np.pi) * np.outer(freq, t)
        num_s = np.sum(np.sin(theta2), axis=1)
        num_c = np.sum(np.cos(theta2), axis=1)
        tau = np.arctan(np.divide(num_s, num_c)) / (4.0 * np.pi * np.maximum(freq, 1e-18))
        theta = (2.0 * np.pi) * (np.outer(freq, t) - (freq * tau)[:, None])
        COS = np.cos(theta)
        SIN = np.sin(theta)
        den_c = np.maximum(np.sum(COS * COS, axis=1), 1e-18)
        den_s = np.maximum(np.sum(SIN * SIN, axis=1), 1e-18)
        cz = COS @ z
        sz = SIN @ z
        LSP = np.sum(cz * cz, axis=1) / den_c + np.sum(sz * sz, axis=1) / den_s
        LSP = LSP / (2.0 * max(col, 1))
        return {"freq_hz": freq, "power": LSP, "dof": 0}

    # fast path (for large n)
    theta = (2.0 * np.pi / sample_dt) * (t - t[0])
    sinv = float(np.sum(np.sin(theta)))
    cosv = float(np.sum(np.cos(theta)))
    # ctmm uses atan(SIN/COS), not atan2.
    t_shift = -sample_dt / (2.0 * np.pi) * np.arctan(np.divide(sinv, cosv))
    idx = (t - t[0] - t_shift) / sample_dt
    if idx[0] < 0:
        idx = idx + 1.0
    n = max(int(np.ceil(np.max(idx))) + 2, n_grid)
    W = np.zeros(n, dtype=float)
    col = z.shape[1]
    Z = np.zeros((n, col), dtype=float)
    for i in range(len(idx)):
        fl = int(np.floor(idx[i]))
        p = float(idx[i] - fl)
        q = 1.0 - p
        if 0 <= fl < n:
            W[fl] += q
            Z[fl, :] += q * z[i, :]
        if 0 <= fl + 1 < n:
            W[fl + 1] += p
            Z[fl + 1, :] += p * z[i, :]
    N2 = 2 * n
    Wf = FFT(pad(W, size=N2))
    Zf = FFT(rpad(Z, size=N2))
    W2 = np.conj(np.r_[Wf, Wf][::2])[: N2]
    den = (Wf[0] ** 2 - np.abs(W2) ** 2)
    den = np.where(np.abs(den) < 1e-12, 1e-12 + 0j, den)
    LSP = np.real((Wf[0] * np.sum(np.abs(Zf) ** 2, axis=1) - W2 * np.sum(Zf ** 2, axis=1)) / den / max(col, 1))
    spec = np.asarray(LSP[1:n_grid], dtype=float)
    m = min(spec.shape[0], freq.shape[0])
    return {"freq_hz": freq[:m], "power": spec[:m], "dof": 0}


def subset_periodogram(x, *args, **kwargs):
    del kwargs
    if isinstance(x, dict):
        idx = args[0] if args else slice(None)
        return {k: (np.asarray(v)[idx] if np.asarray(v).ndim else v) for k, v in x.items()}
    return x

def periodogram_fast(data, *args, **kwargs):
    return periodogram(data, *args, **kwargs)


def periodogram_slow(data, *args, **kwargs):
    return periodogram(data, *args, **kwargs)


def max_periodogram(x):
    power = np.asarray(x.get("power", []), dtype=float)
    if power.size == 0:
        return {"frequency": np.nan, "power": np.nan}
    i = int(np.nanargmax(power))
    freq = np.asarray(x.get("freq_hz", []), dtype=float)
    return {"frequency": float(freq[i]) if i < freq.size else np.nan, "power": float(power[i])}


def plot_periodogram(x, *args, **kwargs):
    del args, kwargs
    return x


def ticker(x, *args, **kwargs):
    del args, kwargs
    return x


__all__ = [
    "max_periodogram",
    "periodogram",
    "periodogram_fast",
    "periodogram_slow",
    "plot_periodogram",
    "subset_periodogram",
    "ticker",
]
