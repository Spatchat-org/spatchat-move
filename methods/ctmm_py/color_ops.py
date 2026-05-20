from __future__ import annotations

import colorsys
from typing import Callable

import numpy as np

from .types import Telemetry


def _clamp01(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def _default_col_fn(index, alpha):
    idx = _clamp01(index)
    a = _clamp01(alpha)
    out = []
    for i, aa in zip(idx, a):
        r = int(round(255 * i))
        g = 0
        b = int(round(255 * (1.0 - i)))
        out.append(f"#{r:02x}{g:02x}{b:02x}{int(round(255*aa)):02x}")
    return np.array(out, dtype=object)


def _individual_col_fn(index, alpha):
    idx = _clamp01(index)
    a = _clamp01(alpha)
    out = []
    for i, aa in zip(idx, a):
        r, g, b = colorsys.hsv_to_rgb(float(i), 1.0, 1.0)
        out.append(f"#{int(round(255*r)):02x}{int(round(255*g)):02x}{int(round(255*b)):02x}{int(round(255*aa)):02x}")
    return np.array(out, dtype=object)


def color(object, by: str = "time", col_fn: Callable | None = None, alpha: float = 1.0, dt=None, **kwargs):
    if kwargs:
        raise TypeError(f"color got unexpected keyword arguments: {sorted(kwargs)}")
    objs = object if isinstance(object, list) else [object]
    if not objs:
        return []
    if not all(isinstance(o, Telemetry) for o in objs):
        raise TypeError("color currently supports Telemetry (or list[Telemetry])")

    if by == "individual":
        ind = np.linspace(0.0, 1.0, num=max(len(objs), 2), endpoint=False)[: len(objs)]
        fn = col_fn or _individual_col_fn
        out = []
        for i, o in enumerate(objs):
            n = len(o.data)
            out.append(fn(np.full(n, ind[i]), np.full(n, alpha)))
        return out[0] if not isinstance(object, list) else out

    idxs = []
    for o in objs:
        if by == "time":
            idx = o.data[o.time_col].astype("int64").to_numpy(dtype=float) / 1e9
        else:
            if by not in o.data.columns:
                raise ValueError("Data are not annotated.")
            idx = o.data[by].to_numpy(dtype=float)
        idxs.append(idx)

    allv = np.concatenate([i[np.isfinite(i)] for i in idxs if np.any(np.isfinite(i))]) if idxs else np.array([])
    vmin = float(np.min(allv)) if allv.size else 0.0
    vmax = float(np.max(allv)) if allv.size else 1.0
    denom = vmax - vmin if vmax > vmin else 1.0
    idxs = [((i - vmin) / denom) for i in idxs]

    fn = col_fn or _default_col_fn
    out = []
    for o, idx in zip(objs, idxs):
        t = o.data[o.time_col].astype("int64").to_numpy(dtype=float) / 1e9
        if len(t) <= 1:
            a = np.full(len(t), float(alpha))
        else:
            dtv = np.diff(t)
            if dt is None:
                dt0 = float(np.median(dtv[np.isfinite(dtv) & (dtv > 0)])) if np.any(np.isfinite(dtv) & (dtv > 0)) else 1.0
            else:
                dt0 = float(dt)
            aa = np.minimum(np.r_[np.inf, dtv], np.r_[dtv, np.inf])
            a = float(alpha) * np.clip(aa / max(dt0, np.finfo(float).eps), 0.0, 1.0)
        out.append(fn(idx, a))
    return out[0] if not isinstance(object, list) else out

