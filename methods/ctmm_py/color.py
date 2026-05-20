"""Parity-focused translation of ctmm 1.3.0 ``R/color.R`` helpers."""

from __future__ import annotations

import colorsys
import re

import numpy as np
import pandas as pd

from .color_ops import color as _color
from .types import Telemetry


def annotate(object, by="all", cores: int = 1, **kwargs):
    del cores, kwargs
    if isinstance(object, list):
        return [annotate(o, by=by) for o in object]
    if not isinstance(object, Telemetry):
        return object
    out = Telemetry(object.data.copy(), id_col=object.id_col, time_col=object.time_col, x_col=object.x_col, y_col=object.y_col, crs=object.crs, metadata=dict(object.metadata))
    if by == "all" or "tropic" in by:
        ts = pd.to_datetime(out.data[out.time_col], errors="coerce", utc=True).dt.tz_convert(None)
        start = pd.to_datetime(ts.dt.year.astype(str) + "-01-01")
        end = pd.to_datetime((ts.dt.year + 1).astype(str) + "-01-01")
        out.data["tropic"] = (ts - start).dt.total_seconds() / (end - start).dt.total_seconds()
    return out


def color(object, by: str = "time", col_fn=None, alpha: float = 1.0, dt=None, cores: int = 1, **kwargs):
    del cores
    return _color(object, by=by, col_fn=col_fn, alpha=alpha, dt=dt, **kwargs)


def check(cby, column, object=None, by=None):
    """R ``color`` local validation helper."""
    if by is not None and by != cby:
        return True
    if object is None:
        return True
    first = object[0] if isinstance(object, list) and object else object
    cols = first.data.columns if isinstance(first, Telemetry) else getattr(first, "columns", [])
    if column not in cols:
        raise ValueError("Data are not annotated.")
    return True


def _hex_to_rgba(col):
    if not isinstance(col, str):
        raise TypeError("colors must be hex strings")
    s = col.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6:
        s += "ff"
    if len(s) != 8 or not re.fullmatch(r"[0-9A-Fa-f]{8}", s):
        raise ValueError(f"unsupported color {col!r}")
    return np.array([int(s[i : i + 2], 16) for i in range(0, 8, 2)], dtype=float)


def _rgba_to_hex(rgba):
    vals = np.clip(np.round(rgba), 0, 255).astype(int)
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}{vals[3]:02x}"


def simplify_color(object):
    if isinstance(object, list):
        return [simplify_color(o) for o in object]
    cols = np.asarray(object, dtype=object).ravel()
    rgba = np.vstack([_hex_to_rgba(c) for c in cols])
    w = rgba[:, 3]
    if np.sum(w) <= 0:
        w = np.ones_like(w)
    rgb = rgba[:, :3].T @ (w / np.sum(w))
    return _rgba_to_hex(np.r_[rgb, 255])


def malpha(col, alpha=1):
    cols = np.asarray(col, dtype=object).ravel()
    al = np.resize(np.asarray(alpha, dtype=float).ravel(), cols.size)
    out = []
    for c, a in zip(cols, al):
        rgba = _hex_to_rgba(str(c))
        rgba[3] *= float(a)
        out.append(_rgba_to_hex(rgba))
    return out[0] if np.asarray(col).ndim == 0 and np.asarray(alpha).ndim == 0 else np.array(out, dtype=object)


def grad_white(col):
    rgba = _hex_to_rgba(col)
    r, g, b = rgba[:3] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    out = []
    for white in np.arange(255, -1, -1) / 255.0:
        rr, gg, bb = colorsys.hsv_to_rgb(h, white, v)
        out.append(_rgba_to_hex(np.array([rr * 255, gg * 255, bb * 255, 255])))
    return np.array(out, dtype=object)


def color_individual(object, cores: int = 1, **kwargs):
    del cores, kwargs
    n = len(object) if isinstance(object, list) else 1
    if n == 1:
        return np.array([0.0])
    if n == 2:
        return np.array([0.0, 0.5])
    if n == 3:
        return np.array([0.0, 1.0 / 3.0, 2.0 / 3.0])
    return np.arange(n, dtype=float) / float(n)


__all__ = [
    "annotate",
    "check",
    "color",
    "color_individual",
    "grad_white",
    "malpha",
    "simplify_color",
]
