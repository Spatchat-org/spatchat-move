"""Parity-focused translation of ctmm 1.3.0 ``R/plot.telemetry.R``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import CTMMModel, Telemetry
from .viz_ops import plot as _plot
from .viz_ops import zoom as _zoom


def zoom_telemetry(x: Telemetry, fraction: float = 1.0, **kwargs):
    del kwargs
    return _zoom(x, fraction=fraction)


def new_plot(*args, **kwargs):
    return {"args": args, "kwargs": kwargs}


def ellipsograph(mu, sigma=None, level: float = 0.95, n: int = 128):
    center = np.asarray(mu, dtype=float).reshape(-1)[:2]
    if center.size < 2:
        center = np.pad(center, (0, 2 - center.size))
    s = np.eye(2) if sigma is None else np.asarray(sigma, dtype=float)[:2, :2]
    vals, vecs = np.linalg.eigh(s)
    vals = np.clip(vals, 0.0, np.inf)
    rad = np.sqrt(-2.0 * np.log(max(1.0 - float(level), np.finfo(float).tiny)))
    th = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=True)
    unit = np.column_stack([np.cos(th), np.sin(th)])
    xy = center + unit @ (vecs @ np.diag(np.sqrt(vals) * rad)).T
    return xy


def plot_telemetry(x: Telemetry, *args, **kwargs):
    del args
    return _plot(x, **kwargs)


def plot_ctmm(x: CTMMModel, *args, **kwargs):
    del args, kwargs
    return {"type": "ctmm", "model": x.model, "params": dict(x.params)}


def format_par(x, digits: int = 3):
    if isinstance(x, dict):
        return {k: format_par(v, digits=digits) for k, v in x.items()}
    try:
        return f"{float(x):.{digits}g}"
    except Exception:
        return str(x)


def pull(data, col):
    if isinstance(data, Telemetry):
        data = data.data
    return data[col].to_numpy() if hasattr(data, "__getitem__") else np.asarray([])


def plot_R(*args, **kwargs):
    return new_plot(*args, **kwargs)


def plot_SP(*args, **kwargs):
    return new_plot(*args, **kwargs)


def plot_UD(x, *args, **kwargs):
    del args, kwargs
    if isinstance(x, dict) and "PDF" in x:
        pdf = np.asarray(x["PDF"], dtype=float)
        return {"type": "UD", "shape": pdf.shape, "mass": float(np.nansum(pdf))}
    return {"type": "UD", "object": x}


def plot_df(x: pd.DataFrame, *args, **kwargs):
    del args
    return _plot(x, **kwargs)


def plot_kde(x, *args, **kwargs):
    return plot_UD(x, *args, **kwargs)


def plot_list(x: list, *args, **kwargs):
    return [_plot(v, **kwargs) if isinstance(v, (Telemetry, pd.DataFrame)) else v for v in x]


def plot_sf(x, *args, **kwargs):
    return plot_df(x, *args, **kwargs) if isinstance(x, pd.DataFrame) else new_plot(x, *args, **kwargs)


def plot_sf_list(x, *args, **kwargs):
    return [plot_sf(v, *args, **kwargs) for v in x]


def ctmm_coloc(*args, **kwargs):
    return new_plot(*args, **kwargs)


plot = plot_telemetry

__all__ = [
    "ctmm_coloc",
    "ellipsograph",
    "format_par",
    "new_plot",
    "plot",
    "plot_R",
    "plot_SP",
    "plot_UD",
    "plot_ctmm",
    "plot_df",
    "plot_kde",
    "plot_list",
    "plot_sf",
    "plot_sf_list",
    "plot_telemetry",
    "pull",
    "zoom_telemetry",
]
