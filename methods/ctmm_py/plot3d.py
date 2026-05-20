"""Parity-focused translation of ctmm 1.3.0 ``R/plot3d.R``."""

from __future__ import annotations

import numpy as np

from .viz_ops import plot


def plot3d(x, **kwargs):
    return plot(x, **kwargs)


def plot_fn(fn, x=None, y=None, **kwargs):
    del kwargs
    if x is None:
        x = np.linspace(0.0, 1.0, 50)
    if y is None:
        y = np.linspace(0.0, 1.0, 50)
    xx, yy = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float), indexing="ij")
    z = np.vectorize(fn)(xx, yy)
    return {"x": xx, "y": yy, "z": z}


def mesh3d_UD(object, **kwargs):
    del kwargs
    if not isinstance(object, dict):
        raise TypeError("mesh3d_UD expects a UD dict")
    return {"x": object.get("r", {}).get("x"), "y": object.get("r", {}).get("y"), "z": object.get("PDF")}


def row_intersect(x, y):
    xx = np.asarray(x)
    yy = np.asarray(y)
    xset = {tuple(r) for r in xx.reshape((xx.shape[0], -1))}
    mask = [tuple(r) in xset for r in yy.reshape((yy.shape[0], -1))]
    return yy[np.asarray(mask)]


def row_unique(x):
    arr = np.asarray(x)
    flat = arr.reshape((arr.shape[0], -1))
    _, idx = np.unique(flat, axis=0, return_index=True)
    return arr[np.sort(idx)]


__all__ = ["mesh3d_UD", "plot", "plot3d", "plot_fn", "row_intersect", "row_unique"]
