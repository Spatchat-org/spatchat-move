"""Partial parity translation of ctmm 1.3.0 ``R/lasso.R``."""
from __future__ import annotations
import numpy as np

def marquee(points, xmin, xmax, ymin, ymax):
    p = np.asarray(points, dtype=float)
    m = (p[:,0]>=xmin)&(p[:,0]<=xmax)&(p[:,1]>=ymin)&(p[:,1]<=ymax)
    return np.where(m)[0]

def lasso(points, polygon):
    p = np.asarray(points, dtype=float)
    poly = np.asarray(polygon, dtype=float)
    x, y = p[:,0], p[:,1]
    inside = np.zeros(len(p), dtype=bool)
    j = len(poly)-1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        inter = ((yi>y)!=(yj>y)) & (x < (xj-xi)*(y-yi)/max(yj-yi, np.finfo(float).eps)+xi)
        inside ^= inter
        j = i
    return np.where(inside)[0]

def cleave(points, indices):
    p = np.asarray(points, dtype=float)
    idx = np.asarray(indices, dtype=int)
    mask = np.zeros(len(p), dtype=bool)
    mask[idx[(idx>=0)&(idx<len(p))]] = True
    return {"inside": p[mask], "outside": p[~mask]}


def selector(points, polygon=None, xmin=None, xmax=None, ymin=None, ymax=None):
    if polygon is not None:
        return lasso(points, polygon)
    if None not in (xmin, xmax, ymin, ymax):
        return marquee(points, xmin, xmax, ymin, ymax)
    return np.arange(len(np.asarray(points)), dtype=int)


__all__ = ["cleave", "lasso", "marquee", "selector"]
