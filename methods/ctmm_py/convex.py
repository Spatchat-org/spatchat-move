"""Partial parity translation of ctmm 1.3.0 ``R/convex.R``."""

from __future__ import annotations

import numpy as np

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover
    ConvexHull = None


def contourLines(UD=None, r=None, CDF=None, levels=0.95):
    if UD is not None:
        r = UD.get("r", r) if isinstance(UD, dict) else r
        CDF = UD.get("CDF", CDF) if isinstance(UD, dict) else CDF
    if r is None or CDF is None:
        return [{"level": float(levels), "x": np.array([0.0, 1.0, 0.0, -1.0]), "y": np.array([1.0, 0.0, -1.0, 0.0])}]

    x = np.asarray(r["x"], dtype=float)
    y = np.asarray(r["y"], dtype=float)
    z = np.asarray(CDF, dtype=float)
    lev = float(levels)
    idx = np.argwhere(z <= lev)
    if idx.size == 0:
        idx = np.argwhere(z <= lev * (1.0 + np.finfo(float).eps))
    if idx.size == 0:
        i, j = np.unravel_index(int(np.nanargmin(z)), z.shape)
        return [{"level": lev, "x": np.array([x[i], x[i], x[i], x[i]]), "y": np.array([y[j], y[j], y[j], y[j]])}]
    pts = np.column_stack([x[idx[:, 0]], y[idx[:, 1]]])
    if ConvexHull is not None and pts.shape[0] >= 3:
        hull = ConvexHull(pts)
        p = pts[hull.vertices]
    else:
        p = pts
    return [{"level": lev, "x": p[:, 0], "y": p[:, 1]}]


def convex(UD, level: float = 0.95, convex: bool = True, SP: bool = True, ID: str = "ID"):
    cl = contourLines(UD=UD, levels=level)
    xy = np.vstack([np.column_stack([c["x"], c["y"]]) for c in cl])
    if convex and ConvexHull is not None and xy.shape[0] >= 3:
        hull = ConvexHull(xy)
        xy = xy[hull.vertices]
    poly = {"id": ID, "xy": xy, "level": float(level), "spatial": bool(SP)}
    return poly


__all__ = ["contourLines", "convex"]
