"""Parity-focused translation of ctmm 1.3.0 ``R/grid.R`` helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .viz_ops import raster, writeRaster, writeVector, zoom


def is_grid_complete(grid) -> bool:
    if grid is None:
        return False
    if isinstance(grid, dict):
        if "r" in grid:
            return True
        if "extent" in grid and "dr" in grid:
            return True
        return False
    if hasattr(grid, "get") and callable(grid.get):
        return bool(grid.get("r") is not None)
    return False


def is_consistent_pair(grid1: dict[str, Any], grid2: dict[str, Any]) -> bool:
    dr1 = np.asarray(grid1["dr"], dtype=float).reshape(-1)
    dr2 = np.asarray(grid2["dr"], dtype=float).reshape(-1)
    mx = np.maximum(dr1, dr2)
    mn = np.maximum(np.minimum(dr1, dr2), np.finfo(float).eps)
    t = np.log2(mx / mn)
    if np.any(np.abs(t - np.round(t)) > np.finfo(float).eps):
        return False
    r1 = np.array([float(grid1["r"]["x"][0]), float(grid1["r"]["y"][0])], dtype=float)
    r2 = np.array([float(grid2["r"]["x"][0]), float(grid2["r"]["y"][0])], dtype=float)
    u = (r1 - r2) / mn
    return bool(np.all(np.abs(u - np.round(u)) <= np.finfo(float).eps))


def is_consistent(grids: list[dict[str, Any]]) -> bool:
    if len(grids) < 2:
        return True
    return all(is_consistent_pair(grids[i], grids[i + 1]) for i in range(len(grids) - 1))


def grid_union(UD: list[dict[str, Any]]) -> dict[str, Any]:
    drs = np.array([[float(ud["dr"]["x"]), float(ud["dr"]["y"])] for ud in UD], dtype=float)
    if np.any(np.abs(np.diff(drs[:, 0])) > 0) or np.any(np.abs(np.diff(drs[:, 1])) > 0):
        raise ValueError("Inconsistent grid resolutions.")
    dr = {"x": float(drs[0, 0]), "y": float(drs[0, 1])}
    rx = [np.asarray(ud["r"]["x"], dtype=float) for ud in UD]
    ry = [np.asarray(ud["r"]["y"], dtype=float) for ud in UD]
    x_min = min(float(v[0]) for v in rx)
    x_max = max(float(v[-1]) for v in rx)
    y_min = min(float(v[0]) for v in ry)
    y_max = max(float(v[-1]) for v in ry)
    x = np.arange(x_min, x_max + dr["x"] / 2.0, dr["x"])
    y = np.arange(y_min, y_max + dr["y"] / 2.0, dr["y"])
    return {"r": {"x": x, "y": y}, "dr": dr}


def grid_intersection(UD: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
    tol = np.sqrt(np.finfo(float).eps)
    drs = np.array([[float(ud["dr"]["x"]), float(ud["dr"]["y"])] for ud in UD], dtype=float)
    if np.any(np.abs(np.diff(drs[:, 0])) > 0) or np.any(np.abs(np.diff(drs[:, 1])) > 0):
        raise ValueError("Inconsistent grid resolutions.")
    drx, dry = float(drs[0, 0]), float(drs[0, 1])
    rx = [np.asarray(ud["r"]["x"], dtype=float) for ud in UD]
    ry = [np.asarray(ud["r"]["y"], dtype=float) for ud in UD]
    x_min = max(float(v[0]) for v in rx)
    x_max = min(float(v[-1]) for v in rx)
    y_min = max(float(v[0]) for v in ry)
    y_max = min(float(v[-1]) for v in ry)
    out = []
    for x, y in zip(rx, ry):
        sx = ((x - x_min) / drx >= -tol) & ((x_max - x) / drx >= -tol)
        sy = ((y - y_min) / dry >= -tol) & ((y_max - y) / dry >= -tol)
        out.append({"x": sx, "y": sy})
    return out


def same_grids(UD: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sub = grid_intersection(UD)
    out = []
    for ud, s in zip(UD, sub):
        u = dict(ud)
        u["r"] = {"x": np.asarray(ud["r"]["x"])[s["x"]], "y": np.asarray(ud["r"]["y"])[s["y"]]}
        u["PDF"] = np.asarray(ud["PDF"])[np.ix_(s["x"], s["y"])]
        out.append(u)
    return out


def format_grid(grid, axes=("x", "y")):
    if grid is None:
        return {"axes": list(axes), "dr_fn": min}
    if isinstance(grid, dict):
        g = dict(grid)
        if all(a in g for a in axes) and "r" not in g:
            g = {"r": {a: np.asarray(g[a], dtype=float) for a in axes}}
        if "dr" in g:
            dr = np.asarray(g["dr"], dtype=float).reshape(-1)
            if dr.size == 1:
                dr = np.repeat(dr[0], len(axes))
            g["dr"] = {axes[i]: float(dr[i]) for i in range(len(axes))}
        g.setdefault("dr_fn", min)
        return g
    raise TypeError("Malformed grid argument.")


def grid_comp(grid1, grid2):
    return is_consistent_pair(grid1, grid2)


def kde_grid(*args, **kwargs):
    from .kde import kde_grid as _kde_grid

    return _kde_grid(*args, **kwargs)


__all__ = [
    "raster",
    "writeRaster",
    "writeVector",
    "zoom",
    "is_grid_complete",
    "is_consistent",
    "is_consistent_pair",
    "grid_union",
    "grid_intersection",
    "grid_comp",
    "kde_grid",
    "same_grids",
    "format_grid",
]
