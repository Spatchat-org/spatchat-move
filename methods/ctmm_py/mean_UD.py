"""Parity translation of ctmm 1.3.0 ``R/mean.UD.R`` for dict-backed UDs."""

from __future__ import annotations

import numpy as np

from .grid import grid_intersection, grid_union
from .kde import pmf2cdf


def _ud_type(ud):
    return ud.get("type") or ud.get("TYPE") or ud.get("@type") or "utilization"


def _dr_value(dr, axis: str, index: int) -> float:
    if isinstance(dr, dict):
        return float(dr[axis])
    return float(np.asarray(dr, dtype=float).reshape(-1)[index])


def mean_UD(x, weights=None, sample: bool = True, **kwargs):
    """Average aligned utilization distributions on their union grid."""
    del sample, kwargs
    uds = list(x.values()) if isinstance(x, dict) else list(x)
    if not uds:
        raise ValueError("mean_UD requires at least one UD")
    axes = list(uds[0].get("axes", ["x", "y"]))
    if weights is None:
        if _ud_type(uds[0]) == "occurrence":
            weights = [float(ud.get("W", 1.0)) for ud in uds]
        else:
            weights = [1.0] * len(uds)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.max(weights)
    weight_sum = float(np.sum(weights))
    types = [_ud_type(ud) for ud in uds]
    if len(set(types)) > 1:
        raise ValueError(f"Distribution types {types} differ.")

    grid = grid_union(uds)
    gx = grid["r"]["x"]
    gy = grid["r"]["y"]
    pdf = np.zeros((len(gx), len(gy)), dtype=float)
    for w, ud in zip(weights, uds):
        sub = grid_intersection([grid, ud])
        pdf[np.ix_(sub[0]["x"], sub[0]["y"])] += w * np.asarray(ud["PDF"], dtype=float)[np.ix_(sub[1]["x"], sub[1]["y"])]
    pdf = pdf / weight_sum

    dr = grid["dr"]
    dV = _dr_value(dr, "x", 0) * _dr_value(dr, "y", 1)
    out = dict(grid)
    out["weights"] = weights
    out["axes"] = axes
    out["PDF"] = pdf
    out["CDF"] = pmf2cdf(pdf * dV)
    out["type"] = types[0]
    out["H"] = None
    return out


__all__ = ["mean_UD"]
