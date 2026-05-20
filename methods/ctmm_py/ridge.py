"""Parity translation of ctmm 1.3.0 ``R/ridge.R`` raster helpers."""

from __future__ import annotations

import numpy as np

from .modes import DiffGrid


def anti_alias(i, j, M, dM=1):
    """``anti.alias`` bilinear raster accumulation."""
    out = np.asarray(M, dtype=float).copy()
    ii = np.unique([np.floor(i), np.ceil(i)]).astype(int)
    jj = np.unique([np.floor(j), np.ceil(j)]).astype(int)
    ii = ii[(0 < ii) & (ii < out.shape[0])]
    jj = jj[(0 < jj) & (jj < out.shape[1])]
    if ii.size == 0 or jj.size == 0:
        return out
    dm = np.full((ii.size, jj.size), float(dM), dtype=float)
    if ii.size > 1:
        w = abs(float(i) - ii[0])
        dm[0, :] = w * dm[0, :]
        dm[1, :] = (1.0 - w) * dm[1, :]
    if jj.size > 1:
        w = abs(float(j) - jj[0])
        dm[:, 0] = w * dm[:, 0]
        dm[:, 1] = (1.0 - w) * dm[:, 1]
    out[np.ix_(ii, jj)] += dm
    return out


def _ud_field(object, name):
    if isinstance(object, dict):
        return object[name]
    return getattr(object, name)


def ridges_UD(object, **kwargs):
    """``ridges.UD`` indicator and average-width raster calculation."""
    del kwargs
    pdf = np.asarray(_ud_field(object, "PDF"), dtype=float)
    with np.errstate(divide="ignore"):
        log_pdf = np.log(pdf)
    dr = _ud_field(object, "dr")
    dx = float(dr["x"] if isinstance(dr, dict) else dr[0])
    dy = float(dr["y"] if isinstance(dr, dict) else dr[1])
    dx2 = dx * dx
    dy2 = dy * dy
    dr2 = dx2 + dy2
    dxdydr2 = dx * dy / dr2
    point = np.zeros_like(log_pdf, dtype=float)
    curve = np.zeros_like(log_pdf, dtype=float)

    for i in range(1, log_pdf.shape[0] - 1):
        for j in range(1, log_pdf.shape[1] - 1):
            sub = log_pdf[i - 1 : i + 2, j - 1 : j + 2]
            if not np.all(sub > -np.inf):
                continue
            diff = DiffGrid(sub, dx, dy, dx2, dy2, dr2, dxdydr2)
            grad = diff["GRAD"]
            hess = diff["HESS"] + np.outer(grad, grad)
            vals, vecs = np.linalg.eigh(hess)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            if vals[1] >= 0:
                continue
            dridge = -float(grad @ vecs[:, 1]) / vals[1] * vecs[:, 1]
            dij = dridge / np.array([dx, dy])
            ip = i + dij[0]
            jp = j + dij[1]
            if np.max(np.abs(dij)) <= 2:
                if np.max(np.abs(dij)) <= 1:
                    point = anti_alias(ip, jp, point, 1.0)
                    u = vecs[:, 0] / np.array([dx, dy])
                    u = u / np.sqrt(np.sum(u * u))
                    point = anti_alias(ip + u[0], jp + u[1], point, 0.5)
                    point = anti_alias(ip - u[0], jp - u[1], point, 0.5)
                    curve = anti_alias(ip + u[0], jp + u[1], curve, -vals[1] / 2.0)
                    curve = anti_alias(ip - u[0], jp - u[1], curve, -vals[1] / 2.0)
                else:
                    point = anti_alias(ip, jp, point, 1.0 / 8.0)
                    curve = anti_alias(ip, jp, curve, -vals[1] / 8.0)

    max_vote = 1.0 + 2.0 / 2.0 + 6.0 / 8.0
    point = point / max_vote
    curve = curve / max_vote
    scale = np.maximum(point, 1.0)
    point = point / scale
    curve = curve / scale
    sub = curve > 0
    width = float("nan")
    if np.any(sub):
        width = float(1.0 / np.sqrt(np.sum(curve[sub] * pdf[sub]) / np.sum(pdf[sub])))
    return {"Indicator": point, "Ave.Width": width}


def ridges2_UD(object, precision: float = 1 / 8, **kwargs):
    """``ridges2.UD`` ridge metric raster."""
    del precision, kwargs
    pdf = np.asarray(_ud_field(object, "PDF"), dtype=float)
    with np.errstate(divide="ignore"):
        log_pdf = np.log(pdf)
    dr = _ud_field(object, "dr")
    dx = float(dr["x"] if isinstance(dr, dict) else dr[0])
    dy = float(dr["y"] if isinstance(dr, dict) else dr[1])
    dx2 = dx * dx
    dy2 = dy * dy
    dr2 = dx2 + dy2
    dxdydr2 = dx * dy / dr2
    ridge = np.full_like(log_pdf, np.nan, dtype=float)
    for i in range(1, log_pdf.shape[0] - 1):
        for j in range(1, log_pdf.shape[1] - 1):
            if log_pdf[i, j] == -np.inf:
                continue
            sub = log_pdf[i - 1 : i + 2, j - 1 : j + 2]
            if not np.all(sub > -np.inf):
                continue
            diff = DiffGrid(sub, dx, dy, dx2, dy2, dr2, dxdydr2)
            grad = diff["GRAD"]
            hess = diff["HESS"] + np.outer(grad, grad)
            vals, vecs = np.linalg.eigh(hess)
            order = np.argsort(vals)[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            if vals[1] < 0:
                ridge[i, j] = float(np.sqrt(np.sum((-(np.array([1 / dx, 1 / dy]) @ vecs[:, 1]) / vals[1] * (grad @ vecs[:, 1])) ** 2)))
    return ridge


def ridges(object, **kwargs):
    return ridges_UD(object, **kwargs)


__all__ = ["anti_alias", "ridges", "ridges2_UD", "ridges_UD"]
