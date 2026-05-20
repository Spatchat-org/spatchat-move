"""Parity-focused translation of ctmm 1.3.0 ``R/median.R``."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .types import Telemetry

R_EQ = 6378137.0
R_PL = 6356752.314245


def ellipsoid2cartesian(data):
    df = data.data if isinstance(data, Telemetry) else data
    lon = np.asarray(df["longitude"], dtype=float) * (2.0 * np.pi / 360.0)
    lat = np.asarray(df["latitude"], dtype=float) * (2.0 * np.pi / 360.0)
    s = np.asarray(df["z"], dtype=float) if "z" in df else 0.0
    z = (R_PL + s) * np.sin(lat)
    r = (R_EQ + s) * np.cos(lat)
    x = r * np.cos(lon)
    y = r * np.sin(lon)
    return np.column_stack([x, y, z])


def cartesian2ellipsoid(mu):
    arr = np.atleast_2d(np.asarray(mu, dtype=float))
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    lon = np.arctan2(y, x)
    r = np.sqrt(x * x + y * y) / R_EQ
    lat = np.arctan2(z / R_PL, r)
    return pd.DataFrame({"longitude": lon * (360.0 / (2.0 * np.pi)), "latitude": lat * (360.0 / (2.0 * np.pi))})


def _gmedian_rowvec(values, init=None, gamma: float = 2.0, alpha: float = 0.75, nstart: int = 1, epsilon: float = 1e-8):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.asarray([], dtype=float)
    if init is not None:
        x = np.vstack([np.asarray(init, dtype=float).reshape(1, -1), x])
    n, p = x.shape
    medvec = x[0].copy()
    medrm = x[0].copy()
    for _ in range(int(nstart)):
        for it in range(1, n):
            diff = x[it] - medrm
            norm = float(np.linalg.norm(diff))
            if norm > epsilon:
                weight = math.sqrt(float(p)) * float(gamma) * float(it + 1) ** (-float(alpha)) / norm
                medrm = medrm + weight * diff
            medvec = medvec + (medrm - medvec) / float(it + 1)
    return medvec


def _gmedian_cov_row_p(values, median, gamma: float = 2.0, alpha: float = 0.75, nstart: int = 1):
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    n, p = x.shape
    diff = x[0] - np.asarray(median, dtype=float)
    medav = np.outer(diff, diff)
    medrm = medav.copy()
    for _ in range(int(nstart)):
        for it in range(1, n):
            diff = x[it] - median
            diffmat = np.outer(diff, diff) - medrm
            norm = float(np.linalg.norm(diffmat, ord="fro"))
            weight = 1.0 if norm == 0.0 else min(1.0, float(p) * float(gamma) * float(it + 1) ** (-float(alpha)) / norm)
            medrm = medrm + weight * diffmat
            medav = medav + (medrm - medav) / float(it + 1)
    return medav


def median_longlat(data, k: int = 1, **kwargs):
    del kwargs
    xyz = ellipsoid2cartesian(data)
    if xyz.shape[0] == 0:
        return pd.DataFrame({"longitude": [], "latitude": []})
    init = np.nanmedian(xyz, axis=0)
    mu = _gmedian_rowvec(xyz, init=init, nstart=10)
    if k == 1 or xyz.shape[0] == 1:
        return cartesian2ellipsoid(mu)
    cov = _gmedian_cov_row_p(xyz, mu, nstart=10)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    proj = (xyz - mu) @ axis
    proj = np.asarray(proj, dtype=float) - float(np.nanmedian(proj))
    pos = xyz[proj >= 0]
    neg = xyz[proj <= 0]
    mu1 = pos[0] if pos.shape[0] == 1 else _gmedian_rowvec(pos, init=np.nanmedian(pos, axis=0), nstart=2)
    mu2 = neg[0] if neg.shape[0] == 1 else _gmedian_rowvec(neg, init=np.nanmedian(neg, axis=0), nstart=2)
    out = cartesian2ellipsoid(np.vstack([mu1, mu2])).sort_values("longitude").reset_index(drop=True)
    return out


def median_telemetry(x: Telemetry | list[Telemetry], na_rm: bool = False, **kwargs):
    del na_rm
    if isinstance(x, list):
        parts = [median_telemetry(v, **kwargs).data for v in x]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return Telemetry(df, x_col="longitude", y_col="latitude")
    if {"longitude", "latitude"}.issubset(x.data.columns):
        mu = median_longlat(x.data, **kwargs)
        return Telemetry(mu, id_col=x.id_col, time_col=x.time_col, x_col="longitude", y_col="latitude", crs=x.crs, metadata=dict(x.metadata))
    cols = [x.x_col, x.y_col]
    mu = np.nanmedian(x.data[cols].to_numpy(dtype=float), axis=0)
    return Telemetry(pd.DataFrame({x.x_col: [mu[0]], x.y_col: [mu[1]]}), id_col=x.id_col, time_col=x.time_col, x_col=x.x_col, y_col=x.y_col, crs=x.crs, metadata=dict(x.metadata))


def median(x, axis=0, **kwargs):
    if isinstance(x, Telemetry) or (isinstance(x, list) and x and isinstance(x[0], Telemetry)):
        return median_telemetry(x, **kwargs)
    return np.nanmedian(np.asarray(x, dtype=float), axis=axis)


__all__ = ["cartesian2ellipsoid", "ellipsoid2cartesian", "median", "median_longlat", "median_telemetry"]
