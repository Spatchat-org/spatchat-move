"""Parity-focused translation of ctmm 1.3.0 ``R/projection.R``."""

from __future__ import annotations

import numpy as np
from pyproj import CRS, Transformer

from .types import Telemetry

DATUM = "+proj=longlat +datum=WGS84"
DATA_EARTH = {"R.EQ": 6378137.0, "R.PL": 6356752.3142}


def projection_NULL(x=None, asText: bool = True):
    del x, asText
    return None


def projection_telemetry(x, asText: bool = True):
    proj = None
    if isinstance(x, Telemetry):
        proj = x.metadata.get("proj4", x.crs)
    elif hasattr(x, "params"):
        proj = x.params.get("projection") or x.params.get("proj4")
    elif isinstance(x, dict):
        proj = x.get("projection") or x.get("proj4")
    if not asText and proj is not None:
        return CRS.from_user_input(proj)
    return proj


def projection(x, asText: bool = True):
    if x is None:
        return None
    if isinstance(x, list):
        return projection_list(x, asText=asText)
    return projection_telemetry(x, asText=asText)


def format_projection(value):
    if value is None:
        return None
    crs = CRS.from_user_input(value)
    try:
        return crs.to_proj4()
    except Exception:
        return crs.to_wkt()


def project(x, from_=DATUM, to=DATUM, **kwargs):
    if "from" in kwargs:
        from_ = kwargs.pop("from")
    if kwargs:
        raise TypeError(f"project got unexpected keyword arguments: {sorted(kwargs)}")
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if str(to) == str(from_):
        return arr.copy()
    tr = Transformer.from_crs(CRS.from_user_input(from_), CRS.from_user_input(to), always_xy=True)
    xx, yy = tr.transform(arr[:, 0], arr[:, 1])
    return np.column_stack([xx, yy])


def projection_list(x, asText: bool = True):
    vals = [projection(v, asText=asText) for v in x]
    if not vals:
        return None
    if vals[0] is None:
        return None
    unique = []
    for v in vals:
        if v not in unique:
            unique.append(v)
    return unique[0] if len(unique) == 1 else vals


def projection_set(x, value):
    if isinstance(x, list):
        return [projection_set(v, value) for v in x]
    if not isinstance(x, Telemetry):
        raise TypeError("projection_set expects telemetry or list")
    out = Telemetry(x.data.copy(), id_col=x.id_col, time_col=x.time_col, x_col=x.x_col, y_col=x.y_col, crs=x.crs, metadata=dict(x.metadata))
    if value is None:
        out.metadata.pop("proj4", None)
        out.crs = None
    else:
        out.metadata["proj4"] = format_projection(value)
        out.crs = out.metadata["proj4"]
    return out


def projection_set_telemetry(x, value):
    return projection_set(x, value)


def projection_set_list(x, value):
    return projection_set(x, value)


def projection_set_ctmm(x, value):
    if hasattr(x, "params"):
        params = dict(x.params)
        params["projection"] = format_projection(value) if value is not None else None
        from .types import CTMMModel

        return CTMMModel(x.model, params)
    out = dict(x)
    out["projection"] = format_projection(value) if value is not None else None
    return out


def northing(x, proj, angle: bool = False):
    if isinstance(x, Telemetry):
        df = x.data
        lon = df["longitude"].to_numpy(dtype=float) if "longitude" in df else df[x.x_col].to_numpy(dtype=float)
        lat = df["latitude"].to_numpy(dtype=float) if "latitude" in df else df[x.y_col].to_numpy(dtype=float)
        xy = df[[x.x_col, x.y_col]].to_numpy(dtype=float)
    else:
        df = x
        lon = np.asarray(df["longitude"], dtype=float)
        lat = np.asarray(df["latitude"], dtype=float)
        xy = np.column_stack([np.asarray(df["x"], dtype=float), np.asarray(df["y"], dtype=float)])
    dlam = 1.0 / np.sqrt((DATA_EARTH["R.EQ"] * np.sin(np.deg2rad(lat))) ** 2 + (DATA_EARTH["R.PL"] * np.cos(np.deg2rad(lat))) ** 2)
    dlam = dlam * (360.0 / (2.0 * np.pi))
    u = project(np.column_stack([lon, lat + dlam]), to=proj)
    u = u - xy
    u = u / np.sqrt(np.sum(u * u, axis=1))[:, None]
    if angle:
        return np.rad2deg(np.arctan2(u[:, 1], u[:, 0]))
    return u


def rotate_north(north, heading):
    n = np.asarray(north, dtype=float)
    h = np.deg2rad(np.asarray(heading, dtype=float))
    c = np.cos(h)
    s = np.sin(h)
    return np.column_stack([n[:, 0] * c - n[:, 1] * s, n[:, 0] * s + n[:, 1] * c])


def validate_projection(proj):
    CRS.from_user_input(proj)
    return True


def validate_grid(grid):
    return isinstance(grid, dict) and ("r" in grid or ("extent" in grid and "dr" in grid))


def check_projections(*objects):
    vals = [projection(o) for o in objects if projection(o) is not None]
    return len(set(vals)) <= 1


def cov_geo2xy(cov, lonlat=None, proj=DATUM):
    del lonlat, proj
    return np.asarray(cov, dtype=float)


def cov_xy2geo(cov, xy=None, proj=DATUM):
    del xy, proj
    return np.asarray(cov, dtype=float)


__all__ = [
    "DATA_EARTH",
    "DATUM",
    "check_projections",
    "cov_geo2xy",
    "cov_xy2geo",
    "format_projection",
    "northing",
    "project",
    "projection",
    "projection_NULL",
    "projection_list",
    "projection_set",
    "projection_set_ctmm",
    "projection_set_list",
    "projection_set_telemetry",
    "projection_telemetry",
    "rotate_north",
    "validate_grid",
    "validate_projection",
]
