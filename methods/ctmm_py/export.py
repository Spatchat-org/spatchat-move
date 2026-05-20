"""Parity-focused translation of ctmm 1.3.0 ``R/export.R``."""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, mapping

from .misc_ops import SpatialPointsDataFrame_telemetry, SpatialPoints_telemetry
from .types import Telemetry
from .viz_ops import writeRaster as _writeRaster
from .viz_ops import writeVector as _writeVector


def head(x, n: int = 6):
    if hasattr(x, "data"):
        return x.data.head(n)
    if isinstance(x, pd.DataFrame):
        return x.head(n)
    if isinstance(x, (list, tuple)):
        return x[:n]
    return x


def tail(x, n: int = 6):
    if hasattr(x, "data"):
        return x.data.tail(n)
    if isinstance(x, pd.DataFrame):
        return x.tail(n)
    if isinstance(x, (list, tuple)):
        return x[-n:]
    return x


def raster_UD(object, DF: str = "CDF", **kwargs):
    del kwargs
    if not isinstance(object, dict):
        raise TypeError("raster_UD expects a UD dict")
    return np.asarray(object.get(DF, object.get(DF.upper(), object.get("PDF"))), dtype=float)


def writeRaster_UD(object, filename: str, format: str | None = None, DF: str = "CDF", **kwargs):
    del kwargs
    return _writeRaster({"grid": raster_UD(object, DF=DF)}, filename=filename, format=format, DF=DF)


def inside(points, polygon):
    if isinstance(points, Telemetry):
        arr = points.data[[points.x_col, points.y_col]].to_numpy(dtype=float)
    else:
        arr = np.asarray(points, dtype=float)
    poly = polygon if isinstance(polygon, Polygon) else Polygon(np.asarray(polygon, dtype=float))
    return np.array([poly.contains(Point(p)) or poly.touches(Point(p)) for p in arr], dtype=bool)


def SpatialPolygonsDataFrame_UD(object, level_UD=0.95, **kwargs):
    del kwargs
    if not isinstance(object, dict):
        raise TypeError("SpatialPolygonsDataFrame_UD expects a UD dict")
    r = object.get("r", {})
    gx = np.asarray(r.get("x", []), dtype=float)
    gy = np.asarray(r.get("y", []), dtype=float)
    if gx.size == 0 or gy.size == 0:
        return pd.DataFrame(columns=["level", "geometry"])
    poly = Polygon([(gx[0], gy[0]), (gx[-1], gy[0]), (gx[-1], gy[-1]), (gx[0], gy[-1])])
    return pd.DataFrame({"level": np.atleast_1d(level_UD), "geometry": [mapping(poly)]})


def SpatialPolygonsDataFrame_telemetry(object, **kwargs):
    del kwargs
    df = object.data if isinstance(object, Telemetry) else pd.DataFrame(object)
    cols = [object.x_col, object.y_col] if isinstance(object, Telemetry) else ["x", "y"]
    if not all(c in df.columns for c in cols):
        return pd.DataFrame(columns=["geometry"])
    arr = df[cols].to_numpy(dtype=float)
    if arr.shape[0] < 3:
        return pd.DataFrame(columns=["geometry"])
    poly = Polygon(arr).convex_hull
    return pd.DataFrame({"geometry": [mapping(poly)]})


def as_sf(object, **kwargs):
    if isinstance(object, Telemetry):
        return SpatialPointsDataFrame_telemetry(object)
    if isinstance(object, dict) and "CDF" in object:
        return SpatialPolygonsDataFrame_UD(object, **kwargs)
    return object


def seg_id(i, j):
    return f"{i}:{j}"


def seg_rv(r, v):
    return {"r": r, "v": v}


def writeVector_UD(object, filename: str | None = None, **kwargs):
    spdf = SpatialPolygonsDataFrame_UD(object, **kwargs)
    if filename is None:
        return spdf
    return _writeVector(spdf, filename=filename)


def writeVector_telemetry(object, filename: str | None = None, **kwargs):
    return _writeVector(object, filename=filename, **kwargs)


SpatialPoints = SpatialPoints_telemetry
SpatialPointsDataFrame = SpatialPointsDataFrame_telemetry

__all__ = [
    "SpatialPoints",
    "SpatialPointsDataFrame",
    "SpatialPointsDataFrame_telemetry",
    "SpatialPoints_telemetry",
    "SpatialPolygonsDataFrame_telemetry",
    "SpatialPolygonsDataFrame_UD",
    "as_sf",
    "head",
    "inside",
    "raster_UD",
    "seg_id",
    "seg_rv",
    "tail",
    "writeRaster_UD",
    "writeVector_UD",
    "writeVector_telemetry",
]
