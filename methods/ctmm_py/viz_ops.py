from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .types import Telemetry


def _to_df(x):
    if isinstance(x, Telemetry):
        return x.data, x.x_col, x.y_col
    if isinstance(x, pd.DataFrame):
        cols = list(x.columns)
        x_col = "x" if "x" in cols else ("longitude" if "longitude" in cols else cols[0])
        y_col = "y" if "y" in cols else ("latitude" if "latitude" in cols else cols[1])
        return x, x_col, y_col
    raise TypeError("Expected Telemetry or DataFrame")


def extent_xy(x):
    df, x_col, y_col = _to_df(x)
    xv = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    xv = xv[np.isfinite(xv)]
    yv = yv[np.isfinite(yv)]
    if xv.size == 0 or yv.size == 0:
        return {"xlim": (np.nan, np.nan), "ylim": (np.nan, np.nan)}
    return {"xlim": (float(np.min(xv)), float(np.max(xv))), "ylim": (float(np.min(yv)), float(np.max(yv)))}


def zoom(x, fraction: float = 1.0):
    ex = extent_xy(x)
    x0, x1 = ex["xlim"]
    y0, y1 = ex["ylim"]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    hx = 0.5 * (x1 - x0) * float(fraction)
    hy = 0.5 * (y1 - y0) * float(fraction)
    return {"xlim": (cx - hx, cx + hx), "ylim": (cy - hy, cy + hy)}


def plot(x, **kwargs):
    if kwargs:
        # Keep callable parity while remaining lightweight.
        kwargs.pop("add", None)
        kwargs.pop("fraction", None)
        kwargs.pop("xlim", None)
        kwargs.pop("ylim", None)
        if kwargs:
            raise TypeError(f"plot got unexpected keyword arguments: {sorted(kwargs)}")
    ex = extent_xy(x)
    df, x_col, y_col = _to_df(x)
    return {
        "type": "scatter",
        "n": int(len(df)),
        "x_col": x_col,
        "y_col": y_col,
        "xlim": ex["xlim"],
        "ylim": ex["ylim"],
    }


def raster(x, bins: int = 64):
    df, x_col, y_col = _to_df(x)
    xv = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    yv = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[ok]
    yv = yv[ok]
    if xv.size == 0:
        return {"grid": np.zeros((bins, bins), dtype=float), "xedges": np.array([]), "yedges": np.array([])}
    h, xedges, yedges = np.histogram2d(xv, yv, bins=int(bins))
    return {"grid": h, "xedges": xedges, "yedges": yedges}


def writeRaster(x, filename: str, format: str | None = None, DF: str = "CDF"):
    r = x if isinstance(x, dict) and "grid" in x else raster(x)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".npy"}:
        np.save(path, r["grid"])
    else:
        pd.DataFrame(r["grid"]).to_csv(path, index=False)
    return str(path)


def writeVector(x, filename: str | None = None, filetype: str = "ESRI Shapefile", **kwargs):
    if kwargs:
        kwargs.pop("level.UD", None)
        kwargs.pop("level", None)
        kwargs.pop("error", None)
        if kwargs:
            raise TypeError(f"writeVector got unexpected keyword arguments: {sorted(kwargs)}")
    df, _, _ = _to_df(x)
    if filename is None:
        return df.copy()
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        df.to_json(path, orient="records")
    else:
        df.to_csv(path, index=False)
    return str(path)

