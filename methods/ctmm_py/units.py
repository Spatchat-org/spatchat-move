"""Partial parity translation of ctmm 1.3.0 ``R/units.R``."""

from __future__ import annotations

from typing import Any

import numpy as np

from .units_conv import pct_hash_pct as _pct_hash_pct


def dimfig(data, dimension: str, thresh: float = 1.0, concise: bool = False, SI: bool = False):
    u = unit(data, dimension=dimension, thresh=thresh, concise=concise, SI=SI)
    arr = np.asarray(data, dtype=float)
    return {"data": arr / float(u["scale"]), "units": (u["name"], u["abrv"])}


def unit(data, dimension: str, thresh: float = 1.0, concise: bool = False, SI: bool = False):
    arr = np.asarray(data, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    vmax = float(np.max(np.abs(arr))) if arr.size else 1.0
    if SI:
        vmax = 1.01
        thresh = 1.0

    if dimension in ("length", "distance"):
        names = ["microns", "milimeters", "centimeters", "meters", "kilometers"]
        abrvs = ["um", "mm", "cm", "m", "km"]
        scales = [1e-6, 1e-3, 1e-2, 1.0, 1e3]
    elif dimension == "area":
        names = ["square microns", "square milimeters", "square centimeters", "square meters", "hectares", "square kilometers"]
        abrvs = ["um^2", "mm^2", "cm^2", "m^2", "hm^2", "km^2"]
        scales = [1e-12, 1e-6, 1e-4, 1.0, 1e4, 1e6]
    elif dimension == "time":
        names = ["microseconds", "miliseconds", "seconds", "minutes", "hours", "days", "months", "years"]
        abrvs = ["us", "ms", "sec", "min", "hr", "day", "mon", "yr"]
        scales = [1e-6, 1e-3, 1.0, 60.0, 3600.0, 86400.0, 2629743.83, 31556926.0]
    elif dimension in ("speed", "velocity"):
        names = ["microns/day", "milimeters/day", "centimeters/day", "meters/day", "kilometers/day"]
        abrvs = ["um/day", "mm/day", "cm/day", "m/day", "km/day"]
        scales = [1e-6 / 86400.0, 1e-3 / 86400.0, 1e-2 / 86400.0, 1.0 / 86400.0, 1e3 / 86400.0]
        if SI:
            names, abrvs, scales = ["meters/second"], ["m/s"], [1.0]
    elif dimension == "diffusion":
        names = ["square microns/day", "square milimeters/day", "square centimeters/day", "square meters/day", "hectares/day", "square kilometers/day"]
        abrvs = ["um^2/day", "mm^2/day", "cm^2/day", "m^2/day", "hm^2/day", "km^2/day"]
        scales = [1e-12 / 86400.0, 1e-6 / 86400.0, 1e-4 / 86400.0, 1.0 / 86400.0, 1e4 / 86400.0, 1e6 / 86400.0]
        if SI:
            names, abrvs, scales = ["square meters/second"], ["m^2/s"], [1.0]
    else:
        return {"scale": 1.0, "name": None, "abrv": None}

    labels = abrvs if concise else names
    idx = [i for i, s in enumerate(scales) if vmax >= thresh * s]
    i = idx[-1] if idx else 0
    return {"scale": float(scales[i]), "name": labels[i], "abrv": abrvs[i]}


def unit_par(par, **kwargs):
    p = np.asarray(par, dtype=float).reshape(-1)
    if p.size >= 3:
        cand = p[1:3]
    else:
        cand = p
    cand = cand[cand > np.finfo(float).eps]
    val = float(np.min(cand)) if cand.size else 0.0
    return unit(val, **kwargs)


def pct_hash_pct(x: Any, y: Any) -> float:
    return _pct_hash_pct(x, y)


def ustring(x: str) -> float:
    s = str(x).strip()
    i = 0
    while i < len(s) and (s[i].isdigit() or s[i] in ".+-eE"):
        i += 1
    num = float(s[:i]) if i else 1.0
    unit_name = s[i:].strip()
    return pct_hash_pct(num, unit_name) if unit_name else num


def add(x, y):
    return x + y


def convert(x, from_unit=None, to_unit=None, dimension: str | None = None):
    if from_unit is None or to_unit is None:
        return x
    f = pct_hash_pct(1.0, from_unit)
    t = pct_hash_pct(1.0, to_unit)
    if dimension == "area":
        f *= f
        t *= t
    return np.asarray(x, dtype=float) * f / t


def generate_units():
    return {
        "length": unit(1.0, "length"),
        "area": unit(1.0, "area"),
        "time": unit(1.0, "time"),
        "speed": unit(1.0, "speed"),
    }


def unit_UD(object, **kwargs):
    ci = object.get("CI.area", object.get("area_km2", 1.0)) if isinstance(object, dict) else 1.0
    return unit(ci, "area", **kwargs)


def unit_ctmm(object, **kwargs):
    sig = object.params.get("sigma_matrix", 1.0) if hasattr(object, "params") else 1.0
    return unit(np.sqrt(np.nanmean(np.diag(np.asarray(sig, dtype=float)))) if np.asarray(sig).ndim == 2 else sig, "length", **kwargs)


def unit_telemetry(object, **kwargs):
    if hasattr(object, "data"):
        vals = object.data[[object.x_col, object.y_col]].to_numpy(dtype=float)
        return unit(np.nanmax(np.ptp(vals, axis=0)), "length", **kwargs)
    return unit(1.0, "length", **kwargs)


def unit_variogram(object, **kwargs):
    vals = object.get("gamma", object.get("SVF", [1.0])) if isinstance(object, dict) else [1.0]
    return unit(vals, "area", **kwargs)


__all__ = [
    "add",
    "convert",
    "dimfig",
    "generate_units",
    "unit",
    "unit_UD",
    "unit_ctmm",
    "unit_par",
    "unit_telemetry",
    "unit_variogram",
    "pct_hash_pct",
    "ustring",
]
