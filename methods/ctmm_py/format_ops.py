from __future__ import annotations

import numpy as np


def _format_num(x: float, digits: int) -> str:
    if np.isposinf(x):
        return "Inf"
    if np.isneginf(x):
        return "-Inf"
    s = format(float(x), f".{max(int(digits),1)}g")
    if "e" not in s and "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def sigfig(est, var=None, sd=None, level: float = 0.95, digits: int = 2):
    if var is not None:
        sd = np.sqrt(np.asarray(var, dtype=float))
    if sd is not None:
        est = np.asarray(est, dtype=float)
        sd = np.asarray(sd, dtype=float)
        z = 1.959963984540054 if abs(level - 0.95) < 1e-12 else 1.959963984540054
        ci = np.column_stack([est - z * sd, est, est + z * sd])
        return sigfig(ci, digits=digits)

    arr = np.asarray(est, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            return np.array([_format_num(v, digits) for v in arr], dtype=object)
        arr = arr[None, :]
    if arr.shape[1] < 3:
        return np.array([_format_num(v, digits) for v in arr.ravel()], dtype=object)

    out = []
    for row in arr:
        low, mid, high = float(row[0]), float(row[1]), float(row[2])
        d1 = abs(mid - low)
        d2 = abs(high - mid)
        ref = min(d1, d2) if (np.isfinite(d1) and np.isfinite(d2) and d1 > 0 and d2 > 0) else max(d1, d2, 1.0)
        pow10 = np.floor(np.log10(ref) + np.finfo(float).eps) - digits + 1
        sig = int(np.floor(np.log10(abs(mid))) - pow10 + 1) if mid != 0 else digits
        low_s = _format_num(low, sig)
        mid_s = _format_num(mid, sig)
        high_s = _format_num(high, sig)
        out.append(f"{mid_s} ({low_s},{high_s})")
    return np.array(out, dtype=object)


def _unit_scale_table(dimension: str):
    dim = str(dimension).lower()
    if dim in {"length", "distance"}:
        return [
            ("microns", "um", 1e-6),
            ("milimeters", "mm", 1e-3),
            ("centimeters", "cm", 1e-2),
            ("meters", "m", 1.0),
            ("kilometers", "km", 1e3),
        ]
    if dim == "area":
        return [
            ("square microns", "um^2", 1e-12),
            ("square milimeters", "mm^2", 1e-6),
            ("square centimeters", "cm^2", 1e-4),
            ("square meters", "m^2", 1.0),
            ("hectares", "hm^2", 1e4),
            ("square kilometers", "km^2", 1e6),
        ]
    if dim in {"speed", "velocity"}:
        day = 86400.0
        return [
            ("microns/day", "um/day", 1e-6 / day),
            ("milimeters/day", "mm/day", 1e-3 / day),
            ("centimeters/day", "cm/day", 1e-2 / day),
            ("meters/day", "m/day", 1.0 / day),
            ("kilometers/day", "km/day", 1e3 / day),
        ]
    if dim == "time":
        day = 86400.0
        month = 2.8 + 60.0 * (44.0 + 60.0 * (12.0 + 24.0 * 29.0))
        year = 365.24217 * day
        return [
            ("microseconds", "us", 1e-6),
            ("miliseconds", "ms", 1e-3),
            ("seconds", "sec", 1.0),
            ("minutes", "min", 60.0),
            ("hours", "hr", 3600.0),
            ("days", "day", day),
            ("months", "mon", month),
            ("years", "yr", year),
        ]
    return [("", "", 1.0)]


def dimfig(data, dimension: str, thresh: float = 1.0):
    arr = np.asarray(data, dtype=float)
    arrf = arr[np.isfinite(arr)]
    max_data = float(np.max(np.abs(arrf))) if arrf.size else 1.0
    table = _unit_scale_table(dimension)
    idx = 0
    for i, (_, _, scale) in enumerate(table):
        if max_data >= thresh * scale:
            idx = i
    name, abrv, scale = table[idx]
    return {"data": arr / scale, "units": (name, abrv)}

