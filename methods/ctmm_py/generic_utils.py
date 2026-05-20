from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype


def is_good(x: Any) -> bool:
    return (x is not None) and (not (isinstance(x, float) and np.isnan(x))) and bool(x)


def is_bad(x: Any) -> bool:
    return not is_good(x)


def not_in(x, table) -> bool:
    return x not in table


def pos_range(start: int, end: int) -> list[int]:
    if start <= end:
        return list(range(start, end + 1))
    return []


def na_replace(x, rep):
    a = np.asarray(x).copy()
    r = np.asarray(rep)
    m = np.isnan(a)
    a[m] = r[m]
    return a


def nant(x, to):
    a = np.asarray(x).copy()
    m = np.isnan(a)
    if np.any(m):
        t = np.full(a.shape, to, dtype=a.dtype if np.isscalar(to) else None)
        a[m] = t[m]
    return a


def inft(x, to: float = 0.0):
    a = np.asarray(x).copy()
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        return a
    d = np.diag(a)
    m = np.isinf(d)
    if np.any(m):
        idx = np.where(m)[0]
        a[idx, :] = to
        a[:, idx] = to
        a[idx, idx] = np.inf
    return a


def is_even(x):
    return np.asarray(x) % 2 == 0


def is_odd(x):
    return np.asarray(x) % 2 != 0


def last(vec):
    return vec[-1]


def first(vec):
    return vec[0]


def prepend(x, values, before: int = 1):
    i = max(int(before) - 1, 0)
    return list(x[:i]) + list(values) + list(x[i:])


def rm_name(obj, name):
    if isinstance(obj, dict):
        out = dict(obj)
        out.pop(name, None)
        return out
    a = np.asarray(obj)
    return a


def listify(x):
    if x is None:
        return x
    if isinstance(x, list):
        return x
    return [x]


def rename_dict(obj: dict, name1: str, name2: str):
    out = dict(obj)
    if name1 in out:
        out[name2] = out.pop(name1)
    return out


def glue(*parts):
    vals = []
    for p in parts:
        if p is None:
            continue
        s = str(p)
        if s not in vals:
            vals.append(s)
    return " ".join(vals)


def capitalize(s: str) -> str:
    if not s:
        return s
    return s[0].upper() + s[1:]


def mid(x):
    a = np.asarray(x, dtype=float)
    if a.size < 2:
        return np.array([], dtype=float)
    return (a[1:] + a[:-1]) / 2.0


def epoch_seconds(col) -> np.ndarray:
    """Convert datetime-like or numeric timestamps to epoch seconds."""
    if isinstance(col, pd.Series):
        s = col
    else:
        s = pd.Series(col)
    if is_datetime64_any_dtype(s.dtype) or is_datetime64tz_dtype(s.dtype):
        v = s.astype("int64").to_numpy(dtype=float)
    else:
        num = pd.to_numeric(s, errors="coerce")
        if float(np.mean(np.isfinite(num))) >= 0.9:
            v = num.to_numpy(dtype=float)
        else:
            dt = pd.to_datetime(s, utc=True, errors="coerce")
            v = dt.astype("int64").to_numpy(dtype=float)
    m = float(np.nanmedian(np.abs(v))) if v.size else 0.0
    if m > 1e17:
        return v / 1e9
    if m > 1e14:
        return v / 1e6
    if m > 1e11:
        return v / 1e3
    return v
