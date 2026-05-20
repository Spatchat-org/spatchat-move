"""Partial parity translation of ctmm 1.3.0 ``R/generic.R``."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def is_good(x: Any) -> bool:
    return x is not None and not (isinstance(x, float) and np.isnan(x)) and bool(x)


def is_bad(x: Any) -> bool:
    return not is_good(x)


def getMethod(fn, signature=None, *args, **kwargs):
    del args, kwargs
    sig = signature[0] if isinstance(signature, (list, tuple)) else signature
    name = f"{fn}_{sig}" if sig else str(fn)
    obj = globals().get(name) or globals().get(str(fn))
    if obj is None:
        raise AttributeError(f"Cannot find method {fn} for class {signature}")
    return obj


def match_arg(arg, choices, *args, **kwargs):
    del args, kwargs
    if arg is None:
        return None
    if isinstance(arg, float) and np.isnan(arg):
        return arg if any(isinstance(c, float) and np.isnan(c) for c in choices) else None
    if arg in choices:
        return arg
    matches = [c for c in choices if str(c).startswith(str(arg))]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"{arg!r} should be one of {choices!r}")


def nin(x, table) -> np.ndarray:
    arr = np.asarray(x)
    return ~np.isin(arr, np.asarray(table))


def pos_seq(x: int, y: int) -> list[int]:
    return list(range(x, y + 1)) if x <= y else []


def composite(n: int) -> int:
    return int(2 ** np.ceil(np.log2(max(int(n), 1))))


def na_replace(x, rep):
    a = np.asarray(x).copy()
    r = np.asarray(rep)
    mask = np.isnan(a)
    a[mask] = r[mask]
    return a


def nant(x, to):
    a = np.asarray(x).copy()
    mask = ~np.isfinite(a)
    if np.any(mask):
        repl = np.broadcast_to(np.asarray(to), a.shape)
        a[mask] = repl[mask]
    return a


def inft(x, to: float = 0.0):
    a = np.asarray(x, dtype=float).copy()
    if a.ndim != 2:
        return a
    d = np.diag(a)
    inf_mask = np.isinf(d)
    if np.any(inf_mask):
        a[inf_mask, :] = to
        a[:, inf_mask] = to
        for i, flag in enumerate(inf_mask):
            if flag:
                a[i, i] = np.inf
    return a


def last(vec):
    return vec[-1]


def first(vec):
    return vec[0]


def prepend(x, values, before: int = 1):
    i = max(int(before) - 1, 0)
    return list(x[:i]) + list(np.atleast_1d(values)) + list(x[i:])


def clamp(num, min_: float = 0.0, max_: float = 1.0):
    return np.clip(num, min_, max_)


def pad(vec, size: int | None = None, padding=0, side: int = +1):
    v = list(vec)
    size = len(v) if size is None else int(size)
    diff = max(size - len(v), 0)
    p = [padding] * diff
    if side > 0:
        return v + p
    if side < 0:
        return p + v
    return v


def rpad(mat, size: int | None = None, padding=0, side: int = +1):
    arr = np.asarray(mat)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    size = arr.shape[0] if size is None else int(size)
    diff = max(size - arr.shape[0], 0)
    pad_arr = np.full((diff, arr.shape[1]), padding, dtype=arr.dtype if arr.size else float)
    return np.vstack([arr, pad_arr]) if side > 0 else np.vstack([pad_arr, arr])


def mpad(mat, size: int | None = None, padding=0, side: int = +1, padname=None, **kwargs):
    del padname, kwargs
    arr = np.asarray(mat)
    size = max(arr.shape) if size is None else int(size)
    out = rpad(arr, size=size, padding=padding, side=side)
    out = rpad(out.T, size=size, padding=padding, side=side).T
    return out


def rm_name(object, name):
    names = set(np.atleast_1d(name).astype(str))
    if isinstance(object, pd.DataFrame):
        keep_i = [i for i in object.index if str(i) not in names]
        keep_c = [c for c in object.columns if str(c) not in names]
        return object.loc[keep_i, keep_c]
    if isinstance(object, dict):
        return {k: v for k, v in object.items() if str(k) not in names}
    return object


def listify(x):
    if x is None:
        return x
    return x if isinstance(x, list) else [x]


def glue(*args: Any) -> str:
    vals = [v for v in args if v is not None]
    uniq = []
    for v in vals:
        if v not in uniq:
            uniq.append(v)
    return " ".join(str(v) for v in uniq)


def capitalize(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def rename(object, name1, name2):
    if isinstance(object, pd.DataFrame):
        return object.rename(columns={name1: name2}, index={name1: name2})
    if isinstance(object, dict):
        out = dict(object)
        if name1 in out:
            out[name2] = out.pop(name1)
        return out
    return object


def rename_matrix(object, name1, name2):
    return rename(object, name1, name2)


def simplify_formula(x):
    return x


def copy(from_, to):
    return copy_fields(from_, to)


def copy_fields(from_: dict[str, Any], to: dict[str, Any]) -> dict[str, Any]:
    for k, v in from_.items():
        to[k] = v
    return to


def mid(x):
    a = np.asarray(x, dtype=float)
    if a.size < 2:
        return np.array([], dtype=float)
    return (a[1:] + a[:-1]) / 2.0


def _list_first_method(x, name, *args, **kwargs):
    if not x:
        return None
    fn = globals().get(name)
    if fn is None:
        return x
    return fn(x, *args, **kwargs)


def zoom_list(x, *args, **kwargs):
    return _list_first_method(x, "zoom", *args, **kwargs)


def log_list(x, *args, **kwargs):
    from .log import log_ctmms

    return log_ctmms(x, *args, **kwargs)


def mean_list(x, *args, **kwargs):
    return np.nanmean(np.asarray(x, dtype=float), *args, **kwargs)


def median_list(x, na_rm: bool = False, *args, **kwargs):
    del args, kwargs
    arr = np.asarray(x, dtype=float)
    return np.nanmedian(arr) if na_rm else np.median(arr)


def plot_list(x, *args, **kwargs):
    from .plot_telemetry import plot_list as _plot_list

    return _plot_list(x, *args, **kwargs)


def summary_list(object, *args, **kwargs):
    return [getattr(o, "summary", lambda *a, **k: o)(*args, **kwargs) for o in object]


def writeVector_list(x, filename=None, *args, **kwargs):
    del args, kwargs
    if filename is None:
        return x
    import json

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(x, f, default=str)
    return filename


def name_list(x):
    if isinstance(x, list):
        return x
    return x


__all__ = [
    "is_good",
    "is_bad",
    "getMethod",
    "match_arg",
    "nin",
    "pos_seq",
    "composite",
    "na_replace",
    "nant",
    "inft",
    "last",
    "first",
    "prepend",
    "clamp",
    "pad",
    "rpad",
    "mpad",
    "rm_name",
    "listify",
    "rename",
    "rename_matrix",
    "glue",
    "capitalize",
    "simplify_formula",
    "copy",
    "copy_fields",
    "mid",
    "zoom_list",
    "log_list",
    "mean_list",
    "median_list",
    "plot_list",
    "summary_list",
    "writeVector_list",
    "name_list",
]
