"""Parity translation of ctmm 1.3.0 ``R/aicc.R``."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import ctmm_loglike
from .emulate import emulate as _emulate
from .krige import simulate_ctmm


def simulate(*args: Any, **kwargs: Any) -> Any:
    model = args[0] if args else None
    data = kwargs.get("data", args[1] if len(args) > 1 else None)
    if data is None:
        return model
    return simulate_ctmm(model, data=data, **{k: v for k, v in kwargs.items() if k != "data"})


def emulate(*args: Any, **kwargs: Any) -> Any:
    model = args[0] if args else None
    data = kwargs.pop("data", None)
    return _emulate(model, data=data, **kwargs)


def AICc_list(object: list[Any], *args: Any, **kwargs: Any) -> np.ndarray:
    """
    Python translation of ``AICc.list``.

    Returns a 2-column array with columns:
    - ``mean``: bootstrap AICc target mean
    - ``error``: standard error
    sorted by ``mean`` ascending.
    """
    rows: list[tuple[float, float]] = []
    for model in object:
        res = AICc_ctmm(model, *args, **kwargs)
        rows.append((float(res["mean"]), float(res["error"])))

    if not rows:
        return np.empty((0, 2), dtype=float)

    arr = np.asarray(rows, dtype=float)
    order = np.argsort(arr[:, 0], kind="quicksort")
    return arr[order]


def AICc_ctmm(
    object: Any,
    data: Any,
    n: int = 100,
    fast: bool = False,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Python translation of ``AICc.ctmm`` double-bootstrap estimate for one model.

    """
    if n <= 0:
        return {"mean": float("nan"), "error": float("nan")}

    ctmm_model = object
    likes = np.empty(int(n), dtype=float)

    for i in range(int(n)):
        sim_data = simulate(ctmm_model, data=data)
        fit = emulate(ctmm_model, data=data, fast=fast, **kwargs)
        likes[i] = float(ctmm_loglike(sim_data, fit, profile=False))

    # R: LIKE <- -2*LIKE
    likes = -2.0 * likes
    mean_val = float(np.mean(likes))
    # R: sqrt(stats::var(LIKE)/n) uses sample variance (ddof=1)
    if likes.size > 1:
        se = float(np.sqrt(np.var(likes, ddof=1) / likes.size))
    else:
        se = 0.0
    return {"mean": mean_val, "error": se}


__all__ = ["AICc_list", "AICc_ctmm"]
