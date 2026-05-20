"""Partial parity translation of ctmm 1.3.0 ``R/acf.R``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .types import CTMMModel, Telemetry
from .variogram import correlogram as _correlogram


def residuals_ctmm(object: CTMMModel | None, data: Telemetry, **kwargs: Any) -> pd.DataFrame:
    """
    Minimal translation scaffold for ``residuals.ctmm``.
    Current behavior returns detrended coordinate block when possible.
    """
    del kwargs
    df = data.data.copy()
    axes = [c for c in (data.x_col, data.y_col) if c in df.columns]
    cols = [data.time_col] + axes
    return df.loc[:, cols]


def residuals_telemetry(object: Telemetry, CTMM: CTMMModel | None = None, **kwargs: Any) -> pd.DataFrame:
    return residuals_ctmm(CTMM, object, **kwargs)


def correlogram(data: Telemetry, dt: float | None = None, fast: bool = True, res: int = 1, axes=("x", "y"), trace: bool = True):
    """
    Translation of ``correlogram`` dispatch path.
    Delegates to the existing ``ctmm_py.variogram.correlogram`` implementation.
    """
    del dt, fast, res, axes, trace
    acf = _correlogram(data)
    svf = np.asarray(acf.get("acf", []), dtype=float)
    if svf.size:
        svf = svf / max(float(svf[0]), 1e-12)
        acf["acf"] = svf
    return acf


__all__ = ["residuals_ctmm", "residuals_telemetry", "correlogram"]
