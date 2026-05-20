"""Parity-focused translation of ctmm 1.3.0 ``R/variogram.test.R``."""

from __future__ import annotations

import numpy as np
from scipy.stats import f as scipy_f

from .types import CTMMModel, Telemetry
from .stats import F_CI
from .variogram import variogram


def svf_test(x, y, test: str = "F", level: float = 0.95):
    del test
    if isinstance(x, CTMMModel):
        x, y = y, x
    if isinstance(x, Telemetry):
        x = variogram(x)
    if not isinstance(x, dict):
        raise TypeError("svf_test expects a variogram dict or Telemetry")
    svf_x = np.asarray(x.get("SVF", x.get("gamma", [])), dtype=float)
    dof_x = np.asarray(x.get("DOF", np.full(svf_x.shape, np.inf)), dtype=float)
    if isinstance(y, CTMMModel):
        svf_y = np.asarray(x.get("model_SVF", svf_x), dtype=float)
        dof_y = np.asarray(x.get("model_DOF", np.full(svf_x.shape, np.inf)), dtype=float)
    else:
        svf_y = np.asarray(y.get("SVF", y.get("gamma", [])), dtype=float)
        dof_y = np.asarray(y.get("DOF", np.full(svf_y.shape, np.inf)), dtype=float)
    n = min(svf_x.size, svf_y.size, dof_x.size, dof_y.size)
    if n == 0:
        return np.array([0.0, np.nan, np.inf], dtype=float)
    svf_x, svf_y, dof_x, dof_y = svf_x[:n], svf_y[:n], dof_x[:n], dof_y[:n]
    ratio = svf_x / svf_y
    p = np.minimum(scipy_f.cdf(ratio, dof_x, dof_y), scipy_f.sf(ratio, dof_x, dof_y))
    p = np.maximum(p, dof_x < 4)
    i = int(np.nanargmin(p))
    if svf_x[i] > svf_y[i]:
        s1 = svf_x[i]
        v1 = 2.0 * s1 * s1 / dof_x[i]
        s2 = 1.0 / svf_y[i]
        v2 = 2.0 * s2 * s2 / dof_y[i]
    else:
        s1 = svf_y[i]
        v1 = 2.0 * s1 * s1 / dof_y[i]
        s2 = 1.0 / svf_x[i]
        v2 = 2.0 * s2 * s2 / dof_x[i]
    return F_CI(float(s1), float(v1), float(s2), float(v2), level=level)


__all__ = ["svf_test"]
