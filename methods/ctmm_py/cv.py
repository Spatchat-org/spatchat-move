"""Parity-focused translation of ctmm 1.3.0 ``R/cv.R``."""

from __future__ import annotations

import numpy as np

from .generic_utils import epoch_seconds
from .types import CTMMModel, Telemetry
from .fit import ctmm_fit
from .models import ctmm_loglike


def _subset_telem(data: Telemetry, keep_idx: np.ndarray) -> Telemetry:
    df = data.data.iloc[np.asarray(keep_idx, dtype=int)].copy()
    return Telemetry(
        data=df,
        id_col=data.id_col,
        time_col=data.time_col,
        x_col=data.x_col,
        y_col=data.y_col,
        crs=data.crs,
        metadata=dict(data.metadata),
    )


def cv_like(data: Telemetry, CTMM: CTMMModel, IN, method: str, **kwargs) -> float:
    idx_all = np.arange(len(data.data), dtype=int)
    idx_in = np.asarray(IN, dtype=int).reshape(-1)
    idx_in = idx_in[(idx_in >= 0) & (idx_in < len(idx_all))]
    if idx_in.size == 0:
        return float("inf")

    d_in = _subset_telem(data, idx_in)
    fit = ctmm_fit(d_in, CTMM, method=method, COV=False, **kwargs)

    ll_all = ctmm_loglike(data, fit, REML=False, profile=False)
    ll_in = ctmm_loglike(d_in, fit, REML=False, profile=False)
    like = ll_all - ll_in
    return float(like)


def LOOCV(data: Telemetry, CTMM: CTMMModel, cores: int = 1, method: str | None = None, **kwargs) -> float:
    del cores
    m = method if method is not None else str(CTMM.params.get("method", "pHREML"))
    n = len(data.data)
    like = 0.0
    for i in range(n):
        like += cv_like_i(i, data=data, CTMM=CTMM, method=m, **kwargs)
    return float(-2.0 * like)


def cv_like_i(i, data: Telemetry, CTMM: CTMMModel, method: str, **kwargs) -> float:
    idx = np.delete(np.arange(len(data.data), dtype=int), int(i))
    return cv_like(data, CTMM, IN=idx, method=method, **kwargs)


def HSCV(data: Telemetry, CTMM: CTMMModel, cores: int = 1, method: str | None = None, **kwargs) -> float:
    del cores
    m = method if method is not None else str(CTMM.params.get("method", "pHREML"))
    t = epoch_seconds(data.data[data.time_col])
    if t.size < 2:
        return float("inf")
    t_mid = float(t[0] + (t[-1] - t[0]) / 2.0)
    idx_in = np.where(t <= t_mid)[0]
    idx_out = np.where(t > t_mid)[0]
    like = cv_like(data, CTMM, IN=idx_in, method=m, **kwargs) + cv_like(data, CTMM, IN=idx_out, method=m, **kwargs)
    if np.any(np.isclose(t, t_mid)):
        idx_in2 = np.where(t < t_mid)[0]
        idx_out2 = np.where(t >= t_mid)[0]
        like2 = cv_like(data, CTMM, IN=idx_in2, method=m, **kwargs) + cv_like(data, CTMM, IN=idx_out2, method=m, **kwargs)
        like = (like + like2) / 2.0
    return float(-2.0 * like)


__all__ = ["cv_like", "cv_like_i", "LOOCV", "HSCV"]
