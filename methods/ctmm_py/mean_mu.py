"""Partial parity translation of ctmm 1.3.0 ``R/mean.mu.R``."""
from __future__ import annotations
import numpy as np

from .diff_ops import midpoint


def flatten_cov_mu(COV_mu):
    arr = np.asarray(COV_mu, dtype=float)
    if arr.ndim == 4:
        arr = np.transpose(arr, (1, 0, 2, 3))
        dim = int(np.prod(arr.shape[:2]))
        arr = arr.reshape((dim, dim), order="C")
    return arr


def mean_mu(data, CTMM, t=None, complete: bool = False):
    return midpoint(data, CTMM=CTMM, t=t, complete=complete)


__all__ = ["flatten_cov_mu", "midpoint", "mean_mu"]
