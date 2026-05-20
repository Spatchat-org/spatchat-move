"""Partial parity translation of ctmm 1.3.0 ``R/funnel.R``."""
from __future__ import annotations
import numpy as np

def funnel(est, se):
    est = np.asarray(est, dtype=float).reshape(-1)
    se = np.asarray(se, dtype=float).reshape(-1)
    z = est / np.maximum(se, np.finfo(float).eps)
    return {"est": est, "se": se, "z": z}

__all__ = ["funnel"]
