"""Math helpers from ctmm 1.3.0 ``R/math.R`` (subset used by Kalman / variogram)."""

from __future__ import annotations

import numpy as np


def sinc(x, sin=None):
    """``sinc(x, SIN=sin(x))`` — R returns ``1`` at ``x==0``."""
    x = np.asarray(x, dtype=float)
    if sin is None:
        sin = np.sin(x)
    else:
        sin = np.asarray(sin, dtype=float)
    out = np.ones_like(x, dtype=float)
    np.divide(sin, x, out=out, where=x != 0)
    return out


def sinch(x, sinh=None):
    """``sinch(x, SINH=sinh(x))`` — R returns ``1`` at ``x==0``."""
    x = np.asarray(x, dtype=float)
    if sinh is None:
        sh = np.sinh(x)
    else:
        sh = np.asarray(sinh, dtype=float)
    out = np.ones_like(x, dtype=float)
    np.divide(sh, x, out=out, where=x != 0)
    return out


__all__ = ["sinc", "sinch"]
