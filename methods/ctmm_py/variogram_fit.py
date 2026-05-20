"""Partial parity translation of ctmm 1.3.0 ``R/variogram.fit.R``."""

from __future__ import annotations

import numpy as np

from .models import ctmm as _ctmm
from .models import ctmm_guess as _ctmm_guess
from .models import variogram_fit as _variogram_fit


def fraction(z=1.0):
    return float(np.power(4.0, float(z) - 1.0))


def storer(**kwargs):
    return dict(kwargs)


def variogram_guess(variogram: dict, CTMM=None):
    model = _ctmm_guess(variogram, model=CTMM)
    return model


def variogram_fit(variogram, CTMM=None, name: str = "GUESS", fraction: float = 0.5, interactive: bool = False, **kwargs):
    del name, fraction, interactive, kwargs
    if CTMM is None:
        CTMM = _ctmm()
    return _variogram_fit(variogram, model=CTMM)


def variogram_fit_backend(variogram, CTMM=None, fraction: float = 0.5, b: float = 4.0):
    del b
    g = variogram_guess(variogram, CTMM=CTMM)
    return {
        "DF": {"fraction": fraction},
        "storer": lambda **k: g if not k else {**g.params, **k} if hasattr(g, "params") else {**dict(g), **k},
        "fraction": fraction,
    }


__all__ = ["fraction", "storer", "variogram_guess", "variogram_fit", "variogram_fit_backend"]
