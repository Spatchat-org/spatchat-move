"""Parity translation of ctmm 1.3.0 ``R/1.R`` runtime helpers."""

from __future__ import annotations

import importlib.util

import numpy as np

from .mag import mag
from .speed_ops import speed, speeds

NAMES_CI = ("low", "est", "high")


def FFT(X, inverse: bool = False):
    arr = np.asarray(X)
    if arr.ndim <= 1:
        return np.fft.ifft(arr) if inverse else np.fft.fft(arr)
    return np.fft.ifft(arr, axis=0) if inverse else np.fft.fft(arr, axis=0)


def FFTW(X, inverse: bool = False):
    return FFT(X, inverse=inverse)


def IFFT(X, plan=None):
    del plan
    return FFT(X, inverse=True)


def is_installed(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def emulate(object, *args, **kwargs):
    from .emulate import emulate as _emulate

    return _emulate(object, *args, **kwargs)


def AICc(object, *args, **kwargs):
    from .aicc import AICc_ctmm, AICc_list

    if isinstance(object, list):
        return AICc_list(object, *args, **kwargs)
    return AICc_ctmm(object, *args, **kwargs)


def ridges(object, *args, **kwargs):
    from .ridge import ridges as _ridges

    return _ridges(object, *args, **kwargs)


def pars(*args, **kwargs):
    if not args:
        return {}
    obj = args[0]
    return dict(getattr(obj, "params", obj if isinstance(obj, dict) else {}))


__all__ = [
    "AICc",
    "FFT",
    "FFTW",
    "IFFT",
    "NAMES_CI",
    "emulate",
    "is_installed",
    "mag",
    "pars",
    "ridges",
    "speed",
    "speeds",
]
