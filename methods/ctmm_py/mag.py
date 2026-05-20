"""Parity translation of ctmm 1.3.0 ``R/mag.R``."""

from __future__ import annotations

import numpy as np

from .types import Telemetry


def mag_numeric(x) -> float:
    """R ``mag.numeric`` behavior: Frobenius norm of a columnized vector/matrix."""
    arr = np.asarray(x)
    arr = np.atleast_2d(arr).reshape(-1, 1)
    return float(np.linalg.norm(arr, ord="fro"))


def mag_complex(x) -> float:
    """R ``mag.complex`` behavior for complex-valued vectors."""
    arr = np.asarray(x, dtype=complex)
    arr = np.atleast_2d(arr).reshape(-1, 1)
    v = np.vdot(arr[:, 0], arr[:, 0])
    return float(np.sqrt(np.real(v)))


def mag_telemetry(x: Telemetry, axes: tuple[str, str] = ("x", "y")) -> np.ndarray:
    """R ``mag.telemetry`` behavior: pointwise magnitude from selected axes."""
    df = x.data
    arr = df.loc[:, list(axes)].to_numpy(dtype=float)
    return np.sqrt(np.sum(arr * arr, axis=1))


def mag(x, axes: tuple[str, str] = ("x", "y")):
    """Dispatch-compatible ``mag`` parity helper."""
    if isinstance(x, Telemetry):
        return mag_telemetry(x, axes=axes)
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        return mag_complex(arr)
    return mag_numeric(arr)


__all__ = ["mag", "mag_numeric", "mag_complex", "mag_telemetry"]
