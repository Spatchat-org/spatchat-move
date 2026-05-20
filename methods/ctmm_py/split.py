"""Parity translation of ctmm 1.3.0 ``R/split.R`` non-interactive cleave."""

from __future__ import annotations

from .types import Telemetry


def cleave(object, fraction: float = 0.5, name: str = "CLEFT", **kwargs):
    del name, kwargs
    if not isinstance(object, Telemetry):
        raise TypeError("cleave expects a Telemetry object")
    data = object.data.sort_values(object.time_col).reset_index(drop=True)
    n = len(data)
    if n == 0:
        first = second = data.copy()
    else:
        t = data[object.time_col].astype("int64").to_numpy(dtype=float) / 1e9
        cutoff = t[0] + float(fraction) * (t[-1] - t[0])
        idx = int((t < cutoff).sum())
        first = data.iloc[:idx].copy()
        second = data.iloc[idx:].copy()
    return {
        "before": Telemetry(first, id_col=object.id_col, time_col=object.time_col, x_col=object.x_col, y_col=object.y_col, crs=object.crs, metadata=dict(object.metadata)),
        "after": Telemetry(second, id_col=object.id_col, time_col=object.time_col, x_col=object.x_col, y_col=object.y_col, crs=object.crs, metadata=dict(object.metadata)),
    }


__all__ = ["cleave"]
