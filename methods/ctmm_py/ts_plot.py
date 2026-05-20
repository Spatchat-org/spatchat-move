"""Partial parity translation of ctmm 1.3.0 ``R/ts.plot.R``."""
from __future__ import annotations
from .dt import dt_plot


def ts_plot(x, **kwargs):
    return dt_plot(x, **kwargs)


__all__ = ["dt_plot", "ts_plot"]
