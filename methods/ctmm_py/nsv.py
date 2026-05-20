"""Partial parity translation of ctmm 1.3.0 ``R/nsv.R``."""
from __future__ import annotations
from .speed_ops import speed


def nsv(data, **kwargs):
    return speed(data, **kwargs)


def plot_nsv(x, *args, **kwargs):
    del args, kwargs
    return {"type": "nsv", "object": x}


__all__ = ["speed", "nsv", "plot_nsv"]
