"""Partial parity translation of ctmm 1.3.0 ``R/diff.R``."""

from __future__ import annotations

from .diff_ops import difference, distances, midpoint, proximity
from .telemetry import tbind


def combine(data, *args, **kwargs):
    if args:
        data = [data, *args]
    return tbind(data, **kwargs)

__all__ = ["combine", "difference", "midpoint", "distances", "proximity"]
