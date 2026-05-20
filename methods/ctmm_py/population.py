"""Parity-focused translation of ctmm 1.3.0 ``R/population.R``."""
from __future__ import annotations

from .meta_chisq import meta


def population(*args, **kwargs):
    return meta(*args, **kwargs)


__all__ = ["meta", "population"]
