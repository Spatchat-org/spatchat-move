"""Partial parity translation of ctmm 1.3.0 ``R/occurrence.R``."""

from __future__ import annotations

import numpy as np

from .occurrence_ops import occurrence as _occurrence


def occurrence(
    data,
    CTMM,
    R=None,
    SP=None,
    SP_in: bool = True,
    H=0,
    variable: str = "utilization",
    res_time: int = 10,
    res_space: int = 10,
    grid=None,
    cor_min: float = 0.05,
    dt_max=None,
    buffer: bool = True,
    **kwargs,
):
    return _occurrence(
        data,
        CTMM,
        R=R,
        SP=SP,
        SP_in=SP_in,
        H=H,
        variable=variable,
        res_time=res_time,
        res_space=res_space,
        grid=grid,
        cor_min=cor_min,
        dt_max=dt_max,
        buffer=buffer,
        **kwargs,
    )


def currence(*args, **kwargs):
    # Single-individual wrapper in R; here we dispatch through occurrence and unwrap.
    out = occurrence(*args, **kwargs)
    return out[0] if isinstance(out, list) else out


__all__ = ["occurrence", "currence"]
