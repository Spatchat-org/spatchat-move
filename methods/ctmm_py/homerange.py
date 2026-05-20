"""Parity-focused translation of ctmm 1.3.0 ``R/homerange.R``."""

from __future__ import annotations

from .types import CTMMModel, Telemetry
from .kde import akde


def agde(data: Telemetry | None = None, CTMM: CTMMModel | None = None, **kwargs):
    # In R this constructs a Gaussian home-range UD.
    # Current parity behavior: route to AKDE with fixed model/data pairing.
    if isinstance(data, CTMMModel):
        data, CTMM = CTMM, data
    if CTMM is None:
        raise ValueError("agde requires CTMM model.")
    if data is None:
        raise ValueError("agde currently requires data in this port.")
    return akde(data, CTMM, **kwargs)


def homerange(data=None, CTMM=None, method: str = "AKDE", **kwargs):
    m = str(method).upper()
    if m not in {"AKDE", "AGDE"}:
        raise ValueError("method must be one of: AKDE, AGDE")

    if isinstance(data, CTMMModel):
        data, CTMM = CTMM, data

    if data is None:
        m = "AGDE"

    if m == "AKDE":
        if data is None or CTMM is None:
            raise ValueError("AKDE homerange requires both data and CTMM.")
        return akde(data, CTMM, **kwargs)
    return agde(data, CTMM, **kwargs)


__all__ = ["homerange", "agde"]
