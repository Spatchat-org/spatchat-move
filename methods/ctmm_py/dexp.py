"""Parity translation of ctmm 1.3.0 ``R/dexp.R``."""

from __future__ import annotations

import math


def dexp2(x: float, Exp: float | None = None) -> float:
    """
    Numerically-stable evaluation of ``1 - exp(-x)^2``.
    """
    exp_term = math.exp(-x) if Exp is None else float(Exp)
    if exp_term < 0.7071068:
        return float(1.0 - exp_term * exp_term)
    return float(2.0 * exp_term * math.sinh(x))


def dexp1(x: float, Exp: float | None = None) -> float:
    """
    Numerically-stable evaluation of ``1 - exp(-x)``.
    """
    exp_term = math.exp(-x) if Exp is None else float(Exp)
    if exp_term < 0.5:
        return float(1.0 - exp_term)
    return float(2.0 * math.sqrt(exp_term) * math.sinh(x / 2.0))


__all__ = ["dexp2", "dexp1"]
