"""Partial parity translation of ctmm 1.3.0 ``R/transition.R``."""

from __future__ import annotations

from .transition_ops import transition as _transition


def transition(data, n: int = 3, filename: str = "transition", height: int = 2160, **kwargs):
    # Delegates to the implemented transition frame-splitting logic.
    return _transition(data, n=n, filename=filename, height=height, **kwargs)


__all__ = ["transition"]
