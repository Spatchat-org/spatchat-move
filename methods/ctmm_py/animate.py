"""Partial parity translation of ctmm 1.3.0 ``R/animate.R``."""
from __future__ import annotations

def video(data, n_frames: int = 100, **kwargs):
    del kwargs
    return {"frames": int(n_frames), "data": data, "format": "in-memory"}

__all__ = ["video"]
