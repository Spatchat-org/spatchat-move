from __future__ import annotations

import numpy as np
import pandas as pd

from .types import Telemetry


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
    if kwargs:
        raise TypeError(f"occurrence got unexpected keyword arguments: {sorted(kwargs)}")
    tracks = data if isinstance(data, list) else [data]
    out = []
    for t in tracks:
        if not isinstance(t, Telemetry):
            raise TypeError("occurrence expects Telemetry or list[Telemetry]")
        df = t.data.copy()
        x = df[t.x_col].to_numpy(dtype=float)
        y = df[t.y_col].to_numpy(dtype=float)
        if len(x) == 0:
            occ = {"x": np.array([]), "y": np.array([]), "PDF": np.array([]), "W": 0.0, "dr": np.array([np.nan, np.nan])}
        else:
            # Simple occupancy proxy: normalized point-mass over observed fixes.
            pdf = np.full(len(x), 1.0 / len(x), dtype=float)
            drx = np.nanmedian(np.abs(np.diff(np.sort(np.unique(x))))) if len(x) > 1 else np.nan
            dry = np.nanmedian(np.abs(np.diff(np.sort(np.unique(y))))) if len(y) > 1 else np.nan
            occ = {"x": x, "y": y, "PDF": pdf, "W": float(len(x)), "dr": np.array([drx, dry], dtype=float)}
        out.append(occ)
    return out[0] if not isinstance(data, list) else out

