from __future__ import annotations

import numpy as np
import pandas as pd

from .speed_ops import speeds
from .types import Telemetry


def outlie(data, plot: bool = True, by: str = "d", units: bool = True, **kwargs):
    if kwargs:
        raise TypeError(f"outlie got unexpected keyword arguments: {sorted(kwargs)}")
    if isinstance(data, list):
        return [outlie(d, plot=plot, by=by, units=units) for d in data]
    if not isinstance(data, Telemetry):
        raise TypeError("outlie expects Telemetry or list[Telemetry]")

    df = data.data.sort_values(data.time_col).copy()
    t = df[data.time_col].astype("int64").to_numpy(dtype=float) / 1e9
    x = df[data.x_col].to_numpy(dtype=float)
    y = df[data.y_col].to_numpy(dtype=float)

    x0 = np.nanmedian(x)
    y0 = np.nanmedian(y)
    d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    sp = speeds(data)
    speed_by_t = dict(zip(sp["t"].to_numpy(dtype=float), sp["speed"].to_numpy(dtype=float)))
    v = np.full(len(t), np.nan, dtype=float)
    for i in range(1, len(t)):
        tm = (t[i - 1] + t[i]) / 2.0
        v[i] = speed_by_t.get(tm, np.nan)
    v[0] = v[1] if len(v) > 1 else np.nan

    out = pd.DataFrame(
        {
            "t": t,
            "distance": d,
            "VAR.distance": np.zeros_like(d),
            "speed": v,
            "VAR.speed": np.zeros_like(v),
        }
    )
    return out

