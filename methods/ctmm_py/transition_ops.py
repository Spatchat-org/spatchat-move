from __future__ import annotations

import numpy as np
import pandas as pd

from .color_ops import color
from .types import Telemetry


def transition(data: Telemetry, n: int = 3, filename: str = "transition", height: int = 2160, **kwargs):
    if kwargs:
        raise TypeError(f"transition got unexpected keyword arguments: {sorted(kwargs)}")
    if not isinstance(data, Telemetry):
        raise TypeError("transition expects a Telemetry object")
    df = data.data.sort_values(data.time_col).copy()
    if df.empty:
        return []
    t = df[data.time_col].astype("int64").to_numpy(dtype=float) / 1e9
    t1, t2 = float(np.min(t)), float(np.max(t))
    base_colors = color(data, by="time")
    frames = []
    for i in range(1, int(n) + 1):
        lo = t1 + (t2 - t1) * (i - 1) / n
        hi = t1 + (t2 - t1) * i / n
        mask = (t >= lo) & (t <= hi)
        c = np.array(base_colors, dtype=object)
        c[~mask] = "#80808020"
        frame = df.copy()
        frame["color"] = c
        frame["frame"] = i
        frames.append(frame)
    return frames

