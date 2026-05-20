"""Partial parity translation of ctmm 1.3.0 ``R/dt.R``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import Telemetry
from .generic import listify


def dt_plot(data):
    """
    Translation of ``dt.plot`` sampling-schedule extraction.
    Returns sorted positive time intervals as a DataFrame.
    """
    tracks = listify(data)
    vals = []
    for d in tracks:
        if isinstance(d, Telemetry):
            t = pd.to_datetime(d.data[d.time_col], utc=True, errors="coerce").astype("int64") / 1e9
        elif isinstance(d, pd.DataFrame) and "t" in d.columns:
            t = pd.to_numeric(d["t"], errors="coerce").to_numpy(dtype=float)
        else:
            continue
        dt = np.diff(np.asarray(t, dtype=float))
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            vals.append(dt)
    if not vals:
        return pd.DataFrame({"dt": np.array([], dtype=float)})
    out = np.sort(np.concatenate(vals))
    return pd.DataFrame({"dt": out, "index_sorted": np.arange(1, out.size + 1)})


__all__ = ["dt_plot"]
