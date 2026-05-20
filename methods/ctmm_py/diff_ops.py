from __future__ import annotations

import numpy as np
import pandas as pd

from .types import Telemetry


def _sync_pair(a: Telemetry, b: Telemetry) -> pd.DataFrame:
    da = a.data[[a.time_col, a.x_col, a.y_col]].copy()
    db = b.data[[b.time_col, b.x_col, b.y_col]].copy()
    da = da.rename(columns={a.time_col: "timestamp", a.x_col: "x1", a.y_col: "y1"})
    db = db.rename(columns={b.time_col: "timestamp", b.x_col: "x2", b.y_col: "y2"})
    m = pd.merge(da, db, on="timestamp", how="inner").sort_values("timestamp")
    if m.empty:
        return m
    m["t"] = m["timestamp"].astype("int64") / 1e9
    return m


def difference(data, CTMM=None, t=None):
    if not isinstance(data, (list, tuple)) or len(data) != 2:
        raise TypeError("difference expects data=[telemetry1, telemetry2]")
    a, b = data
    if not isinstance(a, Telemetry) or not isinstance(b, Telemetry):
        raise TypeError("difference expects Telemetry objects")
    m = _sync_pair(a, b)
    if m.empty:
        return Telemetry(pd.DataFrame(columns=[a.id_col, a.time_col, "x", "y", "t"]), id_col=a.id_col, time_col=a.time_col, x_col="x", y_col="y", crs=a.crs)
    if t is not None:
        if len(t) == 2:
            t1, t2 = float(t[0]), float(t[1])
            m = m[(m["t"] >= t1) & (m["t"] <= t2)]
    out = pd.DataFrame(
        {
            a.id_col: [f"diff({a.data[a.id_col].iloc[0]},{b.data[b.id_col].iloc[0]})"] * len(m),
            a.time_col: m["timestamp"].to_numpy(),
            "t": m["t"].to_numpy(),
            "x": (m["x1"] - m["x2"]).to_numpy(),
            "y": (m["y1"] - m["y2"]).to_numpy(),
        }
    )
    return Telemetry(out, id_col=a.id_col, time_col=a.time_col, x_col="x", y_col="y", crs=a.crs)


def midpoint(data, CTMM=None, t=None, complete: bool = False):
    if not isinstance(data, (list, tuple)) or len(data) != 2:
        raise TypeError("midpoint expects data=[telemetry1, telemetry2]")
    a, b = data
    m = _sync_pair(a, b)
    if m.empty:
        return Telemetry(pd.DataFrame(columns=[a.id_col, a.time_col, "x", "y", "t"]), id_col=a.id_col, time_col=a.time_col, x_col="x", y_col="y", crs=a.crs)
    if t is not None and len(t) == 2:
        t1, t2 = float(t[0]), float(t[1])
        m = m[(m["t"] >= t1) & (m["t"] <= t2)]
    out = pd.DataFrame(
        {
            a.id_col: [f"mean({a.data[a.id_col].iloc[0]},{b.data[b.id_col].iloc[0]})"] * len(m),
            a.time_col: m["timestamp"].to_numpy(),
            "t": m["t"].to_numpy(),
            "x": ((m["x1"] + m["x2"]) / 2.0).to_numpy(),
            "y": ((m["y1"] + m["y2"]) / 2.0).to_numpy(),
        }
    )
    return Telemetry(out, id_col=a.id_col, time_col=a.time_col, x_col="x", y_col="y", crs=a.crs)


def distances(data, CTMM=None, t=None, level: float | None = 0.95):
    d = difference(data, CTMM=CTMM, t=t)
    if d.data.empty:
        return pd.DataFrame(columns=["t", "timestamp", "low", "est", "high"])
    x = d.data["x"].to_numpy(dtype=float)
    y = d.data["y"].to_numpy(dtype=float)
    est = np.sqrt(x * x + y * y)
    out = pd.DataFrame(
        {
            "t": d.data["t"].to_numpy(dtype=float),
            "timestamp": d.data[d.time_col].to_numpy(),
            "est": est,
        }
    )
    if level is None:
        out["DOF"] = np.inf
        out["VAR"] = 0.0
        return out
    out["low"] = est
    out["high"] = est
    return out[["t", "timestamp", "low", "est", "high"]]


def proximity(data, CTMM=None, t=None, level: float = 0.95, debias: bool = True, GUESS=None):
    ds = distances(data, CTMM=CTMM, t=t, level=None)
    if ds.empty:
        return {"low": 0.0, "est": 1.0, "high": float("inf")}
    r = ds["est"].to_numpy(dtype=float)
    mean_d = float(np.nanmean(r))
    est = 1.0 / max(mean_d, np.finfo(float).eps)
    return {"low": est, "est": est, "high": est}

