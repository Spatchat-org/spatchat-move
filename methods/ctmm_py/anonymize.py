"""Partial parity translation of ctmm 1.3.0 ``R/anonymize.R``."""

from __future__ import annotations

import copy
from datetime import timezone

import numpy as np
import pandas as pd

from .types import Telemetry


def _to_list(data):
    return data if isinstance(data, list) else [data]


def anonymize(data, **kwargs):
    del kwargs
    drop = isinstance(data, Telemetry)
    items = _to_list(data)
    out = []
    for t in items:
        if not isinstance(t, Telemetry):
            out.append(t)
            continue
        z = copy.deepcopy(t)
        df = z.data.copy()
        if z.time_col in df.columns:
            ts = pd.to_datetime(df[z.time_col], utc=True, errors="coerce")
            if ts.notna().any():
                t0 = ts[ts.notna()].iloc[0]
                dt = (ts - t0).dt.total_seconds()
                df["t"] = dt
                df[z.time_col] = pd.Timestamp("1970-01-01 00:00:00", tz="UTC") + pd.to_timedelta(
                    np.maximum(dt, 0.0), unit="s"
                )
        for col in ("timestamp", "longitude", "latitude"):
            if col in df.columns:
                df = df.drop(columns=[col])
        z.data = df
        z.metadata.pop("timezone", None)
        z.metadata.pop("projection", None)
        out.append(z)
    return out[0] if drop and out else out


def pseudonymize(data, center=(0.0, 0.0), datum: str = "WGS84", origin: str = "1111-11-11 11:11:11 UTC", tz: str = "GMT", proj: str | None = None):
    del datum, origin
    drop = isinstance(data, Telemetry)
    items = _to_list(data)
    out = []
    if proj is None:
        proj = f"+proj=aeqd +lon_0={center[0]} +lat_0={center[1]} +datum=WGS84"
    for t in items:
        if not isinstance(t, Telemetry):
            out.append(t)
            continue
        z = copy.deepcopy(t)
        df = z.data.copy()
        if "x" in df.columns and "y" in df.columns:
            df["longitude"] = pd.to_numeric(df["x"], errors="coerce")
            df["latitude"] = pd.to_numeric(df["y"], errors="coerce")
        elif z.x_col in df.columns and z.y_col in df.columns:
            df["longitude"] = pd.to_numeric(df[z.x_col], errors="coerce")
            df["latitude"] = pd.to_numeric(df[z.y_col], errors="coerce")
        if "t" in df.columns:
            base = pd.Timestamp("1970-01-01 00:00:00", tz="UTC")
            df["timestamp"] = base + pd.to_timedelta(pd.to_numeric(df["t"], errors="coerce"), unit="s")
        z.data = df
        z.metadata["projection"] = proj
        z.metadata["timezone"] = tz
        out.append(z)
    return out[0] if drop and out else out


__all__ = ["anonymize", "pseudonymize"]
