from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from .models import ctmm_loglike
from .types import CTMMModel, Telemetry

try:
    from scipy import optimize as _sp_opt
except Exception:  # pragma: no cover
    _sp_opt = None


def SpatialPoints_telemetry(x):
    if isinstance(x, list):
        parts = [SpatialPoints_telemetry(t) for t in x]
        return pd.concat(parts, axis=0, ignore_index=True)
    if not isinstance(x, Telemetry):
        raise TypeError("SpatialPoints.telemetry expects Telemetry or list[Telemetry]")
    df = x.data.copy()
    return pd.DataFrame({"x": df[x.x_col].to_numpy(dtype=float), "y": df[x.y_col].to_numpy(dtype=float)})


def SpatialPointsDataFrame_telemetry(x):
    if isinstance(x, list):
        parts = [SpatialPointsDataFrame_telemetry(t) for t in x]
        return pd.concat(parts, axis=0, ignore_index=True)
    if not isinstance(x, Telemetry):
        raise TypeError("SpatialPointsDataFrame.telemetry expects Telemetry or list[Telemetry]")
    df = x.data.copy()
    out = pd.DataFrame(
        {
            "x": df[x.x_col].to_numpy(dtype=float),
            "y": df[x.y_col].to_numpy(dtype=float),
            "identity": df[x.id_col].astype(str).to_numpy(),
            "timestamp": df[x.time_col].to_numpy(),
        }
    )
    return out


def SpatialPolygonsDataFrame_telemetry(x, level_UD: float = 0.95):
    # Lightweight stand-in: points with nominal radius metadata.
    spdf = SpatialPointsDataFrame_telemetry(x)
    spdf = spdf.copy()
    spdf["level_UD"] = float(level_UD)
    spdf["radius"] = 10.0
    return spdf


def as_sf(x, error: bool = False, **kwargs):
    if kwargs:
        raise TypeError(f"as.sf got unexpected keyword arguments: {sorted(kwargs)}")
    if isinstance(x, Telemetry) or (isinstance(x, list) and x and isinstance(x[0], Telemetry)):
        df = SpatialPolygonsDataFrame_telemetry(x) if error else SpatialPointsDataFrame_telemetry(x)
    elif isinstance(x, pd.DataFrame):
        df = x.copy()
    else:
        raise TypeError("as.sf currently supports Telemetry/list[Telemetry]/DataFrame")
    if "x" in df.columns and "y" in df.columns:
        df = df.copy()
        df["geometry"] = "POINT(" + df["x"].astype(str) + " " + df["y"].astype(str) + ")"
    return df


def optimizer(
    par,
    fn,
    *fn_args,
    method: str = "L-BFGS-B",
    lower=None,
    upper=None,
    control: dict[str, Any] | None = None,
    **fn_kwargs,
):
    p0 = np.asarray(par, dtype=float)
    control = control or {}
    maxiter = int(control.get("maxit", control.get("maxiter", 1000)))
    if _sp_opt is None:
        val0 = float(fn(p0, *fn_args, **fn_kwargs))
        return {"par": p0, "value": val0, "counts": 1, "convergence": 1, "method": "fallback"}

    bounds = None
    if lower is not None or upper is not None:
        lo = np.full_like(p0, -np.inf) if lower is None else np.asarray(lower, dtype=float)
        hi = np.full_like(p0, np.inf) if upper is None else np.asarray(upper, dtype=float)
        if lo.shape == ():
            lo = np.full_like(p0, float(lo))
        if hi.shape == ():
            hi = np.full_like(p0, float(hi))
        bounds = list(zip(lo.tolist(), hi.tolist()))

    meth = "Nelder-Mead" if method == "pNewton" else method
    res = _sp_opt.minimize(
        lambda p: float(fn(np.asarray(p, dtype=float), *fn_args, **fn_kwargs)),
        p0,
        method=meth,
        bounds=bounds,
        options={"maxiter": maxiter},
    )
    return {
        "par": np.asarray(res.x, dtype=float),
        "value": float(res.fun),
        "counts": int(getattr(res, "nfev", 0)),
        "convergence": 0 if bool(res.success) else 1,
        "method": meth,
        "message": str(res.message),
    }


def ctmm_boot(data: Telemetry, CTMM: CTMMModel, nboot: int = 100, seed: int | None = None, **kwargs):
    if kwargs:
        raise TypeError(f"ctmm.boot got unexpected keyword arguments: {sorted(kwargs)}")
    if not isinstance(data, Telemetry):
        raise TypeError("ctmm.boot expects Telemetry as first argument")
    if not isinstance(CTMM, CTMMModel):
        raise TypeError("ctmm.boot expects CTMMModel as second argument")
    rng = np.random.default_rng(seed)
    df = data.data.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return {"samples": np.array([]), "mean": np.nan, "sd": np.nan, "nboot": int(nboot)}
    vals = np.empty(int(nboot), dtype=float)
    for i in range(int(nboot)):
        idx = rng.integers(0, n, size=n)
        bdf = df.iloc[idx].sort_values(data.time_col).reset_index(drop=True)
        btele = Telemetry(
            data=bdf,
            id_col=data.id_col,
            time_col=data.time_col,
            x_col=data.x_col,
            y_col=data.y_col,
            crs=data.crs,
            metadata=dict(data.metadata),
        )
        vals[i] = float(ctmm_loglike(btele, CTMM))
    return {"samples": vals, "mean": float(np.nanmean(vals)), "sd": float(np.nanstd(vals, ddof=1)), "nboot": int(nboot)}

