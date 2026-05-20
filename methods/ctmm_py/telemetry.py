from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer

from .generic_utils import epoch_seconds
from .types import Telemetry

DATUM = "+proj=longlat +datum=WGS84"
ATTRIBUTE = {
    "timestamp": [
        "timestamp", "timestamp.of.fix", "Acquisition.Time", "Date.Time", "Date.Time.GMT",
        "UTC.Date.Time", "DT.TM", "Ser.Local", "GPS_YYYY.MM.DD_HH.MM.SS", "Acquisition.Start.Time",
        "start.timestamp", "Time.GMT", "GMT.Time", "Local.Time", "time", "Date.GMT", "Date",
        "Date.Local", "t", "t_dat", "use_date", "event.Date", "observation.Date",
    ],
    "id": [
        "animal.ID", "AID", "individual.local.identifier", "local.identifier", "individual.ID",
        "Name", "ID", "ID.Names", "Animal", "Full.ID", "tag.local.identifier", "tag.ID",
        "band.number", "band.num", "device.info.serial", "Device.ID", "collar.id", "Logger",
        "Logger.ID", "Deployment", "deployment.ID", "track.ID",
    ],
    "long": [
        "location.longitude", "location.long", "Longitude", "longitude.WGS84", "Longitude.deg",
        "long", "lon", "lng", "GPS.Longitude", "decimal.Longitude",
    ],
    "lat": [
        "location.latitude", "location.lat", "Latitude", "latitude.WGS84", "Latitude.deg",
        "latt", "lat", "GPS.Latitude", "decimal.Latitude",
    ],
    "zone": ["GPS.UTM.zone", "UTM.zone", "zone"],
    "east": ["GPS.UTM.Easting", "GPS.UTM.East", "GPS.UTM.x", "UTM.Easting", "UTM.East", "UTM.E", "UTM.x", "Easting", "East", "x"],
    "north": ["GPS.UTM.Northing", "GPS.UTM.North", "GPS.UTM.y", "UTM.Northing", "UTM.North", "UTM.N", "UTM.y", "Northing", "North", "y"],
    "outliers": ["manually.marked.outlier", "algorithm.marked.outlier", "import.marked.outlier", "marked.outlier", "outlier"],
}


def merge_class(a: Optional[pd.Series], b: Optional[pd.Series]) -> Optional[pd.Series]:
    if a is None and b is None:
        return None
    if a is None:
        return b.astype("string")
    if b is None:
        return a.astype("string")
    aa = a.astype("string")
    bb = b.astype("string")
    out = aa.fillna("NA") + " " + bb.fillna("NA")
    return out.astype("string")


def canonical_name(name: str) -> str:
    s = str(name)
    s = s.replace("\ufeff", "")
    for ch in [".", ":", "_", " ", "-", "(", ")", "[", "]", "（", "）"]:
        s = s.replace(ch, "")
    return s.lower()


def pull_column(df: pd.DataFrame, names: list[str], func=None):
    canon_to_real = {canonical_name(c): c for c in df.columns}
    for n in names:
        cn = canonical_name(n)
        if cn in canon_to_real:
            col = df[canon_to_real[cn]]
            vals = col if func is None else func(col)
            if not pd.isna(vals).all():
                return vals
    return None


def telemetry_clean(data: pd.DataFrame, identity: str, occurrence: bool = False) -> pd.DataFrame:
    d = data.copy()
    if d.shape[0] > 1:
        diff = np.diff(d["t"].to_numpy(dtype=float))
        nz = diff[diff != 0]
        if nz.size:
            diff = diff * nz[0]
            ok = bool(np.all(diff >= 0)) if occurrence else bool(np.all(diff > 0))
            if not ok:
                pass
    d = d.sort_values("t").dropna(subset=["t"])
    if not occurrence:
        # remove duplicate observations (full rows), then check duplicate times
        d = d.drop_duplicates()
        # ctmm warns on duplicate times; here we preserve rows for error-model workflows
        # and allow downstream model-specific handling.
    d = d.reset_index(drop=True)
    return d


def rm_incomplete(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if len(present) == 0 or len(present) == len(cols):
        return df
    out = df.copy()
    for c in present:
        out = out.drop(columns=[c])
    return out


def missing_class(data: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = data.copy()
    levels = [f"[{kind}]", f"[NA-{kind}]"]
    col = "speed" if kind == "speed" else ("z" if kind == "vertical" else kind)
    if col not in out.columns:
        return out
    nas = out[col].isna()
    if not nas.any():
        return out
    if "class" in out.columns:
        miss = pd.Series(np.where(nas, levels[1], levels[0]), index=out.index, dtype="string")
        out["class"] = merge_class(out["class"], miss)
    else:
        out["class"] = pd.Series(np.where(nas, levels[1], levels[0]), index=out.index, dtype="string")
    out.loc[nas, col] = 0
    if kind == "speed" and "heading" in out.columns:
        out.loc[nas, "heading"] = 0
    return out


def _as_telemetry_dataframe(
    object_df: pd.DataFrame,
    *,
    timezone: str = "UTC",
    projection: Optional[str] = None,
    datum: str = "WGS84",
    mark_rm: bool = False,
    na_rm: str = "row",
    occurrence: bool = False,
) -> pd.DataFrame:
    df = object_df.copy()

    # emulate datum parsing behavior (can be plain token or +datum= token)
    datum = str(datum)
    if "+datum=" in datum:
        datum = datum.split("+datum=", 1)[1].split(" +", 1)[0].strip()
    if not datum:
        datum = "WGS84"

    # outlier flag handling (mark.rm path)
    out = pull_column(df, ATTRIBUTE["outliers"], func=lambda s: s.astype("boolean"))
    if out is not None and mark_rm:
        keep = ~(out.fillna(False).astype(bool))
        df = df.loc[keep].copy()

    id_col = pull_column(df, ATTRIBUTE["id"], func=lambda s: s.astype(str))
    if id_col is None:
        df["id"] = "unknown"
    else:
        df["id"] = id_col.astype(str)

    # time column
    tcol = pull_column(df, ATTRIBUTE["timestamp"], func=lambda s: s.astype(str))
    if tcol is None:
        raise ValueError("No timestamp column found")

    def _parse_timestamp_series(series: pd.Series) -> pd.Series:
        raw = series.copy()
        # Try numeric POSIX first (R POSIXct in .rda often arrives as float seconds).
        num = pd.to_numeric(raw, errors="coerce")
        frac_num = float(np.mean(np.isfinite(num))) if len(num) else 0.0
        if frac_num >= 0.95:
            aval = np.abs(num[np.isfinite(num)].to_numpy(dtype=float))
            med = float(np.nanmedian(aval)) if aval.size else np.nan
            # Infer epoch unit by scale; fallback to seconds for POSIXct-like ranges.
            if np.isfinite(med):
                if med > 1e17:
                    unit = "ns"
                elif med > 1e14:
                    unit = "us"
                elif med > 1e11:
                    unit = "ms"
                else:
                    unit = "s"
            else:
                unit = "s"
            ts_num = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
            # If conversion failed badly, fallback to generic parser.
            if float(np.mean(ts_num.notna())) >= 0.9:
                return ts_num

        # Generic parser handles ISO strings and datetime-like values. Pandas can
        # otherwise coerce date-only strings to NaT when mixed with full datetimes.
        ts = pd.to_datetime(raw, utc=True, errors="coerce")
        missing = ts.isna() & raw.notna()
        if missing.any():
            try:
                ts_mixed = pd.to_datetime(raw, utc=True, errors="coerce", format="mixed")
                if ts_mixed.notna().sum() > ts.notna().sum():
                    ts = ts_mixed
            except (TypeError, ValueError):
                for idx in raw[missing].index:
                    val = raw.loc[idx]
                    parsed = pd.to_datetime([val], utc=True, errors="coerce")
                    if len(parsed) and not pd.isna(parsed[0]):
                        ts.loc[idx] = parsed[0]
        return ts

    ts = _parse_timestamp_series(tcol)
    df["timestamp"] = ts.dt.tz_convert("UTC")
    df["t"] = epoch_seconds(df["timestamp"])

    # location columns
    lon = pull_column(df, ATTRIBUTE["long"], func=lambda s: pd.to_numeric(s, errors="coerce"))
    lat = pull_column(df, ATTRIBUTE["lat"], func=lambda s: pd.to_numeric(s, errors="coerce"))
    if lon is None or lat is None:
        # fallback UTM-ish columns if present (ctmm: east/north/zone branch)
        east = pull_column(df, ATTRIBUTE["east"], func=lambda s: pd.to_numeric(s, errors="coerce"))
        north = pull_column(df, ATTRIBUTE["north"], func=lambda s: pd.to_numeric(s, errors="coerce"))
        zone = pull_column(df, ATTRIBUTE["zone"], func=lambda s: s.astype(str))
        if east is None or north is None or zone is None:
            raise ValueError("No supported coordinate columns found")
        zone_vals = zone.to_numpy(dtype=str)
        e = east.to_numpy(dtype=float)
        n = north.to_numpy(dtype=float)
        lon_out = np.full(e.shape, np.nan, dtype=float)
        lat_out = np.full(e.shape, np.nan, dtype=float)
        # convert per-zone, preserving hemisphere suffix if present
        for z in np.unique(zone_vals):
            m = zone_vals == z
            if not np.any(m):
                continue
            zc = str(z).strip()
            hemi = "N"
            if zc and zc[-1].isalpha():
                hemi = zc[-1].upper()
                znum = zc[:-1]
            else:
                znum = zc
            try:
                znum_i = int(float(znum))
            except Exception:
                continue
            # preserve incoming datum token where possible; pyproj EPSG path assumes WGS84
            epsg = 32600 + znum_i if hemi != "S" else 32700 + znum_i
            tr_ll = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
            lo, la = tr_ll.transform(e[m], n[m])
            lon_out[m] = np.asarray(lo, dtype=float)
            lat_out[m] = np.asarray(la, dtype=float)
        df["x"] = e
        df["y"] = n
        df["longitude"] = lon_out
        df["latitude"] = lat_out
    else:
        df["longitude"] = lon.to_numpy(dtype=float)
        df["latitude"] = lat.to_numpy(dtype=float)

    parity_cols = [
        "class",
        "HDOP",
        "VDOP",
        "SDOP",
        "VAR.xy",
        "VAR.z",
        "VAR.v",
        "COV.x.x",
        "COV.x.y",
        "COV.y.y",
        "COV.major",
        "COV.minor",
        "COV.angle",
        "light.time",
        "dark.time",
        "light",
        "sundial",
        "suntime",
    ]
    keep = ["id", "timestamp", "t", "longitude", "latitude"] + ([c for c in ["x", "y"] if c in df.columns])
    keep += [c for c in parity_cols if c in df.columns and c not in keep]
    out_df = df[keep].copy()
    out_df = out_df.dropna(subset=["timestamp"])
    if na_rm == "row":
        out_df = out_df.dropna()
    elif na_rm == "col":
        bad_cols = [c for c in out_df.columns if out_df[c].isna().any()]
        # preserve essential columns
        essential = {"id", "timestamp", "t", "longitude", "latitude"}
        bad_cols = [c for c in bad_cols if c not in essential]
        if bad_cols:
            out_df = out_df.drop(columns=bad_cols)
    # basic parity hooks for missing-class behavior
    if "HDOP" in out_df.columns:
        out_df = missing_class(out_df, "HDOP")
    out_df = rm_incomplete(out_df, ["speed", "heading"])
    return out_df


def _ellipsoid_to_cartesian(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    r_eq = 6378137.0
    r_pl = 6356752.3142
    z = r_pl * np.sin(lat)
    r = r_eq * np.cos(lat)
    x = r * np.cos(lon)
    y = r * np.sin(lon)
    return np.column_stack([x, y, z])


def _cartesian_to_lonlat(cart: np.ndarray) -> np.ndarray:
    x = cart[:, 0]
    y = cart[:, 1]
    z = cart[:, 2]
    lon = np.rad2deg(np.arctan2(y, x))
    r = np.sqrt(x * x + y * y) / 6378137.0
    zz = z / 6356752.3142
    lat = np.rad2deg(np.arctan2(zz, r))
    return np.column_stack([lon, lat])


def _gmedian_rowvec(
    values: np.ndarray,
    *,
    init: np.ndarray | None = None,
    gamma: float = 2.0,
    alpha: float = 0.75,
    nstart: int = 1,
    epsilon: float = 1e-8,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.asarray([], dtype=float)
    if init is not None:
        x = np.vstack([np.asarray(init, dtype=float).reshape(1, -1), x])
    n, p = x.shape
    medvec = x[0].copy()
    medrm = x[0].copy()
    for _ in range(int(nstart)):
        for it in range(1, n):
            diff = x[it] - medrm
            norm = float(np.linalg.norm(diff))
            if norm > epsilon:
                weight = math.sqrt(float(p)) * float(gamma) * float(it + 1) ** (-float(alpha)) / norm
                medrm = medrm + weight * diff
            medvec = medvec + (medrm - medvec) / float(it + 1)
    return medvec


def _gmedian_cov_row_p(
    values: np.ndarray,
    median: np.ndarray,
    *,
    gamma: float = 2.0,
    alpha: float = 0.75,
    nstart: int = 1,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, 0), dtype=float)
    n, p = x.shape
    diff = x[0] - np.asarray(median, dtype=float)
    medav = np.outer(diff, diff)
    medrm = medav.copy()
    for _ in range(int(nstart)):
        for it in range(1, n):
            diff = x[it] - median
            diffmat = np.outer(diff, diff) - medrm
            norm = float(np.linalg.norm(diffmat, ord="fro"))
            weight = 1.0 if norm == 0.0 else min(1.0, float(p) * float(gamma) * float(it + 1) ** (-float(alpha)) / norm)
            medrm = medrm + weight * diffmat
            medav = medav + (medrm - medav) / float(it + 1)
    return medav


def _median_lonlat_two_focus(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    xyz = _ellipsoid_to_cartesian(lon_deg, lat_deg)
    if xyz.shape[0] <= 1:
        return np.repeat(_cartesian_to_lonlat(xyz[:1]), 2, axis=0)
    if xyz.shape[0] == 2:
        foci = _cartesian_to_lonlat(xyz)
    else:
        init = np.median(xyz, axis=0)
        mu = _gmedian_rowvec(xyz, init=init, nstart=10)
        cov = _gmedian_cov_row_p(xyz, mu, nstart=10)
        v1 = np.linalg.eigh(cov)[1][:, -1]
        proj = (xyz - mu) @ v1
        proj = np.asarray(proj, dtype=float) - float(np.median(proj))
        pos = xyz[proj >= 0]
        neg = xyz[proj <= 0]
        if pos.shape[0] == 0:
            pos = xyz
        if neg.shape[0] == 0:
            neg = xyz
        mu1 = pos[0] if pos.shape[0] == 1 else _gmedian_rowvec(pos, init=np.median(pos, axis=0), nstart=2)
        mu2 = neg[0] if neg.shape[0] == 1 else _gmedian_rowvec(neg, init=np.median(neg, axis=0), nstart=2)
        foci = _cartesian_to_lonlat(np.vstack([mu1, mu2]))
    if foci[0, 0] > foci[1, 0]:
        foci = foci[[1, 0], :]
    return foci


def _median_lonlat_two_focus_by_id(lon_deg: np.ndarray, lat_deg: np.ndarray, ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=object)
    unique_ids = pd.unique(ids)
    if unique_ids.size <= 1:
        return _median_lonlat_two_focus(lon_deg, lat_deg)

    foci = []
    for value in unique_ids:
        mask = ids == value
        if np.any(mask):
            foci.append(_median_lonlat_two_focus(lon_deg[mask], lat_deg[mask]))
    if not foci:
        return _median_lonlat_two_focus(lon_deg, lat_deg)
    stacked = np.vstack(foci)
    return _median_lonlat_two_focus(stacked[:, 0], stacked[:, 1])


def as_telemetry(
    df: pd.DataFrame,
    *,
    id_col: str = "animal_id",
    time_col: str = "timestamp",
    x_col: str = "longitude",
    y_col: str = "latitude",
    crs: Optional[str] = None,
    timeformat: str = "auto",
    timezone: str = "UTC",
    projection: Optional[str] = None,
    datum: str = "WGS84",
    dt_hot: float | None = None,
    timeout: float = np.inf,
    na_rm: str = "row",
    mark_rm: bool = False,
    keep: bool = False,
    drop: bool = True,
    occurrence: bool = False,
) -> Telemetry:
    """Canonical telemetry normalizer for parity harness fixtures."""
    data_raw = _as_telemetry_dataframe(
        df,
        timezone=timezone,
        projection=projection,
        datum=datum,
        mark_rm=mark_rm,
        na_rm=na_rm,
        occurrence=occurrence,
    )
    data = data_raw.rename(columns={"id": id_col, "timestamp": time_col}).copy()
    data = telemetry_clean(data, identity="unknown", occurrence=occurrence)
    if x_col not in data.columns or y_col not in data.columns:
        if "longitude" in data.columns and "latitude" in data.columns:
            x_col = "longitude"
            y_col = "latitude"
    data = data.dropna(subset=[time_col, x_col, y_col]).sort_values([id_col, time_col])

    if (x_col, y_col) != ("longitude", "latitude"):
        return Telemetry(
            data=data,
            id_col=id_col,
            time_col=time_col,
            x_col=x_col,
            y_col=y_col,
            crs=crs or projection,
            metadata={"source_x_col": x_col, "source_y_col": y_col, "projected": True},
        )

    # Match ctmm semantics: analyses operate on projected planar coordinates.
    lon = data[x_col].to_numpy(dtype=float)
    lat = data[y_col].to_numpy(dtype=float)
    if projection is not None:
        proj = str(projection).strip()
        if proj and proj.lower() not in {"none", "auto"}:
            tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(proj), always_xy=True)
            xx, yy = tr.transform(lon, lat)
            data["x"] = np.asarray(xx, dtype=float)
            data["y"] = np.asarray(yy, dtype=float)
            return Telemetry(
                data=data,
                id_col=id_col,
                time_col=time_col,
                x_col="x",
                y_col="y",
                crs=projection,
                metadata={"source_lon_col": x_col, "source_lat_col": y_col, "proj4": proj, "projected": True},
            )

    foci = _median_lonlat_two_focus_by_id(lon, lat, data[id_col].to_numpy())
    lon1, lat1 = float(foci[0, 0]), float(foci[0, 1])
    lon2, lat2 = float(foci[1, 0]), float(foci[1, 1])
    proj4 = (
        f"+proj=tpeqd +lon_1={lon1} +lat_1={lat1} "
        f"+lon_2={lon2} +lat_2={lat2} +datum=WGS84 +units=m +no_defs"
    )
    tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_proj4(proj4), always_xy=True)
    xx, yy = tr.transform(lon, lat)
    data["x"] = np.asarray(xx, dtype=float)
    data["y"] = np.asarray(yy, dtype=float)
    x_col_use = "x"
    y_col_use = "y"

    return Telemetry(
        data=data,
        id_col=id_col,
        time_col=time_col,
        x_col=x_col_use,
        y_col=y_col_use,
        crs=crs,
        metadata={"source_lon_col": x_col, "source_lat_col": y_col, "proj4": proj4},
    )


def tbind(*tracks: Telemetry) -> Telemetry:
    """Bind telemetry tracks, preserving canonical schema and sort order."""
    if len(tracks) == 0:
        raise ValueError("tbind requires at least one telemetry object")
    if len(tracks) == 1:
        return tracks[0]

    first = tracks[0]
    frames = []
    for trk in tracks:
        if not isinstance(trk, Telemetry):
            raise TypeError("tbind expects Telemetry objects")
        frames.append(trk.data)

    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    merged = merged.sort_values([first.id_col, first.time_col]).reset_index(drop=True)
    return Telemetry(
        data=merged,
        id_col=first.id_col,
        time_col=first.time_col,
        x_col=first.x_col,
        y_col=first.y_col,
        crs=first.crs,
        metadata={"bound_tracks": len(tracks)},
    )


def dt_plot(telem: Telemetry) -> pd.DataFrame:
    """Return timestamp-interval summary data used by dt.plot-like diagnostics."""
    df = telem.data.sort_values([telem.id_col, telem.time_col])
    out = []
    for aid, grp in df.groupby(telem.id_col, sort=False):
        t = epoch_seconds(grp[telem.time_col])
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            continue
        out.append(
            {
                "id": aid,
                "n": int(grp.shape[0]),
                "dt_min_s": float(np.min(dt)),
                "dt_median_s": float(np.median(dt)),
                "dt_mean_s": float(np.mean(dt)),
                "dt_max_s": float(np.max(dt)),
            }
        )
    return pd.DataFrame(out)


def subset_telemetry(x: Telemetry, *args, **kwargs) -> Telemetry:
    df = x.data
    if args:
        mask = args[0]
        df = df.loc[mask]
    for key, value in kwargs.items():
        df = df.loc[df[key] == value]
    return Telemetry(df.copy(), id_col=x.id_col, time_col=x.time_col, x_col=x.x_col, y_col=x.y_col, crs=x.crs, metadata=dict(x.metadata))


def head_telemetry(x: Telemetry, n: int = 6, **kwargs):
    return x.data.head(n, **kwargs)


def tail_telemetry(x: Telemetry, n: int = 6, **kwargs):
    return x.data.tail(n, **kwargs)


def ubind(x):
    return x


def get_telemetry(data: Telemetry | pd.DataFrame, axes=("x", "y")):
    if isinstance(data, Telemetry):
        df = data.data
        cols = [data.x_col if a in ("x", data.x_col) else data.y_col if a in ("y", data.y_col) else a for a in axes]
    else:
        df = data
        cols = list(axes)
    return df.loc[:, cols].to_numpy(dtype=float)


def set_telemetry(data: Telemetry, values, axes=("x", "y")) -> Telemetry:
    out = Telemetry(data.data.copy(), id_col=data.id_col, time_col=data.time_col, x_col=data.x_col, y_col=data.y_col, crs=data.crs, metadata=dict(data.metadata))
    vals = np.asarray(values)
    cols = [out.x_col if a in ("x", out.x_col) else out.y_col if a in ("y", out.y_col) else a for a in axes]
    for i, col in enumerate(cols):
        out.data[col] = vals[:, i] if vals.ndim > 1 else vals
    return out


def set_name(data: Telemetry, name) -> Telemetry:
    out = Telemetry(data.data.copy(), id_col=data.id_col, time_col=data.time_col, x_col=data.x_col, y_col=data.y_col, crs=data.crs, metadata=dict(data.metadata))
    out.data[out.id_col] = name
    return out


def classnames(uere):
    if isinstance(uere, dict):
        val = uere.get("UERE", uere)
        if hasattr(val, "index"):
            return list(val.index)
    return []


def new_telemetry(data, info=None, UERE=None, **kwargs):
    telem = as_telemetry(data, **kwargs)
    if info:
        telem.metadata.update(info)
    if UERE is not None:
        telem.metadata["UERE"] = UERE
    return telem


def asPOSIXct(x, tz: str = "UTC", **kwargs):
    del kwargs
    return pd.to_datetime(x, utc=True, errors="coerce").dt.tz_convert(tz) if hasattr(x, "dt") else pd.to_datetime(x, utc=True, errors="coerce")


def try_dop(data):
    df = data.data if isinstance(data, Telemetry) else pd.DataFrame(data)
    if "HDOP" not in df.columns:
        df = df.copy()
        df["HDOP"] = 1.0
    return df


def check_class(data):
    df = data.data if isinstance(data, Telemetry) else pd.DataFrame(data)
    if "class" not in df.columns:
        df = df.copy()
        df["class"] = "all"
    return df


def summary_telemetry(object, *args, **kwargs):
    del args, kwargs
    df = object.data if isinstance(object, Telemetry) else pd.DataFrame(object)
    out = {"n": int(len(df))}
    if isinstance(object, Telemetry) and object.time_col in df:
        t = pd.to_datetime(df[object.time_col], utc=True, errors="coerce")
        out["start"] = t.min()
        out["stop"] = t.max()
    return out


def Move2CSV(object, *args, **kwargs):
    del args, kwargs
    return pd.DataFrame(object)


def as_telemetry_data_frame(object, **kwargs):
    return as_telemetry(pd.DataFrame(object), **kwargs)


def as_telemetry_character(object, **kwargs):
    return as_telemetry(pd.read_csv(object), **kwargs)


def as_telemetry_Move(object, **kwargs):
    return as_telemetry(pd.DataFrame(object), **kwargs)


def as_telemetry_MoveStack(object, **kwargs):
    return as_telemetry(pd.DataFrame(object), **kwargs)


def UNFINISHED_as_telemetry_sf(object, **kwargs):
    return as_telemetry(pd.DataFrame(object), **kwargs)

__all__ = [
    "DATUM",
    "as_telemetry",
    "as_telemetry_character",
    "as_telemetry_data_frame",
    "as_telemetry_Move",
    "as_telemetry_MoveStack",
    "asPOSIXct",
    "canonical_name",
    "check_class",
    "classnames",
    "dt_plot",
    "get_telemetry",
    "head_telemetry",
    "merge_class",
    "missing_class",
    "Move2CSV",
    "new_telemetry",
    "pull_column",
    "rm_incomplete",
    "set_name",
    "set_telemetry",
    "subset_telemetry",
    "summary_telemetry",
    "tail_telemetry",
    "telemetry_clean",
    "tbind",
    "try_dop",
    "ubind",
    "UNFINISHED_as_telemetry_sf",
]
