# estimators/akde.py
import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from pyproj import Transformer
from skimage import measure

import storage  # absolute import


@dataclass
class AKDEParams:
    bandwidth_m: Optional[float] = None
    grid_res_m: Optional[float] = None
    grid_size: int = 200
    extent_buffer_mult: float = 3.0
    min_points: int = 15
    variogram_fast: bool = True
    variogram_res: int = 1
    variogram_dt: Optional[float] = None  # seconds; if None, use median dt
    model: str = "auto"  # "auto", "ou", "ouf"
    use_effective_n: bool = True
    estimate_velocity_tau: bool = True
    smooth: bool = True


_DEFAULT_PARAMS = AKDEParams()


def _write_variogram_plot(meta: dict, out_path: str) -> str | None:
    variogram = (meta or {}).get("variogram") or {}
    lags_s = np.asarray(variogram.get("lags_s", []), dtype=float)
    gamma_m2 = np.asarray(variogram.get("gamma_m2", []), dtype=float)
    counts = np.asarray(variogram.get("counts", []), dtype=float)
    if lags_s.size == 0 or gamma_m2.size == 0:
        return None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True, height_ratios=[3.0, 1.0])
    axes[0].plot(lags_s / 3600.0, gamma_m2, color="#5b86c5", linewidth=2.0)
    axes[0].scatter(lags_s / 3600.0, gamma_m2, color="#d7e3ff", s=18, zorder=3)
    axes[0].set_title(f"AKDE Variogram ({str((meta or {}).get('model', 'auto')).upper()})")
    axes[0].set_xlabel("Lag (hours)")
    axes[0].set_ylabel("Semivariance (m²)")

    if counts.size == lags_s.size:
        axes[1].bar(lags_s / 3600.0, counts, width=max(float(np.nanmedian(np.diff(lags_s / 3600.0))) if lags_s.size > 1 else 0.25, 0.05), color="#7d8ca6")
        axes[1].set_ylabel("Pairs")
        axes[1].set_xlabel("Lag (hours)")
    else:
        axes[1].axis("off")

    tau_pos_s = (meta or {}).get("tau_pos_s")
    if tau_pos_s:
        axes[0].axvline(float(tau_pos_s) / 3600.0, color="#ffb86c", linestyle="--", linewidth=1.4, label="tau_pos")
    tau_vel_s = (meta or {}).get("tau_vel_s")
    if tau_vel_s:
        axes[0].axvline(float(tau_vel_s) / 3600.0, color="#8be9a8", linestyle=":", linewidth=1.4, label="tau_vel")
    if tau_pos_s or tau_vel_s:
        axes[0].legend(loc="best")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _prepare_track(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    timestamps: np.ndarray,
    params: AKDEParams = _DEFAULT_PARAMS,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[dict]]:
    if latitudes is None or longitudes is None or timestamps is None:
        return None, None, None, None

    lat = np.asarray(latitudes)
    lon = np.asarray(longitudes)
    ts = np.asarray(timestamps)

    if len(lat) != len(lon) or len(lat) != len(ts):
        return None, None, None, None

    if len(lat) < max(2, params.min_points):
        return None, None, None, None

    ts_parsed = pd.to_datetime(ts, errors="coerce", utc=True)

    lat_num = pd.to_numeric(lat, errors="coerce")
    lon_num = pd.to_numeric(lon, errors="coerce")
    mask = np.isfinite(lat_num) & np.isfinite(lon_num) & (~pd.isna(ts_parsed))
    if mask.sum() < max(2, params.min_points):
        return None, None, None, None

    lat = lat_num[mask].astype(float)
    lon = lon_num[mask].astype(float)
    ts_parsed = ts_parsed[mask]

    order = np.argsort(ts_parsed.view("int64"))
    lat = lat[order]
    lon = lon[order]
    ts_parsed = ts_parsed[order]

    t_ns = ts_parsed.view("int64")
    keep = np.ones(len(t_ns), dtype=bool)
    keep[1:] = t_ns[1:] > t_ns[:-1]
    lat = lat[keep]
    lon = lon[keep]
    ts_parsed = ts_parsed[keep]

    if len(lat) < max(2, params.min_points):
        return None, None, None, None

    t_s = (ts_parsed.view("int64") - ts_parsed.view("int64")[0]) / 1e9
    t_s = t_s.astype(float)

    if len(t_s) < 2:
        return None, None, None, None

    dt = np.diff(t_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return None, None, None, None

    meta = {
        "n": int(len(lat)),
        "duration_s": float(t_s[-1] - t_s[0]),
        "median_dt_s": float(np.median(dt)),
        "start_time": ts_parsed[0].isoformat(),
        "end_time": ts_parsed[-1].isoformat(),
    }

    if meta["duration_s"] <= 0:
        return None, None, None, None

    return lat, lon, t_s, meta


def _project_track_to_utm(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Transformer], Optional[Transformer], Optional[int]]:
    if latitudes is None or longitudes is None:
        return None, None, None, None, None

    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)

    if len(lat) == 0 or len(lon) == 0 or len(lat) != len(lon):
        return None, None, None, None, None
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        return None, None, None, None, None

    lon0 = float(np.mean(lon))
    lat0 = float(np.mean(lat))

    zone = int((lon0 + 180.0) // 6.0) + 1
    zone = max(1, min(60, zone))
    epsg_utm = 32600 + zone if lat0 >= 0 else 32700 + zone

    try:
        to_utm = Transformer.from_crs("epsg:4326", f"epsg:{epsg_utm}", always_xy=True)
        to_ll = Transformer.from_crs(f"epsg:{epsg_utm}", "epsg:4326", always_xy=True)
        x, y = to_utm.transform(lon, lat)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    except Exception:
        return None, None, None, None, None

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None, None, None, None, None

    return x, y, to_utm, to_ll, epsg_utm


def _effective_sample_size(
    n: int,
    duration_s: float,
    tau_pos_s: float,
    tau_vel_s: Optional[float] = None,
) -> float:
    if not np.isfinite(n) or n <= 0:
        return 1.0
    if not np.isfinite(duration_s) or duration_s <= 0:
        return 1.0
    if not np.isfinite(tau_pos_s) or tau_pos_s <= 0:
        return float(n)

    neff = float(duration_s / tau_pos_s)
    if tau_vel_s is not None and np.isfinite(tau_vel_s) and tau_vel_s > 0:
        ratio = tau_vel_s / tau_pos_s
        ratio = max(0.0, min(1.0, ratio))
        neff *= (1.0 - 0.5 * ratio)

    neff = max(1.0, min(float(n), neff))
    return neff


def _empirical_variogram(
    x: np.ndarray,
    y: np.ndarray,
    t_s: np.ndarray,
    params: AKDEParams = _DEFAULT_PARAMS,
) -> Dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t_s = np.asarray(t_s, dtype=float)

    n = len(x)
    if n < max(3, params.min_points):
        return {"lags_s": np.array([]), "gamma_m2": np.array([]), "counts": np.array([])}

    dt_all = np.diff(t_s)
    dt_all = dt_all[np.isfinite(dt_all) & (dt_all > 0)]
    if len(dt_all) == 0:
        return {"lags_s": np.array([]), "gamma_m2": np.array([]), "counts": np.array([])}

    dt0 = float(params.variogram_dt) if (params.variogram_dt is not None and params.variogram_dt > 0) else float(np.median(dt_all))
    duration_s = float(t_s[-1] - t_s[0])
    if duration_s <= 0 or dt0 <= 0:
        return {"lags_s": np.array([]), "gamma_m2": np.array([]), "counts": np.array([])}

    max_lag = 0.5 * duration_s
    if max_lag <= dt0:
        max_lag = duration_s

    n_bins = max(8, min(60, int(np.floor(max_lag / dt0))))
    edges = np.linspace(dt0, max_lag, n_bins + 1)
    lag_sums = np.zeros(n_bins, dtype=float)
    gamma_sums = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    if params.variogram_fast and params.variogram_res > 1:
        anchor_idx = np.arange(0, n - 1, int(params.variogram_res))
    else:
        anchor_idx = np.arange(0, n - 1, 1)

    for i in anchor_idx:
        dt = t_s[i + 1:] - t_s[i]
        if len(dt) == 0:
            continue
        valid = (dt >= dt0) & (dt <= max_lag)
        if not np.any(valid):
            continue

        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        sq_disp = dx * dx + dy * dy

        dt_v = dt[valid]
        sd_v = sq_disp[valid]

        bin_idx = np.searchsorted(edges, dt_v, side="right") - 1
        ok = (bin_idx >= 0) & (bin_idx < n_bins)
        if not np.any(ok):
            continue

        for b, lag_val, sd_val in zip(bin_idx[ok], dt_v[ok], sd_v[ok]):
            lag_sums[b] += lag_val
            gamma_sums[b] += 0.5 * sd_val
            counts[b] += 1

    keep = counts > 0
    if not np.any(keep):
        return {"lags_s": np.array([]), "gamma_m2": np.array([]), "counts": np.array([])}

    lags_s = lag_sums[keep] / counts[keep]
    gamma_m2 = gamma_sums[keep] / counts[keep]
    counts = counts[keep]

    finite = np.isfinite(lags_s) & np.isfinite(gamma_m2) & (counts > 0)
    return {
        "lags_s": lags_s[finite],
        "gamma_m2": gamma_m2[finite],
        "counts": counts[finite],
    }


def _fit_autocorr_model(
    variogram: Dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    t_s: np.ndarray,
    params: AKDEParams = _DEFAULT_PARAMS,
) -> Dict[str, Any]:
    lags = np.asarray(variogram.get("lags_s", []), dtype=float)
    gamma = np.asarray(variogram.get("gamma_m2", []), dtype=float)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t_s = np.asarray(t_s, dtype=float)

    n = len(x)
    duration_s = float(t_s[-1] - t_s[0]) if len(t_s) >= 2 else 0.0

    if n >= 2:
        cov_xy = np.cov(np.vstack([x, y]), ddof=1)
        sigma2_x = float(max(cov_xy[0, 0], 1e-9))
        sigma2_y = float(max(cov_xy[1, 1], 1e-9))
        sigma_xy = float(cov_xy[0, 1])
    else:
        sigma2_x = sigma2_y = 1.0
        sigma_xy = 0.0

    dt_all = np.diff(t_s)
    dt_all = dt_all[np.isfinite(dt_all) & (dt_all > 0)]
    median_dt = float(np.median(dt_all)) if len(dt_all) else 1.0

    if len(lags) < 4 or len(gamma) < 4:
        tau_pos_s = max(median_dt, duration_s / max(n - 1, 1))
        n_eff = _effective_sample_size(n=n, duration_s=duration_s, tau_pos_s=tau_pos_s, tau_vel_s=None)
        return {
            "model": "ou",
            "tau_pos_s": float(tau_pos_s),
            "tau_vel_s": None,
            "sigma2_x": sigma2_x,
            "sigma2_y": sigma2_y,
            "sigma_xy": sigma_xy,
            "n_eff": float(n_eff),
        }

    upper_start = max(0, int(np.floor(0.7 * len(gamma))))
    plateau = float(np.median(gamma[upper_start:])) if upper_start < len(gamma) else float(np.max(gamma))
    if not np.isfinite(plateau) or plateau <= 0:
        plateau = float(np.max(gamma))
    plateau = max(plateau, 1e-9)

    target = 0.632 * plateau
    idx_tau = np.where(gamma >= target)[0]
    if len(idx_tau):
        tau_pos_s = float(lags[idx_tau[0]])
    else:
        tau_pos_s = float(lags[min(len(lags) - 1, max(0, len(lags) // 2))])
    tau_pos_s = max(tau_pos_s, median_dt)

    tau_vel_s = None
    model = "ou"

    if params.estimate_velocity_tau and len(gamma) >= 6:
        k = min(5, len(gamma) - 1)
        early_lags = lags[:k]
        early_gamma = gamma[:k]
        dlag = np.diff(early_lags)
        dgam = np.diff(early_gamma)
        valid = dlag > 0
        if np.any(valid):
            slopes = dgam[valid] / dlag[valid]
            slope_rise = False
            if len(slopes) >= 3:
                slope_rise = (slopes[1] > slopes[0] * 1.1) and (slopes[-1] < np.max(slopes) * 0.9)
            target_vel = 0.25 * plateau
            idx_vel = np.where(gamma >= target_vel)[0]
            if slope_rise and len(idx_vel):
                tau_vel_s = float(lags[idx_vel[0]])
                tau_vel_s = max(median_dt, min(tau_vel_s, tau_pos_s))
                model = "ouf"

    if params.model.lower() == "ou":
        model = "ou"
        tau_vel_s = None
    elif params.model.lower() == "ouf":
        model = "ouf"
        if tau_vel_s is None:
            tau_vel_s = max(median_dt, 0.25 * tau_pos_s)

    n_eff = _effective_sample_size(n=n, duration_s=duration_s, tau_pos_s=tau_pos_s, tau_vel_s=tau_vel_s)

    return {
        "model": model,
        "tau_pos_s": float(tau_pos_s),
        "tau_vel_s": None if tau_vel_s is None else float(tau_vel_s),
        "sigma2_x": sigma2_x,
        "sigma2_y": sigma2_y,
        "sigma_xy": sigma_xy,
        "n_eff": float(n_eff),
    }


def _bandwidth_matrix(
    x: np.ndarray,
    y: np.ndarray,
    fit: Dict[str, Any],
    params: AKDEParams = _DEFAULT_PARAMS,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or len(y) < 2:
        h = float(params.bandwidth_m) if (params.bandwidth_m is not None and params.bandwidth_m > 0) else 30.0
        return np.array([[h * h, 0.0], [0.0, h * h]], dtype=float)

    if params.bandwidth_m is not None and params.bandwidth_m > 0:
        h = float(params.bandwidth_m)
        return np.array([[h * h, 0.0], [0.0, h * h]], dtype=float)

    sigma2_x = float(max(fit.get("sigma2_x", np.var(x, ddof=1)), 1e-9))
    sigma2_y = float(max(fit.get("sigma2_y", np.var(y, ddof=1)), 1e-9))
    n_eff = float(fit.get("n_eff", len(x)))
    n_eff = max(1.0, min(float(len(x)), n_eff))

    scale = n_eff ** (-1.0 / 6.0)
    std_x = np.sqrt(sigma2_x)
    std_y = np.sqrt(sigma2_y)

    hx = max(5.0, std_x * scale)
    hy = max(5.0, std_y * scale)

    ratio = hx / hy if hy > 0 else 1.0
    if ratio > 10.0:
        hx = hy * 10.0
    elif ratio < 0.1:
        hy = hx * 10.0

    return np.array([[hx * hx, 0.0], [0.0, hy * hy]], dtype=float)


def _make_grid(
    x: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    params: AKDEParams = _DEFAULT_PARAMS,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[tuple], Optional[float], Optional[float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    H = np.asarray(H, dtype=float)

    if len(x) == 0 or len(y) == 0 or H.shape != (2, 2):
        return None, None, None, None, None, None

    hx = float(np.sqrt(max(H[0, 0], 1e-9)))
    hy = float(np.sqrt(max(H[1, 1], 1e-9)))
    h_ref = max(hx, hy, 1.0)
    pad = float(params.extent_buffer_mult) * h_ref

    xmin = float(np.min(x) - pad)
    xmax = float(np.max(x) + pad)
    ymin = float(np.min(y) - pad)
    ymax = float(np.max(y) + pad)

    if params.grid_res_m is not None and params.grid_res_m > 0:
        dx = dy = float(params.grid_res_m)
        gx = np.arange(xmin, xmax + dx, dx)
        gy = np.arange(ymin, ymax + dy, dy)
    else:
        gsz = max(25, int(params.grid_size))
        gx = np.linspace(xmin, xmax, gsz)
        gy = np.linspace(ymin, ymax, gsz)
        dx = float(gx[1] - gx[0]) if len(gx) > 1 else h_ref
        dy = float(gy[1] - gy[0]) if len(gy) > 1 else h_ref

    if len(gx) < 2 or len(gy) < 2:
        return None, None, None, None, None, None

    Xg, Yg = np.meshgrid(gx, gy)
    grid_xy = np.column_stack([Xg.ravel(), Yg.ravel()])
    bbox = (xmin, ymin, xmax, ymax)
    return Xg, Yg, grid_xy, bbox, dx, dy


def _evaluate_gaussian_ud(
    xy: np.ndarray,
    grid_xy: np.ndarray,
    H: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    grid_xy = np.asarray(grid_xy, dtype=float)
    H = np.asarray(H, dtype=float)

    if xy.ndim != 2 or xy.shape[1] != 2:
        return np.array([])
    if grid_xy.ndim != 2 or grid_xy.shape[1] != 2:
        return np.array([])
    if H.shape != (2, 2):
        return np.array([])

    n = xy.shape[0]
    m = grid_xy.shape[0]
    if n == 0 or m == 0:
        return np.array([])

    if weights is None:
        w = np.full(n, 1.0 / n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if len(w) != n:
            return np.array([])
        w = np.where(np.isfinite(w) & (w >= 0), w, 0.0)
        s = float(w.sum())
        w = np.full(n, 1.0 / n, dtype=float) if s <= 0 else (w / s)

    eps = 1e-9
    H = H.copy()
    H[0, 0] = max(H[0, 0], eps)
    H[1, 1] = max(H[1, 1], eps)

    try:
        H_inv = np.linalg.inv(H)
        det_H = float(np.linalg.det(H))
    except np.linalg.LinAlgError:
        return np.array([])

    if not np.isfinite(det_H) or det_H <= 0:
        return np.array([])

    norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_H))
    chunk_size = 5000
    out = np.zeros(m, dtype=float)

    for start in range(0, m, chunk_size):
        stop = min(start + chunk_size, m)
        G = grid_xy[start:stop]
        diffs = G[:, None, :] - xy[None, :, :]
        q = np.einsum("gni,ij,gnj->gn", diffs, H_inv, diffs)
        K = norm_const * np.exp(-0.5 * q)
        out[start:stop] = K @ w

    return out


def _normalize_ud(Z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    if Z.size == 0 or not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
        return Z
    Z = np.where(np.isfinite(Z), Z, 0.0)
    total = float(Z.sum() * dx * dy)
    return (Z / total) if total > 0 else Z


def _threshold_mask(
    Z: np.ndarray,
    dx: float,
    dy: float,
    percent: int,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    Z = np.asarray(Z, dtype=float)
    if Z.size == 0 or Z.ndim != 2:
        return None, None
    if not np.isfinite(dx) or not np.isfinite(dy) or dx <= 0 or dy <= 0:
        return None, None

    pct = int(percent)
    if pct < 1 or pct > 100:
        return None, None

    cell_area = float(dx * dy)
    Zf = np.where(np.isfinite(Z.ravel()), Z.ravel(), 0.0)
    if np.all(Zf <= 0):
        return None, None

    idx = np.argsort(Zf)[::-1]
    mass = np.cumsum(Zf[idx] * cell_area)
    target = pct / 100.0
    k = int(min(np.searchsorted(mass, target), len(idx) - 1))
    thr = float(Zf[idx][k])
    return Z >= thr, thr


def _mask_to_polygons_utm(mask: np.ndarray, Xg: np.ndarray, Yg: np.ndarray) -> Optional[MultiPolygon]:
    mask = np.asarray(mask)
    Xg = np.asarray(Xg, dtype=float)
    Yg = np.asarray(Yg, dtype=float)
    if mask.ndim != 2 or Xg.ndim != 2 or Yg.ndim != 2:
        return None
    if mask.shape != Xg.shape or mask.shape != Yg.shape:
        return None

    contours = measure.find_contours(mask.astype(float), 0.5)
    polys = []
    gx = Xg[0, :]
    gy = Yg[:, 0]

    for c in contours:
        if c.shape[0] < 3:
            continue
        px = c[:, 1]
        py = c[:, 0]
        xs = np.interp(px, np.arange(len(gx)), gx)
        ys = np.interp(py, np.arange(len(gy)), gy)
        try:
            p = Polygon(zip(xs, ys)).buffer(0)
        except Exception:
            continue
        if p.is_empty:
            continue
        if isinstance(p, Polygon):
            if p.is_valid and p.area > 0:
                polys.append(p)
        elif isinstance(p, MultiPolygon):
            for q in p.geoms:
                if q.is_valid and q.area > 0:
                    polys.append(q)

    if not polys:
        return None

    mp = unary_union(polys)
    if isinstance(mp, Polygon):
        return MultiPolygon([mp])
    if isinstance(mp, MultiPolygon):
        return mp
    return None


def _utm_geom_to_lonlat(geom, to_ll: Transformer):
    if geom is None or to_ll is None:
        return None

    def _poly_to_ll(poly: Polygon) -> Optional[Polygon]:
        if poly.is_empty:
            return None
        try:
            ex_x, ex_y = poly.exterior.xy
            ex_lon, ex_lat = to_ll.transform(ex_x, ex_y)
            holes = []
            for ring in poly.interiors:
                rx, ry = ring.xy
                rlon, rlat = to_ll.transform(rx, ry)
                holes.append(list(zip(rlon, rlat)))
            out = Polygon(list(zip(ex_lon, ex_lat)), holes)
            if out.is_valid and not out.is_empty and out.area > 0:
                return out.buffer(0)
        except Exception:
            return None
        return None

    if isinstance(geom, Polygon):
        return _poly_to_ll(geom)
    if isinstance(geom, MultiPolygon):
        parts = []
        for p in geom.geoms:
            q = _poly_to_ll(p)
            if q is not None and not q.is_empty:
                parts.append(q)
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return MultiPolygon(parts)
    return None


def _write_geotiff_wgs84(Zm: np.ndarray, bbox_utm: tuple, to_ll: Transformer, out_path: str) -> str:
    Zm = np.asarray(Zm, dtype=float)
    xmin, ymin, xmax, ymax = bbox_utm
    lon_sw, lat_sw = to_ll.transform(xmin, ymin)
    lon_ne, lat_ne = to_ll.transform(xmax, ymax)

    width = int(Zm.shape[1])
    height = int(Zm.shape[0])
    xres = (lon_ne - lon_sw) / width
    yres = (lat_ne - lat_sw) / height

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=Zm.dtype,
        crs="EPSG:4326",
        transform=from_origin(lon_sw, lat_ne, xres, yres),
    ) as dst:
        dst.write(np.flipud(Zm), 1)
    return out_path


def _write_geojson(geom_ll, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mapping(geom_ll), f)
    return out_path


def _akde_surface(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    timestamps: np.ndarray,
    params: AKDEParams = _DEFAULT_PARAMS,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[tuple],
    Optional[Transformer],
    Optional[dict]
]:
    lat, lon, t_s, prep = _prepare_track(latitudes=latitudes, longitudes=longitudes, timestamps=timestamps, params=params)
    if lat is None or lon is None or t_s is None or prep is None:
        return None, None, None, None, None, None

    x, y, _, to_ll, epsg_utm = _project_track_to_utm(lat, lon)
    if x is None or y is None or to_ll is None:
        return None, None, None, None, None, None

    xy = np.column_stack([x, y])
    variogram = _empirical_variogram(x, y, t_s, params=params)
    fit = _fit_autocorr_model(variogram, x, y, t_s, params=params)

    if not params.use_effective_n:
        fit = dict(fit)
        fit["n_eff"] = float(len(x))

    H = _bandwidth_matrix(x, y, fit, params=params)
    Xg, Yg, grid_xy, bbox_utm, dx, dy = _make_grid(x, y, H, params=params)
    if Xg is None or Yg is None or grid_xy is None or bbox_utm is None:
        return None, None, None, None, None, None

    z_flat = _evaluate_gaussian_ud(xy, grid_xy, H, weights=None)
    if z_flat.size == 0:
        return None, None, None, None, None, None

    Z = _normalize_ud(z_flat.reshape(Xg.shape), dx, dy)
    meta = {
        "n": int(prep["n"]),
        "duration_s": float(prep["duration_s"]),
        "median_dt_s": float(prep["median_dt_s"]),
        "start_time": prep["start_time"],
        "end_time": prep["end_time"],
        "model": fit.get("model"),
        "tau_pos_s": None if fit.get("tau_pos_s") is None else float(fit.get("tau_pos_s")),
        "tau_vel_s": None if fit.get("tau_vel_s") is None else float(fit.get("tau_vel_s")),
        "n_eff": None if fit.get("n_eff") is None else float(fit.get("n_eff")),
        "bandwidth_matrix": np.asarray(H, dtype=float).tolist(),
        "epsg_utm": int(epsg_utm),
        "grid_res_m": None if params.grid_res_m is None else float(params.grid_res_m),
        "dx": float(dx),
        "dy": float(dy),
        "variogram": {
            "lags_s": np.asarray(variogram.get("lags_s", []), dtype=float).tolist(),
            "gamma_m2": np.asarray(variogram.get("gamma_m2", []), dtype=float).tolist(),
            "counts": np.asarray(variogram.get("counts", []), dtype=int).tolist(),
        },
    }
    return Z, Xg, Yg, bbox_utm, to_ll, meta


def _akde_core(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    timestamps: np.ndarray,
    percent: int = 95,
    params: AKDEParams = _DEFAULT_PARAMS,
) -> Tuple[Optional[MultiPolygon], Optional[float], Optional[np.ndarray], Optional[tuple], Optional[Transformer], Optional[dict]]:
    Z, Xg, Yg, bbox_utm, to_ll, meta = _akde_surface(latitudes=latitudes, longitudes=longitudes, timestamps=timestamps, params=params)
    if Z is None or Xg is None or Yg is None or bbox_utm is None or to_ll is None or meta is None:
        return None, None, None, None, None, None

    dx = float(meta["dx"])
    dy = float(meta["dy"])
    mask, thr = _threshold_mask(Z, dx, dy, percent=percent)
    if mask is None:
        return None, None, None, None, None, None

    Zm = _normalize_ud(np.where(mask, Z, 0.0), dx, dy)
    geom_utm = _mask_to_polygons_utm(mask, Xg, Yg)
    if geom_utm is None:
        return None, None, None, None, None, None

    geom_ll = _utm_geom_to_lonlat(geom_utm, to_ll)
    if geom_ll is None:
        return None, None, None, None, None, None

    area_km2 = float(geom_utm.area / 1e6)
    meta = dict(meta)
    meta["threshold"] = float(thr) if thr is not None else None
    meta["percent"] = int(percent)
    return geom_ll, area_km2, Zm, bbox_utm, to_ll, meta


def add_akdes(df: pd.DataFrame, percent_list, params: AKDEParams = _DEFAULT_PARAMS):
    outputs_dir = storage.get_output_dir()
    os.makedirs(outputs_dir, exist_ok=True)
    if not hasattr(storage, "akde_results"):
        storage.akde_results = {}

    percent_list = list(percent_list) if percent_list else [95]
    animal_ids = df["animal_id"].unique() if "animal_id" in df.columns else ["sample"]

    for animal in animal_ids:
        trk = df[df["animal_id"] == animal] if "animal_id" in df.columns else df
        if "latitude" not in trk.columns or "longitude" not in trk.columns or "timestamp" not in trk.columns:
            continue

        lat = trk["latitude"].values
        lon = trk["longitude"].values
        ts = trk["timestamp"].values
        storage.akde_results.setdefault(animal, {})

        Z, Xg, Yg, bbox_utm, to_ll, surface_meta = _akde_surface(
            latitudes=lat,
            longitudes=lon,
            timestamps=ts,
            params=params or _DEFAULT_PARAMS,
        )
        if Z is None or Xg is None or Yg is None or bbox_utm is None or to_ll is None or surface_meta is None:
            continue

        dx = float(surface_meta["dx"])
        dy = float(surface_meta["dy"])
        safe = str(animal).replace(" ", "_").replace("/", "_")
        variogram_plot = _write_variogram_plot(surface_meta, os.path.join(outputs_dir, f"akde_variogram_{safe}.png"))
        if variogram_plot:
            surface_meta = dict(surface_meta)
            surface_meta["variogram_plot"] = variogram_plot

        for percent in percent_list:
            percent = int(percent)
            if percent < 1 or percent > 100:
                continue
            if percent in storage.akde_results[animal]:
                continue

            mask, thr = _threshold_mask(Z, dx, dy, percent=percent)
            if mask is None:
                continue

            Zm = _normalize_ud(np.where(mask, Z, 0.0), dx, dy)
            geom_utm = _mask_to_polygons_utm(mask, Xg, Yg)
            if geom_utm is None:
                continue

            geom_ll = _utm_geom_to_lonlat(geom_utm, to_ll)
            if geom_ll is None:
                continue

            area_km2 = float(geom_utm.area / 1e6)
            tif = os.path.join(outputs_dir, f"akde_{safe}_{percent}.tif")
            gj = os.path.join(outputs_dir, f"akde_{safe}_{percent}.geojson")

            _write_geotiff_wgs84(Zm, bbox_utm, to_ll, tif)
            _write_geojson(geom_ll, gj)

            meta = dict(surface_meta)
            meta["threshold"] = float(thr) if thr is not None else None
            meta["percent"] = int(percent)

            storage.akde_results[animal][percent] = {
                "contour": geom_ll,
                "area": area_km2,
                "geotiff": tif,
                "geojson": gj,
                "meta": meta,
            }
