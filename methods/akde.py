from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import shapes as rio_shapes
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from skimage import measure
from shapely.geometry import MultiPolygon, Polygon, mapping, shape as shp_shape
from shapely.ops import transform as shp_transform, unary_union

import storage
from methods.ctmm_py.home_range import akde as ctmm_akde
from methods.ctmm_py.models import ctmm, ctmm_fit, ctmm_guess, ctmm_select
from methods.ctmm_py.kde import CI_UD
from methods.ctmm_py.plot_variogram import svf_func
from methods.ctmm_py.telemetry import as_telemetry
from methods.ctmm_py.types import Telemetry
from methods.ctmm_py.variogram import variogram


@dataclass
class AKDEParams:
    bandwidth_m: Optional[float] = None
    grid_res_m: Optional[float] = None
    grid_size: int = 200
    extent_buffer_mult: float = 3.0
    min_points: int = 15
    variogram_fast: bool = True
    variogram_res: int = 1
    variogram_dt: Optional[float] = None
    model: str = "auto"
    use_effective_n: bool = True
    estimate_velocity_tau: bool = True
    smooth: bool = True
    debias_area: bool = True
    debias_strength: float = 0.22
    cores: int = 0


def _safe_name(value) -> str:
    safe = str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return safe or "animal"


def _pick_model(params: AKDEParams):
    model = str(params.model or "auto").lower()
    if model == "ouf":
        return ctmm(tau=[10 * 24 * 3600.0, 1 * 24 * 3600.0], range=True, isotropic=False), "ouf"
    return ctmm(tau=[10 * 24 * 3600.0], range=True, isotropic=False), "ou"


def _fit_model(telem, params: AKDEParams):
    model = str(params.model or "auto").lower()
    if model == "ouf":
        model0, model_name = _pick_model(params)
        return ctmm_fit(telem, model0), model_name
    if model == "auto":
        guess = ctmm_guess(variogram(telem), ctmm(tau=[10 * 24 * 3600.0, 1 * 24 * 3600.0], range=True, isotropic=False))
        fit = ctmm_select(telem, [guess], IC="AICc", MSPE="position", iterate=True, cores=int(params.cores))
        return fit, str(getattr(fit, "model", "auto"))

    candidates = [
        ctmm(tau=[10 * 24 * 3600.0], range=True, isotropic=False),
        ctmm(tau=[10 * 24 * 3600.0], range=True, isotropic=True),
    ]
    fitted = [ctmm_fit(telem, m) for m in candidates]
    best = min(fitted, key=lambda m: float(m.params.get("AICc", np.inf)))
    tau = best.params.get("tau", {})
    selected = "ouf" if isinstance(tau, dict) and "velocity" in tau else "ou"
    return best, selected


def _column(df: pd.DataFrame, *names: str) -> str | None:
    lowered = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        hit = lowered.get(name.lower())
        if hit is not None:
            return hit
    return None


def _normalise_input(df: pd.DataFrame) -> tuple[str, str, str, str]:
    animal_col = _column(df, "animal_id", "id")
    ts_col = _column(df, "timestamp", "time", "datetime", "date")
    lon_col = _column(df, "longitude", "lon", "x")
    lat_col = _column(df, "latitude", "lat", "y")
    if animal_col is None:
        df["animal_id"] = "Animal_1"
        animal_col = "animal_id"
    if ts_col is None:
        raise ValueError("AKDE input requires a timestamp column.")
    if lon_col is None or lat_col is None:
        raise ValueError("AKDE input requires longitude/latitude columns.")
    return animal_col, ts_col, lon_col, lat_col


def _parse_timestamps_utc(values) -> pd.Series:
    ts = pd.to_datetime(values, errors="coerce", utc=True)
    raw = pd.Series(values)
    missing = ts.isna() & raw.notna()
    if missing.any():
        try:
            mixed = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
            if mixed.notna().sum() > ts.notna().sum():
                return mixed
        except (TypeError, ValueError):
            pass
    return ts


def _grid_edges(values: np.ndarray, step: float) -> tuple[float, float]:
    return float(values[0] - step / 2.0), float(values[-1] + step / 2.0)


def _projected_to_wgs_transform(telem):
    proj4 = (telem.metadata or {}).get("proj4")
    if proj4:
        return Transformer.from_crs(CRS.from_proj4(proj4), CRS.from_epsg(4326), always_xy=True)
    if telem.crs:
        return Transformer.from_crs(CRS.from_user_input(telem.crs), CRS.from_epsg(4326), always_xy=True)
    raise ValueError("AKDE telemetry is missing projection metadata.")


def _projected_crs(telem):
    proj4 = (telem.metadata or {}).get("proj4")
    if proj4:
        return CRS.from_proj4(proj4)
    if telem.crs:
        return CRS.from_user_input(telem.crs)
    raise ValueError("AKDE telemetry is missing projection metadata.")


def _write_ud_raster(ud: dict, telem, out_path: str) -> None:
    gx = np.asarray(ud["r"]["x"], dtype=float)
    gy = np.asarray(ud["r"]["y"], dtype=float)
    pdf = np.asarray(ud["PDF"], dtype=float)
    dx = float(ud["dr"]["x"])
    dy = float(ud["dr"]["y"])
    if pdf.shape != (gx.size, gy.size):
        raise ValueError(f"AKDE PDF shape {pdf.shape} does not match grid {(gx.size, gy.size)}")

    # ctmm_py stores arrays as [x, y]. GeoTIFF rows are north-to-south [y, x].
    x_min, x_max = _grid_edges(gx, dx)
    y_min, y_max = _grid_edges(gy, dy)
    src_raster = np.flipud(pdf.T).astype(np.float32)
    src_transform = from_origin(x_min, y_max, dx, dy)
    src_crs = _projected_crs(telem)
    dst_crs = CRS.from_epsg(4326)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        dst_crs,
        src_raster.shape[1],
        src_raster.shape[0],
        x_min,
        y_min,
        x_max,
        y_max,
    )
    dst_raster = np.zeros((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=src_raster,
        destination=dst_raster,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=0.0,
        dst_nodata=0.0,
        resampling=Resampling.bilinear,
    )
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=dst_raster.shape[0],
        width=dst_raster.shape[1],
        count=1,
        dtype=rasterio.float32,
        crs=dst_crs,
        transform=dst_transform,
        nodata=0.0,
        compress="lzw",
    ) as dst:
        dst.write(dst_raster, 1)


def _contour_from_mask(mask_xy: np.ndarray, gx: np.ndarray, gy: np.ndarray, dx: float, dy: float):
    x_min, _ = _grid_edges(gx, dx)
    _, y_max = _grid_edges(gy, dy)
    mask_raster = np.flipud(mask_xy.T).astype(np.uint8)
    transform = from_origin(x_min, y_max, dx, dy)
    polygons = []
    for geom, val in rio_shapes(mask_raster, mask=mask_raster.astype(bool), transform=transform):
        if int(val) != 1:
            continue
        poly = shp_shape(geom)
        if not poly.is_empty and poly.area > 0:
            polygons.append(poly)
    if not polygons:
        return None
    return unary_union(polygons).buffer(0)


def _smooth_contour_from_cdf(cdf: np.ndarray, gx: np.ndarray, gy: np.ndarray, level: float, dx: float, dy: float):
    z = np.asarray(cdf, dtype=float)
    finite = np.isfinite(z)
    if z.shape != (gx.size, gy.size) or not np.any(finite):
        return None

    zmin = float(np.nanmin(z[finite]))
    zmax = float(np.nanmax(z[finite]))
    level = float(level)
    if not (zmin < level < zmax):
        return None

    # skimage uses array coordinates [row, col]. The AKDE grid is [x, y],
    # so transpose to rows=y and cols=x before interpolating back to meters.
    z_yx = z.T
    fill = zmax
    z_yx = np.where(np.isfinite(z_yx), z_yx, fill)
    contours = measure.find_contours(z_yx, level=level)
    polygons = []
    max_gap = 4.0 * max(abs(float(dx)), abs(float(dy)))
    x_index = np.arange(gx.size, dtype=float)
    y_index = np.arange(gy.size, dtype=float)

    for line in contours:
        if line.shape[0] < 4:
            continue
        xs = np.interp(line[:, 1], x_index, gx)
        ys = np.interp(line[:, 0], y_index, gy)
        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
            continue
        if np.hypot(xs[0] - xs[-1], ys[0] - ys[-1]) > max_gap:
            continue
        coords = list(zip(xs, ys))
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords).buffer(0)
        if not poly.is_empty and poly.area > 0:
            polygons.append(poly)

    if not polygons:
        return None
    return unary_union(polygons).buffer(0)


def _contour_from_ud(ud: dict, telem, level: float):
    gx = np.asarray(ud["r"]["x"], dtype=float)
    gy = np.asarray(ud["r"]["y"], dtype=float)
    cdf = np.asarray(ud["CDF"], dtype=float)
    dx = float(ud["dr"]["x"])
    dy = float(ud["dr"]["y"])
    if cdf.shape != (gx.size, gy.size):
        raise ValueError(f"AKDE CDF shape {cdf.shape} does not match grid {(gx.size, gy.size)}")

    mask_xy = np.isfinite(cdf) & (cdf <= float(level))
    if not np.any(mask_xy):
        return None

    geom_m = _smooth_contour_from_cdf(cdf, gx, gy, float(level), dx, dy)
    if geom_m is None:
        geom_m = _contour_from_mask(mask_xy, gx, gy, dx, dy)
    if geom_m is None:
        return None

    to_wgs = _projected_to_wgs_transform(telem)
    geom_ll = shp_transform(lambda x, y, z=None: to_wgs.transform(x, y), geom_m)
    if isinstance(geom_ll, (Polygon, MultiPolygon)) and not geom_ll.is_empty:
        return geom_ll
    return None


def _ci_ud_object(ud: dict) -> dict:
    return {
        "CDF": np.asarray(ud["CDF"], dtype=float),
        "dr": {"x": float(ud["dr"]["x"]), "y": float(ud["dr"]["y"])},
        "DOF.area": np.asarray(ud.get("DOF.area", [np.nan]), dtype=float),
    }


def _area_ci_for_level(ud: dict, level: float) -> np.ndarray:
    ci_m2 = CI_UD(
        _ci_ud_object(ud),
        level_UD=float(level),
        level=0.95,
    )
    return np.asarray(ci_m2, dtype=float) / 1e6


def _area_ci_contour_levels_for_level(ud: dict, level: float) -> np.ndarray:
    return np.asarray(
        CI_UD(
            _ci_ud_object(ud),
            level_UD=float(level),
            level=0.95,
            P=True,
        ),
        dtype=float,
    )


def _write_variogram_plot(telem, model, model_name: str, out_path: str, params: AKDEParams) -> tuple[str | None, dict | None]:
    vg = variogram(telem, dt=params.variogram_dt)
    lags_s = np.asarray(vg.get("lags_s", []), dtype=float)
    gamma_m2 = np.asarray(vg.get("gamma", []), dtype=float)
    counts = np.asarray(vg.get("counts", []), dtype=float)
    keep = np.isfinite(lags_s) & np.isfinite(gamma_m2)
    if counts.size == lags_s.size:
        keep &= counts > 0
    lags_s = lags_s[keep]
    gamma_m2 = gamma_m2[keep]
    counts = counts[keep] if counts.size == keep.size else np.array([], dtype=float)
    if lags_s.size == 0 or gamma_m2.size == 0:
        return None, None

    lags_h = lags_s / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True, height_ratios=[3.0, 1.0])
    axes[0].plot(lags_h, gamma_m2, color="#5b86c5", linewidth=1.8)
    axes[0].scatter(lags_h, gamma_m2, color="#d7e3ff", edgecolor="#5b86c5", linewidth=0.4, s=18, zorder=3)

    try:
        svf = svf_func(model)["svf"]
        curve_s = np.linspace(float(np.nanmin(lags_s)), float(np.nanmax(lags_s)), 240)
        curve_gamma = np.asarray(svf(curve_s), dtype=float)
        valid = np.isfinite(curve_gamma)
        if np.any(valid):
            axes[0].plot(curve_s[valid] / 3600.0, curve_gamma[valid], color="#1f2937", linewidth=1.5, label="Fitted CTMM")
    except Exception:
        pass

    axes[0].set_title(f"AKDE Variogram ({str(model_name).upper()})")
    axes[0].set_xlabel("Lag (hours)")
    axes[0].set_ylabel("Semivariance (m^2)")
    axes[0].grid(alpha=0.22, linewidth=0.7)

    tau = model.params.get("tau", {}) if hasattr(model, "params") else {}
    if not isinstance(tau, dict):
        tau = {}
    tau_pos_s = tau.get("position")
    tau_vel_s = tau.get("velocity")
    if tau_pos_s:
        axes[0].axvline(float(tau_pos_s) / 3600.0, color="#ff9f1c", linestyle="--", linewidth=1.4, label="tau_pos")
    if tau_vel_s:
        axes[0].axvline(float(tau_vel_s) / 3600.0, color="#2a9d8f", linestyle=":", linewidth=1.5, label="tau_vel")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="best")

    if counts.size == lags_h.size:
        diffs = np.diff(lags_h[np.isfinite(lags_h)])
        width = max(float(np.nanmedian(diffs)) if diffs.size else 0.25, 0.05)
        axes[1].bar(lags_h, counts, width=width, color="#7d8ca6")
        axes[1].set_ylabel("Pairs")
        axes[1].set_xlabel("Lag (hours)")
        axes[1].grid(axis="y", alpha=0.2, linewidth=0.7)
    else:
        axes[1].axis("off")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    variogram_meta = {
        "lags_s": lags_s.tolist(),
        "gamma_m2": gamma_m2.tolist(),
        "counts": counts.tolist() if counts.size else [],
    }
    return out_path, variogram_meta


def _model_meta(model, model_name: str, ud: dict) -> dict:
    tau = model.params.get("tau", {}) if hasattr(model, "params") else {}
    if not isinstance(tau, dict):
        tau = {}
    meta = {
        "model": model_name,
        "isotropic": bool(model.params.get("isotropic", False)) if hasattr(model, "params") else False,
        "tau_pos_s": tau.get("position"),
        "tau_vel_s": tau.get("velocity"),
        "dof_area": float(np.asarray(ud.get("DOF.area", [np.nan]), dtype=float).reshape(-1)[0]),
        "dof_h": float(ud.get("DOF.H", np.nan)),
    }
    candidates = model.params.get("_select_candidates") if hasattr(model, "params") else None
    if candidates:
        meta["auto_candidates"] = candidates
    errors = model.params.get("_select_errors") if hasattr(model, "params") else None
    if errors:
        meta["auto_errors"] = errors
    return meta


def add_akdes(df: pd.DataFrame, percent_list, params: AKDEParams | None = None):
    params = params or AKDEParams()
    outputs_dir = storage.get_output_dir()
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    if _column(df, "animal_id", "id") is None:
        df = df.copy()
        df["animal_id"] = "Animal_1"

    animal_col, ts_col, lon_col, lat_col = _normalise_input(df)
    percents = sorted({int(p) for p in percent_list if 0 < int(p) <= 100})
    if not percents:
        return

    track_all = df[[animal_col, ts_col, lon_col, lat_col]].copy()
    track_all.columns = ["id", "timestamp", "longitude", "latitude"]
    track_all["id"] = track_all["id"].astype(str)
    track_all["timestamp"] = _parse_timestamps_utc(track_all["timestamp"])
    track_all["longitude"] = pd.to_numeric(track_all["longitude"], errors="coerce")
    track_all["latitude"] = pd.to_numeric(track_all["latitude"], errors="coerce")
    track_all = track_all.dropna().sort_values(["id", "timestamp"])
    track_all = track_all.drop_duplicates(subset=["id", "timestamp", "longitude", "latitude"], keep="first")
    valid_ids = [
        animal
        for animal, group in track_all.groupby("id", sort=False)
        if len(group) >= int(params.min_points)
    ]
    if not valid_ids:
        return
    track_all = track_all[track_all["id"].isin(valid_ids)].copy()
    telemetry_all = as_telemetry(track_all, id_col="id", time_col="timestamp", x_col="longitude", y_col="latitude")

    for animal, group in telemetry_all.data.groupby(telemetry_all.id_col, sort=False):
        if len(group) < int(params.min_points):
            continue

        telem = Telemetry(
            data=group.copy(),
            id_col=telemetry_all.id_col,
            time_col=telemetry_all.time_col,
            x_col=telemetry_all.x_col,
            y_col=telemetry_all.y_col,
            crs=telemetry_all.crs,
            metadata=dict(telemetry_all.metadata),
        )
        model, model_name = _fit_model(telem, params)
        ud = ctmm_akde(
            telem,
            model,
            debias=bool(params.debias_area),
            res=max(2, int(round(float(params.grid_size) / 20.0))),
        )

        safe = _safe_name(animal)
        variogram_path = os.path.join(outputs_dir, f"akde_variogram_{safe}.png")
        variogram_plot, variogram_meta = _write_variogram_plot(telem, model, model_name, variogram_path, params)
        tif_path = os.path.join(outputs_dir, f"akde_{safe}.tif")
        _write_ud_raster(ud, telem, tif_path)

        storage.akde_results.setdefault(str(animal), {})
        meta = _model_meta(model, model_name, ud)
        if variogram_plot:
            meta["variogram_plot"] = variogram_plot
        if variogram_meta:
            meta["variogram"] = variogram_meta
        for percent in percents:
            level = float(percent) / 100.0
            contour = _contour_from_ud(ud, telem, level)
            ci = _area_ci_for_level(ud, level)
            ci_levels = _area_ci_contour_levels_for_level(ud, level)
            ci_contours = {}
            ci_contour_levels = {}
            for label, idx in (("low", 0), ("high", 2)):
                ci_level = float(ci_levels[idx]) if idx < ci_levels.size else float("nan")
                if not np.isfinite(ci_level) or ci_level <= 0.0:
                    continue
                ci_level = min(max(ci_level, 0.0), 1.0)
                ci_contour = _contour_from_ud(ud, telem, ci_level)
                if ci_contour is not None:
                    ci_contours[label] = ci_contour
                    ci_contour_levels[label] = ci_level
            area_km2 = float(ci[1])
            gj_path = os.path.join(outputs_dir, f"akde_{safe}_{percent}.geojson")
            if contour is not None:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "animal_id": str(animal),
                        "percent": int(percent),
                        "area_km2": area_km2,
                        "area_ci95_low_km2": float(ci[0]),
                        "area_ci95_high_km2": float(ci[2]),
                    },
                    "geometry": mapping(contour),
                }
                with open(gj_path, "w", encoding="utf-8") as f:
                    json.dump(feature, f)
            else:
                gj_path = None

            ci_gj_path = os.path.join(outputs_dir, f"akde_{safe}_{percent}_ci.geojson")
            if ci_contours:
                ci_features = []
                for label, geom in ci_contours.items():
                    idx = 0 if label == "low" else 2
                    ci_features.append(
                        {
                            "type": "Feature",
                            "properties": {
                                "animal_id": str(animal),
                                "percent": int(percent),
                                "contour_type": f"ci_{label}",
                                "area_km2": float(ci[idx]),
                                "area_ci95_low_km2": float(ci[0]),
                                "area_ci95_high_km2": float(ci[2]),
                                "contour_level": float(ci_contour_levels[label]),
                            },
                            "geometry": mapping(geom),
                        }
                    )
                with open(ci_gj_path, "w", encoding="utf-8") as f:
                    json.dump({"type": "FeatureCollection", "features": ci_features}, f)
            else:
                ci_gj_path = None

            storage.akde_results[str(animal)][int(percent)] = {
                "contour": contour,
                "ci_contours": ci_contours,
                "ci_contour_levels": ci_contour_levels,
                "area": area_km2,
                "area_km2": area_km2,
                "area_ci95_km2": [float(ci[0]), float(ci[2])],
                "ci_low_km2": float(ci[0]),
                "ci_high_km2": float(ci[2]),
                "dof_area": meta["dof_area"],
                "dof_h": meta["dof_h"],
                "model": model_name,
                "geotiff": tif_path,
                "geojson": gj_path,
                "ci_geojson": ci_gj_path,
                "raster_path": tif_path,
                "geojson_path": gj_path,
                "meta": dict(meta),
            }


__all__ = ["AKDEParams", "add_akdes"]
