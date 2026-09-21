# methods/kde.py

import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_origin

from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union

from pyproj import CRS, Transformer

from skimage import measure
from sklearn.neighbors import KernelDensity

import storage


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

@dataclass
class KDEParams:
    bandwidth_m: Optional[float] = None
    kernel: str = "gaussian"
    grid_res_m: Optional[float] = None
    grid_size: int = 200
    extent_buffer_mult: float = 3.0


_DEFAULT_PARAMS = KDEParams()


# ---------------------------------------------------------------------
# CRS helper
# ---------------------------------------------------------------------

def _local_utm_crs(longitudes, latitudes):
    """
    Select a local WGS84 UTM CRS based on the mean longitude/latitude
    of the animal's locations.
    """

    lon0 = float(np.mean(longitudes))
    lat0 = float(np.mean(latitudes))

    zone = int(np.floor((lon0 + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))

    if lat0 >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    return CRS.from_epsg(epsg)


# ---------------------------------------------------------------------
# Polygon conversion helper
# ---------------------------------------------------------------------

def _projected_to_lonlat(geom, transformer):
    """
    Convert a projected Polygon or MultiPolygon back to WGS84 lon/lat.
    """

    if geom is None or geom.is_empty:
        return None

    if isinstance(geom, Polygon):

        x, y = geom.exterior.xy
        lon, lat = transformer.transform(x, y)

        holes = []

        for ring in geom.interiors:
            hx, hy = ring.xy
            hlon, hlat = transformer.transform(hx, hy)
            holes.append(list(zip(hlon, hlat)))

        return Polygon(
            list(zip(lon, lat)),
            holes
        )

    if isinstance(geom, MultiPolygon):

        parts = []

        for part in geom.geoms:

            converted = _projected_to_lonlat(
                part,
                transformer
            )

            if converted is not None:
                parts.append(converted)

        return MultiPolygon(parts)

    return None


# ---------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------

def _density_contours(
    Z: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    threshold: float
):
    """
    Extract polygon contours directly from the KDE density surface.

    No additional polygon smoothing is applied so that the result
    stays closer to the raster-derived contours produced by amt.
    """

    z = np.asarray(Z, dtype=float)

    if z.ndim != 2:
        return []

    if z.shape != (len(gy), len(gx)):
        return []

    finite = np.isfinite(z)

    if not np.any(finite):
        return []

    zmin = float(np.nanmin(z[finite]))
    zmax = float(np.nanmax(z[finite]))

    if not (zmin < threshold < zmax):
        return []

    contours = measure.find_contours(
        np.where(finite, z, zmin),
        level=float(threshold)
    )

    polygons = []

    x_index = np.arange(len(gx), dtype=float)
    y_index = np.arange(len(gy), dtype=float)

    for contour in contours:

        if contour.shape[0] < 4:
            continue

        xs = np.interp(
            contour[:, 1],
            x_index,
            gx
        )

        ys = np.interp(
            contour[:, 0],
            y_index,
            gy
        )

        if not (
            np.all(np.isfinite(xs))
            and np.all(np.isfinite(ys))
        ):
            continue

        coords = np.column_stack((xs, ys))

        # Ensure closed ring
        if not np.allclose(coords[0], coords[-1]):
            coords = np.vstack([coords, coords[0]])

        poly = Polygon(coords).buffer(0)

        if (
            not poly.is_empty
            and poly.is_valid
            and poly.area > 0
        ):
            polygons.append(poly)

    return polygons


# ---------------------------------------------------------------------
# Core KDE
# ---------------------------------------------------------------------

def _kde_core(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    percent: int = 95,
    params: KDEParams = _DEFAULT_PARAMS,
) -> Tuple[
    Optional[MultiPolygon],
    Optional[float],
    Optional[np.ndarray],
    Optional[tuple],
    Optional[CRS]
]:
    """
    KDE in projected metric coordinates.

    Returns:
        polygon_ll
            Final contour polygon in WGS84 lon/lat.

        area_km2
            Area of the contour in square kilometers.

        Z_masked
            KDE raster values within the requested isopleth.

        raster_info
            Dictionary describing projected raster geometry.

        projected_crs
            CRS used for KDE calculation.
    """

    latitudes = np.asarray(
        latitudes,
        dtype=float
    )

    longitudes = np.asarray(
        longitudes,
        dtype=float
    )

    # Remove invalid coordinates
    valid = (
        np.isfinite(latitudes)
        & np.isfinite(longitudes)
    )

    latitudes = latitudes[valid]
    longitudes = longitudes[valid]

    if len(latitudes) < 3:
        return None, None, None, None, None

    # -----------------------------------------------------------------
    # 1. Project locations to local UTM
    # -----------------------------------------------------------------

    projected_crs = _local_utm_crs(
        longitudes,
        latitudes
    )

    to_proj = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True
    )

    to_ll = Transformer.from_crs(
        projected_crs,
        "EPSG:4326",
        always_xy=True
    )

    x, y = to_proj.transform(
        longitudes,
        latitudes
    )

    XY = np.column_stack((x, y))

    n = len(XY)

    # -----------------------------------------------------------------
    # 2. Bandwidth
    #
    # Match the reference-bandwidth logic used by amt::hr_kde_ref():
    #
    # h = 0.5 * (sd(x) + sd(y)) * n^(-1/6)
    #
    # One isotropic bandwidth is used in both dimensions.
    # -----------------------------------------------------------------

    if (
        params
        and params.bandwidth_m is not None
        and params.bandwidth_m > 0
    ):

        h = float(params.bandwidth_m)

    else:

        if n > 1:

            std_x = np.std(
                XY[:, 0],
                ddof=1
            )

            std_y = np.std(
                XY[:, 1],
                ddof=1
            )

            h = (
                0.5
                * (std_x + std_y)
                * (n ** (-1.0 / 6.0))
            )

            if (
                not np.isfinite(h)
                or h <= 0
            ):
                h = 30.0

        else:
            h = 30.0

    # -----------------------------------------------------------------
    # 3. Raster/grid definition
    # -----------------------------------------------------------------

    buffer_distance = (
        params.extent_buffer_mult * h
        if params
        else 3.0 * h
    )

    xmin = float(np.min(x) - buffer_distance)
    xmax = float(np.max(x) + buffer_distance)

    ymin = float(np.min(y) - buffer_distance)
    ymax = float(np.max(y) + buffer_distance)

    if (
        params
        and params.grid_res_m is not None
        and params.grid_res_m > 0
    ):

        step = float(params.grid_res_m)

        gx = np.arange(
            xmin,
            xmax + step,
            step
        )

        gy = np.arange(
            ymin,
            ymax + step,
            step
        )

    else:

        grid_size = (
            params.grid_size
            if params
            else 200
        )

        gx = np.linspace(
            xmin,
            xmax,
            grid_size
        )

        gy = np.linspace(
            ymin,
            ymax,
            grid_size
        )

    if len(gx) < 2 or len(gy) < 2:
        return None, None, None, None, None

    dx = float(gx[1] - gx[0])
    dy = float(gy[1] - gy[0])

    Xg, Yg = np.meshgrid(
        gx,
        gy
    )

    grid = np.column_stack(
        (
            Xg.ravel(),
            Yg.ravel()
        )
    )

    # -----------------------------------------------------------------
    # 4. KDE evaluation
    # -----------------------------------------------------------------

    kernel = (
        params.kernel.lower()
        if params
        else "gaussian"
    )

    kde = KernelDensity(
        bandwidth=h,
        kernel=kernel
    )

    kde.fit(XY)

    log_density = kde.score_samples(
        grid
    )

    Z = np.exp(
        log_density
    ).reshape(
        Xg.shape
    )

    # -----------------------------------------------------------------
    # 5. Normalize raster to integrate to 1
    # -----------------------------------------------------------------

    cell_area = abs(dx * dy)

    total_probability = (
        Z.sum()
        * cell_area
    )

    if (
        not np.isfinite(total_probability)
        or total_probability <= 0
    ):
        return None, None, None, None, None

    Z = Z / total_probability

    # -----------------------------------------------------------------
    # 6. Find utilization-distribution threshold
    # -----------------------------------------------------------------

    level = float(percent) / 100.0

    if not (0 < level < 1):
        raise ValueError(
            "KDE percent must be greater than 0 "
            "and less than 100."
        )

    Z_flat = Z.ravel()

    order = np.argsort(
        Z_flat
    )[::-1]

    cumulative_probability = np.cumsum(
        Z_flat[order]
        * cell_area
    )

    threshold_index = np.searchsorted(
        cumulative_probability,
        level
    )

    threshold_index = min(
        threshold_index,
        len(order) - 1
    )

    threshold = Z_flat[
        order[threshold_index]
    ]

    mask = Z >= threshold

    # Keep the original normalized UD values inside the isopleth.
    # Do not renormalize them to sum to 1 inside the contour.
    Z_masked = np.where(
        mask,
        Z,
        np.nan
    )

    # -----------------------------------------------------------------
    # 7. Extract contour polygons in projected coordinates
    # -----------------------------------------------------------------

    polygons = _density_contours(
        Z,
        gx,
        gy,
        threshold
    )

    # Fallback to binary mask contour if needed
    if not polygons:

        contours = measure.find_contours(
            mask.astype(float),
            level=0.5
        )

        for contour in contours:

            if contour.shape[0] < 4:
                continue

            xs = np.interp(
                contour[:, 1],
                np.arange(len(gx)),
                gx
            )

            ys = np.interp(
                contour[:, 0],
                np.arange(len(gy)),
                gy
            )

            coords = np.column_stack(
                (xs, ys)
            )

            if not np.allclose(
                coords[0],
                coords[-1]
            ):
                coords = np.vstack(
                    [coords, coords[0]]
                )

            poly = Polygon(
                coords
            ).buffer(0)

            if (
                not poly.is_empty
                and poly.is_valid
                and poly.area > 0
            ):
                polygons.append(poly)

    if not polygons:
        return None, None, None, None, None

    contour_projected = unary_union(
        polygons
    )

    if contour_projected.is_empty:
        return None, None, None, None, None

    # -----------------------------------------------------------------
    # 8. Area in projected CRS
    # -----------------------------------------------------------------

    area_km2 = (
        contour_projected.area
        / 1e6
    )

    # -----------------------------------------------------------------
    # 9. Convert contour to lon/lat for web mapping
    # -----------------------------------------------------------------

    contour_ll = _projected_to_lonlat(
        contour_projected,
        to_ll
    )

    # -----------------------------------------------------------------
    # 10. Raster metadata
    # -----------------------------------------------------------------

    raster_info = {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "dx": abs(dx),
        "dy": abs(dy),
        "bandwidth_m": float(h),
    }

    return (
        contour_ll,
        area_km2,
        Z_masked,
        raster_info,
        projected_crs
    )


# ---------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------

def add_kdes(
    df,
    percent_list,
    params: KDEParams = _DEFAULT_PARAMS
):
    """
    Compute KDE utilization distributions for each animal
    and requested isopleth.

    Outputs:
        - GeoTIFF in projected UTM coordinates
        - GeoJSON contour in WGS84 lon/lat
        - area in square kilometers

    The projected raster is retained because KDE distances,
    bandwidths, and areas are defined in meters.
    """

    outputs_dir = storage.get_output_dir()

    os.makedirs(
        outputs_dir,
        exist_ok=True
    )

    if "animal_id" not in df.columns:

        df = df.copy()

        df["animal_id"] = "Animal_1"

    required = {
        "latitude",
        "longitude"
    }

    missing = required.difference(
        df.columns
    )

    if missing:

        raise ValueError(
            "KDE requires latitude and longitude columns. "
            f"Missing: {sorted(missing)}"
        )

    for percent in percent_list:

        percent = int(percent)

        for animal in df["animal_id"].unique():

            storage.kde_results.setdefault(
                animal,
                {}
            )

            if percent in storage.kde_results[animal]:
                continue

            track = df[
                df["animal_id"] == animal
            ]

            (
                polygon_ll,
                area_km2,
                Z_masked,
                raster_info,
                projected_crs
            ) = _kde_core(
                track["latitude"].values,
                track["longitude"].values,
                percent=percent,
                params=params or _DEFAULT_PARAMS
            )

            if polygon_ll is None:
                continue

            safe_name = (
                str(animal)
                .replace(" ", "_")
                .replace("/", "_")
            )

            # ---------------------------------------------------------
            # Write GeoTIFF in projected CRS
            # ---------------------------------------------------------

            tif_path = os.path.join(
                outputs_dir,
                f"kde_{safe_name}_{percent}.tif"
            )

            xmin = raster_info["xmin"]
            ymax = raster_info["ymax"]

            dx = raster_info["dx"]
            dy = raster_info["dy"]

            transform = from_origin(
                xmin,
                ymax,
                dx,
                dy
            )

            raster_data = np.flipud(
                Z_masked
            )

            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=raster_data.shape[0],
                width=raster_data.shape[1],
                count=1,
                dtype="float64",
                crs=projected_crs.to_wkt(),
                transform=transform,
                nodata=np.nan
            ) as dst:

                dst.write(
                    raster_data.astype(
                        np.float64
                    ),
                    1
                )

            # ---------------------------------------------------------
            # Write contour as GeoJSON in WGS84
            # ---------------------------------------------------------

            geojson_path = os.path.join(
                outputs_dir,
                f"kde_{safe_name}_{percent}.geojson"
            )

            with open(
                geojson_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    mapping(polygon_ll),
                    f
                )

            # ---------------------------------------------------------
            # Save result for Spatchat
            # ---------------------------------------------------------

            storage.kde_results[
                animal
            ][percent] = {

                "contour": polygon_ll,

                "area": area_km2,

                "geotiff": tif_path,

                "geojson": geojson_path,

                "bandwidth_m": (
                    raster_info[
                        "bandwidth_m"
                    ]
                ),

                "analysis_crs": (
                    projected_crs.to_string()
                ),

                "grid_resolution_m": (
                    raster_info["dx"]
                ),
            }
