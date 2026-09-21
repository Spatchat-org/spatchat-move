# methods/mcp.py

import numpy as np

from pyproj import CRS, Transformer
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

import storage


def _local_utm_crs(longitudes, latitudes):
    """
    Choose a local WGS84 UTM CRS from the mean longitude/latitude
    of one animal's locations.

    Northern hemisphere: EPSG 326xx
    Southern hemisphere: EPSG 327xx
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


def _mcp_polygon(latitudes, longitudes, percent=95):
    """
    Calculate an MCP using logic similar to amt::hr_mcp():

    1. Transform lon/lat to a projected metric CRS.
    2. Calculate the mean center.
    3. Calculate squared Euclidean distance from the mean center.
    4. Find the requested distance quantile.
    5. Retain all points at or below that quantile.
    6. Construct the convex hull.

    Returns:
        polygon_ll  : Shapely Polygon in WGS84 lon/lat
        polygon_proj: Shapely Polygon in projected meters
        projected_crs: CRS used for calculation
    """

    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)

    # Remove rows with missing or non-finite coordinates
    valid = (
        np.isfinite(latitudes)
        & np.isfinite(longitudes)
    )

    latitudes = latitudes[valid]
    longitudes = longitudes[valid]

    if len(latitudes) < 3:
        return None

    # ---------------------------------------------------------
    # 1. Select a local projected CRS
    # ---------------------------------------------------------

    projected_crs = _local_utm_crs(
        longitudes,
        latitudes
    )

    to_proj = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True
    )

    x, y = to_proj.transform(
        longitudes,
        latitudes
    )

    pts = np.column_stack((x, y))

    # ---------------------------------------------------------
    # 2. Mean center
    # ---------------------------------------------------------

    centroid = pts.mean(axis=0)

    # ---------------------------------------------------------
    # 3. Squared Euclidean distance from mean center
    #
    # This follows the logic used by amt::hr_mcp()
    # ---------------------------------------------------------

    sqd = np.sum(
        (pts - centroid) ** 2,
        axis=1
    )

    # Convert percent such as 95 -> 0.95
    level = float(percent) / 100.0

    if not (0 < level <= 1):
        raise ValueError(
            "MCP percent must be greater than 0 and less than or equal to 100."
        )

    # ---------------------------------------------------------
    # 4. Distance quantile
    # ---------------------------------------------------------

    threshold = np.quantile(
        sqd,
        level
    )

    # ---------------------------------------------------------
    # 5. Keep locations inside the requested quantile
    # ---------------------------------------------------------

    pts_kept = pts[sqd <= threshold]

    if len(pts_kept) < 3:
        return None

    # ---------------------------------------------------------
    # 6. Convex hull
    # ---------------------------------------------------------

    hull = ConvexHull(pts_kept)

    hull_xy = pts_kept[hull.vertices]

    polygon_proj = Polygon(hull_xy)

    if polygon_proj.is_empty or not polygon_proj.is_valid:
        polygon_proj = polygon_proj.buffer(0)

    if polygon_proj.is_empty:
        return None

    # ---------------------------------------------------------
    # 7. Convert polygon back to longitude / latitude
    #    for displaying on the Spatchat map
    # ---------------------------------------------------------

    to_ll = Transformer.from_crs(
        projected_crs,
        "EPSG:4326",
        always_xy=True
    )

    hull_lon, hull_lat = to_ll.transform(
        hull_xy[:, 0],
        hull_xy[:, 1]
    )

    polygon_ll = Polygon(
        zip(hull_lon, hull_lat)
    )

    return {
        "polygon": polygon_ll,
        "polygon_projected": polygon_proj,
        "projected_crs": projected_crs,
        "n_total": len(pts),
        "n_used": len(pts_kept),
    }


def add_mcps(df, percent_list):
    """
    Calculate MCPs separately for each animal and requested percentage.

    Results are saved to storage.mcp_results in the same structure
    expected by the rest of Spatchat.
    """

    if "animal_id" not in df.columns:
        df = df.copy()
        df["animal_id"] = "Animal_1"

    required = {"latitude", "longitude"}

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "MCP requires latitude and longitude columns. "
            f"Missing: {sorted(missing)}"
        )

    for percent in percent_list:

        percent = int(percent)

        for animal in df["animal_id"].unique():

            storage.mcp_results.setdefault(
                animal,
                {}
            )

            # Skip if this result has already been calculated
            if percent in storage.mcp_results[animal]:
                continue

            track = df[
                df["animal_id"] == animal
            ]

            result = _mcp_polygon(
                track["latitude"].values,
                track["longitude"].values,
                percent=percent
            )

            if result is None:
                continue

            polygon_ll = result["polygon"]
            polygon_proj = result["polygon_projected"]

            # Area is calculated in projected square meters
            area_km2 = polygon_proj.area / 1e6

            storage.mcp_results[animal][percent] = {
                "polygon": polygon_ll,
                "area": area_km2,

                # Useful metadata for reproducibility/debugging
                "n_total": result["n_total"],
                "n_used": result["n_used"],
                "analysis_crs": result[
                    "projected_crs"
                ].to_string(),
            }
