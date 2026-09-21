# methods/locoh.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from shapely.geometry import MultiPoint, Polygon, mapping
from shapely.ops import unary_union, transform as shp_transform

from scipy.spatial import cKDTree

from pyproj import CRS, Transformer


# -----------------------------------------------------------------------------
# Projection helper
# -----------------------------------------------------------------------------

def _local_utm_crs(longitudes, latitudes):
    """
    Select a local WGS84 UTM CRS based on the mean longitude/latitude
    of the input locations.

    Northern hemisphere: EPSG:326xx
    Southern hemisphere: EPSG:327xx
    """

    longitudes = np.asarray(longitudes, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)

    lon0 = float(np.mean(longitudes))
    lat0 = float(np.mean(latitudes))

    zone = int(np.floor((lon0 + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))

    if lat0 >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    return CRS.from_epsg(epsg)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

@dataclass
class LoCoHParams:
    """
    Parameters for Local Convex Hull (LoCoH).

    method:
      - "k": k-nearest neighbours
      - "a": cumulative-distance threshold in meters
      - "r": fixed-radius threshold in meters

    isopleths:
      Percent envelopes to return, e.g., (50, 95).

    Per-hull facets are returned separately for visualization.
    """

    method: str = "k"
    k: int = 10
    a: Optional[float] = None
    r: Optional[float] = None
    isopleths: Tuple[int, ...] = (50, 95)


def compute_locoh(
    df: pd.DataFrame,
    id_col: str,
    x_col: str,
    y_col: str,
    params: Optional[LoCoHParams] = None,
) -> Dict:
    """
    Compute Local Convex Hull (LoCoH) home ranges by animal.

    Parameters
    ----------
    df
        DataFrame containing animal IDs and coordinates.

    id_col
        Column containing animal/individual identifiers.

    x_col, y_col
        Longitude and latitude column names in WGS84.

    params
        LoCoHParams object.

    Returns
    -------
    dict
        Dictionary containing:

        {
            "method": "k" | "a" | "r",
            "isopleths": [50, 95, ...],
            "analysis_crs": "EPSG:32617",
            "animals": {
                "<animal_id>": {
                    "n_points": int,
                    "isopleths": [
                        {
                            "isopleth": 50,
                            "area_sq_km": float,
                            "geometry": <GeoJSON>
                        },
                        ...
                    ],
                    "facets": [
                        {
                            "cum_frac": float,
                            "cum_percent": int,
                            "area_sq_km": float,
                            "geometry": <GeoJSON>
                        },
                        ...
                    ]
                }
            }
        }

    Notes
    -----
    - Input coordinates are expected to be WGS84 longitude/latitude.
    - A local UTM CRS is selected automatically from the mean location
      of the dataset.
    - All distance, nearest-neighbour, convex-hull, and area calculations
      are performed in projected UTM coordinates (meters).
    - Output geometries are transformed back to WGS84 for web mapping.
    """

    if params is None:
        params = LoCoHParams()

    method = params.method.lower()

    if method not in {"k", "a", "r"}:
        raise ValueError(
            "LoCoH method must be one of {'k', 'a', 'r'}."
        )

    # -------------------------------------------------------------------------
    # Parameter validation
    # -------------------------------------------------------------------------

    if method == "a":
        if params.a is None or float(params.a) <= 0:
            raise ValueError(
                "a-LoCoH requires a positive 'a' value in meters."
            )

    if method == "r":
        if params.r is None or float(params.r) <= 0:
            raise ValueError(
                "r-LoCoH requires a positive 'r' value in meters."
            )

    # -------------------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------------------

    if id_col not in df.columns:
        df = df.copy()
        df[id_col] = "Animal_1"

    required = {id_col, x_col, y_col}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required LoCoH columns: {sorted(missing)}"
        )

    gdf = (
        df[[id_col, x_col, y_col]]
        .dropna()
        .copy()
    )

    gdf.columns = [
        "animal_id",
        "lon",
        "lat"
    ]

    gdf["lon"] = pd.to_numeric(
        gdf["lon"],
        errors="coerce"
    )

    gdf["lat"] = pd.to_numeric(
        gdf["lat"],
        errors="coerce"
    )

    gdf = gdf.dropna(
        subset=["lon", "lat"]
    )

    if gdf.empty:
        return {
            "method": method,
            "isopleths": list(params.isopleths),
            "analysis_crs": None,
            "animals": {},
        }

    # -------------------------------------------------------------------------
    # Select local UTM projection
    # -------------------------------------------------------------------------

    projected_crs = _local_utm_crs(
        gdf["lon"].values,
        gdf["lat"].values
    )

    to_proj = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True
    )

    to_wgs = Transformer.from_crs(
        projected_crs,
        "EPSG:4326",
        always_xy=True
    )

    # Helper for transforming Shapely geometries back to WGS84
    def reproj_geom(geom):
        return shp_transform(
            lambda x, y, z=None: to_wgs.transform(x, y),
            geom
        )

    # Project all coordinates to meters
    xs, ys = to_proj.transform(
        gdf["lon"].values,
        gdf["lat"].values
    )

    gdf["x_m"] = xs
    gdf["y_m"] = ys

    # -------------------------------------------------------------------------
    # Output object
    # -------------------------------------------------------------------------

    out: Dict = {
        "method": method,
        "isopleths": list(params.isopleths),
        "analysis_crs": projected_crs.to_string(),
        "animals": {},
    }

    # -------------------------------------------------------------------------
    # Analyze each animal separately
    # -------------------------------------------------------------------------

    for animal, sub in gdf.groupby("animal_id"):

        sub = sub.reset_index(drop=True)

        if len(sub) < 3:

            out["animals"][str(animal)] = {
                "n_points": int(len(sub)),
                "isopleths": [],
                "facets": [],
            }

            continue

        coords_m = np.column_stack(
            [
                sub["x_m"].values,
                sub["y_m"].values
            ]
        )

        tree = cKDTree(coords_m)

        # ---------------------------------------------------------------------
        # Construct local hulls
        # ---------------------------------------------------------------------

        hulls: List[Polygon] = []

        n = len(coords_m)

        k_cap = max(
            3,
            min(
                params.k if method == "k" else 10,
                n,
                50
            )
        )

        for i, p in enumerate(coords_m):

            # -------------------------------------------------------------
            # k-LoCoH
            # -------------------------------------------------------------

            if method == "k":

                k = max(
                    3,
                    min(
                        params.k,
                        n,
                        50
                    )
                )

                dists, idxs = tree.query(
                    p,
                    k
                )

                idxs = np.atleast_1d(
                    idxs
                )

                pts = coords_m[
                    idxs
                ]

            # -------------------------------------------------------------
            # r-LoCoH
            # -------------------------------------------------------------

            elif method == "r":

                idxs = tree.query_ball_point(
                    p,
                    float(params.r)
                )

                if len(idxs) < 3:

                    _, idxs = tree.query(
                        p,
                        3
                    )

                    idxs = np.atleast_1d(
                        idxs
                    )

                pts = coords_m[
                    idxs
                ]

            # -------------------------------------------------------------
            # a-LoCoH
            # -------------------------------------------------------------

            else:

                # Search a local neighborhood first
                query_k = min(
                    max(
                        10,
                        min(
                            2 * k_cap,
                            n
                        )
                    ),
                    n
                )

                dists, idxs = tree.query(
                    p,
                    k=query_k
                )

                dists = np.atleast_1d(
                    dists
                )

                idxs = np.atleast_1d(
                    idxs
                )

                order = np.argsort(
                    dists
                )

                cumdist = 0.0
                kept: List[int] = []

                for j in order:

                    nd = float(
                        dists[j]
                    )

                    cumdist += nd

                    kept.append(
                        int(idxs[j])
                    )

                    if cumdist >= float(params.a):
                        break

                if len(kept) < 3:

                    _, extra = tree.query(
                        p,
                        3
                    )

                    kept = list(
                        np.unique(
                            np.atleast_1d(
                                extra
                            )
                        )
                    )

                pts = coords_m[
                    kept
                ]

            # -------------------------------------------------------------
            # Construct convex hull
            # -------------------------------------------------------------

            if len(pts) >= 3:

                hull = MultiPoint(
                    pts
                ).convex_hull

                if (
                    isinstance(hull, Polygon)
                    and not hull.is_empty
                    and hull.area > 0
                ):
                    hulls.append(
                        hull
                    )

        # ---------------------------------------------------------------------
        # Handle failure to create hulls
        # ---------------------------------------------------------------------

        if not hulls:

            out["animals"][str(animal)] = {
                "n_points": int(len(sub)),
                "isopleths": [],
                "facets": [],
            }

            continue

        # ---------------------------------------------------------------------
        # Sort local hulls by area
        # -------------------------------------------------------------------------

        hulls.sort(
            key=lambda h: h.area
        )

        # ---------------------------------------------------------------------
        # Per-hull facets for visualization
        # -------------------------------------------------------------------------

        total_area = (
            sum(
                h.area
                for h in hulls
            )
            or 1.0
        )

        facets = []

        cum = 0.0

        for h in hulls:

            cum += h.area

            frac = min(
                1.0,
                cum / total_area
            )

            pct = int(
                round(
                    frac * 100
                )
            )

            facets.append(
                {
                    "cum_frac": float(frac),

                    "cum_percent": int(pct),

                    "area_sq_km": float(
                        h.area / 1e6
                    ),

                    "geometry": mapping(
                        reproj_geom(h)
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # Cumulative unions at requested isopleths
        # -------------------------------------------------------------------------

        targets = sorted(
            {
                int(t)
                for t in params.isopleths
                if 1 <= int(t) <= 100
            }
        )

        polys_by_iso: Dict[
            int,
            Polygon
        ] = {}

        cum_union = None
        cum_area = 0.0

        t_idx = 0

        target_fracs = [
            t / 100.0
            for t in targets
        ]

        for h in hulls:

            cum_area += h.area

            if cum_union is None:
                cum_union = h
            else:
                cum_union = (
                    cum_union.union(h)
                )

            while (
                t_idx < len(target_fracs)
                and
                (cum_area / total_area)
                >= target_fracs[t_idx]
            ):

                polys_by_iso[
                    targets[t_idx]
                ] = cum_union

                t_idx += 1

            if t_idx >= len(
                target_fracs
            ):
                break

        if (
            targets
            and targets[-1]
            not in polys_by_iso
        ):

            polys_by_iso[
                targets[-1]
            ] = (
                cum_union
                if cum_union is not None
                else unary_union(hulls)
            )

        # ---------------------------------------------------------------------
        # Reproject results and calculate areas
        # -------------------------------------------------------------------------

        envelopes = []

        for iso, geom_m in polys_by_iso.items():

            geom_wgs = reproj_geom(
                geom_m
            )

            envelopes.append(
                {
                    "isopleth": int(iso),

                    "area_sq_km": float(
                        geom_m.area / 1e6
                    ),

                    "geometry": mapping(
                        geom_wgs
                    ),
                }
            )

        # ---------------------------------------------------------------------
        # Store result
        # -------------------------------------------------------------------------

        out["animals"][
            str(animal)
        ] = {

            "n_points": int(
                len(sub)
            ),

            "isopleths": envelopes,

            "facets": facets,
        }

    return out
