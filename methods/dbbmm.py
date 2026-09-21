# methods/dbbmm.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import os

import numpy as np
import pandas as pd

from affine import Affine

import rasterio
from rasterio.features import shapes as rio_shapes

from shapely.geometry import (
    shape as shp_shape,
    mapping as shp_mapping,
    MultiPolygon,
    Polygon,
)

from shapely.ops import (
    unary_union,
    transform as shp_transform,
)

from pyproj import CRS, Transformer


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

@dataclass
class DBBMMParams:
    location_error_m: float = 30.0
    window_size: int = 31
    margin: int = 11
    raster_resolution_m: float = 50.0
    buffer_m: float = 1000.0
    n_substeps: int = 40
    isopleths: Tuple[int, ...] = (50, 95)


@dataclass
class DBBMMResult:
    geotiff: str
    isopleths: List[Dict]


# -----------------------------------------------------------------------------
# Projection helper
# -----------------------------------------------------------------------------

def _local_utm_crs(longitudes, latitudes):
    """
    Select a local WGS84 UTM CRS based on the mean longitude/latitude
    of one animal's locations.

    Northern hemisphere: EPSG:326xx
    Southern hemisphere: EPSG:327xx
    """

    longitudes = np.asarray(
        longitudes,
        dtype=float
    )

    latitudes = np.asarray(
        latitudes,
        dtype=float
    )

    lon0 = float(
        np.mean(longitudes)
    )

    lat0 = float(
        np.mean(latitudes)
    )

    zone = int(
        np.floor(
            (lon0 + 180.0) / 6.0
        )
    ) + 1

    zone = max(
        1,
        min(60, zone)
    )

    if lat0 >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone

    return CRS.from_epsg(
        epsg
    )


# -----------------------------------------------------------------------------
# Main dBBMM function
# -----------------------------------------------------------------------------

def compute_dbbmm(
    df: pd.DataFrame,
    id_col: str,
    x_col: str,
    y_col: str,
    ts_col: str,
    params: Optional[DBBMMParams] = None,
    outputs_dir: str = "outputs",
) -> Dict[str, DBBMMResult]:
    """
    Compute dBBMM per animal using a local UTM meter grid.

    Input coordinates are expected to be WGS84 longitude/latitude.

    For each animal:
      1. A local UTM CRS is selected from the animal's mean location.
      2. All movement distances, velocities, Brownian bridge calculations,
         raster operations, and area calculations are performed in meters.
      3. The GeoTIFF is written in the projected UTM CRS.
      4. Final isopleth geometries are transformed back to WGS84
         for web mapping.

    Returns
    -------
    {
        animal_id: DBBMMResult(
            geotiff=<projected GeoTIFF>,
            isopleths=[
                {
                    "percent": int,
                    "area_sq_km": float,
                    "geometry": <GeoJSON WGS84>,
                    "analysis_crs": str
                },
                ...
            ]
        )
    }
    """

    if params is None:
        params = DBBMMParams()

    # -------------------------------------------------------------------------
    # Normalize input data
    # -------------------------------------------------------------------------

    if id_col not in df.columns:
        df = df.copy()
        df[id_col] = "Animal_1"

    required = {
        id_col,
        x_col,
        y_col,
        ts_col,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"dBBMM missing required columns: {sorted(missing)}"
        )

    df0 = (
        df[
            [
                id_col,
                x_col,
                y_col,
                ts_col,
            ]
        ]
        .dropna()
        .copy()
    )

    df0.columns = [
        "animal_id",
        "lon",
        "lat",
        "timestamp",
    ]

    df0["lon"] = pd.to_numeric(
        df0["lon"],
        errors="coerce"
    )

    df0["lat"] = pd.to_numeric(
        df0["lat"],
        errors="coerce"
    )

    df0["timestamp"] = pd.to_datetime(
        df0["timestamp"],
        errors="coerce"
    )

    df0 = df0.dropna(
        subset=[
            "lon",
            "lat",
            "timestamp",
        ]
    )

    results: Dict[
        str,
        DBBMMResult
    ] = {}

    os.makedirs(
        outputs_dir,
        exist_ok=True
    )

    # -------------------------------------------------------------------------
    # Analyze each animal separately
    # -------------------------------------------------------------------------

    for animal, sub in df0.groupby(
        "animal_id"
    ):

        sub = (
            sub.sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        if len(sub) < 2:
            continue

        # ---------------------------------------------------------------------
        # Select local UTM CRS for this animal
        # ---------------------------------------------------------------------

        projected_crs = _local_utm_crs(
            sub["lon"].values,
            sub["lat"].values
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

        def reproj_geom(geom):
            return shp_transform(
                lambda x, y, z=None:
                    to_wgs.transform(
                        x,
                        y
                    ),
                geom
            )

        # ---------------------------------------------------------------------
        # Project lon/lat to UTM meters
        # ---------------------------------------------------------------------

        xs, ys = to_proj.transform(
            sub["lon"].values,
            sub["lat"].values
        )

        xs = np.asarray(
            xs,
            dtype=float
        )

        ys = np.asarray(
            ys,
            dtype=float
        )

        ts = (
            sub["timestamp"]
            .astype("int64")
            .to_numpy()
            / 1e9
        )

        # ---------------------------------------------------------------------
        # Drop zero or negative time gaps
        # ---------------------------------------------------------------------

        dt = np.diff(
            ts
        )

        valid = (
            dt > 0
        )

        if not np.all(valid):

            keep_idx = np.insert(
                valid,
                0,
                True
            )

            xs = xs[
                keep_idx
            ]

            ys = ys[
                keep_idx
            ]

            ts = ts[
                keep_idx
            ]

            if len(xs) < 2:
                continue

            dt = np.diff(
                ts
            )

        # ---------------------------------------------------------------------
        # Movement steps and velocity
        # ---------------------------------------------------------------------

        coords = np.column_stack(
            [
                xs,
                ys
            ]
        )

        steps = np.diff(
            coords,
            axis=0
        )

        d = np.hypot(
            steps[:, 0],
            steps[:, 1]
        )

        v = (
            d
            / np.maximum(
                dt,
                1e-6
            )
        )

        # ---------------------------------------------------------------------
        # Rolling variance of velocity
        # ---------------------------------------------------------------------

        w = int(
            max(
                5,
                (
                    params.window_size
                    if params.window_size % 2 == 1
                    else params.window_size + 1
                )
            )
        )

        pad = (
            w // 2
        )

        v_pad = np.pad(
            v,
            (
                pad,
                pad
            ),
            mode="edge"
        )

        v2 = (
            pd.Series(
                v_pad
            )
            .rolling(
                window=w,
                center=True,
                min_periods=max(
                    5,
                    w // 3
                )
            )
            .var()
            .to_numpy()[
                pad:-pad
            ]
        )

        if len(v2) != len(v):
            v2 = np.resize(
                v2,
                len(v)
            )

        baseline_var_v = (
            params.raster_resolution_m
            / 5.0
        ) ** 2

        v2 = np.where(
            np.isfinite(v2)
            & (v2 > 0),
            v2,
            baseline_var_v
        )

        T = dt
        sigma2 = v2

        # ---------------------------------------------------------------------
        # Raster grid in local UTM coordinates
        # ---------------------------------------------------------------------

        res = float(
            params.raster_resolution_m
        )

        buf = float(
            params.buffer_m
        )

        minx = float(
            np.min(xs)
            - buf
        )

        maxx = float(
            np.max(xs)
            + buf
        )

        miny = float(
            np.min(ys)
            - buf
        )

        maxy = float(
            np.max(ys)
            + buf
        )

        width = int(
            max(
                1,
                math.ceil(
                    (maxx - minx)
                    / res
                )
            )
        )

        height = int(
            max(
                1,
                math.ceil(
                    (maxy - miny)
                    / res
                )
            )
        )

        transform = (
            Affine.translation(
                minx,
                maxy
            )
            * Affine.scale(
                res,
                -res
            )
        )

        # ---------------------------------------------------------------------
        # World -> raster col,row
        # ---------------------------------------------------------------------

        def world_to_cr(
            x: float,
            y: float
        ) -> Tuple[
            float,
            float
        ]:

            c = (
                x - minx
            ) / res

            r = (
                maxy - y
            ) / res

            return c, r

        # ---------------------------------------------------------------------
        # Initialize UD
        # ---------------------------------------------------------------------

        UD = np.zeros(
            (
                height,
                width
            ),
            dtype=np.float64
        )

        cell_area = (
            res
            * res
        )

        nseg = len(
            steps
        )

        n_sub = int(
            max(
                5,
                params.n_substeps
            )
        )

        loc_err2 = (
            float(
                params.location_error_m
            )
            ** 2
        )

        # ---------------------------------------------------------------------
        # Brownian bridge accumulation
        # ---------------------------------------------------------------------

        updated_windows = 0

        for i in range(
            nseg
        ):

            x0, y0 = coords[i]

            x1, y1 = coords[
                i + 1
            ]

            Ti = max(
                T[i],
                1e-3
            )

            sig2 = max(
                sigma2[i],
                (
                    params.raster_resolution_m
                    / 10.0
                )
                ** 2
            )

            # -------------------------------------------------------------
            # Spatial window around segment
            # -------------------------------------------------------------

            seg_len = max(
                1.0,
                np.hypot(
                    x1 - x0,
                    y1 - y0
                )
            )

            sigma_max = math.sqrt(
                loc_err2
                + sig2
                * (
                    0.25
                    * Ti
                )
            )

            radius = (
                3.0
                * (
                    sigma_max
                    + 0.5
                    * seg_len
                )
            )

            minx_w = (
                min(
                    x0,
                    x1
                )
                - radius
            )

            maxx_w = (
                max(
                    x0,
                    x1
                )
                + radius
            )

            miny_w = (
                min(
                    y0,
                    y1
                )
                - radius
            )

            maxy_w = (
                max(
                    y0,
                    y1
                )
                + radius
            )

            # -------------------------------------------------------------
            # Convert spatial window to raster indices
            # -------------------------------------------------------------

            c0f, r_topf = world_to_cr(
                minx_w,
                maxy_w
            )

            c1f, r_botf = world_to_cr(
                maxx_w,
                miny_w
            )

            c0 = int(
                max(
                    0,
                    math.floor(
                        min(
                            c0f,
                            c1f
                        )
                    )
                )
            )

            c1 = int(
                min(
                    width - 1,
                    math.ceil(
                        max(
                            c0f,
                            c1f
                        )
                    )
                )
            )

            r0 = int(
                max(
                    0,
                    math.floor(
                        min(
                            r_topf,
                            r_botf
                        )
                    )
                )
            )

            r1 = int(
                min(
                    height - 1,
                    math.ceil(
                        max(
                            r_topf,
                            r_botf
                        )
                    )
                )
            )

            if (
                r1 < r0
                or c1 < c0
            ):
                continue

            # -------------------------------------------------------------
            # Substep positions along segment
            # -------------------------------------------------------------

            ss = np.linspace(
                0.0,
                1.0,
                n_sub,
                endpoint=True
            )

            xs_sub = (
                x0
                + ss
                * (
                    x1 - x0
                )
            )

            ys_sub = (
                y0
                + ss
                * (
                    y1 - y0
                )
            )

            # -------------------------------------------------------------
            # Pixel-center coordinates
            # -------------------------------------------------------------

            rows = np.arange(
                r0,
                r1 + 1
            )

            cols = np.arange(
                c0,
                c1 + 1
            )

            if (
                rows.size == 0
                or cols.size == 0
            ):
                continue

            xx = (
                minx
                + (
                    cols
                    + 0.5
                )
                * res
            )

            yy = (
                maxy
                - (
                    rows
                    + 0.5
                )
                * res
            )

            XX, YY = np.meshgrid(
                xx,
                yy
            )

            # -------------------------------------------------------------
            # Accumulate Gaussian kernels
            # -------------------------------------------------------------

            for (
                x_s,
                y_s,
                s
            ) in zip(
                xs_sub,
                ys_sub,
                ss
            ):

                var_s = (
                    loc_err2
                    + sig2
                    * (
                        s
                        * (
                            1.0 - s
                        )
                        * Ti
                    )
                )

                if var_s <= 0:
                    continue

                dx = (
                    XX - x_s
                )

                dy = (
                    YY - y_s
                )

                inv_two = (
                    1.0
                    / (
                        2.0
                        * var_s
                    )
                )

                kernel = (
                    np.exp(
                        -(
                            dx * dx
                            + dy * dy
                        )
                        * inv_two
                    )
                    / (
                        2.0
                        * math.pi
                        * var_s
                    )
                )

                UD[
                    rows[:, None],
                    cols[None, :]
                ] += kernel

            updated_windows += 1

        # ---------------------------------------------------------------------
        # Debug before normalization
        # ---------------------------------------------------------------------

        nz = int(
            np.count_nonzero(
                UD
            )
        )

        print(
            f"[dBBMM] animal={animal} "
            f"pre-norm UD stats: "
            f"sum={UD.sum() * cell_area:.6e}, "
            f"max={UD.max():.6e}, "
            f"nonzero={nz}/{UD.size}, "
            f"windows={updated_windows}, "
            f"crs={projected_crs.to_string()}"
        )

        # ---------------------------------------------------------------------
        # Normalize UD
        # ---------------------------------------------------------------------

        total = (
            UD.sum()
            * cell_area
        )

        if total > 0:

            UD /= total

        else:

            print(
                f"[dBBMM] animal={animal} "
                f"WARNING: UD total mass is zero; "
                f"check timestamps/units."
            )

        # ---------------------------------------------------------------------
        # Write projected GeoTIFF
        # ---------------------------------------------------------------------

        safe_animal = (
            str(animal)
            .replace(
                " ",
                "_"
            )
            .replace(
                "/",
                "_"
            )
        )

        tif_path = os.path.join(
            outputs_dir,
            f"dbbmm_{safe_animal}.tif"
        )

        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=UD.shape[0],
            width=UD.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs=projected_crs.to_wkt(),
            transform=transform,
            compress="lzw",
        ) as dst:

            dst.write(
                UD.astype(
                    np.float32
                ),
                1
            )

        # ---------------------------------------------------------------------
        # Extract isopleth polygons
        # ---------------------------------------------------------------------

        flat = UD.ravel()

        order = np.argsort(
            flat
        )[::-1]

        flat_sorted = flat[
            order
        ]

        mass = np.cumsum(
            flat_sorted
            * cell_area
        )

        total_mass = (
            mass[-1]
            if mass.size
            else 0.0
        )

        env_list: List[
            Dict
        ] = []

        if (
            total_mass > 0
            and len(
                flat_sorted
            ) > 0
        ):

            levels = sorted(
                set(
                    int(x)
                    for x in params.isopleths
                    if 1 <= int(x) <= 100
                )
            )

            for p in levels:

                target = (
                    p
                    / 100.0
                ) * total_mass

                idx = int(
                    np.searchsorted(
                        mass,
                        target,
                        side="left"
                    )
                )

                thr = float(
                    flat_sorted[
                        min(
                            idx,
                            len(
                                flat_sorted
                            ) - 1
                        )
                    ]
                )

                mask = (
                    UD >= thr
                ).astype(
                    np.uint8
                )

                polygons: List[
                    Polygon
                ] = []

                for (
                    geom,
                    val
                ) in rio_shapes(
                    mask,
                    mask=mask.astype(
                        bool
                    ),
                    transform=transform
                ):

                    if val == 1:

                        poly = shp_shape(
                            geom
                        )

                        if (
                            not poly.is_empty
                            and poly.area > 0
                        ):

                            polygons.append(
                                poly
                            )

                if not polygons:
                    continue

                mp = unary_union(
                    polygons
                )

                if isinstance(
                    mp,
                    (
                        Polygon,
                        MultiPolygon
                    )
                ):

                    area_sq_km = float(
                        mp.area
                        / 1e6
                    )

                    mp_wgs = reproj_geom(
                        mp
                    )

                    env_list.append(
                        {
                            "percent": int(p),

                            "area_sq_km": area_sq_km,

                            "geometry": shp_mapping(
                                mp_wgs
                            ),

                            "analysis_crs": (
                                projected_crs.to_string()
                            ),
                        }
                    )

        # ---------------------------------------------------------------------
        # Debug after normalization
        # ---------------------------------------------------------------------

        nz_post = int(
            np.count_nonzero(
                UD
            )
        )

        print(
            f"[dBBMM] animal={animal} "
            f"post-norm UD stats: "
            f"sum={UD.sum() * cell_area:.6e}, "
            f"max={UD.max():.6e}, "
            f"nonzero={nz_post}/{UD.size}, "
            f"tif={tif_path}, "
            f"crs={projected_crs.to_string()}"
        )

        # ---------------------------------------------------------------------
        # Save result
        # ---------------------------------------------------------------------

        results[
            str(animal)
        ] = DBBMMResult(
            geotiff=tif_path,
            isopleths=env_list
        )

    return results
