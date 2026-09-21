# methods/dbbmm.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence

import math
import os

import numpy as np
import pandas as pd

from affine import Affine

import rasterio
from rasterio.features import shapes as rio_shapes

from scipy.optimize import minimize_scalar

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


# =============================================================================
# Parameters
# =============================================================================

@dataclass
class DBBMMParams:
    """
    Parameters modeled after move::brownian.bridge.dyn().

    location_error_m
        Location error in projected map units (meters). May be a scalar
        or a sequence with one value per location.

    window_size
        Sliding window size used to estimate dynamic Brownian motion
        variance. Must be odd.

    margin
        Margin used within the sliding variance window. Must be odd.

    raster_resolution_m
        Raster cell size in meters.

    ext
        Proportional extension of the track bounding box, matching the
        semantics of move::brownian.bridge.dyn(ext=...).

        ext=0.3 expands each side by 30% of the original range.

    time_step_min
        Integration time step in minutes. If None, use the move default:
        shortest positive time lag / 15.

    isopleths
        Requested utilization-distribution contours.
    """

    location_error_m: float | Sequence[float] = 30.0
    window_size: int = 31
    margin: int = 11
    raster_resolution_m: float = 50.0
    ext: float = 0.3
    time_step_min: Optional[float] = None
    isopleths: Tuple[int, ...] = (50, 95)

    # -------------------------------------------------------------------------
    # Deprecated compatibility arguments.
    #
    # Older Spatchat app.py versions may still pass these. They are deliberately
    # ignored because they do not correspond to move::brownian.bridge.dyn().
    # -------------------------------------------------------------------------
    buffer_m: Optional[float] = None
    n_substeps: Optional[int] = None


@dataclass
class DBBMMResult:
    geotiff: str
    isopleths: List[Dict]


# =============================================================================
# Projection
# =============================================================================

def _local_utm_crs(longitudes, latitudes):
    """
    Select a local WGS84 UTM CRS based on mean longitude/latitude.

    This is Spatchat preprocessing. The move package itself requires the
    input Move object to already use a projected coordinate system.
    """

    lon = np.asarray(longitudes, dtype=float)
    lat = np.asarray(latitudes, dtype=float)

    lon0 = float(np.mean(lon))
    lat0 = float(np.mean(lat))

    zone = int(np.floor((lon0 + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))

    epsg = (
        32600 + zone
        if lat0 >= 0
        else 32700 + zone
    )

    return CRS.from_epsg(epsg)


# =============================================================================
# Location error
# =============================================================================

def _location_error_vector(
    value,
    n: int
) -> np.ndarray:
    """
    Match move behavior:
    - scalar -> repeat for every location
    - vector -> must equal number of locations
    """

    arr = np.asarray(value, dtype=float)

    if arr.ndim == 0:
        arr = np.repeat(float(arr), n)

    else:
        arr = arr.reshape(-1)

        if arr.size == 1:
            arr = np.repeat(float(arr[0]), n)

    if arr.size != n:
        raise ValueError(
            "Location error must be a scalar or have exactly one value "
            "for every location."
        )

    if np.any(~np.isfinite(arr)):
        raise ValueError(
            "Location error contains missing or non-finite values."
        )

    if np.any(arr <= 0):
        raise ValueError(
            "Location error values must be positive."
        )

    return arr.astype(float)


# =============================================================================
# Brownian motion variance
# =============================================================================

def _brownian_motion_variance(
    time_lag_min: np.ndarray,
    location_error: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    """
    Port of move::brownian.motion.variance().

    Estimates Brownian motion variance by maximum likelihood using
    alternating location triplets.

    Returns
    -------
    bmvar
        Maximum-likelihood Brownian motion variance.

    cll
        Maximized log likelihood.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    err = np.asarray(location_error, dtype=float)
    lag = np.asarray(time_lag_min, dtype=float)

    n = len(x)

    if not (
        len(y) == n
        and len(err) == n
    ):
        raise ValueError(
            "Coordinate and location-error vectors must have equal length."
        )

    if n < 3:
        raise ValueError(
            "At least three locations are required for motion variance."
        )

    # move's R function evaluates centers 2,4,6,... in one-based indexing.
    # In Python these are indices 1,3,5,...
    centers = np.arange(
        1,
        n - 1,
        2,
        dtype=int
    )

    if centers.size == 0:
        raise ValueError(
            "Insufficient locations for Brownian motion variance estimation."
        )

    t_jump = []
    alpha = []
    ztz = []
    err1 = []
    err2 = []

    for i in centers:

        # Segment i-1 -> i and i -> i+1
        dt1 = float(lag[i - 1])
        dt2 = float(lag[i])

        if (
            not np.isfinite(dt1)
            or not np.isfinite(dt2)
            or dt1 <= 0
            or dt2 <= 0
        ):
            continue

        total_t = dt1 + dt2

        a = dt1 / total_t

        ux = x[i - 1] + a * (
            x[i + 1] - x[i - 1]
        )

        uy = y[i - 1] + a * (
            y[i + 1] - y[i - 1]
        )

        residual2 = (
            (x[i] - ux) ** 2
            + (y[i] - uy) ** 2
        )

        t_jump.append(total_t)
        alpha.append(a)
        ztz.append(residual2)
        err1.append(err[i - 1])
        err2.append(err[i + 1])

    if not t_jump:
        raise ValueError(
            "No valid location triplets were available for "
            "Brownian motion variance estimation."
        )

    t_jump = np.asarray(t_jump, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    ztz = np.asarray(ztz, dtype=float)
    err1 = np.asarray(err1, dtype=float)
    err2 = np.asarray(err2, dtype=float)

    def neg_log_likelihood(bmvar: float) -> float:

        if bmvar < 0:
            return np.inf

        variance = (
            t_jump
            * alpha
            * (1.0 - alpha)
            * bmvar
            + ((1.0 - alpha) ** 2)
            * (err1 ** 2)
            + (alpha ** 2)
            * (err2 ** 2)
        )

        if np.any(
            ~np.isfinite(variance)
            | (variance <= 0)
        ):
            return np.inf

        # Algebraically equivalent to move's likelihood calculation,
        # but evaluated in a numerically stable form.
        return float(
            np.sum(
                np.log(
                    2.0
                    * math.pi
                    * variance
                )
                + ztz
                / (
                    2.0
                    * variance
                )
            )
        )

    result = minimize_scalar(
        neg_log_likelihood,
        bounds=(0.0, 1.0e15),
        method="bounded",
    )

    if (
        not result.success
        or not np.isfinite(result.x)
        or result.x <= 0.0
        or result.x >= 1.0e15
    ):
        raise RuntimeError(
            "Brownian motion variance optimization failed. "
            "Consider checking coordinate units and location error."
        )

    bmvar = float(result.x)
    cll = -float(result.fun)

    return bmvar, cll


# =============================================================================
# Dynamic variance
# =============================================================================

def _dynamic_bm_variance(
    x: np.ndarray,
    y: np.ndarray,
    time_min: np.ndarray,
    location_error: np.ndarray,
    window_size: int,
    margin: int,
) -> Dict:
    """
    Port of move::brownian.motion.variance.dyn().

    Implements:
      - sliding windows
      - single-window BM variance
      - candidate behavioral breakpoints
      - BIC comparison
      - segment-wise averaging across overlapping windows
      - move-style interest mask
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    time_min = np.asarray(time_min, dtype=float)
    location_error = np.asarray(
        location_error,
        dtype=float
    )

    n = len(x)

    if n < window_size:
        raise ValueError(
            "window_size cannot be larger than the number of locations."
        )

    if window_size % 2 != 1:
        raise ValueError(
            "window_size must be odd."
        )

    if margin % 2 != 1:
        raise ValueError(
            "margin must be odd."
        )

    if window_size < 2 * margin:
        raise ValueError(
            "window_size must be large enough relative to margin."
        )

    # move::timeLag(..., units="mins")
    time_lag = np.diff(
        time_min
    )

    if np.any(
        ~np.isfinite(time_lag)
        | (time_lag <= 0)
    ):
        raise ValueError(
            "dBBMM requires strictly increasing timestamps."
        )

    # Add one trailing NA solely so window slicing mirrors the R code.
    time_lag_full = np.concatenate(
        [
            time_lag,
            [np.nan]
        ]
    )

    # R:
    # breaks <- margin:(window.size - margin + 1)
    breaks_r = np.arange(
        margin,
        window_size - margin + 2,
        dtype=int
    )

    if breaks_r.size < 2:
        raise ValueError(
            "Margin to window ratio is not appropriate."
        )

    # Only odd candidate break locations
    uneven_breaks_r = breaks_r[
        breaks_r % 2 == 1
    ]

    # Each segment/location may receive estimates from several windows.
    estimates_by_loc: List[List[float]] = [
        []
        for _ in range(n)
    ]

    breaks_found: List[int] = []

    # Equivalent to R:
    # 1:(n.locs(object) - window.size + 1)
    for w0 in range(
        0,
        n - window_size + 1
    ):

        stop = (
            w0
            + window_size
        )

        x_sub = x[
            w0:stop
        ]

        y_sub = y[
            w0:stop
        ]

        err_sub = location_error[
            w0:stop
        ]

        lag_sub = time_lag_full[
            w0:stop
        ]

        # -------------------------------------------------------------
        # Whole-window variance
        # -------------------------------------------------------------

        whole_var, whole_cll = (
            _brownian_motion_variance(
                lag_sub,
                err_sub,
                x_sub,
                y_sub,
            )
        )

        whole_bic = (
            -2.0 * whole_cll
            + math.log(
                window_size
            )
        )

        # -------------------------------------------------------------
        # Search candidate change points
        # -------------------------------------------------------------

        best_break = None

        for b_r in uneven_breaks_r:

            # R uses 1:b and b:window.size.
            # b_r is one-based.
            before_stop = b_r

            after_start = (
                b_r - 1
            )

            try:

                before_var, before_cll = (
                    _brownian_motion_variance(
                        lag_sub[
                            :before_stop
                        ],
                        err_sub[
                            :before_stop
                        ],
                        x_sub[
                            :before_stop
                        ],
                        y_sub[
                            :before_stop
                        ],
                    )
                )

                after_var, after_cll = (
                    _brownian_motion_variance(
                        lag_sub[
                            after_start:
                        ],
                        err_sub[
                            after_start:
                        ],
                        x_sub[
                            after_start:
                        ],
                        y_sub[
                            after_start:
                        ],
                    )
                )

            except (
                ValueError,
                RuntimeError
            ):
                continue

            break_bic = (
                -2.0
                * (
                    before_cll
                    + after_cll
                )
                + 2.0
                * math.log(
                    window_size
                )
            )

            if (
                best_break is None
                or break_bic
                < best_break["bic"]
            ):

                best_break = {
                    "b_r": int(b_r),
                    "before_var": float(
                        before_var
                    ),
                    "after_var": float(
                        after_var
                    ),
                    "bic": float(
                        break_bic
                    ),
                }

        # -------------------------------------------------------------
        # Select whole-window versus broken-window solution
        # -------------------------------------------------------------

        if (
            best_break is not None
            and best_break["bic"]
            < whole_bic
        ):

            b_r = best_break[
                "b_r"
            ]

            n_before = int(
                np.sum(
                    breaks_r
                    < b_r
                )
            )

            n_after = int(
                np.sum(
                    breaks_r
                    > b_r
                )
            )

            window_variance = np.concatenate(
                [
                    np.repeat(
                        best_break[
                            "before_var"
                        ],
                        n_before
                    ),
                    np.repeat(
                        best_break[
                            "after_var"
                        ],
                        n_after
                    ),
                ]
            )

            # R: w - 1 + breakWindow$b
            # Convert to Python zero-based index.
            breaks_found.append(
                int(
                    w0
                    + b_r
                    - 1
                )
            )

        else:

            window_variance = np.repeat(
                whole_var,
                breaks_r.size - 1
            )

        # -------------------------------------------------------------
        # R:
        # loc = w - 1 + margin:(window.size-margin)
        #
        # With zero-based Python indexes:
        # w0 + margin-1 ... w0 + window_size-margin-1
        # -------------------------------------------------------------

        locs = np.arange(
            w0 + margin - 1,
            w0 + window_size - margin,
            dtype=int
        )

        if (
            window_variance.size
            != locs.size
        ):
            raise RuntimeError(
                "Internal dBBMM variance alignment error."
            )

        for loc, value in zip(
            locs,
            window_variance
        ):

            estimates_by_loc[
                int(loc)
            ].append(
                float(value)
            )

    # -------------------------------------------------------------------------
    # Aggregate overlapping windows exactly as move does
    # -------------------------------------------------------------------------

    means = np.full(
        n,
        np.nan,
        dtype=float
    )

    in_windows = np.full(
        n,
        np.nan,
        dtype=float
    )

    counts = np.array(
        [
            len(v)
            for v
            in estimates_by_loc
        ],
        dtype=int
    )

    valid_locations = np.where(
        counts > 0
    )[0]

    if valid_locations.size == 0:
        raise RuntimeError(
            "No dynamic Brownian variance estimates were produced."
        )

    for loc in valid_locations:

        vals = np.asarray(
            estimates_by_loc[
                int(loc)
            ],
            dtype=float
        )

        means[
            loc
        ] = float(
            np.mean(
                vals
            )
        )

        in_windows[
            loc
        ] = float(
            len(vals)
        )

    max_count = int(
        np.max(
            counts[
                valid_locations
            ]
        )
    )

    interest = (
        counts
        == max_count
    )

    # Last location is not a movement segment.
    interest[-1] = False

    return {
        "means": means,
        "in_windows": in_windows,
        "interest": interest,
        "breaks": breaks_found,
    }


# =============================================================================
# Raster grid matching move
# =============================================================================

def _move_grid(
    x: np.ndarray,
    y: np.ndarray,
    cell_size: float,
    ext: float,
):
    """
    Reproduce move's .extcalc() + numeric-raster construction.

    move's scalar ext expands each side by:
        coordinate_range * ext

    It then symmetrically adjusts the expanded extent so that an exact
    integer number of square cells fits.
    """

    if cell_size <= 0:
        raise ValueError(
            "Raster resolution must be positive."
        )

    if ext < 0:
        raise ValueError(
            "ext must be non-negative."
        )

    xmin0 = float(
        np.min(x)
    )

    xmax0 = float(
        np.max(x)
    )

    ymin0 = float(
        np.min(y)
    )

    ymax0 = float(
        np.max(y)
    )

    x_range0 = (
        xmax0 - xmin0
    )

    y_range0 = (
        ymax0 - ymin0
    )

    # move::.extcalc
    range_xmin = (
        xmin0
        - x_range0
        * ext
    )

    range_xmax = (
        xmax0
        + x_range0
        * ext
    )

    range_ymin = (
        ymin0
        - y_range0
        * ext
    )

    range_ymax = (
        ymax0
        + y_range0
        * ext
    )

    x_range = (
        range_xmax
        - range_xmin
    )

    y_range = (
        range_ymax
        - range_ymin
    )

    ncol = int(
        math.ceil(
            x_range
            / cell_size
        )
    )

    nrow = int(
        math.ceil(
            y_range
            / cell_size
        )
    )

    # Symmetric padding so square cells fit exactly.
    extra_x = (
        ncol
        * cell_size
        - x_range
    )

    extra_y = (
        nrow
        * cell_size
        - y_range
    )

    xmin = (
        range_xmin
        - extra_x / 2.0
    )

    xmax = (
        range_xmax
        + extra_x / 2.0
    )

    ymin = (
        range_ymin
        - extra_y / 2.0
    )

    ymax = (
        range_ymax
        + extra_y / 2.0
    )

    # Cell centers, analogous to xFromCol() / yFromRow()
    x_grid = (
        xmin
        + (
            np.arange(
                ncol,
                dtype=float
            )
            + 0.5
        )
        * cell_size
    )

    # Ascending south -> north, as passed to move's dbbmm2.
    y_grid = (
        ymin
        + (
            np.arange(
                nrow,
                dtype=float
            )
            + 0.5
        )
        * cell_size
    )

    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "ncol": ncol,
        "nrow": nrow,
        "x_grid": x_grid,
        "y_grid": y_grid,
    }


# =============================================================================
# dBBMM grid evaluator
# =============================================================================

def _dbbmm_grid(
    x: np.ndarray,
    y: np.ndarray,
    time_min: np.ndarray,
    means: np.ndarray,
    interest: np.ndarray,
    location_error: np.ndarray,
    grid: Dict,
    time_step_min: float,
    sd_extent: float = 4.0,
) -> np.ndarray:
    """
    Python implementation of move's dbbmm2 C kernel.

    Key parity details:
      - integration starts at half the remainder of total duration / time.step
      - segment interpolation uses alpha in time
      - variance equation matches dbbmm2
      - probability is only evaluated within 4 SD
      - each Gaussian density is multiplied by raster cell area
      - final raster is normalized so all cells sum to 1
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    t = np.asarray(
        time_min,
        dtype=float
    )

    means = np.asarray(
        means,
        dtype=float
    )

    interest = np.asarray(
        interest,
        dtype=bool
    )

    location_error = np.asarray(
        location_error,
        dtype=float
    )

    x_grid = grid[
        "x_grid"
    ]

    y_grid = grid[
        "y_grid"
    ]

    nx = len(
        x_grid
    )

    ny = len(
        y_grid
    )

    if (
        nx < 1
        or ny < 1
    ):
        raise ValueError(
            "dBBMM raster contains no cells."
        )

    if time_step_min <= 0:
        raise ValueError(
            "time_step_min must be positive."
        )

    x_res = (
        float(
            x_grid[1]
            - x_grid[0]
        )
        if nx > 1
        else float(
            grid["xmax"]
            - grid["xmin"]
        )
    )

    y_res = (
        float(
            y_grid[1]
            - y_grid[0]
        )
        if ny > 1
        else float(
            grid["ymax"]
            - grid["ymin"]
        )
    )

    cell_area = (
        x_res
        * y_res
    )

    # Work internally with y ascending south -> north.
    ud_asc = np.zeros(
        (
            ny,
            nx
        ),
        dtype=np.float64
    )

    total_duration = (
        t[-1]
        - t[0]
    )

    # C:
    # ti = t0 + fmod((t_last - t0), dT)/2
    remainder = math.fmod(
        total_duration,
        time_step_min
    )

    ti = (
        t[0]
        + remainder / 2.0
    )

    k = 0

    while ti <= t[-1] + 1e-12:

        while (
            k + 1
            < len(t) - 1
            and t[
                k + 1
            ]
            < ti
        ):
            k += 1

        if (
            k >= len(t) - 1
        ):
            break

        if (
            interest[k]
            and np.isfinite(
                means[k]
            )
        ):

            segment_dt = (
                t[k + 1]
                - t[k]
            )

            if segment_dt <= 0:
                raise ValueError(
                    "dBBMM encountered a non-positive time interval."
                )

            alpha = (
                ti
                - t[k]
            ) / segment_dt

            mux = (
                x[k]
                + (
                    x[k + 1]
                    - x[k]
                )
                * alpha
            )

            muy = (
                y[k]
                + (
                    y[k + 1]
                    - y[k]
                )
                * alpha
            )

            sigma = (
                segment_dt
                * alpha
                * (
                    1.0
                    - alpha
                )
                * means[k]
                + (
                    (
                        1.0
                        - alpha
                    )
                    ** 2
                )
                * (
                    location_error[k]
                    ** 2
                )
                + (
                    alpha
                    ** 2
                )
                * (
                    location_error[
                        k + 1
                    ]
                    ** 2
                )
            )

            if (
                np.isfinite(
                    sigma
                )
                and sigma > 0
            ):

                sd = math.sqrt(
                    sigma
                )

                radius = (
                    sd_extent
                    * sd
                )

                # ---------------------------------------------------------
                # Match move behavior: the raster must contain the full
                # requested 4-SD computation neighborhood.
                # ---------------------------------------------------------

                if (
                    mux - radius
                    < x_grid[0]
                ):
                    raise ValueError(
                        "Lower x grid not large enough; increase ext."
                    )

                if (
                    mux + radius
                    > x_grid[-1]
                ):
                    raise ValueError(
                        "Higher x grid not large enough; increase ext."
                    )

                if (
                    muy - radius
                    < y_grid[0]
                ):
                    raise ValueError(
                        "Lower y grid not large enough; increase ext."
                    )

                if (
                    muy + radius
                    > y_grid[-1]
                ):
                    raise ValueError(
                        "Higher y grid not large enough; increase ext."
                    )

                # Candidate cells within +/- 4 SD.
                ix0 = max(
                    0,
                    int(
                        np.searchsorted(
                            x_grid,
                            mux - radius,
                            side="left"
                        )
                    )
                )

                ix1 = min(
                    nx,
                    int(
                        np.searchsorted(
                            x_grid,
                            mux + radius,
                            side="right"
                        )
                    )
                )

                iy0 = max(
                    0,
                    int(
                        np.searchsorted(
                            y_grid,
                            muy - radius,
                            side="left"
                        )
                    )
                )

                iy1 = min(
                    ny,
                    int(
                        np.searchsorted(
                            y_grid,
                            muy + radius,
                            side="right"
                        )
                    )
                )

                if (
                    ix1 > ix0
                    and iy1 > iy0
                ):

                    xx = x_grid[
                        ix0:ix1
                    ]

                    yy = y_grid[
                        iy0:iy1
                    ]

                    XX, YY = np.meshgrid(
                        xx,
                        yy
                    )

                    ztz = (
                        (
                            XX
                            - mux
                        )
                        ** 2
                        + (
                            YY
                            - muy
                        )
                        ** 2
                    )

                    kernel_mass = (
                        (
                            1.0
                            / (
                                2.0
                                * math.pi
                                * sigma
                            )
                        )
                        * np.exp(
                            -ztz
                            / (
                                2.0
                                * sigma
                            )
                        )
                        * cell_area
                    )

                    ud_asc[
                        iy0:iy1,
                        ix0:ix1
                    ] += kernel_mass

        ti += (
            time_step_min
        )

    total = float(
        np.sum(
            ud_asc
        )
    )

    if (
        not np.isfinite(
            total
        )
        or total <= 0
    ):
        raise RuntimeError(
            "dBBMM utilization distribution has zero probability mass."
        )

    # move:
    # ans <- ans / sum(ans)
    ud_asc /= total

    # GeoTIFF rows run north -> south.
    return np.flipud(
        ud_asc
    )


# =============================================================================
# Isopleths
# =============================================================================

def _extract_isopleths(
    ud: np.ndarray,
    transform: Affine,
    cell_area_m2: float,
    levels: Sequence[int],
    reproj_geom,
) -> List[Dict]:
    """
    Extract highest-use UD regions from a raster whose cell values sum to 1.
    """

    flat = np.asarray(
        ud,
        dtype=float
    ).ravel()

    valid = np.isfinite(
        flat
    )

    flat_valid = flat[
        valid
    ]

    if flat_valid.size == 0:
        return []

    order = np.argsort(
        flat_valid
    )[::-1]

    sorted_prob = flat_valid[
        order
    ]

    cumulative = np.cumsum(
        sorted_prob
    )

    results: List[Dict] = []

    for percent in sorted(
        {
            int(v)
            for v
            in levels
            if 1
            <= int(v)
            <= 100
        }
    ):

        target = (
            percent
            / 100.0
        )

        idx = int(
            np.searchsorted(
                cumulative,
                target,
                side="left"
            )
        )

        idx = min(
            idx,
            len(
                sorted_prob
            )
            - 1
        )

        threshold = float(
            sorted_prob[
                idx
            ]
        )

        mask = (
            ud
            >= threshold
        ).astype(
            np.uint8
        )

        polygons: List[
            Polygon
        ] = []

        for geom, val in rio_shapes(
            mask,
            mask=mask.astype(
                bool
            ),
            transform=transform
        ):

            if int(val) != 1:
                continue

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

        merged = unary_union(
            polygons
        )

        if not isinstance(
            merged,
            (
                Polygon,
                MultiPolygon
            )
        ):
            continue

        area_sq_km = float(
            merged.area
            / 1e6
        )

        geom_wgs = reproj_geom(
            merged
        )

        results.append(
            {
                "percent": int(
                    percent
                ),
                "area_sq_km": (
                    area_sq_km
                ),
                "geometry": (
                    shp_mapping(
                        geom_wgs
                    )
                ),
            }
        )

    return results


# =============================================================================
# Public function
# =============================================================================

def compute_dbbmm(
    df: pd.DataFrame,
    id_col: str,
    x_col: str,
    y_col: str,
    ts_col: str,
    params: Optional[
        DBBMMParams
    ] = None,
    outputs_dir: str = "outputs",
) -> Dict[
    str,
    DBBMMResult
]:
    """
    Compute dBBMMs using calculations designed to closely reproduce
    move::brownian.bridge.dyn().

    Spatchat accepts longitude/latitude input and automatically projects
    each animal to a local UTM coordinate system before running the
    move-style calculations.
    """

    if params is None:
        params = DBBMMParams()

    # -------------------------------------------------------------------------
    # Parameter checks
    # -------------------------------------------------------------------------

    window_size = int(
        params.window_size
    )

    margin = int(
        params.margin
    )

    if window_size % 2 != 1:
        raise ValueError(
            "window_size must be odd."
        )

    if margin % 2 != 1:
        raise ValueError(
            "margin must be odd."
        )

    if window_size < 2 * margin:
        raise ValueError(
            "window_size must be at least twice margin."
        )

    raster_res = float(
        params.raster_resolution_m
    )

    if raster_res <= 0:
        raise ValueError(
            "raster_resolution_m must be positive."
        )

    ext = float(
        params.ext
    )

    # -------------------------------------------------------------------------
    # Input preparation
    # -------------------------------------------------------------------------

    if id_col not in df.columns:
        df = df.copy()
        df[
            id_col
        ] = "Animal_1"

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
            f"dBBMM missing required columns: "
            f"{sorted(missing)}"
        )

    df0 = df[
        [
            id_col,
            x_col,
            y_col,
            ts_col,
        ]
    ].copy()

    df0.columns = [
        "animal_id",
        "lon",
        "lat",
        "timestamp",
    ]

    df0[
        "lon"
    ] = pd.to_numeric(
        df0[
            "lon"
        ],
        errors="coerce"
    )

    df0[
        "lat"
    ] = pd.to_numeric(
        df0[
            "lat"
        ],
        errors="coerce"
    )

    df0[
        "timestamp"
    ] = pd.to_datetime(
        df0[
            "timestamp"
        ],
        errors="coerce",
        utc=True,
    )

    df0 = df0.dropna(
        subset=[
            "lon",
            "lat",
            "timestamp",
        ]
    )

    os.makedirs(
        outputs_dir,
        exist_ok=True
    )

    results: Dict[
        str,
        DBBMMResult
    ] = {}

    # -------------------------------------------------------------------------
    # Animal-by-animal analysis
    # -------------------------------------------------------------------------

    for animal, sub in df0.groupby(
        "animal_id",
        sort=False
    ):

        sub = (
            sub
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        n = len(
            sub
        )

        if n < window_size:
            # Same fundamental constraint as move variance calculation.
            continue

        # ---------------------------------------------------------------------
        # Projection
        # ---------------------------------------------------------------------

        projected_crs = (
            _local_utm_crs(
                sub[
                    "lon"
                ].values,
                sub[
                    "lat"
                ].values,
            )
        )

        to_proj = (
            Transformer.from_crs(
                "EPSG:4326",
                projected_crs,
                always_xy=True,
            )
        )

        to_wgs = (
            Transformer.from_crs(
                projected_crs,
                "EPSG:4326",
                always_xy=True,
            )
        )

        def reproj_geom(
            geom
        ):
            return shp_transform(
                lambda x, y, z=None:
                    to_wgs.transform(
                        x,
                        y
                    ),
                geom
            )

        x, y = to_proj.transform(
            sub[
                "lon"
            ].to_numpy(
                dtype=float
            ),
            sub[
                "lat"
            ].to_numpy(
                dtype=float
            ),
        )

        x = np.asarray(
            x,
            dtype=float
        )

        y = np.asarray(
            y,
            dtype=float
        )

        # Relative time in MINUTES, matching move.
        timestamp_ns = (
            sub[
                "timestamp"
            ]
            .astype(
                "int64"
            )
            .to_numpy()
        )

        time_min = (
            timestamp_ns
            - timestamp_ns[0]
        ) / (
            60.0
            * 1e9
        )

        time_lag = np.diff(
            time_min
        )

        if np.any(
            ~np.isfinite(
                time_lag
            )
            | (
                time_lag
                <= 0
            )
        ):
            raise ValueError(
                f"dBBMM for animal {animal} requires "
                "strictly increasing unique timestamps."
            )

        # ---------------------------------------------------------------------
        # Location error
        # ---------------------------------------------------------------------

        location_error = (
            _location_error_vector(
                params.location_error_m,
                n,
            )
        )

        # ---------------------------------------------------------------------
        # Dynamic Brownian variance
        # ---------------------------------------------------------------------

        variance = (
            _dynamic_bm_variance(
                x=x,
                y=y,
                time_min=time_min,
                location_error=location_error,
                window_size=window_size,
                margin=margin,
            )
        )

        means = variance[
            "means"
        ]

        interest = variance[
            "interest"
        ]

        # ---------------------------------------------------------------------
        # move-style integration time step
        # ---------------------------------------------------------------------

        if (
            params.time_step_min
            is None
        ):

            time_step_min = (
                float(
                    np.min(
                        time_lag
                    )
                )
                / 15.0
            )

        else:

            time_step_min = float(
                params.time_step_min
            )

        if (
            not np.isfinite(
                time_step_min
            )
            or time_step_min <= 0
        ):
            raise ValueError(
                "time_step_min must be positive."
            )

        # ---------------------------------------------------------------------
        # move-style raster
        # ---------------------------------------------------------------------

        grid = _move_grid(
            x=x,
            y=y,
            cell_size=raster_res,
            ext=ext,
        )

        transform = (
            Affine.translation(
                grid[
                    "xmin"
                ],
                grid[
                    "ymax"
                ],
            )
            * Affine.scale(
                raster_res,
                -raster_res,
            )
        )

        # ---------------------------------------------------------------------
        # Computational size, analogous to move's message
        # ---------------------------------------------------------------------

        total_interest_time = float(
            np.sum(
                time_lag[
                    interest[
                        :len(
                            time_lag
                        )
                    ]
                ]
            )
        )

        computational_size = (
            grid[
                "nrow"
            ]
            * grid[
                "ncol"
            ]
            * (
                total_interest_time
                / time_step_min
            )
        )

        print(
            f"[dBBMM] animal={animal} "
            f"Computational size: "
            f"{computational_size:.1e}"
        )

        # ---------------------------------------------------------------------
        # Brownian bridge UD
        # ---------------------------------------------------------------------

        ud = _dbbmm_grid(
            x=x,
            y=y,
            time_min=time_min,
            means=means,
            interest=interest,
            location_error=location_error,
            grid=grid,
            time_step_min=time_step_min,
            sd_extent=4.0,
        )

        # ---------------------------------------------------------------------
        # Write projected GeoTIFF
        # ---------------------------------------------------------------------

        safe_animal = (
            str(
                animal
            )
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
            height=ud.shape[0],
            width=ud.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs=projected_crs.to_wkt(),
            transform=transform,
            compress="lzw",
        ) as dst:

            dst.write(
                ud.astype(
                    np.float32
                ),
                1
            )

        # ---------------------------------------------------------------------
        # Isopleths
        # ---------------------------------------------------------------------

        env_list = (
            _extract_isopleths(
                ud=ud,
                transform=transform,
                cell_area_m2=(
                    raster_res
                    ** 2
                ),
                levels=params.isopleths,
                reproj_geom=reproj_geom,
            )
        )

        # Attach useful reproducibility metadata.
        for item in env_list:

            item[
                "analysis_crs"
            ] = (
                projected_crs.to_string()
            )

            item[
                "window_size"
            ] = window_size

            item[
                "margin"
            ] = margin

            item[
                "raster_resolution_m"
            ] = raster_res

            item[
                "ext"
            ] = ext

            item[
                "time_step_min"
            ] = time_step_min

            item[
                "breaks"
            ] = variance[
                "breaks"
            ]

        results[
            str(
                animal
            )
        ] = DBBMMResult(
            geotiff=tif_path,
            isopleths=env_list,
        )

    return results
