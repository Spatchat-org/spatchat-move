# methods/kde.py
import os, json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from pyproj import Transformer
from skimage import measure
from sklearn.neighbors import KernelDensity
import storage  # absolute import
from dataclasses import dataclass
from typing import Optional, Tuple

# ---- New: Parameter object for KDE ----
@dataclass
class KDEParams:
    bandwidth_m: Optional[float] = None       # Explicit KDE bandwidth in meters (UTM)
    kernel: str = "gaussian"                  # 'gaussian','epanechnikov','tophat','exponential','linear','cosine'
    grid_res_m: Optional[float] = None        # Desired grid cell size in meters; if None, uses grid_size
    grid_size: int = 200                      # Fallback grid size if grid_res_m is None
    extent_buffer_mult: float = 3.0           # How many bandwidths to pad the bounding box

# Keep a default, so old callers work
_DEFAULT_PARAMS = KDEParams()


def _chaikin_closed(coords, iterations: int = 2):
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return coords
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    ring = pts[:-1]
    for _ in range(max(int(iterations), 0)):
        if ring.shape[0] < 4:
            break
        q = 0.75 * ring + 0.25 * np.roll(ring, -1, axis=0)
        r = 0.25 * ring + 0.75 * np.roll(ring, -1, axis=0)
        ring = np.empty((q.shape[0] + r.shape[0], 2), dtype=float)
        ring[0::2] = q
        ring[1::2] = r
    return [tuple(p) for p in np.vstack([ring, ring[0]])]


def _smooth_density_contours(Z: np.ndarray, gx: np.ndarray, gy: np.ndarray, threshold: float):
    z = np.asarray(Z, dtype=float)
    if z.ndim != 2 or z.shape != (len(gy), len(gx)):
        return []
    finite = np.isfinite(z)
    if not np.any(finite):
        return []
    level = float(threshold)
    zmin = float(np.nanmin(z[finite]))
    zmax = float(np.nanmax(z[finite]))
    if not (zmin < level < zmax):
        return []

    contours = measure.find_contours(np.where(finite, z, zmin), level=level)
    polys = []
    x_idx = np.arange(len(gx), dtype=float)
    y_idx = np.arange(len(gy), dtype=float)
    dx = float(np.nanmedian(np.diff(gx))) if len(gx) > 1 else 0.0
    dy = float(np.nanmedian(np.diff(gy))) if len(gy) > 1 else 0.0
    max_gap = 4.0 * max(abs(dx), abs(dy), 1.0)

    for c in contours:
        if c.shape[0] < 4:
            continue
        xs = np.interp(c[:, 1], x_idx, gx)
        ys = np.interp(c[:, 0], y_idx, gy)
        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
            continue
        if np.hypot(xs[0] - xs[-1], ys[0] - ys[-1]) > max_gap:
            continue
        p = Polygon(_chaikin_closed(np.column_stack([xs, ys]), iterations=2)).buffer(0)
        if p.is_valid and p.area > 0:
            polys.append(p)
    return polys


def _kde_core(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    percent: int = 95,
    params: KDEParams = _DEFAULT_PARAMS,
) -> Tuple[Optional[MultiPolygon], Optional[float], Optional[np.ndarray], Optional[tuple], Optional[Transformer]]:
    """
    Core KDE in UTM meters. Returns (polygon_ll, area_km2, Z_masked, (xmin,ymin,xmax,ymax in UTM), utm->ll transformer)
    """
    # 1) Project to a suitable UTM zone
    lon0, lat0 = float(np.mean(longitudes)), float(np.mean(latitudes))
    zone = int((lon0 + 180) // 6) + 1
    epsg_utm = 32600 + zone if lat0 >= 0 else 32700 + zone
    to_utm = Transformer.from_crs("epsg:4326", f"epsg:{epsg_utm}", always_xy=True)
    to_ll  = Transformer.from_crs(f"epsg:{epsg_utm}", "epsg:4326", always_xy=True)

    x, y = to_utm.transform(longitudes, latitudes)  # meters
    XY = np.vstack([x, y]).T
    n = len(XY)

    # 2) Bandwidth in meters
    if params and params.bandwidth_m and params.bandwidth_m > 0:
        h = float(params.bandwidth_m)
    else:
        # Silverman's rule-of-thumb (in meters, since we're in UTM)
        if n > 1:
            stds = np.std(XY, axis=0, ddof=1)
            h = (4/(3*n))**(1/5) * float(np.mean(stds))
            if h < 1:
                h = 30.0  # fallback minimum
        else:
            h = 30.0

    # 3) Grid definition
    m = params.extent_buffer_mult * h if params else 3.0 * h
    xmin, xmax = x.min()-m, x.max()+m
    ymin, ymax = y.min()-m, y.max()+m

    if params and params.grid_res_m and params.grid_res_m > 0:
        # Use a fixed cell size in meters
        step = float(params.grid_res_m)
        gx = np.arange(xmin, xmax + step, step)
        gy = np.arange(ymin, ymax + step, step)
    else:
        # Fall back to fixed grid size
        gsz = params.grid_size if params else 200
        gx = np.linspace(xmin, xmax, gsz)
        gy = np.linspace(ymin, ymax, gsz)

    Xg, Yg = np.meshgrid(gx, gy)
    grid = np.vstack([Xg.ravel(), Yg.ravel()]).T

    # 4) KDE evaluation
    kernel = (params.kernel if params else "gaussian").lower()
    kde = KernelDensity(bandwidth=h, kernel=kernel).fit(XY)
    Z = np.exp(kde.score_samples(grid)).reshape(Xg.shape)

    # 5) Normalize to integrate to 1 over the raster (units are 1/m^2)
    dx = gx[1]-gx[0] if len(gx) > 1 else h
    dy = gy[1]-gy[0] if len(gy) > 1 else h
    cell_area = dx*dy
    Z /= (Z.sum() * cell_area)

    # 6) Find threshold for N% volume contour
    Zf = Z.ravel()
    idx = np.argsort(Zf)[::-1]
    csum = np.cumsum(Zf[idx]*cell_area)
    k = min(np.searchsorted(csum, percent/100.0), len(idx)-1)
    thr = Zf[idx][k]
    mask = Z >= thr

    Zm = np.where(mask, Z, 0)
    tot = Zm.sum()*cell_area
    if tot > 0:
        Zm /= tot

    # 7) Contours → polygons (in UTM)
    polys = _smooth_density_contours(Z, gx, gy, thr)
    if not polys:
        contours = measure.find_contours(mask.astype(float), 0.5)
        polys = []
        for c in contours:
            px, py = c[:, 1], c[:, 0]
            xs = np.interp(px, np.arange(Xg.shape[1]), gx)
            ys = np.interp(py, np.arange(Xg.shape[0]), gy)
            p = Polygon(_chaikin_closed(np.column_stack([xs, ys]), iterations=1)).buffer(0)
            if p.is_valid and p.area > 0:
                polys.append(p)
    if not polys:
        return None, None, None, None, None

    mp_utm = unary_union(polys)

    # 8) Back to lat/lon for outputs & area in km^2
    def utm_to_ll(poly):
        if poly.is_empty:
            return None
        if isinstance(poly, Polygon):
            elon, elat = to_ll.transform(*poly.exterior.xy)
            holes = [to_ll.transform(*ring.xy) for ring in poly.interiors]
            return Polygon(list(zip(elon, elat)),
                           [list(zip(hlon, hlat)) for hlon, hlat in holes])
        if isinstance(poly, MultiPolygon):
            parts = [utm_to_ll(p) for p in poly.geoms if not p.is_empty]
            return MultiPolygon([p for p in parts if p is not None])
        return None

    mp_ll = utm_to_ll(mp_utm)
    area_km2 = mp_utm.area / 1e6

    return mp_ll, area_km2, Zm, (xmin, ymin, xmax, ymax), to_ll

def add_kdes(df, percent_list, params: KDEParams = _DEFAULT_PARAMS):
    """
    Compute KDE UDs for each requested percent and animal. Writes GeoTIFF + GeoJSON.
    `params` controls bandwidth (m), kernel, and grid resolution (m).
    """
    outputs_dir = storage.get_output_dir()
    os.makedirs(outputs_dir, exist_ok=True)
    if "animal_id" not in df.columns:
        df = df.copy()
        df["animal_id"] = "Animal_1"
    for percent in percent_list:
        for animal in df["animal_id"].unique():
            storage.kde_results.setdefault(animal, {})
            if percent in storage.kde_results[animal]:
                continue

            trk = df[df["animal_id"]==animal]
            poly_ll, area_km2, Zm, bbox, to_ll = _kde_core(
                trk['latitude'].values,
                trk['longitude'].values,
                percent=percent,
                params=params or _DEFAULT_PARAMS
            )

            if poly_ll is None:
                continue

            xmin, ymin, xmax, ymax = bbox
            lon_sw, lat_sw = to_ll.transform(xmin, ymin)
            lon_ne, lat_ne = to_ll.transform(xmax, ymax)

            safe = str(animal).replace(" ","_").replace("/","_")
            tif = os.path.join(outputs_dir, f"kde_{safe}_{percent}.tif")
            with rasterio.open(
                tif, "w", driver="GTiff",
                height=Zm.shape[0], width=Zm.shape[1], count=1, dtype=Zm.dtype,
                crs="EPSG:4326",
                transform=from_origin(lon_sw, lat_ne,
                                      (lon_ne-lon_sw)/Zm.shape[1],
                                      (lat_ne-lat_sw)/Zm.shape[0])
            ) as dst:
                # flipud because array row 0 is north in our mesh, but GeoTIFF row 0 is top
                dst.write(np.flipud(Zm), 1)

            gj = os.path.join(outputs_dir, f"kde_{safe}_{percent}.geojson")
            with open(gj, "w") as f:
                json.dump(mapping(poly_ll), f)

            storage.kde_results[animal][percent] = {
                "contour": poly_ll, "area": area_km2,
                "geotiff": tif, "geojson": gj
            }
