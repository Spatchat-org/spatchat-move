"""Partial parity translation of ctmm 1.3.0 ``R/kde.R``."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm
from pyproj import CRS, Transformer

from .types import CTMMModel, Telemetry
from .generic_utils import epoch_seconds
from .summary_ctmm import DOF_area
from .stats import chisq_ci
from .units import unit
from .plot_variogram import svf_func


def akde_bias(CTMM: CTMMModel, H, lag, DOF=None, weights=None):
    sigma = np.asarray(CTMM.params.get("sigma_matrix", np.eye(2)), dtype=float)
    h = np.asarray(H, dtype=float)
    lag = np.asarray(lag, dtype=float)
    if weights is None:
        weights = np.ones(lag.shape[0], dtype=float) / max(lag.shape[0], 1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if DOF is None:
        if lag.ndim == 1:
            lag = np.abs(lag[:, None] - lag[None, :])
        acf = np.asarray(svf_func(CTMM, moment=False)["ACF"](lag), dtype=float)
        mm = float(w @ acf @ w)
    else:
        acf = np.asarray(svf_func(CTMM, moment=False)["ACF"](lag), dtype=float)
        d = np.asarray(DOF, dtype=float)
        mm = float(np.sum(d * acf))
    var_rel = float(max(1.0 - mm, 0.0))
    cov = var_rel * sigma
    bias = float((np.linalg.det(cov + h) / max(np.linalg.det(sigma), 1e-18)) ** (1.0 / sigma.shape[0]))
    return {"bias": bias, "COV": cov}


def pkde(data, UD, kernel: str = "individual", weights=False, ref: str = "Gaussian", **kwargs):
    del kernel, ref
    return akde(data=data, CTMM=UD, weights=weights, **kwargs)


def akde(
    data,
    CTMM,
    VMM=None,
    R=None,
    SP=None,
    SP_in: bool = True,
    variable: str = "utilization",
    debias: bool = True,
    weights=False,
    smooth: bool = True,
    error: float = 0.001,
    res: int = 10,
    grid=None,
    **kwargs,
):
    del VMM, R, SP, SP_in, smooth, grid, kwargs
    if variable != "utilization":
        raise ValueError(f"variable={variable} not yet supported by akde().")

    if isinstance(data, Telemetry):
        data_list = [data]
    else:
        data_list = list(data)
    if isinstance(CTMM, CTMMModel):
        model_list = [CTMM]
    else:
        model_list = list(CTMM)
    if len(data_list) != len(model_list):
        raise ValueError(f"length(data)=={len(data_list)}, but length(CTMM)=={len(model_list)}")

    out = []
    for d, m in zip(data_list, model_list):
        dof = DOF_area(m)
        ud = _akde_single(d, m, level=0.95, error=error, debias=debias, res=res)
        if dof < error:
            ud["warning"] = f"DOF[area]={dof} below error threshold {error}"
        out.append(ud)
    return out[0] if isinstance(data, Telemetry) else out


def _akde_single(telem: Telemetry, model: CTMMModel, level: float = 0.95, error: float = 0.001, debias: bool = True, res: int = 10):
    df = telem.data
    if telem.x_col in df.columns and telem.y_col in df.columns:
        x = df[telem.x_col].to_numpy(dtype=float)
        y = df[telem.y_col].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
    elif {"longitude", "latitude"}.issubset(df.columns):
        lon = df["longitude"].to_numpy(dtype=float)
        lat = df["latitude"].to_numpy(dtype=float)
        ok = np.isfinite(lon) & np.isfinite(lat)
        lon = lon[ok]
        lat = lat[ok]
        epsg = 32600 + int(np.floor((float(np.nanmedian(lon)) + 180.0) / 6.0) + 1)
        tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
        x, y = tr.transform(lon, lat)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    else:
        raise ValueError("Telemetry requires projected x/y or longitude/latitude.")
    if len(x) < 5:
        raise ValueError("insufficient points for AKDE")

    from .bandwidth import bandwidth

    bw = bandwidth(telem, model, weights=False)
    H = np.asarray(bw["H"], dtype=float)
    W = np.asarray(bw["weights"], dtype=float)
    sigma = _sigma_matrix(model)
    desired_dr = np.sqrt(np.maximum(np.minimum(np.diag(H), np.diag(sigma)), 0.0)) / max(int(res), 1)
    ext_min = _ctmm_extent_min(model, level=1.0 - error, level_UD=0.95)
    dof_area = float(DOF_area(model))
    if not np.isfinite(dof_area) or dof_area <= 1.0:
        t = epoch_seconds(df[telem.time_col])
        t = t[np.isfinite(t)]
        tau = model.params.get("tau", {})
        tau_pos = float(tau.get("position", 86400.0)) if isinstance(tau, dict) else 86400.0
        if t.size > 1:
            dof_area = max(float((t[-1] - t[0]) / max(tau_pos, 1.0)), 1.0)
        else:
            dof_area = 1.0
    dof_h = float(bw.get("DOF.H", np.nan))
    grid_obj = kde_grid(telem, H=H, axes=(telem.x_col, telem.y_col), alpha=error, res=res, dr=desired_dr, EXT_min=ext_min)
    kd = kde(telem, H=H, axes=(telem.x_col, telem.y_col), W=W, alpha=error, grid=grid_obj, bias=bw.get("bias", 1.0), debias=debias)
    Z = kd["PDF"]
    gx = np.asarray(grid_obj["r"]["x"], dtype=float)
    gy = np.asarray(grid_obj["r"]["y"], dtype=float)
    dx = float(grid_obj["dr"][0])
    dy = float(grid_obj["dr"][1])

    cdf = kd["CDF"]
    ci_m2 = CI_UD({"CDF": cdf, "dr": {"x": dx, "y": dy}, "DOF.area": np.array([dof_area], dtype=float)}, level_UD=level, level=level)
    ci = np.asarray(ci_m2, dtype=float) / 1e6
    area_km2 = float(ci[1])
    return {
        "type": "range",
        "variable": "utilization",
        "CDF": cdf,
        "PDF": Z,
        "r": {"x": gx, "y": gy},
        "dr": {"x": dx, "y": dy},
        "DOF.area": np.array([dof_area], dtype=float),
        "DOF.H": float(dof_h),
        "CTMM": model,
        "CI.area": ci,
        "area_km2": area_km2,
    }


def _sigma_matrix(model: CTMMModel) -> np.ndarray:
    sigma = model.params.get("sigma_matrix")
    if sigma is not None:
        return np.asarray(sigma, dtype=float)
    sigma = model.params.get("sigma")
    if hasattr(sigma, "matrix"):
        return np.asarray(sigma.matrix, dtype=float)
    if sigma is not None:
        return np.asarray(sigma, dtype=float)
    return np.eye(2, dtype=float)


def _qmvnorm(p: float, dim: int) -> float:
    alpha = 1.0 - float(p)
    if dim == 1:
        return float(norm.ppf(1.0 - alpha / 2.0))
    if dim == 2:
        return float(np.sqrt(-2.0 * np.log(alpha)))
    return float(np.sqrt(chi2.ppf(float(p), df=dim)))


def _ctmm_extent_min(model: CTMMModel, level: float = 0.999, level_UD: float = 0.95):
    mu = model.params.get("mu")
    if mu is None:
        return None
    mu = np.asarray(mu, dtype=float).reshape(-1)
    sigma = _sigma_matrix(model)
    dim = min(mu.size, sigma.shape[0], 2)
    if dim == 0:
        return None
    alpha_ud = 1.0 - float(level_UD)
    if dim == 2:
        z = np.sqrt(-2.0 * np.log(alpha_ud))
    else:
        z = norm.ppf(1.0 - alpha_ud / 2.0)
    const = 1.0
    dof_area = DOF_area(model)
    if np.isfinite(dof_area) and dof_area > 0:
        alpha = 1.0 - float(level)
        df = 2.0 * float(dof_area)
        upper = chi2.ppf(1.0 - alpha / 2.0, df) / df
        if np.isfinite(upper) and upper > 0:
            const = float(upper)
    buff = float(z) * np.sqrt(np.maximum(const * np.diag(sigma)[:dim], 0.0))
    return np.vstack([mu[:dim] - buff, mu[:dim] + buff])


def _normalize_grid(grid, axes):
    if grid is None:
        return {}
    if isinstance(grid, dict):
        g = dict(grid)
        if "dr" in g:
            dr = g["dr"]
            if isinstance(dr, dict):
                g["dr"] = np.array([float(dr[a]) for a in axes], dtype=float)
            else:
                drr = np.asarray(dr, dtype=float).reshape(-1)
                if drr.size == 1:
                    drr = np.repeat(drr[0], len(axes))
                g["dr"] = drr.astype(float)
        return g
    return {}


def _extent_union(ext, ext_min):
    if ext_min is None:
        return ext
    e1 = np.asarray(ext, dtype=float)
    e2 = np.asarray(ext_min, dtype=float)
    if e2.shape != e1.shape:
        e2 = e2[:, : e1.shape[1]]
    return np.vstack([np.minimum(e1[0], e2[0]), np.maximum(e1[1], e2[1])])


def _seq_len(start: float, stop: float, length: int) -> np.ndarray:
    return np.linspace(float(start), float(stop), int(length), dtype=float)


def kde_grid(data, H, axes=("x", "y"), alpha: float = 0.001, res: int | None = None, dr=None, grid=None, EXT_min=None):
    grid = _normalize_grid(grid, axes)
    if isinstance(data, Telemetry):
        arr = data.data[list(axes)].to_numpy(dtype=float)
    else:
        arr = np.asarray(data, dtype=float)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.shape[0] == 0:
        raise ValueError("kde_grid requires finite coordinates")
    dim = len(axes)
    h = prepare_H(H, n=arr.shape[0], axes=axes)
    z = _qmvnorm(1.0 - alpha, dim)
    dH = z * np.sqrt(np.maximum(np.diagonal(h, axis1=1, axis2=2), 0.0))

    if "r" in grid:
        r = grid["r"]
        gx = np.asarray(r[axes[0]], dtype=float)
        gy = np.asarray(r[axes[1]], dtype=float)
        if "dr" in grid:
            drr = np.asarray(grid["dr"], dtype=float).reshape(-1)
        else:
            drr = np.array([float(np.mean(np.diff(gx))), float(np.mean(np.diff(gy)))], dtype=float)
        return {"r": {axes[0]: gx, axes[1]: gy}, "dr": drr, "dH": dH}

    if "extent" in grid and "dr" in grid:
        drr = np.asarray(grid["dr"], dtype=float).reshape(-1)
        if drr.size == 1:
            drr = np.repeat(drr[0], dim)
        ext = np.asarray(grid["extent"], dtype=float)
        if bool(grid.get("align.to.origin", False)):
            ext[0] = np.floor(ext[0] / drr) * drr
            ext[1] = np.ceil(ext[1] / drr) * drr
        resv = np.round((ext[1] - ext[0]) / drr).astype(int)
        gx = _seq_len(ext[0, 0], ext[1, 0], 1 + resv[0])
        gy = _seq_len(ext[0, 1], ext[1, 1], 1 + resv[1])
        return {"r": {axes[0]: gx, axes[1]: gy}, "dr": drr, "dH": dH}

    if "extent" in grid:
        ext = np.asarray(grid["extent"], dtype=float)
        d_ext = ext[1] - ext[0]
        if dr is None:
            if res is None:
                res = 10
            drr = d_ext / max(int(res), 1)
        else:
            drr = np.asarray(dr, dtype=float).reshape(-1)
            if drr.size == 1:
                drr = np.repeat(drr[0], dim)
        resv = np.maximum(np.ceil(d_ext / drr).astype(int), 1)
        drr = d_ext / resv
        gx = _seq_len(ext[0, 0], ext[1, 0], 1 + resv[0])
        gy = _seq_len(ext[0, 1], ext[1, 1], 1 + resv[1])
        return {"r": {axes[0]: gx, axes[1]: gy}, "dr": drr, "dH": dH}

    ext = np.vstack([np.min(arr - dH, axis=0), np.max(arr + dH, axis=0)])
    ext = _extent_union(ext, EXT_min)
    d_ext = ext[1] - ext[0]
    if dr is None and "dr" in grid:
        dr = grid["dr"]
    if dr is None:
        if res is None:
            res = 10
        drr = d_ext / max(int(res), 1)
        resv = np.maximum(np.ceil(d_ext / drr).astype(int), 1)
        drr = d_ext / resv
    else:
        drr = np.asarray(dr, dtype=float).reshape(-1)
        if drr.size == 1:
            drr = np.repeat(drr[0], dim)
        resv = np.maximum(np.ceil(d_ext / drr).astype(int), 1)
        if "dr" in grid:
            d_ext = resv * drr
            mu = np.mean(ext, axis=0)
            ext = np.vstack([mu - d_ext / 2.0, mu + d_ext / 2.0])
            if bool(grid.get("align.to.origin", False)):
                ext[0] = np.floor(ext[0] / drr) * drr
                ext[1] = np.ceil(ext[1] / drr) * drr
                d_ext = ext[1] - ext[0]
                resv = np.round(d_ext / drr).astype(int)
        else:
            drr = d_ext / resv
    gx = _seq_len(ext[0, 0] - drr[0], ext[1, 0] + drr[0], 1 + resv[0] + 2)
    gy = _seq_len(ext[0, 1] - drr[1], ext[1, 1] + drr[1], 1 + resv[1] + 2)
    return {"r": {axes[0]: gx, axes[1]: gy}, "dr": np.asarray(drr, dtype=float), "dH": dH}


def kde(data, H, axes=("x", "y"), CTMM=None, SP=None, SP_in=True, RASTER=None, bias=False, W=None, alpha: float = 0.001, res=None, dr=None, grid=None, variable=np.nan, normalize: bool = True, trace: bool = False, grad: bool = False, truncate: bool = True, **kwargs):
    del CTMM, SP, SP_in, RASTER, res, dr, variable, trace, grad, truncate, kwargs
    if isinstance(data, Telemetry):
        arr = data.data[list(axes)].to_numpy(dtype=float)
    else:
        arr = np.asarray(data, dtype=float)
    keep = np.all(np.isfinite(arr), axis=1)
    arr = arr[keep]
    n = arr.shape[0]
    if n == 0:
        raise ValueError("kde requires finite coordinates")
    if grid is None:
        grid = kde_grid(data, H=H, axes=axes)
    gx = np.asarray(grid["r"]["x"], dtype=float)
    gy = np.asarray(grid["r"]["y"], dtype=float)
    dx, dy = float(grid["dr"][0]), float(grid["dr"][1])
    if W is None:
        W = np.ones(n, dtype=float)
    else:
        W = np.asarray(W, dtype=float).reshape(-1)
        if W.size != keep.size:
            raise ValueError("W length mismatch")
        W = W[keep]
    if normalize:
        W = W / max(np.sum(W), 1e-18)

    Hn = prepare_H(H, n=n, axes=axes)
    dH = np.asarray(grid.get("dH", np.nan), dtype=float)
    if dH.ndim != 2 or dH.shape[1] != len(axes):
        z = _qmvnorm(1.0 - alpha, len(axes))
        dH = z * np.sqrt(np.maximum(np.diagonal(Hn, axis1=1, axis2=2), 0.0))
    elif dH.shape[0] == keep.size:
        dH = dH[keep]
    elif dH.shape[0] != n:
        z = _qmvnorm(1.0 - alpha, len(axes))
        dH = z * np.sqrt(np.maximum(np.diagonal(Hn, axis1=1, axis2=2), 0.0))

    mass = np.zeros((gx.size, gy.size), dtype=float)
    r0 = np.array([gx[0], gy[0]], dtype=float)
    drv = np.array([dx, dy], dtype=float)
    dims = np.array([gx.size, gy.size], dtype=int)
    for i in range(n):
        lo = np.floor((arr[i] - dH[i] - r0) / drv).astype(int)
        hi = np.ceil((arr[i] + dH[i] - r0) / drv).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(np.maximum(hi, 0), dims - 1)
        if np.any(hi < lo):
            continue
        sx = slice(int(lo[0]), int(hi[0]) + 1)
        sy = slice(int(lo[1]), int(hi[1]) + 1)
        dpmf = pnorm2(gx[sx] - arr[i, 0], gy[sy] - arr[i, 1], Hn[i], np.array([dx, dy], dtype=float), alpha=alpha)
        mass[sx, sy] += W[i] * dpmf
    mass = np.clip(mass, 0.0, np.inf)
    if bias is not False and np.sum(np.asarray(bias, dtype=float)) != 0:
        deb = debias_volume(mass, bias=float(np.min(np.asarray(bias, dtype=float))))
        mass = deb["PMF"]
        CDF = deb["CDF"]
    else:
        CDF = pmf2cdf(mass, finish=True)
    PDF = mass / max(dx * dy, 1e-300)
    return {"PDF": PDF, "CDF": CDF, "r": {"x": gx, "y": gy}, "dr": {"x": dx, "y": dy}}


def prepare_H(H, n: int, axes=("x", "y")):
    d = len(axes)
    h = np.asarray(H, dtype=float)
    if h.size == 1:
        h = np.eye(d, dtype=float) * float(h.reshape(-1)[0])
    elif h.ndim == 1:
        mats = [np.eye(d, dtype=float) * float(v) for v in h]
        h = np.stack(mats, axis=0)
    if h.ndim == 2 and h.shape == (d, d):
        h = np.broadcast_to(h[None, :, :], (int(n), d, d)).copy()
    if h.ndim != 3 or h.shape[1:] != (d, d):
        raise ValueError("prepare_H expects scalar, [n] vector, or [d,d]/[n,d,d] matrix input")
    return h


def debias_volume(PMF, bias=1.0):
    z = np.asarray(PMF, dtype=float)
    info = pmf2cdf(z, finish=False)
    cdf = np.asarray(info["CDF"], dtype=float)
    ind = np.asarray(info["IND"], dtype=int)
    dim = info["DIM"]
    b = float(np.sqrt(float(bias)) ** len(dim))
    vol = np.arange(1, cdf.size + 1, dtype=float)
    vol0 = np.r_[0.0, vol]
    cdf0 = np.r_[0.0, cdf]
    if not np.isfinite(b) or b <= 0:
        b = 1.0
    cdf_new = np.interp(vol0, vol0 / b, cdf0, left=0.0, right=1.0)[1:]
    cdf_new = np.sort(cdf_new)
    pmf_sorted = cdf2pmf(cdf_new)
    cdf_sorted = np.cumsum(pmf_sorted)
    pmf_flat = np.empty_like(pmf_sorted)
    cdf_flat = np.empty_like(cdf_sorted)
    pmf_flat[ind] = pmf_sorted
    cdf_flat[ind] = cdf_sorted
    return {"PMF": pmf_flat.reshape(dim), "CDF": cdf_flat.reshape(dim)}


def debias_area(PMF, bias=1.0):
    z = np.asarray(PMF, dtype=float)
    b = float(np.mean(np.asarray(bias, dtype=float)))
    if not np.isfinite(b) or b <= 0:
        return z
    if abs(b - 1.0) <= np.finfo(float).eps:
        return z
    # Area-preserving-ish transformation from ctmm intent.
    out = np.power(np.clip(z, 0.0, np.inf), 1.0 / b)
    s0 = float(np.sum(z))
    s1 = float(np.sum(out))
    if s1 > 0 and s0 > 0:
        out *= s0 / s1
    return out


def pmf2cdf(PMF, finish: bool = True):
    z = np.asarray(PMF, dtype=float)
    dim = z.shape
    flat = z.ravel()
    order = np.argsort(flat)[::-1]
    c = np.cumsum(flat[order])
    if not finish:
        return {"CDF": c, "IND": order, "DIM": dim}
    out = np.empty_like(flat)
    out[order] = c
    return out.reshape(z.shape)


def cdf2pmf(CDF):
    c = np.asarray(CDF, dtype=float)
    pmf = np.diff(np.r_[0.0, c.ravel()])
    if pmf.size > 1:
        for i in range(1, pmf.size):
            pmf[i] = min(pmf[i - 1], pmf[i])
    return np.maximum(pmf, 0.0)


def Gauss(X, Y, sigma=None, sigma_inv=None, sigma_GM=None):
    x = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float)
    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y, indexing="ij")
    if sigma is None:
        sigma = np.eye(2, dtype=float)
    s = np.asarray(sigma, dtype=float)
    if sigma_inv is None:
        sigma_inv = np.linalg.inv(s)
    if sigma_GM is None:
        sigma_GM = np.sqrt(max(np.linalg.det(s), 1e-18))
    q = sigma_inv[0, 0] * x * x + 2 * sigma_inv[0, 1] * x * y + sigma_inv[1, 1] * y * y
    return np.exp(-0.5 * q) / (2 * np.pi * sigma_GM)


def Gauss1(X, sigma=None):
    x = np.asarray(X, dtype=float)
    v = float(np.asarray(sigma if sigma is not None else 1.0, dtype=float).reshape(-1)[0])
    v = max(v, 1e-18)
    return np.exp(-0.5 * x * x / v) / np.sqrt(2 * np.pi * v)


def Gauss3(X, Y, Z, sigma=None, sigma_inv=None, sigma_GM=None):
    # Horizontal kernel in ctmm's pnorm3 pathway (XY marginalized/conditioned; Z handled separately)
    return Gauss(X, Y, sigma=sigma, sigma_inv=sigma_inv, sigma_GM=sigma_GM) * Gauss1(Z, sigma=1.0)


def pnorm1(X, sigma, dr, alpha: float = 0.001):
    x = np.asarray(X, dtype=float)
    v = float(np.asarray(sigma, dtype=float).reshape(-1)[0])
    v = max(v, 1e-18)
    sd = np.sqrt(v)
    lo = norm.ppf(alpha / 2.0) * sd
    hi = norm.ppf(1 - alpha / 2.0) * sd
    p = norm.cdf((x + dr / 2.0) / sd) - norm.cdf((x - dr / 2.0) / sd)
    p[(x < lo) | (x > hi)] = 0.0
    return p


def pnorm2(X, Y, sigma, dr, alpha: float = 0.001):
    x = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float)
    s = np.asarray(sigma, dtype=float)
    dx, dy = float(dr[0]), float(dr[1])
    cdf = np.zeros((x.size, y.size), dtype=float)
    vals, vecs = np.linalg.eigh(s)
    vals = np.sort(vals)[::-1]
    finite = vals > 0
    if not np.any(finite):
        r = np.abs(x) == np.min(np.abs(x))
        c = np.abs(y) == np.min(np.abs(y))
        cdf[np.ix_(r, c)] = 1.0 / max(float(np.sum(r) * np.sum(c)), 1.0)
        return cdf
    zero = int(np.sum((vals <= 0) | (((min(dx, dy) / 2.0) ** 2 / np.maximum(vals, np.finfo(float).tiny)) > -2.0 * np.log(alpha))))
    S = float(np.sqrt(max(s[0, 0] * s[1, 1], 0.0)))
    rho = float(np.clip(s[0, 1] / S, -1.0, 1.0)) if S > 0 else 0.0
    if zero == 0 and abs(rho) < 1.0:
        z = float(np.sqrt((dx * dx + dy * dy) / max(vals[-1], np.finfo(float).tiny)))
        if z**3 / 12.0 <= alpha:
            gx, gy = np.meshgrid(x, y, indexing="ij")
            return np.maximum((dx * dy) * Gauss(gx, gy, sigma=s), 0.0)
        if z**5 / 2880.0 <= alpha:
            return np.maximum(NewtonCotes(x, y, s, np.array([1.0, 4.0, 1.0]), dx=dx, dy=dy), 0.0)
        if z**7 / 1935360.0 <= alpha:
            return np.maximum(NewtonCotes(x, y, s, np.array([7.0, 32.0, 12.0, 32.0, 7.0]), dx=dx, dy=dy), 0.0)
        from scipy.stats import multivariate_normal

        xc = np.r_[x - dx / 2.0, x[-1] + dx / 2.0]
        yc = np.r_[y - dy / 2.0, y[-1] + dy / 2.0]
        Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
        pts = np.column_stack([Xc.ravel(), Yc.ravel()])
        C = multivariate_normal(mean=np.zeros(2), cov=s, allow_singular=False).cdf(pts).reshape(Xc.shape)
        return np.maximum(C[1:, 1:] - C[:-1, 1:] - C[1:, :-1] + C[:-1, :-1], 0.0)
    if zero == 1 or abs(rho) == 1.0:
        imax = int(np.argmax(vals))
        v = vecs[:, np.argsort(np.linalg.eigvalsh(s))[::-1][imax]]
        svar = max(float(vals[imax]), np.finfo(float).tiny)
        x_cross = []
        y_cross = []
        if abs(v[0]) > 0:
            xc = np.r_[x - dx / 2.0, x[-1] + dx / 2.0]
            x_cross.extend(xc.tolist())
            y_cross.extend((xc * v[1] / v[0]).tolist())
        if abs(v[1]) > 0:
            yc = np.r_[y - dy / 2.0, y[-1] + dy / 2.0]
            y_cross.extend(yc.tolist())
            x_cross.extend((yc * v[0] / v[1]).tolist())
        zc = (np.asarray(x_cross) * v[0] + np.asarray(y_cross) * v[1]) / np.sqrt(svar)
        zc = np.unique(np.sort(zc[np.isfinite(zc)]))
        for i in range(max(zc.size - 1, 0)):
            zmid = float((zc[i] + zc[i + 1]) / 2.0)
            xm = np.sqrt(svar) * v[0] * zmid
            ym = np.sqrt(svar) * v[1] * zmid
            r = np.abs(xm - x) == np.min(np.abs(xm - x))
            c = np.abs(ym - y) == np.min(np.abs(ym - y))
            cdf[np.ix_(r, c)] += (norm.cdf(zc[i + 1]) - norm.cdf(zc[i])) / max(float(np.sum(r) * np.sum(c)), 1.0)
        return cdf
    if zero == 2:
        r = np.abs(x) == np.min(np.abs(x))
        c = np.abs(y) == np.min(np.abs(y))
        cdf[np.ix_(r, c)] += 1.0 / max(float(np.sum(r) * np.sum(c)), 1.0)
        return cdf
    raise ValueError(f"something is wrong in matrix: sigma == {sigma}")


def pnorm3(X, Y, Z, sigma, dr, alpha: float = 0.001):
    del alpha
    s = np.asarray(sigma, dtype=float)
    return np.prod(np.asarray(dr, dtype=float)) * Gauss3(X, Y, Z, sigma=s)


def NewtonCotes(X, Y, sigma, W, dx=None, dy=None):
    x = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float)
    w = np.asarray(W, dtype=float)
    w = w / max(np.sum(w), 1e-18)
    if dx is None:
        dx = float(np.mean(np.diff(x)))
    if dy is None:
        dy = float(np.mean(np.diff(y)))
    n = len(w)
    if n <= 1:
        gx, gy = np.meshgrid(x, y, indexing="ij")
        return dx * dy * Gauss(gx, gy, sigma=np.asarray(sigma, dtype=float))
    offsets_x = np.linspace(-dx / 2.0, dx / 2.0, n, dtype=float)
    offsets_y = np.linspace(-dy / 2.0, dy / 2.0, n, dtype=float)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    out = np.zeros((x.size, y.size), dtype=float)
    for i, ox in enumerate(offsets_x):
        for j, oy in enumerate(offsets_y):
            out += w[i] * w[j] * Gauss(gx + ox, gy + oy, sigma=np.asarray(sigma, dtype=float))
    return dx * dy * out


def CI_UD(object, level_UD: float = 0.95, level: float = 0.95, P: bool = False, convex: bool = False):
    del convex
    dof_area = object.get("DOF.area", None)
    if dof_area is None and P:
        return np.array([level_UD], dtype=float)
    cdf = np.asarray(object.get("CDF"), dtype=float)
    dr = object.get("dr", {"x": 1.0, "y": 1.0})
    dV = float(dr["x"]) * float(dr["y"])
    sortv = np.sort(cdf, axis=None)

    def interpolate(y, val):
        idx = np.where(y < val)[0]
        if idx.size == 0:
            return 0.0
        x0 = int(idx[-1])
        if x0 >= len(y) - 1:
            return float(len(y))
        y0 = y[x0] - val
        y1 = y[x0 + 1] - val
        beta = y1 - y0
        if abs(beta) < 1e-18:
            return float(x0 + 1)
        dx = -y0 / beta
        return float(x0 + 1 + dx)

    area = interpolate(sortv, level_UD) * dV
    if dof_area is not None:
        da = float(np.asarray(dof_area).reshape(-1)[0])
        area = chisq_ci(area, dof=2.0 * da, level=level)
    else:
        area = np.array([area, area, area], dtype=float)
    if not P:
        return area
    ind = area / dV
    # linear index interpolation for probabilities
    pvals = np.interp(ind, np.arange(1, len(sortv) + 1), sortv, left=0.0, right=1.0)
    pvals[0] = max(pvals[0], 0.0)
    pvals[1] = level_UD
    pvals[2] = min(pvals[2], 1.0)
    return pvals


def summary_UD_format(CI, DOF, units: bool = True):
    ci = np.asarray(CI, dtype=float).reshape(3)
    u = unit(ci[1], "area", SI=not units)
    sc = float(u["scale"])
    name = u["name"]
    ci2 = (ci / sc).reshape(1, 3)
    dof = np.asarray(DOF, dtype=float).reshape(-1)
    if dof.size == 1:
        dof = np.array([dof[0], np.nan], dtype=float)
    return {
        "DOF": {"area": float(dof[0]), "bandwidth": float(dof[1]) if np.isfinite(dof[1]) else np.nan},
        "CI": ci2,
        "rowname": f"area ({name})",
        "colnames": ("low", "est", "high"),
    }


def summary_UD(object, convex: bool = False, level: float = 0.95, level_UD: float = 0.95, units: bool = True, **kwargs):
    del kwargs
    typ = object.get("type", "range")
    if typ not in ("range", "revisitation"):
        raise ValueError(f"{typ} area is not generally meaningful, biologically.")
    area = CI_UD(object, level_UD=level_UD, level=level, convex=convex)
    if np.asarray(area).size == 1:
        raise ValueError("Object is not a range distribution.")
    dof = np.array([np.asarray(object.get("DOF.area", [np.nan])).reshape(-1)[0], float(object.get("DOF.H", np.nan))], dtype=float)
    return summary_UD_format(area, DOF=dof, units=units)


def extract(r, UD, DF: str = "CDF", **kwargs):
    del kwargs
    arr = np.asarray(r, dtype=float)
    if arr.ndim > 1:
        arr = arr.T
    else:
        arr = arr.reshape(2, -1)
    gx = np.asarray(UD["r"]["x"], dtype=float)
    gy = np.asarray(UD["r"]["y"], dtype=float)
    dr = UD.get("dr", {"x": gx[1] - gx[0], "y": gy[1] - gy[0]})
    ix = (arr[0, :] - gx[0]) / float(dr["x"]) + 1.0
    iy = (arr[1, :] - gy[0]) / float(dr["y"]) + 1.0
    z = np.asarray(UD[DF], dtype=float)
    # bilinear interpolation
    x0 = np.floor(ix - 1).astype(int)
    y0 = np.floor(iy - 1).astype(int)
    x1 = np.clip(x0 + 1, 0, z.shape[0] - 1)
    y1 = np.clip(y0 + 1, 0, z.shape[1] - 1)
    x0 = np.clip(x0, 0, z.shape[0] - 1)
    y0 = np.clip(y0, 0, z.shape[1] - 1)
    fx = (ix - 1) - x0
    fy = (iy - 1) - y0
    v00 = z[x0, y0]
    v10 = z[x1, y0]
    v01 = z[x0, y1]
    v11 = z[x1, y1]
    return (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + (1 - fx) * fy * v01 + fx * fy * v11


__all__ = [
    "akde_bias",
    "pkde",
    "akde",
    "prepare_H",
    "kde_grid",
    "kde",
    "debias_volume",
    "debias_area",
    "pmf2cdf",
    "cdf2pmf",
    "Gauss",
    "Gauss1",
    "Gauss3",
    "pnorm1",
    "pnorm2",
    "pnorm3",
    "NewtonCotes",
    "CI_UD",
    "summary_UD",
    "summary_UD_format",
    "extract",
]
