"""2D (and 1D) covariance helper type ported from ctmm 1.3.0 ``R/covm.R``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core_math import clamp
from .pd_matrix import nant
from .pd_matrix import _he as He


@dataclass
class Covm:
    """Analog of S4 ``covm``: PSD matrix + canonical parameters."""

    sigma: np.ndarray
    par: dict[str, float]
    isotropic: bool
    axes: tuple[str, ...] = ("x", "y")

    def __post_init__(self) -> None:
        self.sigma = np.asarray(self.sigma, dtype=float)
        if self.sigma.ndim == 1:
            self.sigma = self.sigma.reshape(1, -1)
        if self.sigma.shape[0] != self.sigma.shape[1]:
            raise ValueError("sigma must be square")
        if len(self.axes) != self.sigma.shape[0]:
            raise ValueError("axes length must match sigma dimension")

    def copy(self) -> Covm:
        return Covm(self.sigma.copy(), dict(self.par), self.isotropic, self.axes)

    @property
    def ndim(self) -> int:
        return int(self.sigma.shape[0])


def sigma_construct(pars: Sequence[float] | np.ndarray) -> np.ndarray:
    """``sigma.construct`` — build 2×2 covariance from 1 or 3 parameters."""
    p = np.asarray(pars, dtype=float).ravel()
    if p.size == 1:
        v = float(p[0])
        return np.diag([v, v])
    if p.size != 3:
        raise ValueError("sigma_construct expects 1 or 3 parameters for 2D")
    major, minor, theta = float(p[0]), float(p[1]), float(p[2])
    u = np.array([np.cos(theta), np.sin(theta)])
    v = np.array([-np.sin(theta), np.cos(theta)])
    return major * np.outer(u, u) + minor * np.outer(v, v)


def sigma_destruct(sigma: np.ndarray, *, isotropic: bool = False) -> dict[str, float]:
    """``sigma.destruct`` — eigen decomposition to major/minor/angle."""
    s = np.asarray(sigma, dtype=float)
    if s.shape == (1, 1):
        return {"major": float(s[0, 0])}
    inf = np.diag(s) == np.inf
    if np.any(inf):
        d = np.diag(s)
        fin = d[np.isfinite(d)]
        return {
            "major": float(np.max(d)),
            "minor": float(np.min(fin) if fin.size else 0.0),
            "angle": 0.0,
        }
    w, v = np.linalg.eigh(s)
    w = np.clip(np.real(w), 0.0, np.inf)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = np.real(v[:, order])
    major, minor = float(w[0]), float(w[1])
    if major == minor:
        theta = 0.0
    else:
        t = V[:, 0]
        theta = float(np.arctan2(t[1], t[0]))
    return {"major": major, "minor": minor, "angle": theta}


def covm(
    pars: Any,
    *,
    isotropic: bool = False,
    axes: Sequence[str] = ("x", "y"),
) -> Covm | None:
    """``covm`` — universal covariance format (1D or 2D)."""
    if pars is None:
        return None
    ax = tuple(axes)
    if isinstance(pars, Covm):
        pars = dict(pars.par)

    if len(ax) == 1:
        if isinstance(pars, dict):
            val = float(pars.get("major", next(iter(pars.values()))))
        else:
            p = np.asarray(pars, dtype=float).ravel()
            val = float(p[0])
        sigma = np.array([[val]])
        par_d = {"major": val}
        return Covm(sigma, par_d, True, ax)

    # 2D
    if isinstance(pars, dict):
        if "major" in pars and "minor" in pars:
            pvec = [float(pars["major"]), float(pars["minor"]), float(pars.get("angle", 0.0))]
        else:
            raise ValueError("dict pars for 2D covm need major/minor")
        pars_arr = np.asarray(pvec, dtype=float)
    else:
        pars_arr = np.asarray(pars, dtype=float).ravel()

    if pars_arr.size == 1:
        v = float(pars_arr[0])
        sigma = np.diag([v, v])
        par_d = {"major": v, "minor": v, "angle": 0.0}
    elif pars_arr.size == 3:
        major = max(pars_arr[0], pars_arr[1])
        minor = min(pars_arr[0], pars_arr[1])
        theta = float(pars_arr[2]) if pars_arr[0] == major else float(pars_arr[2] + np.pi / 2)
        par_d = {"major": major, "minor": minor, "angle": theta}
        sigma = sigma_construct((major, minor, theta))
    elif pars_arr.size == 4:
        # R ``as.vector(matrix)`` is column-major
        sigma = pars_arr.reshape(2, 2, order="F")
        if isinstance(pars, Covm):
            par_d = dict(pars.par)
        else:
            par_d = sigma_destruct(sigma, isotropic=isotropic)
    else:
        raise ValueError(f"covm: unsupported pars length {pars_arr.size}")

    if isotropic:
        m = np.mean([par_d["major"], par_d["minor"]])
        par_d = {"major": m, "minor": m, "angle": 0.0}
        sigma = np.diag([m, m])

    return Covm(sigma, par_d, bool(isotropic), ax)


def eigenvalues_covm(sigma: Covm) -> np.ndarray:
    if sigma.ndim == 1:
        return np.array([sigma.par["major"]])
    m, n = float(sigma.par["major"]), float(sigma.par["minor"])
    return np.sort(np.array([m, n]), kind="mergesort")[::-1]


def var_covm(sigma: Covm, *, ave: bool = False) -> float:
    ev = eigenvalues_covm(sigma)
    return float(np.mean(ev)) if ave else float(np.sum(ev))


def det_covm(sigma: Covm, *, ave: bool = False) -> float:
    ev = eigenvalues_covm(sigma)
    dim = ev.size
    p = float(np.prod(ev))
    return p ** (1.0 / dim) if ave else p


def area_covm(sigma: Covm) -> float:
    return det_covm(sigma, ave=True)


def scale_covm(sigma: Covm, value: float) -> Covm:
    out = sigma.copy()
    out.sigma = out.sigma * value
    if "major" in out.par:
        out.par["major"] *= value
    if "minor" in out.par:
        out.par["minor"] *= value
    return out


def squeezable_covm(CTMM: dict) -> dict[str, Any]:
    """``squeezable.covm(CTMM)`` — eccentricity squeeze factor (R ignores ``axes`` in formula)."""
    sigma = CTMM["sigma"]
    if not isinstance(sigma, Covm):
        raise TypeError("CTMM['sigma'] must be a Covm instance")
    vars_ = eigenvalues_covm(sigma)
    smin = float(np.min(vars_))
    smax = float(np.max(vars_))
    if smin <= 0 or not np.isfinite(smin):
        fact = float("nan")
    else:
        fact = (smax / smin) ** 0.25
    eps = np.finfo(float).eps
    able = not np.isnan(fact) and 4 * abs(np.log(fact)) < np.log(1 / eps)
    return {"fact": fact, "able": bool(able)}


def rotate_covm(sigma: Covm, theta: float | None = None) -> Covm:
    if sigma.ndim == 1:
        return sigma
    if theta is None:
        theta = -sigma.par["angle"]
    par = dict(sigma.par)
    par["angle"] = par["angle"] + float(theta)
    return covm(par, isotropic=sigma.isotropic, axes=sigma.axes)


def squeeze_covm(sigma: Covm, smgm: float | None = None, *, circle: bool = False) -> Covm:
    if sigma.ndim == 1:
        return sigma
    if smgm is None:
        circle = True
    par = dict(sigma.par)
    if circle:
        g = float(np.sqrt(par["major"] * par["minor"]))
        par["major"] = par["minor"] = g
    else:
        assert smgm is not None
        f1 = (1.0 / smgm) ** 2
        f2 = smgm**2
        par["major"] *= f1
        par["minor"] *= f2
    return covm(par, isotropic=sigma.isotropic, axes=sigma.axes)


def solve_covm(sigma: Covm, *, pseudo: bool = False) -> Covm:
    _ = pseudo
    if sigma.ndim == 1:
        par = {"major": 1.0 / sigma.par["major"]}
        return covm(par, isotropic=sigma.isotropic, axes=sigma.axes)
    maj, min_ = float(sigma.par["major"]), float(sigma.par["minor"])
    ang = float(sigma.par["angle"])
    # R: sigma[PARS] <- rev(1/sigma[PARS]); angle <- angle + pi/2
    new_maj = 1.0 / min_
    new_min = 1.0 / maj
    new_ang = ang + np.pi / 2
    return covm((new_maj, new_min, new_ang), isotropic=sigma.isotropic, axes=sigma.axes)


def sqrtm_covm(sigma: Covm) -> Covm:
    if sigma.ndim == 1:
        v = float(np.sqrt(max(sigma.par["major"], 0.0)))
        return covm(v, isotropic=sigma.isotropic, axes=sigma.axes)
    maj = float(np.sqrt(max(sigma.par["major"], 0.0)))
    min_ = float(np.sqrt(max(sigma.par["minor"], 0.0)))
    ang = float(sigma.par["angle"])
    return covm((maj, min_, ang), isotropic=sigma.isotropic, axes=sigma.axes)


def fn_covm(sigma: Covm, fn: Callable[[np.ndarray], np.ndarray]) -> Covm:
    if sigma.ndim == 1:
        v = float(fn(np.array([sigma.par["major"]]))[0])
        return covm(v, isotropic=sigma.isotropic, axes=sigma.axes)
    maj = float(fn(np.array([sigma.par["major"]]))[0])
    min_ = float(fn(np.array([sigma.par["minor"]]))[0])
    ang = float(sigma.par["angle"])
    return covm((maj, min_, ang), isotropic=sigma.isotropic, axes=sigma.axes)


def mpow_covm(sigma: Covm, pow: float) -> Covm:
    return fn_covm(sigma, lambda s: s**pow)


def log_covm(sigma: Covm) -> Covm:
    return fn_covm(sigma, np.log)


def exp_covm(sigma: Covm) -> Covm:
    return fn_covm(sigma, np.exp)


def J_sigma_par(par: dict[str, float] | Covm) -> np.ndarray:
    """``J.sigma.par`` — rows ``(xx,yy,xy)``, cols ``(major,minor,angle)``."""
    if isinstance(par, Covm):
        par = par.par
    major = float(par["major"])
    minor = float(par["minor"])
    theta = float(par["angle"])
    grad = np.zeros((3, 3))
    grad[:, 0] = [np.cos(theta) ** 2, np.sin(theta) ** 2, np.sin(2 * theta) / 2]
    grad[:, 1] = [np.sin(theta) ** 2, np.cos(theta) ** 2, -np.sin(2 * theta) / 2]
    grad[0, 2] = (minor - major) * np.sin(2 * theta)
    grad[1, 2] = (major - minor) * np.sin(2 * theta)
    grad[2, 2] = (major - minor) * np.cos(2 * theta)
    return grad


def J_par_sigma(sigma_vec: np.ndarray) -> np.ndarray:
    """``J.par.sigma`` — rows ``(major,minor,angle)``, cols ``(xx,yy,xy)``."""
    s = np.asarray(sigma_vec, dtype=float).ravel()
    if s.size != 3:
        raise ValueError("expected length-3 sigma vector (xx,yy,xy)")
    sxx, syy, sxy = float(s[0]), float(s[1]), float(s[2])
    d = (sxx - syy) ** 2 + 4 * sxy**2
    sqrt_d = np.sqrt(d)
    r1 = nant(np.array([(sxx - syy) / sqrt_d]), 0.0)[0]
    r2 = nant(np.array([4 * sxy / sqrt_d]), 0.0)[0]
    grad = np.zeros((3, 3))
    grad[0, :] = np.array([1, 1, 0]) / 2 + np.array([r1, -r1, r2]) / 2
    grad[1, :] = np.array([1, 1, 0]) / 2 + np.array([-r1, r1, -r2]) / 2
    grad[2, :] = nant(np.array([-sxy, sxy, sxx - syy]) / d, 0.0)
    return grad


def COV_covm(sigma: Covm, n: int, k: int = 1, *, REML: bool = True) -> dict[str, Any]:
    """``COV.covm`` — asymptotic covariance of covm parameters (real PSD case)."""
    par = dict(sigma.par)
    s = np.asarray(sigma.sigma, dtype=float)
    dim = int(np.sqrt(s.size))
    dof_mu = n
    cov_mu = s / n
    n_eff = n - k if REML else n

    if sigma.isotropic or dim == 1:
        c = np.array([[2 * par["major"] ** 2 / (n_eff * dim)]])
        return {"COV": c, "COV_mu": cov_mu, "DOF_mu": dof_mu}

    cov = np.zeros((3, 3))
    cov[np.ix_([0, 1], [0, 1])] = 2 / n_eff * (s**2)
    cov[2, :] = np.array(
        [2 * s[0, 0] * s[0, 1], 2 * s[1, 1] * s[0, 1], s[0, 0] * s[1, 1] + s[0, 1] ** 2]
    ) / n_eff
    cov[:, 2] = cov[2, :]
    grad = J_sigma_par(par)
    inv_grad = np.linalg.inv(grad)  # pd.solve(sym=FALSE)
    cov = inv_grad @ cov @ inv_grad.T
    cov = nant(cov, 0.0)
    cov = He(cov)
    return {"COV": cov, "COV_mu": cov_mu, "DOF_mu": dof_mu}


def pars_covm(cov: Covm) -> dict[str, float] | float:
    if cov.isotropic:
        return float(cov.par["major"])
    return dict(cov.par)


def axes2var(CTMM, MEAN: bool = True):
    cov = None
    if hasattr(CTMM, "params"):
        cov = CTMM.params.get("COV")
        isotropic = CTMM.params.get("isotropic", False)
    elif isinstance(CTMM, dict):
        cov = CTMM.get("COV")
        isotropic = CTMM.get("isotropic", False)
    if cov is None:
        return cov
    import pandas as pd

    C = pd.DataFrame(cov).copy()
    if "minor" not in C.index or "angle" not in C.index:
        return C.rename(index={"major": "variance"}, columns={"major": "variance"})
    other = [x for x in C.index if x not in ("major", "minor", "angle")]
    new = ["variance"] + other
    grad = pd.DataFrame(0.0, index=new, columns=C.index)
    if other:
        for name in other:
            grad.loc[name, name] = 1.0
    if isinstance(isotropic, (list, tuple, np.ndarray)):
        iso = bool(np.asarray(isotropic).reshape(-1)[0])
    else:
        iso = bool(isotropic)
    grad.loc["variance", "major"] = 1.0
    if not iso:
        grad.loc["variance", "minor"] = 1.0
    if MEAN:
        grad.loc["variance", ["major"] + ([] if iso else ["minor"])] /= 2.0 if not iso else 1.0
    out = grad.to_numpy() @ C.to_numpy(dtype=float) @ grad.to_numpy().T
    return pd.DataFrame(out, index=new, columns=new)


def par_fn(par, fn):
    vals = par.par if isinstance(par, Covm) else par
    return fn(np.asarray(list(vals.values()) if isinstance(vals, dict) else vals, dtype=float))


def sigma_COV(sigma, COV=None):
    if COV is None:
        return COV_covm(sigma, n=1).get("COV")
    if isinstance(sigma, Covm):
        jac = J_sigma_par(sigma)
    else:
        jac = J_sigma_par(covm(sigma))
    c = np.asarray(COV, dtype=float)
    return jac @ c @ jac.T

__all__ = [
    "Covm",
    "COV_covm",
    "covm",
    "sigma_construct",
    "sigma_destruct",
    "eigenvalues_covm",
    "var_covm",
    "det_covm",
    "area_covm",
    "scale_covm",
    "squeezable_covm",
    "rotate_covm",
    "squeeze_covm",
    "solve_covm",
    "sqrtm_covm",
    "fn_covm",
    "mpow_covm",
    "log_covm",
    "exp_covm",
    "axes2var",
    "par_fn",
    "pars_covm",
    "sigma_COV",
    "J_par_sigma",
    "J_sigma_par",
]
