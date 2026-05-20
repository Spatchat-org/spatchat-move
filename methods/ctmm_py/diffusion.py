"""Parity translation of ctmm 1.3.0 ``R/diffusion.R`` analytic paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from .covm import Covm, covm, var_covm
from .series_utils import series
from .types import CTMMModel
from .stats import chisq_ci


def _scalar_or_array(value):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return arr


def Diff_OUF_fn(z, deriv: int = 0):
    """``Diff.OUF.fn``: OUF diffusion-rate coefficient and derivative."""
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    w = 1.0 - z_arr
    regular = w > 0.004
    if deriv == 0:
        out[regular] = np.power(z_arr[regular], z_arr[regular] / w[regular])
        out[~regular] = np.exp(-1.0) * series(
            w[~regular],
            np.array([1.0, 1.0 / 2.0, 7.0 / 24.0, 3.0 / 16.0, 743.0 / 5760.0, 215.0 / 2304.0]),
        )
    elif deriv == 1:
        base = np.asarray(Diff_OUF_fn(z_arr, deriv=0), dtype=float)
        out[regular] = base[regular] * (w[regular] + np.log(z_arr[regular])) / (w[regular] ** 2)
        out[~regular] = -base[~regular] * series(w[~regular], 1.0 / np.arange(2.0, 11.0, 1.0))
    else:
        raise ValueError("Diff_OUF_fn supports deriv=0 or deriv=1")
    return _scalar_or_array(out)


def Diff_OUO_fn(x, deriv: int = 0):
    """``Diff.OUO.fn``: OU-omega diffusion-rate coefficient and derivative."""
    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr, dtype=float)
    if deriv == 0:
        ratio = np.divide(np.arctan(x_arr), x_arr, out=np.ones_like(x_arr), where=x_arr != 0)
        out[...] = np.sqrt(1.0 + x_arr**2) * np.exp(-ratio)
    elif deriv == 1:
        regular = x_arr > 0.1
        base = np.asarray(Diff_OUO_fn(x_arr, deriv=0), dtype=float)
        out[regular] = base[regular] / (x_arr[regular] ** 2) * (
            x_arr[regular] - 2.0 * x_arr[regular] / (1.0 + x_arr[regular] ** 2) + np.arctan(x_arr[regular])
        )
        n = 10
        coef = (np.arange(5.0, 5.0 + 4.0 * n, 4.0) / np.arange(3.0, 3.0 + 2.0 * n, 2.0)) * (
            (-1.0) ** (1 + np.arange(1, n + 1))
        )
        out[~regular] = base[~regular] * x_arr[~regular] * series(x_arr[~regular] ** 2, coef)
    else:
        raise ValueError("Diff_OUO_fn supports deriv=0 or deriv=1")
    return _scalar_or_array(out)


def _tau_values(model: CTMMModel | dict[str, Any]) -> tuple[list[str], list[float]]:
    params = model.params if isinstance(model, CTMMModel) else model
    tau = params.get("tau", {})
    if isinstance(tau, dict) and tau:
        ordered = sorted(((str(k), float(v)) for k, v in tau.items()), key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in ordered], [v for _, v in ordered]
    vals = [float(v) for v in (params.get("tau_list", []) or [])]
    names = ["position", "velocity", "acceleration"][: len(vals)]
    return names, vals


def _sigma_covm(model: CTMMModel | dict[str, Any]) -> Covm | None:
    params = model.params if isinstance(model, CTMMModel) else model
    sigma = params.get("sigma")
    if isinstance(sigma, Covm):
        return sigma
    if sigma is not None:
        try:
            return covm(sigma, isotropic=bool(params.get("isotropic", False)), axes=tuple(params.get("axes", ("x", "y"))))
        except Exception:
            pass
    sigma_matrix = params.get("sigma_matrix")
    if sigma_matrix is None:
        return None
    try:
        return covm(np.asarray(sigma_matrix, dtype=float).reshape(-1, order="F"), isotropic=bool(params.get("isotropic", False)), axes=tuple(params.get("axes", ("x", "y"))))
    except Exception:
        return None


def _feature_grad_variance(params: dict[str, Any], grad: dict[str, float]) -> float:
    cov = params.get("COV")
    if cov is None:
        return float("inf")
    try:
        cov_arr = np.asarray(cov, dtype=float)
    except Exception:
        return float("inf")
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        return float("inf")

    rownames = params.get("COV_rownames")
    if rownames is None and hasattr(cov, "index"):
        rownames = list(cov.index)
    if rownames is None:
        rownames = params.get("features")
    if not rownames:
        return float("inf")
    names = [str(n) for n in rownames]
    g = np.zeros(len(names), dtype=float)
    for i, name in enumerate(names):
        g[i] = float(grad.get(name, 0.0))
    if not np.any(g):
        return float("inf")
    sub = cov_arr[: len(names), : len(names)]
    var = float(g @ sub @ g)
    return var if np.isfinite(var) else float("inf")


def diffusion(CTMM: CTMMModel | dict[str, Any], level: float = 0.95, finish: bool = True):
    """``diffusion``: maximum-lag diffusion rate for a movement model."""
    params = CTMM.params if isinstance(CTMM, CTMMModel) else CTMM
    names, tau = _tau_values(CTMM)
    sigma = _sigma_covm(CTMM)
    variance = var_covm(sigma, ave=False) if sigma is not None else float("nan")
    omega = float(params.get("omega", 0.0) or 0.0)
    range_ = bool(params.get("range", True))

    if not tau or all(float(t) == 0.0 for t in tau):
        if not finish:
            return {"D": float("inf"), "grad": {}, "VAR": float("inf"), "DOF": 0.0, "J": {}}
        return np.array([0.0, float("inf"), float("inf")], dtype=float)

    coef_grad: dict[str, float] = {}
    jac: dict[str, float] = {}
    if not range_:
        coeff = 1.0
    elif len(tau) == 1 or (len(tau) > 1 and float(tau[1]) == 0.0):
        t0 = float(tau[0])
        coeff = 1.0 / t0
        key = f"tau {names[0]}"
        coef_grad[key] = -1.0 / (t0**2)
        jac[key] = coef_grad[key]
    elif len(tau) >= 2:
        t0 = float(tau[0])
        t1 = float(tau[1])
        key0 = f"tau {names[0]}"
        key1 = f"tau {names[1]}"
        if not omega and not np.isclose(t0, t1, rtol=1e-10, atol=1e-12):
            z = t1 / t0
            coeff = float(Diff_OUF_fn(z) / t0)
            deriv = float(Diff_OUF_fn(z, deriv=1))
            z_grad0 = -z / t0
            z_grad1 = z / t1
            coef_grad[key0] = deriv / t0 * z_grad0 - coeff / t0
            coef_grad[key1] = deriv / t0 * z_grad1
            jac.update(coef_grad)
        elif omega and np.isclose(t0, t1, rtol=1e-10, atol=1e-12):
            z = t0 * omega
            coeff = float(Diff_OUO_fn(z) / t0)
            deriv = float(Diff_OUO_fn(z, deriv=1))
            coef_grad[key0] = deriv / t0 * omega - coeff / t0
            coef_grad["omega"] = deriv
            jac.update(coef_grad)
        else:
            coeff = float(np.exp(-1.0) / t0)
            coef_grad[key0] = -coeff / t0
            jac[key0] = coef_grad[key0]
    else:
        coeff = float("nan")

    grad = {k: variance * v for k, v in coef_grad.items()}
    jac = {k: variance * v for k, v in jac.items()}
    grad["variance"] = coeff
    if bool(params.get("isotropic", False)):
        jac["major"] = 2.0 * coeff
    else:
        jac["major"] = coeff
        jac["minor"] = coeff

    d = float(variance * coeff)
    var = _feature_grad_variance(params, grad)
    dof = 0.0 if not np.isfinite(var) or var <= 0 else float(2.0 * d * d / var)

    if not finish:
        return {"D": d, "grad": grad, "VAR": var, "DOF": dof, "J": jac}
    return chisq_ci(d, var=var, level=level)


DiffOUFfn = Diff_OUF_fn
DiffOUOfn = Diff_OUO_fn

__all__ = [
    "Diff_OUF_fn",
    "Diff_OUO_fn",
    "DiffOUFfn",
    "DiffOUOfn",
    "diffusion",
]
