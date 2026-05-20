"""Parity translation of ctmm 1.3.0 ``R/math.R`` helpers."""

from __future__ import annotations

import numpy as np
from scipy import special

from .core_math import clamp
from .pd_matrix import nant
from .r_math import sinc, sinch

EulerGamma = 0.57721566490153286061
Zeta3 = 1.2020569031595942854


def _scalar_or_array(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return arr


def _psigamma(x, deriv: int = 0):
    if deriv == 0:
        return special.digamma(x)
    return special.polygamma(deriv, x)


def ipsigamma(x, deriv: int = 0, precision: float = 0.5):
    """Inverse ``psigamma`` by the Newton iteration in ``R/math.R``."""
    x_arr = np.asarray(x, dtype=float)
    if deriv == 0:
        y = np.exp(x_arr)
    else:
        y = ((-1.0) ** (deriv + 1) / x_arr) ** (1.0 / deriv)
    tol = np.finfo(float).eps ** precision
    err = float("inf")
    while err > tol:
        dx = x_arr - _psigamma(y, deriv=deriv)
        dy = dx / _psigamma(y, deriv=deriv + 1)
        dy = nant(dy, 1.0)
        y = y + dy
        err = float(np.nanmax(np.abs(dy / y)))
    return _scalar_or_array(y)


def itrigamma(x, precision: float = 0.5):
    return ipsigamma(x, deriv=1, precision=precision)


def legendre(n: int, x):
    """Shifted Legendre polynomials implemented up to R's n=5 branch."""
    x = np.asarray(x, dtype=float)
    n = int(n)
    if n == 0:
        out = np.ones_like(x, dtype=float)
    elif n == 1:
        out = 2 * x - 1
    elif n == 2:
        out = 6 * x**2 - 6 * x + 1
    elif n == 3:
        out = 20 * x**3 - 30 * x**2 + 12 * x - 1
    elif n == 4:
        out = 70 * x**4 - 140 * x**3 + 90 * x**2 - 20 * x + 1
    elif n == 5:
        out = 252 * x**5 - 630 * x**4 + 560 * x**3 - 210 * x**2 + 30 * x - 1
    else:
        raise ValueError("ctmm legendre is defined for n=0..5")
    return _scalar_or_array(out)


def mpsigamma(x, deriv: int = 0, dim: int = 1):
    psi = np.asarray(x, dtype=float) + (1.0 - np.arange(1, int(dim) + 1, dtype=float)) / 2.0
    if deriv >= 0:
        vals = _psigamma(psi, deriv=deriv)
    elif deriv == -1:
        vals = special.gammaln(psi)
    else:
        raise ValueError(f"Derivative {deriv + 1} of log(Gamma(x)) not supported.")
    return float(np.sum(vals))


def logit(p):
    p = clamp(np.asarray(p, dtype=float), 0.0, 1.0)
    return _scalar_or_array(np.log(p / (1.0 - p)))


def ilogit(z):
    return _scalar_or_array(special.expit(np.asarray(z, dtype=float)))


def binom(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return _scalar_or_array(1.0 / (x + 1.0) / special.beta(y + 1.0, x - y + 1.0))


def lbinom(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return _scalar_or_array(-np.log(x + 1.0) - special.betaln(y + 1.0, x - y + 1.0))


__all__ = [
    "EulerGamma",
    "Zeta3",
    "binom",
    "ilogit",
    "ipsigamma",
    "itrigamma",
    "lbinom",
    "legendre",
    "logit",
    "mpsigamma",
    "sinc",
    "sinch",
]
