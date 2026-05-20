from __future__ import annotations

import math

import numpy as np
from scipy import special

from .generic_utils import nant


def series(x, coef):
    c = np.asarray(coef, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    p = np.ones_like(x, dtype=float)
    for a in c:
        out = out + a * p
        p = p * x
    return out


def cosm1(x):
    x = np.asarray(x, dtype=float)
    y = (x + np.pi) % (2 * np.pi) - np.pi
    out = np.empty_like(y)
    mask = y > 0.2
    out[mask] = np.cos(y[mask]) - 1.0
    coef = np.array([0, 0, -(1 / 2), 0, 1 / 24, 0, -(1 / 720), 0, 1 / 40320, 0, -(1 / 3628800)], dtype=float)
    out[~mask] = series(y[~mask], coef)
    return out


def log1pxdx(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    mask = x > 0.05
    out[mask] = np.log1p(x[mask]) / x[mask]
    coef = np.array([1, 1 / 2, -(1 / 12), 1 / 24, -(19 / 720), 3 / 160, -(863 / 60480), 275 / 24192, -(33953 / 3628800), 8183 / 1036800, -(3250433 / 479001600), 4671 / 788480], dtype=float)
    out[~mask] = 1.0 / series(x[~mask], coef)
    return out


def lbetaplog(a, b, x):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.empty_like(np.broadcast_arrays(a, b, x)[0], dtype=float)
    aa, bb, xx = np.broadcast_arrays(a, b, x)
    mask = bb < 1 / np.sqrt(np.finfo(float).eps)
    out[mask] = special.betaln(aa[mask], bb[mask]) + aa[mask] * np.log(bb[mask] + aa[mask] * xx[mask])
    if np.any(~mask):
        ai = aa[~mask]
        bi = bb[~mask]
        xi = xx[~mask]
        c0 = special.gammaln(ai)
        c1 = ai / 2 - ai**2 / 2 + ai**2 * xi
        c2 = ai / 12 - ai**2 / 4 + ai**3 / 6 - (ai**3 * xi**2) / 2
        out[~mask] = c0 + c1 * (1 / bi) + c2 * (1 / bi) ** 2
    return out


def sqrtxp1(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    mask = x > 0.5
    out[mask] = (np.sqrt(x[mask] + 1.0) - 1.0) / x[mask]
    y = np.sqrt(x[~mask])
    cn = np.array([1 / 2, 0, 19 / 8, 0, 153 / 32, 0, 85 / 16, 0, 455 / 128, 0, 3003 / 2048, 0, 3003 / 8192, 0, 429 / 8192, 0, 495 / 131072, 0, 55 / 524288, 0, 1 / 2097152], dtype=float)
    cd = np.array([1, 0, 5, 0, 171 / 16, 0, 51 / 4, 0, 595 / 64, 0, 273 / 64, 0, 5005 / 4096, 0, 429 / 2048, 0, 1287 / 65536, 0, 55 / 65536, 0, 11 / 1048576], dtype=float)
    out[~mask] = series(y, cn) / series(y, cd)
    return out


def BesselK(x, nu, expon_scaled=False, log=False):
    x = np.asarray(x, dtype=float)
    nu = np.abs(np.asarray(nu, dtype=float))
    y = special.kv(nu, x)
    if expon_scaled:
        y = y * np.exp(x)
    if log:
        y = np.log(y)
    return y


def lKK(r, n, t, s):
    t = float(t)
    s = np.asarray(s, dtype=float)
    if math.isinf(t):
        return np.zeros_like(s)
    return -s * sqrtxp1(s / t) + nant(BesselK(np.sqrt(1 + s / t) * t, n / 2 + r / 2, expon_scaled=True, log=True) - BesselK(t, r / 2, expon_scaled=True, log=True), 0)


def log_chi2_bias(n):
    n = np.asarray(n, dtype=float)
    b = n.copy()
    n1 = 0.000003
    n2 = 40
    m = n == 0
    b[m] = -np.inf
    m = (n > 0) & (n <= n1)
    b[m] = -2 / n[m] - np.log(n[m] / 2) - np.euler_gamma + (np.pi**2 / 12) * n[m]
    m = (n > n1) & (n < n2)
    b[m] = special.digamma(n[m] / 2) - np.log(n[m] / 2)
    m = n >= n2
    if np.any(m):
        coef = np.array([0, -1, -(1 / 3), 0, 2 / 15, 0, -(16 / 63), 0, 16 / 15, 0, -(256 / 33)], dtype=float)
        b[m] = series(1 / n[m], coef)
    return b
