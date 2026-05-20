"""Partial parity translation of ctmm 1.3.0 ``R/stats.R``."""

from __future__ import annotations

import math

import numpy as np
from scipy import special
from scipy.stats import beta as beta_dist
from scipy.stats import f as scipy_f
from scipy.stats import chi2, invgauss, norm

from .generic import clamp, nant

NAMES_CI = ("low", "est", "high")


def cov_loglike(hess, grad=None, tol: float = np.finfo(float).eps) -> np.ndarray:
    """
    Translation of ``cov.loglike`` core logic:
    generalized covariance from Hessian/gradient around optimum.
    """
    h = np.asarray(hess, dtype=float)
    g = np.zeros(h.shape[0], dtype=float) if grad is None else np.asarray(grad, dtype=float)
    g = nant(g, np.inf)
    h = nant(h, 0.0)

    if np.all(np.diag(h) > 0):
        try:
            c = np.linalg.pinv(h)
        except Exception:
            c = np.full_like(h, np.nan, dtype=float)
        if np.all(np.diag(c) > 0):
            return c

    v = np.sqrt(np.abs(np.diag(h)))
    v = np.maximum(v, np.abs(g))
    v[v <= tol] = 1.0
    w = np.outer(v, v)

    g = nant(g / v, 1.0)
    h = nant(h / w, 1.0)

    mx = np.sqrt(np.abs(np.diag(h)))
    mx = np.outer(mx, mx)
    h = clamp(h, -mx, mx)

    vals, vecs = np.linalg.eigh(h)
    vals = clamp(vals, 0.0, np.inf)
    g2 = vecs.T @ g

    out_vals = np.zeros_like(vals)
    for i, lam in enumerate(vals):
        det = lam + g2[i] ** 2
        if lam == 0.0:
            out_vals[i] = 1.0 / max((2.0 * g2[i]) ** 2, 1e-18)
        elif det >= 0.0:
            out_vals[i] = ((math.sqrt(det) - g2[i]) / lam) ** 2
        else:
            out_vals[i] = 1.0 / max((2.0 * g2[i]) ** 2, 1e-18)
    out_vals = nant(out_vals, 0.0)

    cov = vecs @ np.diag(out_vals) @ vecs.T
    return cov / w


def norm_ci(mle: float, var: float, level: float = 0.95, alpha: float | None = None) -> np.ndarray:
    if alpha is None:
        alpha = 1.0 - level
    z = norm.ppf(1.0 - alpha / 2.0)
    ci = np.array([mle - z * math.sqrt(max(var, 0.0)), mle, mle + z * math.sqrt(max(var, 0.0))], dtype=float)
    return ci


def chisq_ci(
    mle: float,
    var: float | None = None,
    level: float = 0.95,
    alpha: float | None = None,
    dof: float | None = None,
) -> np.ndarray:
    if alpha is None:
        alpha = 1.0 - level
    mle = float(mle)
    if dof is None:
        if var is None or not np.isfinite(var) or var <= 0:
            dof = np.inf
        else:
            dof = 2.0 * mle * mle / float(var)
    dof = float(dof)
    if np.isnan(dof) or dof <= 0:
        return np.array([0.0, mle, np.inf], dtype=float)
    if np.isinf(dof):
        return np.array([mle, mle, mle], dtype=float)
    lo = mle * (chi2.ppf(alpha / 2.0, dof) / dof)
    hi = mle * (chi2.ppf(1.0 - alpha / 2.0, dof) / dof)
    out = np.array([lo, mle, hi], dtype=float)
    if var is not None and np.isfinite(var):
        out[2] = max(out[2], norm_ci(mle, var, alpha=alpha)[2])
    return out


def chisq_hdr(df: float, level: float = 0.95, alpha: float | None = None, pow: float = 1.0):
    if alpha is None:
        alpha = 1.0 - level
    df = float(df)
    if df <= pow:
        return np.array([0.0, 0.0, chi2.ppf(level, df)], dtype=float)
    mode = df - pow
    return np.array([chi2.ppf(alpha / 2.0, df), mode, chi2.ppf(1.0 - alpha / 2.0, df)], dtype=float)


def idchisq(p, df):
    # Numeric inverse density branch; returns the two chi-square ordinates around the mode when available.
    df = float(df)
    p = float(p)
    if df <= 2 or p <= 0:
        return np.array([0.0, np.inf], dtype=float)
    xs = np.linspace(np.finfo(float).tiny, chi2.ppf(0.999999, df), 4096)
    dens = chi2.pdf(xs, df)
    mask = dens >= p
    if not np.any(mask):
        return np.array([np.nan, np.nan], dtype=float)
    return np.array([xs[mask][0], xs[mask][-1]], dtype=float)


def lognorm_ci(mle: float, var: float, level: float = 0.95, alpha: float | None = None) -> np.ndarray:
    if alpha is None:
        alpha = 1.0 - level
    sigma = math.log(1.0 + var / max(mle * mle, 1e-18))
    mu = math.log(max(mle, 1e-18)) - sigma / 2.0
    ci = norm_ci(mu, sigma, alpha=alpha)
    out = np.exp(ci)
    out[1] = mle
    return out


def beta_ci(mle: float, var: float, level: float = 0.95, alpha: float | None = None) -> np.ndarray:
    if alpha is None:
        alpha = 1.0 - level
    mle = float(nant(np.array([mle]), 0.0)[0])
    var = float(nant(np.array([var]), np.inf)[0])
    if var == 0:
        return np.array([mle, mle, mle], dtype=float)
    n = mle * (1.0 - mle) / var - 1.0
    if n <= 0:
        return np.array([0.0, mle, 1.0], dtype=float)
    a = n * mle
    b = n * (1.0 - mle)
    lo = beta_dist.ppf(alpha / 2.0, a, b)
    hi = beta_dist.ppf(1.0 - alpha / 2.0, a, b)
    return np.array([lo, mle, hi], dtype=float)


def F_CI(E1: float, VAR1: float, E2: float, VAR2: float, level: float = 0.95) -> np.ndarray:
    """``F.CI`` ratio confidence interval with R's bias correction."""
    est = float(nant(np.array([E1 * E2], dtype=float), 0.0)[0])
    n1 = 2.0 * E1 * E1 / VAR1 if VAR1 != 0 else np.inf
    n2 = float(nant(np.array([2.0 * E2 * E2 / VAR2 + 4.0 if VAR2 != 0 else np.inf]), 0.0)[0])
    if n1 <= 0 or n2 <= 0:
        return np.array([0.0, est, np.inf], dtype=float)
    bias = n2 / (n2 - 2.0)
    if bias <= 0:
        bias = np.inf
    alpha = (1.0 - level) / 2.0
    qs = scipy_f.ppf([alpha, 1.0 - alpha], n1, n2)
    ci = np.array([qs[0] / bias, 1.0, qs[1] / bias], dtype=float) * est
    return ci


def Log_F_CI(E1, VAR1, E2, VAR2, level: float = 0.95):
    n1 = 2.0 * E1 * E1 / VAR1 if VAR1 != 0 else np.inf
    n2 = 2.0 * E2 * E2 / VAR2 if VAR2 != 0 else np.inf
    alpha = (1.0 - level) / 2.0
    ci = np.zeros(3, dtype=float)
    ci[[0, 2]] = np.log(scipy_f.ppf([alpha, 1.0 - alpha], n1, n2))
    ci[1] = math.log(E1) - math.log(E2)
    return ci


def chi_bias(dof):
    dof = np.asarray(dof, dtype=float)
    out = np.ones_like(dof)
    mask = dof > 0
    d = dof[mask]
    out[mask] = np.sqrt(2.0 / d) * np.exp(np.vectorize(math.lgamma)((d + 1.0) / 2.0) - np.vectorize(math.lgamma)(d / 2.0))
    out[dof == 0] = 0.0
    return out


def chi_dof(m1: float, m2: float, precision: float = 0.5) -> float:
    """``chi.dof``: chi degrees-of-freedom matching first two moments."""
    m1 = float(m1)
    m2 = float(m2)
    if not np.isfinite(m1) or not np.isfinite(m2):
        return 0.0
    if m1 == 0.0 and m2 == 0.0:
        return float("inf")
    if m2 <= 0.0:
        return 0.0

    r = m1 * m1 / m2
    if 1.0 - r <= 0.0:
        return float("inf")
    if r <= 0.0:
        return 0.0

    dof = max(m1 * m1 / max(m2 - m1 * m1, np.finfo(float).tiny) / 2.0, np.finfo(float).tiny)
    error = np.finfo(float).eps ** precision
    rel_error = float("inf")
    while rel_error >= error:
        old = dof
        old_error = rel_error
        r0 = 2.0 * math.pi / dof / (special.beta(dof / 2.0, 0.5) ** 2)
        g0 = (special.digamma((dof + 1.0) / 2.0) - special.digamma(dof / 2.0) - 1.0 / dof) * r0
        if not np.isfinite(g0) or g0 == 0.0:
            return old
        delta = (r - r0) / g0
        if dof + delta <= 0.0:
            dof = dof / 2.0
        else:
            dof = dof + delta
        rel_error = abs(delta) / max(dof, np.finfo(float).tiny)
        if rel_error > old_error:
            return old
    return float(dof)


def chi_var(DOF, M1=1.0):
    scalar = np.asarray(DOF).ndim == 0
    d = np.atleast_1d(np.asarray(DOF, dtype=float))
    r = np.empty_like(d, dtype=float)
    mask = d > 0
    r[~mask] = np.inf
    r[mask] = d[mask] / (2.0 * (special.gamma((d[mask] + 1.0) / 2.0) / special.gamma(d[mask] / 2.0)) ** 2)
    out = (r - 1.0) * float(M1) ** 2
    return float(out[0]) if scalar else out


def chisq_dof(MED, IQR, alpha: float = 0.25):
    del alpha
    if IQR == 0:
        return float("inf")
    if IQR == np.inf or MED == 0:
        return 0.0
    target = float(MED) / float(IQR)

    def cost(nu):
        q1 = chi2.ppf(0.25, nu) / nu
        med = chi2.ppf(0.5, nu) / nu
        q2 = chi2.ppf(0.75, nu) / nu
        return (med / (q2 - q1) - target) ** 2

    from scipy.optimize import minimize_scalar

    fit = minimize_scalar(cost, bounds=(np.finfo(float).eps, 1e6), method="bounded")
    return float(fit.x)


def tnorm_hdr(mu=0.0, VAR=1.0, lower=0.0, upper=np.inf, level: float = 0.95):
    sd = math.sqrt(float(VAR))
    mass = norm.cdf(upper, loc=mu, scale=sd) - norm.cdf(lower, loc=mu, scale=sd)
    if mass <= 0:
        return np.array([lower, mu, upper], dtype=float)
    alpha = (1.0 - level) / 2.0
    lo = norm.ppf(norm.cdf(lower, mu, sd) + alpha * mass, mu, sd)
    hi = norm.ppf(norm.cdf(upper, mu, sd) - alpha * mass, mu, sd)
    return np.array([max(lo, lower), mu, min(hi, upper)], dtype=float)


def IG_ci(mu, VAR, k=None, level: float = 0.95, precision: float = 0.5):
    del precision
    mu = float(mu)
    VAR = float(VAR)
    if k is None:
        k = VAR / max(mu**3, np.finfo(float).tiny)
    if np.isnan(level):
        return np.array([2.0 * mu * mu / VAR, mu, VAR], dtype=float)
    if k == np.inf:
        return np.array([0.0, mu, np.inf], dtype=float)
    if k <= 0:
        return np.array([mu, mu, mu], dtype=float)
    alpha = (1.0 - level) / 2.0
    shape = 1.0 / k
    # scipy parameterization: mean=mu_scale*scale, shape=mu_shape.
    vals = invgauss.ppf([alpha, 1.0 - alpha], mu=mu / shape, scale=shape)
    return np.array([vals[0], mu, vals[1]], dtype=float)


def chisq_IG_ci(M, VAR, w, level: float = 0.95, precision: float = 0.5):
    w = np.asarray([1.0 - w, w], dtype=float)
    c1 = chisq_ci(M, var=VAR, level=level)
    c2 = IG_ci(M, VAR=VAR, level=level, precision=precision)
    out = w[0] * c1 + w[1] * c2
    out[2] = np.inf if np.isnan(out[2]) else out[2]
    return out


def DD_IG_ratio(par, VAR, n):
    del par
    if VAR <= 0 or n <= 0:
        return 0.0
    return float(np.clip(1.0 / VAR, 0.0, n) / n)


def loglike_chisq(sigma, dof, constant: bool = False):
    df2 = float(dof) / 2.0
    r = -df2 * (np.log(sigma) + 1.0 / sigma)
    if constant:
        r = r + df2 * np.log(df2) - special.gammaln(df2)
    return r


def pfbinom(q, size, prob, lower_tail: bool = True, log_p: bool = False):
    x = (1.0 - prob) / prob * (q + 1.0 / size) / (1.0 - q)
    df1 = 2.0 * size * (1.0 - q)
    df2 = 2.0 * size * (q + 1.0 / size)
    p = scipy_f.cdf(x, df1, df2) if lower_tail else scipy_f.sf(x, df1, df2)
    return np.log(p) if log_p else p


def qfbinom(p, size, prob):
    from scipy.optimize import brentq

    return float(brentq(lambda q: pfbinom(q, size, prob) - p, np.finfo(float).eps, 1.0 - np.finfo(float).eps))


def rcov(x, *args, **kwargs):
    del args, kwargs
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    med = np.nanmedian(arr, axis=1)
    mad = np.nanmedian(np.abs(arr - med[:, None]), axis=1) / norm.ppf(0.75)
    cov = np.diag(np.nan_to_num(mad * mad, nan=0.0, posinf=0.0))
    return {"median": med, "COV": cov}


def mtmean(x, lower=-np.inf, upper=np.inf, func=np.mean):
    arr = np.sort(np.asarray(x, dtype=float))
    if arr.size == 0:
        return np.nan
    n = int(np.sum(arr <= lower))
    if n == arr.size:
        return lower
    if 2 * n >= arr.size:
        return float(arr[n])
    m = int(np.sum(arr >= upper))
    if m == arr.size:
        return upper
    if 2 * m >= arr.size:
        return float(arr[arr.size - m - 1])
    trim = max(n, m)
    if trim:
        arr = arr[trim:-trim]
    return float(func(arr))


def qmvnorm(p, dim: int = 1, tol: float = 0.5):
    del tol
    alpha = 1.0 - float(p)
    if dim == 1:
        return float(norm.ppf(1.0 - alpha / 2.0))
    if dim == 2:
        return float(math.sqrt(-2.0 * math.log(alpha)))
    return float(math.sqrt(chi2.ppf(float(p), df=dim)))


def X2(p, df=1):
    return idchisq(p, df)


def CDF(x, df=1):
    return chi2.cdf(x, df)


def p_fn(z, p=0.95, dim=3):
    del p
    return chi2.cdf(np.asarray(z) ** 2, dim)


def dp_fn(z, dim=3):
    return 2.0 * np.asarray(z) * chi2.pdf(np.asarray(z) ** 2, dim)


def dP(x, df=1):
    return chi2.cdf(x, df)


def dP2(x, df=1):
    return chi2.pdf(x, df)


def cost(*args, **kwargs):
    del args, kwargs
    return np.nan


def fn(*args, **kwargs):
    del args, kwargs
    return None


__all__ = [
    "NAMES_CI",
    "cov_loglike",
    "norm_ci",
    "chisq_ci",
    "lognorm_ci",
    "beta_ci",
    "F_CI",
    "Log_F_CI",
    "chisq_hdr",
    "idchisq",
    "chi_dof",
    "chi_var",
    "chi_bias",
    "chisq_dof",
    "tnorm_hdr",
    "IG_ci",
    "chisq_IG_ci",
    "DD_IG_ratio",
    "loglike_chisq",
    "pfbinom",
    "qfbinom",
    "rcov",
    "mtmean",
    "qmvnorm",
    "X2",
    "CDF",
    "p_fn",
    "dp_fn",
    "dP",
    "dP2",
    "cost",
    "fn",
]
