"""Positive-definite matrix helpers ported from ctmm 1.3.0 ``R/matrix.R``.

Numeric behavior targets R's ``pd.solve``, ``pd.logdet``, and ``pd.sqrtm``
(``sqrtm``) for real symmetric (or general) double matrices without ``covm``
S4 wrappers. Uses ``numpy`` / ``scipy`` only.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

try:
    from scipy.linalg import sqrtm as scipy_sqrtm
except ImportError:  # pragma: no cover
    scipy_sqrtm = None

_MACHINE_EPS = float(np.finfo(float).eps)


def _inv_eigenvalues(z: np.ndarray | float) -> np.ndarray:
    """Spectral inverse ``1/λ`` with ``1/0 → ∞`` (matches R ``function(m){1/m}`` on eigenvalues)."""
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.full(zz.shape, np.inf, dtype=float)
    np.divide(1.0, zz, out=out, where=zz != 0)
    return out


def _func_at_zero_for_PDfunc(func: Callable[[np.ndarray], np.ndarray]) -> float:
    """Evaluate ``func`` at eigenvalue 0 without Python division-by-zero for ``1/x``."""
    try:
        out = func(np.array([0.0]))
        return float(np.asarray(out, dtype=float).ravel()[0])
    except ZeroDivisionError:
        return float(np.inf)


def _scalar_map(func: Callable[[np.ndarray], np.ndarray], x: float) -> float:
    try:
        out = func(np.array([float(x)]))
        return float(np.asarray(out, dtype=float).ravel()[0])
    except ZeroDivisionError:
        if x == 0.0:
            return float(np.inf)
        raise


def _he(m: np.ndarray) -> np.ndarray:
    """Hermitian part; real matrices: ``(M + M.T) / 2`` (``R/matrix.R`` ``He``)."""
    m = np.asarray(m, dtype=float)
    return (m + m.T) / 2.0


def nant(x: np.ndarray, to: float | np.ndarray) -> np.ndarray:
    """Map NaN/NA to ``to`` (``R/generic.R`` ``nant``)."""
    x = np.asarray(x, dtype=float)
    out = x.copy()
    nan = np.isnan(out)
    if not np.any(nan):
        return out
    to_arr = np.asarray(to, dtype=float)
    if to_arr.shape == ():
        out[nan] = float(to_arr)
    else:
        to_arr = np.broadcast_to(to_arr, out.shape).astype(float, copy=False)
        out[nan] = to_arr[nan]
    return out


def _last(vec: np.ndarray) -> float:
    v = np.asarray(vec).ravel()
    if v.size == 0:
        return float("nan")
    return float(v[-1])


def eigen_extrapolate(vals: np.ndarray) -> np.ndarray:
    """``R/matrix.R`` ``eigen.extrapolate`` on eigenvalues (increasing order not assumed)."""
    m = np.asarray(vals, dtype=float).ravel()
    if m.size == 0:
        return m
    if np.all(m > 0):
        return m
    pos = m[m > 0]
    if pos.size == 0:
        return np.ones_like(m)  # R: LAST <- 1 if no positive values
    log_ratio = np.log(pos / pos[0])
    log_diff = np.diff(log_ratio)
    if log_diff.size:
        log_step = float(_last(log_diff))
    else:
        log_step = float(np.log(_MACHINE_EPS))
    last_pos = float(_last(pos))
    if not np.isfinite(last_pos) or last_pos == 0:
        last_pos = 1.0
    out = m.copy()
    bad_idx = np.where(m <= 0)[0]
    for k, j in enumerate(bad_idx):
        out[j] = last_pos * np.exp(log_step * float(k + 1))
    return out


def PDfunc(
    m: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    sym: bool = True,
    semi: bool = True,
    pseudo: bool = False,
    tol: float = _MACHINE_EPS,
) -> np.ndarray:
    """Spectral map ``f(M)`` for real PSD-ish matrices (``R/matrix.R`` ``PDfunc``)."""
    if func is None:
        func = _inv_eigenvalues

    m = np.asarray(m, dtype=float)
    if m.ndim == 0:
        m = np.reshape(m, (1, 1))
    if m.ndim == 1:
        m = np.reshape(m, (m.size, 1))

    dim = m.shape[0]
    if dim == 0:
        return m

    inf = np.diag(m) == np.inf
    if _func_at_zero_for_PDfunc(func) < np.inf:
        zero = np.array([np.all(m[i, :] == 0) for i in range(dim)], dtype=bool)
    else:
        zero = np.zeros(dim, dtype=bool)

    # R: ``if(DIM==1) { M <- c(M) } else if(any(INF)||any(ZERO)) { ... }``
    if dim == 1:
        m = np.reshape(m, (-1,))
    elif np.any(inf) or np.any(zero):
        if np.any(inf):
            if _scalar_map(func, float(np.inf)) == 0:
                m = m.copy()
                m[inf, :] = 0
                m[:, inf] = 0
            elif _scalar_map(func, float(np.inf)) == np.inf:
                m = m.copy()
                m[inf, :] = 0
                m[:, inf] = 0
                ii = np.ix_(inf, inf)
                m[ii] = np.inf
        if np.any(zero):
            m = m.copy()
            np.fill_diagonal(m, np.where(zero, _scalar_map(func, 0.0), np.diag(m)))
        rem = ~(inf | zero)
        if np.any(rem):
            idx = np.ix_(rem, rem)
            m = m.copy()
            m[idx] = PDfunc(m[idx], func, sym=sym, semi=semi, pseudo=pseudo, tol=tol)
        return m

    if dim == 1:
        eigvals = np.array([float(m[0])], dtype=float)
        v = np.ones((1, 1, 1))
    elif dim == 2:
        w, vecs = np.linalg.eigh(m)
        eigvals = np.real(w)
        v = np.stack([np.outer(vecs[:, i], vecs[:, i]) for i in range(2)], axis=2)
    elif dim > 2:
        if sym:
            w, vecs = np.linalg.eigh(m)
            eigvals = np.real(w)
            v = np.stack([np.outer(vecs[:, i], vecs[:, i]) for i in range(dim)], axis=2)
        else:
            w, vecs = np.linalg.eig(m)
            eigvals = np.real(w)
            vecs = np.real(vecs)
            v = np.stack([np.outer(vecs[:, i], vecs[:, i]) for i in range(dim)], axis=2)
    else:
        raise RuntimeError("unreachable")

    pseudo_mask = eigvals < tol
    ev = eigvals.copy()
    if not semi:
        ev = eigen_extrapolate(ev)
    if np.any(pseudo_mask) and pseudo:
        ev[pseudo_mask] = 0.0
    ev = func(ev)
    if np.any(pseudo_mask) and pseudo:
        ev[pseudo_mask] = 0.0

    if dim == 1:
        return np.reshape(ev, (1, 1))

    idx = np.argsort(np.abs(ev))
    acc = np.zeros((dim, dim))
    for i in idx:
        acc += nant(ev[i] * v[:, :, i], 0.0)
    return acc


def pd_solve(m: np.ndarray, *, sym: bool = True, semi: bool = True) -> np.ndarray:
    """``R/matrix.R`` ``pd.solve`` for real matrices."""
    m = np.asarray(m, dtype=float)
    if m.ndim == 0:
        m = np.reshape(m, (1, 1))
    dim = m.shape[0]
    if dim == 0:
        return m

    inf = np.diag(m) == np.inf
    zero = (np.diag(m) <= 0) & sym
    if np.any(inf) or np.any(zero):
        m = m.copy()
        if np.any(inf):
            m[inf, :] = 0
            m[:, inf] = 0
        if np.any(zero):
            m[zero, :] = 0
            m[:, zero] = 0
            np.fill_diagonal(m, np.where(zero, np.inf, np.diag(m)))
        rem = ~(inf | zero)
        if np.any(rem):
            idx = np.ix_(rem, rem)
            m[idx] = pd_solve(m[idx], sym=sym, semi=semi)
        return m

    if semi:
        if dim == 1:
            return np.array([[1.0 / m[0, 0]]])
        if dim == 2:
            det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
            if det <= 0:
                return np.diag(np.full(2, np.inf))
            return np.array([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]], dtype=float) / det

    if sym:
        m = _he(m)

    w = np.sqrt(np.abs(np.diag(m)))
    zero_w = w <= _MACHINE_EPS
    if np.any(zero_w):
        if np.any(~zero_w):
            w[zero_w] = np.min(w[~zero_w])
        else:
            w[zero_w] = 1.0
    w_outer = np.outer(w, w)
    m_scaled = m / w_outer

    inv_scaled: np.ndarray | None
    try:
        inv_scaled = np.linalg.solve(m_scaled, np.eye(dim))
    except np.linalg.LinAlgError:
        inv_scaled = None

    if inv_scaled is not None and (not sym or np.all(np.diag(inv_scaled) >= 0)):
        inv = inv_scaled
    else:
        inv = PDfunc(m_scaled, None, sym=sym, semi=semi)

    inv = inv / w_outer
    if sym:
        inv = _he(inv)
    return inv


def pd_logdet(m: np.ndarray, *, sym: bool = True, semi: bool = True) -> float:
    """``R/matrix.R`` ``pd.logdet`` (sum of logs of eigenvalues, clamped)."""
    m = np.asarray(m, dtype=float)
    if m.ndim == 0:
        m = np.reshape(m, (1, 1))
    dim = m.shape[0]
    if dim == 0:
        return 0.0

    inf = np.diag(m) == np.inf
    zero = (np.diag(m) <= 0) & sym
    if np.any(inf) or np.any(zero):
        sign = int(np.sum(inf) - np.sum(zero))
        if sign != 0:
            return float(sign * np.inf)
        rem = ~(inf | zero)
        if np.any(rem):
            idx = np.ix_(rem, rem)
            return float(pd_logdet(m[idx], sym=sym, semi=semi))
        return 0.0

    if sym:
        m = _he(m)
    vals = np.linalg.eigvalsh(m)
    vals = np.real(vals)
    vals = np.clip(vals, 0.0, np.inf)
    if not semi:
        vals = eigen_extrapolate(vals)
    return float(np.sum(np.log(vals)))


def pd_sqrtm(m: np.ndarray, *, semi: bool = True) -> np.ndarray:
    """``R/matrix.R`` ``sqrtm`` / ``pd.sqrtm`` for real matrices (no ``covm`` branch)."""
    m = np.asarray(m, dtype=float)
    if m.ndim == 0:
        dim = 1
        m = np.reshape(m, (1, 1))
    else:
        dim = m.shape[0]

    tol = dim * _MACHINE_EPS

    if dim == 1:
        v = float(m[0, 0])
        if v >= 0:
            return np.array([[np.sqrt(v)]])
        return np.array([[0.0]])

    if dim == 2:
        tr = m[0, 0] + m[1, 1]
        det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
        if det < 0 or tr * tr < 4 * det or np.any(np.diag(m) < 0):
            return PDfunc(m, lambda x: np.sqrt(np.maximum(x, 0.0)), semi=semi)
        s = float(np.sqrt(det))
        denom = np.sqrt(tr + 2 * s)
        out = (m + s * np.eye(2)) / denom
        return nant(out, 0.0)

    fail = np.diag(np.full(dim, -1.0))
    if np.all(np.diag(m) >= -tol):
        if scipy_sqrtm is None:
            r = fail
        else:
            try:
                r = scipy_sqrtm(m)
            except Exception:
                r = fail
    else:
        r = fail

    if np.all(np.real(np.diag(r)) >= -tol) and np.all(np.abs(np.imag(np.diag(r))) <= tol):
        out = np.real(r)
        bad = np.diag(out) <= 0
        if np.any(bad):
            out = out.copy()
            np.fill_diagonal(out, np.where(bad, 0.0, np.diag(out)))
        return out
    return PDfunc(m, lambda x: np.sqrt(np.maximum(x, 0.0)), semi=semi)


__all__ = [
    "PDfunc",
    "eigen_extrapolate",
    "nant",
    "pd_logdet",
    "pd_solve",
    "pd_sqrtm",
]
