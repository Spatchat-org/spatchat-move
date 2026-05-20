"""Parity translation of ctmm 1.3.0 ``R/matrix.R`` helpers."""

from __future__ import annotations

import numpy as np

from .pd_matrix import PDfunc, eigen_extrapolate, nant, pd_logdet, pd_solve, pd_sqrtm


def tr(x) -> float:
    return float(np.trace(np.atleast_2d(np.asarray(x, dtype=float))))


def isotrope(x) -> np.ndarray:
    m = np.atleast_2d(np.asarray(x, dtype=float))
    return np.eye(m.shape[0]) * float(np.mean(np.diag(m)))


def rotate(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=float)


def rotate_vec(z, theta: float) -> np.ndarray:
    return np.asarray(z, dtype=float) @ rotate(theta).T


def rotate_mat(M, theta: float) -> np.ndarray:
    arr = np.asarray(M, dtype=float)
    r = rotate(theta)
    if arr.ndim == 2:
        return r @ arr @ r.T
    return np.stack([r @ arr[i] @ r.T for i in range(arr.shape[0])], axis=0)


def rotates(theta) -> np.ndarray:
    th = np.asarray(theta, dtype=float).ravel()
    return np.stack([rotate(v) for v in th], axis=0)


def rotates_vec(z, R) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    r_arr = np.asarray(R, dtype=float)
    dim = z_arr.shape
    zz = z_arr.reshape(dim[0], 2, int(np.prod(dim[1:]) / 2))
    out = np.stack([r_arr[i] @ zz[i] for i in range(zz.shape[0])], axis=0)
    return out.reshape(dim)


def rotates_mat(M, R) -> np.ndarray:
    m = np.asarray(M, dtype=float)
    r = np.asarray(R, dtype=float)
    return np.stack([r[i] @ m[i] @ r[i].T for i in range(m.shape[0])], axis=0)


def squeeze(z, smgm: float) -> np.ndarray:
    out = np.asarray(z, dtype=float).copy()
    out[:, 0] = out[:, 0] / float(smgm)
    out[:, 1] = out[:, 1] * float(smgm)
    return out


def squeeze_mat(M, smgm: float) -> np.ndarray:
    arr = np.asarray(M, dtype=float).copy()
    scale = np.array([1.0 / float(smgm), float(smgm)], dtype=float)
    return arr * scale[None, :, None] * scale[None, None, :]


def det2(x) -> float:
    m = np.asarray(x, dtype=float)
    return float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])


def outer(X, Y=None, FUN="*", **kwargs):
    if Y is None:
        Y = X
    x = np.asarray(X)
    y = np.asarray(Y)
    if callable(FUN):
        return np.vectorize(lambda a, b: FUN(a, b, **kwargs))(x[:, None], y[None, :])
    if kwargs:
        raise TypeError(f"outer got unused keyword arguments for FUN={FUN!r}: {sorted(kwargs)}")
    ops = {
        "*": np.multiply,
        "+": np.add,
        "-": np.subtract,
        "/": np.divide,
        "^": np.power,
        "**": np.power,
        "==": np.equal,
        "!=": np.not_equal,
        "<": np.less,
        "<=": np.less_equal,
        ">": np.greater,
        ">=": np.greater_equal,
        "pmin": np.minimum,
        "pmax": np.maximum,
        "min": np.minimum,
        "max": np.maximum,
    }
    fn = ops.get(str(FUN))
    if fn is None:
        raise ValueError(f"unsupported outer FUN={FUN!r}")
    return fn(x[:, None], y[None, :])


def riffle(*args, by: int = 1):
    if by > 1:
        if len(args) != 2:
            raise ValueError("riffle.by expects two matrices")
        return riffle_by(args[0], args[1], by=by)
    mats = [np.atleast_2d(np.asarray(a)) for a in args]
    rows, cols = mats[0].shape
    cat = np.vstack(mats)
    return cat.reshape(rows, cols * len(mats), order="F")


def riffle_by(u, v, by: int = 1):
    uu = np.asarray(u)
    vv = np.asarray(v)
    rows, cols = uu.shape
    parts = []
    for i in range(0, cols, by):
        parts.append(np.hstack([uu[:, i : i + by], vv[:, i : i + by]]))
    return np.hstack(parts).reshape(rows, 2 * cols)


def viffle(*args):
    return np.vstack([np.asarray(a).reshape(1, -1) for a in args]).reshape(-1, order="F")


def Adj(M) -> np.ndarray:
    return np.asarray(M).conj().T


def He(M) -> np.ndarray:
    m = np.asarray(M)
    return (m + Adj(m)) / 2.0


def sqrtm(M, semi: bool = True, **kwargs) -> np.ndarray:
    del kwargs
    return pd_sqrtm(np.asarray(M, dtype=float), semi=semi)


def isqrtm(M, semi: bool = True, **kwargs) -> np.ndarray:
    del kwargs
    return pd_solve(pd_sqrtm(np.asarray(M, dtype=float), semi=semi), semi=semi)


def fixInf(M) -> np.ndarray:
    out = np.asarray(M, dtype=float).copy()
    inf = np.isposinf(out)
    diag = np.eye(out.shape[0], dtype=bool)
    out[inf] = 0.0
    out[inf & diag] = float("inf")
    return out


def fixNaN(M) -> np.ndarray:
    out = np.asarray(M, dtype=float).copy()
    nan = np.isnan(out)
    diag = np.eye(out.shape[0], dtype=bool)
    out[nan] = 0.0
    out[nan & diag] = float("inf")
    return out


def conditionNumber(M) -> float:
    try:
        vals = np.linalg.eigvalsh(np.asarray(M, dtype=float))
        vals = np.sort(vals)
        return float(nant(np.array([vals[0] / vals[-1]]), float("inf"))[0])
    except Exception:
        return float("inf")


def PDclamp(M, lower: float = 0.0, upper: float = float("inf"), **kwargs) -> np.ndarray:
    del kwargs
    m = He(np.asarray(M, dtype=float))
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, lower, upper)
    return He(vecs @ np.diag(vals) @ vecs.T)


def mat_min(M) -> float:
    m = np.asarray(M, dtype=float).copy()
    if np.any(np.isnan(m)) or np.any(np.isinf(np.abs(m))) or np.any(np.diag(m) == 0):
        return 0.0
    np.fill_diagonal(m, np.abs(np.diag(m)))
    d = np.sqrt(np.diag(m))
    corr = m / np.outer(d, d)
    return float(np.min(np.linalg.eigvalsh(corr)))


def ext_mat(*mats, MAX: bool = True) -> np.ndarray:
    eigs = [np.linalg.eigh(np.asarray(m, dtype=float)) for m in mats]
    vals = np.concatenate([e[0] for e in eigs])
    vecs = np.column_stack([e[1] for e in eigs])
    order = np.argsort(vals)
    if MAX:
        order = order[::-1]
    dim = np.asarray(mats[0]).shape[0]
    vals = vals[order]
    vecs = vecs[:, order]
    out = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        v = vecs[:, i]
        out += vals[i] * np.outer(v, v)
        for j in range(i + 1, vecs.shape[1]):
            vecs[:, j] = vecs[:, j] - float(v @ vecs[:, j]) * v
    return out


def mvrnorm(Mu, Sigma) -> np.ndarray:
    mu = np.asarray(Mu, dtype=float).reshape(-1)
    sigma = np.asarray(Sigma, dtype=float)
    scale = np.sqrt(np.abs(np.diag(sigma)))
    scale[scale <= np.finfo(float).eps] = 1.0
    corr = sigma / np.outer(scale, scale)
    root = pd_sqrtm(corr)
    return mu + scale * (root @ np.random.normal(size=mu.size))


pd_sqrtm_alias = pd_sqrtm


def det_numeric(x, *args, **kwargs):
    del args, kwargs
    return float(x)


def determinant_numeric(x, logarithm: bool = True, *args, **kwargs):
    del args, kwargs
    val = float(x)
    sign = 1.0 if val >= 0 else -1.0
    modulus = np.log(abs(val)) if logarithm else abs(val)
    return {"modulus": modulus, "sign": sign}

__all__ = [
    "Adj",
    "He",
    "PDclamp",
    "PDfunc",
    "conditionNumber",
    "det2",
    "det_numeric",
    "determinant_numeric",
    "eigen_extrapolate",
    "ext_mat",
    "fixInf",
    "fixNaN",
    "isotrope",
    "isqrtm",
    "mat_min",
    "mvrnorm",
    "outer",
    "pd_logdet",
    "pd_solve",
    "pd_sqrtm",
    "riffle",
    "riffle_by",
    "rotate",
    "rotate_mat",
    "rotate_vec",
    "rotates",
    "rotates_mat",
    "rotates_vec",
    "sqrtm",
    "squeeze",
    "squeeze_mat",
    "tr",
    "viffle",
]
