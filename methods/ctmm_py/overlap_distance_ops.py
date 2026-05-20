from __future__ import annotations

import math
from typing import Any

import numpy as np

from .types import CTMMModel, Telemetry


def _from_telemetry(t: Telemetry):
    x = t.data[t.x_col].to_numpy(dtype=float)
    y = t.data[t.y_col].to_numpy(dtype=float)
    m = np.array([np.nanmean(x), np.nanmean(y)], dtype=float)
    xy = np.column_stack([x, y])
    if len(xy) < 2:
        s = np.eye(2, dtype=float)
    else:
        s = np.cov(xy.T)
        if s.shape != (2, 2) or not np.all(np.isfinite(s)):
            s = np.eye(2, dtype=float)
    s = s + np.eye(2) * 1e-9
    return m, s


def _from_ctmm(m: CTMMModel):
    mu = m.params.get("mu")
    if mu is None:
        mu_vec = np.zeros(2, dtype=float)
    else:
        mu_arr = np.asarray(mu, dtype=float).reshape(-1)
        if mu_arr.size >= 2:
            mu_vec = mu_arr[:2]
        else:
            mu_vec = np.array([float(mu_arr[0]), 0.0], dtype=float)
    sigma = m.params.get("sigma")
    if hasattr(sigma, "to_matrix"):
        s = np.asarray(sigma.to_matrix(), dtype=float)
    elif hasattr(sigma, "sigma"):
        s = np.asarray(sigma.sigma, dtype=float)
    elif sigma is not None:
        s = np.asarray(sigma, dtype=float)
    else:
        s = np.eye(2, dtype=float)
    if s.shape != (2, 2) or not np.all(np.isfinite(s)):
        s = np.eye(2, dtype=float)
    s = s + np.eye(2) * 1e-9
    return mu_vec, s


def _gauss_pair(obj: Any):
    if isinstance(obj, Telemetry):
        return _from_telemetry(obj)
    if isinstance(obj, CTMMModel):
        return _from_ctmm(obj)
    raise TypeError("Expected Telemetry or CTMMModel")


def _mahalanobis2(mu1, s1, mu2, s2):
    sigma = 0.5 * (s1 + s2)
    dmu = mu1 - mu2
    inv = np.linalg.pinv(sigma)
    return float(dmu.T @ inv @ dmu)


def _bhattacharyya(mu1, s1, mu2, s2):
    sigma = 0.5 * (s1 + s2)
    dmu = mu1 - mu2
    inv = np.linalg.pinv(sigma)
    t1 = 0.125 * float(dmu.T @ inv @ dmu)
    det_sigma = max(np.linalg.det(sigma), 1e-18)
    det1 = max(np.linalg.det(s1), 1e-18)
    det2 = max(np.linalg.det(s2), 1e-18)
    t2 = 0.5 * math.log(det_sigma / math.sqrt(det1 * det2))
    return float(max(t1 + t2, 0.0))


def _encounter_distance(mu1, s1, mu2, s2):
    sigma = 0.5 * (s1 + s2)
    dmu = mu1 - mu2
    inv = np.linalg.pinv(sigma)
    t1 = 0.25 * float(dmu.T @ inv @ dmu)
    det_sigma = max(np.linalg.det(sigma), 1e-18)
    det1 = max(np.linalg.det(s1), 1e-18)
    det2 = max(np.linalg.det(s2), 1e-18)
    t2 = 0.5 * math.log(det_sigma / math.sqrt(det1 * det2))
    return float(max(t1 + t2, 0.0))


def _rate_distance(mu1, s1, mu2, s2):
    sigma = 0.5 * (s1 + s2)
    dmu = mu1 - mu2
    inv = np.linalg.pinv(sigma)
    return float(0.25 * (dmu.T @ inv @ dmu) + 0.5 * math.log(max(np.linalg.det(sigma), 1e-18)) + sigma.shape[0] / 2.0 * math.log(4.0 * math.pi))


def distance_pair(a: Any, b: Any, method: str = "Mahalanobis", sqrt: bool = False):
    mu1, s1 = _gauss_pair(a)
    mu2, s2 = _gauss_pair(b)
    m = method.lower()
    if m == "euclidean":
        d2 = float(np.sum((mu1 - mu2) ** 2))
    elif m == "bhattacharyya":
        d2 = _bhattacharyya(mu1, s1, mu2, s2)
    elif m == "encounter":
        d2 = _encounter_distance(mu1, s1, mu2, s2)
    elif m == "rate":
        d2 = _rate_distance(mu1, s1, mu2, s2)
    else:
        d2 = _mahalanobis2(mu1, s1, mu2, s2)
    est = math.sqrt(max(d2, 0.0)) if sqrt else d2
    return {"low": est, "est": est, "high": est, "DOF": float("inf")}


def distance_matrix(objects: list[Any], method: str = "Mahalanobis", sqrt: bool = False):
    n = len(objects)
    dof = np.full((n, n), np.inf, dtype=float)
    ci = np.zeros((n, n, 3), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = distance_pair(objects[i], objects[j], method=method, sqrt=sqrt)
            ci[i, j, :] = [d["low"], d["est"], d["high"]]
    return {"DOF": dof, "CI": ci}


def overlap_pair(a: Any, b: Any, method: str = "Bhattacharyya"):
    m = method.lower()
    if m == "rate":
        d = distance_pair(a, b, method="Rate", sqrt=False)["est"]
        o = float(np.exp(-max(d, 0.0)))
    else:
        d = distance_pair(a, b, method=method, sqrt=False)["est"]
        o = float(np.exp(-max(d, 0.0)))
    return {"low": o, "est": o, "high": o, "DOF": float("inf")}


def overlap_matrix(objects: list[Any], method: str = "Bhattacharyya"):
    n = len(objects)
    dof = np.full((n, n), np.inf, dtype=float)
    ci = np.ones((n, n, 3), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            o = overlap_pair(objects[i], objects[j], method=method)
            ci[i, j, :] = [o["low"], o["est"], o["high"]]
            ci[j, i, :] = ci[i, j, :]
    return {"DOF": dof, "CI": ci}
