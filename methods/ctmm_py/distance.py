"""Partial parity translation of ctmm 1.3.0 ``R/distance.R``."""

from __future__ import annotations

import numpy as np

from .overlap_distance_ops import (
    _gauss_pair,
    distance_matrix as _distance_matrix,
    distance_pair as _distance_pair,
)


def _sigma_mu(CTMM):
    mu1, s1 = _gauss_pair(CTMM[0])
    mu2, s2 = _gauss_pair(CTMM[1])
    sigma = 0.5 * (s1 + s2)
    mu = mu1 - mu2
    return sigma, mu, s1, s2


def BhattacharyyaD(CTMM):
    return float(_distance_pair(CTMM[0], CTMM[1], method="Bhattacharyya", sqrt=False)["est"])


def RateD(CTMM):
    sigma, mu, _, _ = _sigma_mu(CTMM)
    inv = np.linalg.pinv(sigma)
    d = float(mu.T @ inv @ mu) / 4.0 + float(np.linalg.slogdet(sigma)[1]) / 2.0 + sigma.shape[0] / 2.0 * np.log(4.0 * np.pi)
    return float(d)


def EncounterD(CTMM):
    sigma, mu, s1, s2 = _sigma_mu(CTMM)
    inv = np.linalg.pinv(sigma)
    det_sigma = max(np.linalg.det(sigma), np.finfo(float).tiny)
    det1 = max(np.linalg.det(s1), np.finfo(float).tiny)
    det2 = max(np.linalg.det(s2), np.finfo(float).tiny)
    return float((mu.T @ inv @ mu) / 4.0 + np.log(det_sigma / np.sqrt(det1 * det2)) / 2.0)


def MahalanobisD(CTMM):
    return float(_distance_pair(CTMM[0], CTMM[1], method="Mahalanobis", sqrt=False)["est"])


def EuclideanD(CTMM):
    return float(_distance_pair(CTMM[0], CTMM[1], method="Euclidean", sqrt=False)["est"])


def distance(object, method: str = "Mahalanobis", sqrt: bool = False, level: float = 0.95, debias: bool = True, **kwargs):
    del level, debias, kwargs
    objs = object if isinstance(object, list) else [object]
    return _distance_matrix(objs, method=method, sqrt=sqrt)


__all__ = [
    "BhattacharyyaD",
    "RateD",
    "EncounterD",
    "MahalanobisD",
    "EuclideanD",
    "distance",
]
