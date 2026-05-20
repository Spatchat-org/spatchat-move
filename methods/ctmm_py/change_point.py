"""Partial parity translation of ctmm 1.3.0 ``R/change.point.R``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pd_matrix import pd_logdet


def _as_df(data):
    if hasattr(data, "data"):
        return data.data
    return data


def _get_xy(data, axes=("x", "y")):
    df = _as_df(data)
    if hasattr(data, "x_col") and hasattr(data, "y_col"):
        cols = (data.x_col, data.y_col)
    else:
        cols = axes
    z = df.loc[:, list(cols)].to_numpy(dtype=float)
    return z


def change_point_iid(data, axes=("x", "y"), IC: str = "AICc"):
    z = _get_xy(data, axes=axes)
    n, ax = z.shape
    if n < 4:
        return {"MIN": 0, "IC": 0.0}

    m1 = z.copy()
    s1 = np.zeros((n, ax, ax), dtype=float)
    for i in range(1, n):
        k = i + 1
        m1[i] = ((k - 1) * m1[i - 1] + z[i]) / k
        d = z[i] - m1[i - 1]
        s1[i] = s1[i - 1] + (k - 1) / k * np.outer(d, d)
    n1 = np.arange(1, n + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = s1 / np.maximum(n1[:, None, None] - 1.0, 1.0)

    m2 = z.copy()
    s2 = np.zeros((n, ax, ax), dtype=float)
    for i in range(1, n):
        j = n - i - 1
        k = i + 1
        m2[j] = ((k - 1) * m2[j + 1] + z[j]) / k
        d = z[j] - m2[j + 1]
        s2[j] = s2[j + 1] + (k - 1) / k * np.outer(d, d)
    n2 = np.arange(n, 0, -1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = s2 / np.maximum(n2[:, None, None] - 1.0, 1.0)

    s1[0] = np.eye(ax) * np.inf
    s2[-1] = np.eye(ax) * np.inf

    with np.errstate(divide="ignore", invalid="ignore"):
        nllu1 = ax * (n1 * (np.sum(np.log(np.clip(np.stack([s1[:, d, d] for d in range(ax)], axis=1), 1e-300, np.inf)), axis=1) + 1.0) - 1.0)
        nllu2 = ax * (n2 * (np.sum(np.log(np.clip(np.stack([s2[:, d, d] for d in range(ax)], axis=1), 1e-300, np.inf)), axis=1) + 1.0) - 1.0)
        nllc1 = ax * (n1 * (np.array([pd_logdet(s1[i]) for i in range(n)]) + 1.0) - 1.0)
        nllc2 = ax * (n2 * (np.array([pd_logdet(s2[i]) for i in range(n)]) + 1.0) - 1.0)

    nllu1[0] = np.inf
    nllu2[-1] = np.inf
    kpar = ax + (ax * ax + ax) / 2.0
    mx = int(np.floor(kpar / ax))
    nllc1[:mx] = np.inf
    nllc2[n - mx :] = np.inf

    icu1 = nllu1.copy()
    icu2 = nllu2.copy()
    icc1 = nllc1.copy()
    icc2 = nllc2.copy()
    if IC == "AIC":
        icu1 += 2.0 * (2.0 * ax)
        icu2 += 2.0 * (2.0 * ax)
        icc1 += 2.0 * kpar
        icc2 += 2.0 * kpar
    elif IC == "BIC":
        icu1 += np.log(n) * (2.0 * ax)
        icu2 += np.log(n) * (2.0 * ax)
        icc1 += np.log(n) * kpar
        icc2 += np.log(n) * kpar
    else:
        icu1 += ax * (n1 - 1.0) * 4.0 / np.maximum(n1 - 3.0, 1.0)
        icu2 += ax * (n2 - 1.0) * 4.0 / np.maximum(n2 - 3.0, 1.0)
        icc1 += ax * (n1 - 1.0) * (ax + 3.0) / np.maximum(n1 - ax - 2.0, 1.0)
        icc2 += ax * (n2 - 1.0) * (ax + 3.0) / np.maximum(n2 - ax - 2.0, 1.0)

    ic1 = np.minimum(icu1, icc1)
    ic2 = np.minimum(icu2, icc2)
    total = np.r_[0.0, ic1] + np.r_[ic2, 0.0]
    idx = int(np.argmin(total))
    min_cp = idx
    delta = float(total[idx] - total[0])
    return {"MIN": min_cp, "IC": delta}


def ctmm_commute(models):
    ang = []
    for m in models:
        if isinstance(m, dict):
            a = m.get("angle", m.get("sigma_angle", 0.0))
        else:
            a = 0.0
        ang.append(float(a))
    ang = np.asarray(ang, dtype=float)
    if ang.size <= 1:
        return True
    return bool(not np.any(np.abs(np.diff(ang)) > np.finfo(float).eps))


def change_point_guess(data, n: int = 1, axes=("x", "y"), **kwargs):
    del kwargs
    df = _as_df(data)
    cp = []
    nrow = len(df)
    for _ in range(max(int(n), 0)):
        ap = np.unique(np.r_[0, cp, nrow]).astype(int)
        m = len(ap) - 1
        mins = np.zeros(m, dtype=int)
        ics = np.full(m, np.inf, dtype=float)
        for j in range(m):
            sub = df.iloc[ap[j] : ap[j + 1]]
            stuff = change_point_iid(sub, axes=axes)
            mins[j] = int(stuff["MIN"] + ap[j])
            ics[j] = float(stuff["IC"])
        valid = mins > 0
        if not np.any(valid):
            break
        mm = mins[valid]
        ii = ics[valid]
        cp.append(int(mm[np.argmin(ii)]))
        cp = sorted(set(cp))
    tcol = "t" if "t" in df.columns else None
    change_t = df.iloc[cp][tcol].to_numpy() if (tcol and len(cp)) else np.array([], dtype=float)
    return {
        "axes": tuple(axes),
        "dynamics": "change.point",
        "change.point": change_t,
        "indices": np.asarray(cp, dtype=int),
        "commute": True,
    }


def get_max(x):
    """Return the 1-based index of the first maximum, matching R helper use."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return None
    return int(np.nanargmax(arr) + 1)


def change_point_set(CTMM):
    c = dict(CTMM) if isinstance(CTMM, dict) else {"value": CTMM}
    c["dynamics"] = "change.point"
    return c


__all__ = ["change_point_iid", "change_point_guess", "ctmm_commute", "get_max", "change_point_set"]
