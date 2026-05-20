from __future__ import annotations

import numpy as np
import pandas as pd

from .types import Telemetry


def _ensure_uere(telem: Telemetry):
    u = telem.metadata.get("UERE")
    if isinstance(u, dict):
        return u
    u = {
        "UERE": {"horizontal": 1.0},
        "DOF": {"horizontal": 0.0},
        "AICc": {"horizontal": float("inf")},
        "Zsq": {"horizontal": float("inf")},
        "VAR.Zsq": {"horizontal": float("inf")},
        "N": {"horizontal": 0.0},
    }
    telem.metadata["UERE"] = u
    return u


def uere(data):
    if isinstance(data, Telemetry):
        return _ensure_uere(data)
    if isinstance(data, list):
        vals = [uere(d) for d in data]
        first = vals[0] if vals else None
        if all(v == first for v in vals):
            return first
        return vals
    raise TypeError("uere expects Telemetry or list[Telemetry]")


def uere_set(data, value):
    if isinstance(data, list):
        return [uere_set(d, value) for d in data]
    if not isinstance(data, Telemetry):
        raise TypeError("uere<- expects Telemetry or list[Telemetry]")
    out = Telemetry(
        data=data.data.copy(),
        id_col=data.id_col,
        time_col=data.time_col,
        x_col=data.x_col,
        y_col=data.y_col,
        crs=data.crs,
        metadata=dict(data.metadata),
    )
    u = _ensure_uere(out)
    if value is None:
        u["UERE"]["horizontal"] = np.nan
        u["DOF"]["horizontal"] = 0.0
        return out
    if isinstance(value, (int, float)):
        u["UERE"]["horizontal"] = float(value)
        u["DOF"]["horizontal"] = float("inf")
        u["N"]["horizontal"] = float("inf")
        return out
    if isinstance(value, dict):
        out.metadata["UERE"] = value
        return out
    raise TypeError("unsupported uere<- value")


def uere_fit(data, precision: float = 0.5):
    tracks = data if isinstance(data, list) else [data]
    prepared = []
    classes: list[str] = []
    for t in tracks:
        if not isinstance(t, Telemetry):
            continue
        df = t.data.sort_values(t.time_col)
        x = df[t.x_col].to_numpy(dtype=float)
        y = df[t.y_col].to_numpy(dtype=float)
        if "HDOP" in df.columns:
            dop = pd.to_numeric(df["HDOP"], errors="coerce").to_numpy(dtype=float)
        else:
            dop = np.ones(len(df), dtype=float)
        if "class" in df.columns:
            cls = df["class"].astype(str).to_numpy()
        else:
            cls = np.repeat("all", len(df))
        keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(dop) & (dop < np.inf) & (dop > 0)
        if np.count_nonzero(keep) < 2:
            continue
        z = np.column_stack([x[keep], y[keep]])
        dop = dop[keep]
        cls = cls[keep]
        prepared.append((z, dop, cls))
        for c in cls:
            if c not in classes:
                classes.append(str(c))
    if not prepared:
        est = 1.0
        return {
            "UERE": {"horizontal": est, "class": {"all": est}},
            "DOF": {"horizontal": 0.0, "class": {"all": 0.0}},
            "AICc": {"horizontal": float("nan")},
            "Zsq": {"horizontal": float("nan")},
            "VAR.Zsq": {"horizontal": float("nan")},
            "N": {"horizontal": 0.0},
        }

    class_index = {c: i for i, c in enumerate(classes)}
    cnum = len(classes)
    axes = 2.0
    tol = np.finfo(float).eps ** float(precision)
    uere_vals = np.full(cnum, 10.0, dtype=float)
    fixed = np.array(["argos" in c.lower() for c in classes], dtype=bool)
    uere_vals[fixed] = 1.0
    est_mask = ~fixed
    dof_ml = np.zeros(cnum, dtype=float)
    z_list = []
    w_list = []
    ci_list = []
    cim_list = []
    for z, dop, cls in prepared:
        ci = np.asarray([class_index[str(c)] for c in cls], dtype=int)
        cim = np.zeros((ci.size, cnum), dtype=float)
        cim[np.arange(ci.size), ci] = 1.0
        w = axes / np.maximum(dop * dop, np.finfo(float).tiny)
        dof_ml += np.sum(cim, axis=0)
        z_list.append(z)
        w_list.append(w)
        ci_list.append(ci)
        cim_list.append(cim)
    missing = dof_ml <= 0
    uere_vals[missing] = np.inf
    est_mask[missing] = False

    Kc = np.zeros(cnum, dtype=float)
    Pck = np.zeros((cnum, len(prepared)), dtype=float)
    Pk = np.ones(len(prepared), dtype=float)
    mu = np.zeros((len(prepared), 2), dtype=float)
    while True:
        inv_u2 = np.zeros_like(uere_vals)
        good = np.isfinite(uere_vals) & (uere_vals > 0)
        inv_u2[good] = 1.0 / (uere_vals[good] ** 2)
        for k, (z, w, ci, cim) in enumerate(zip(z_list, w_list, ci_list, cim_list)):
            Pck[:, k] = (w @ cim) * inv_u2
            Pk[k] = max(float(np.sum(Pck[:, k])), np.finfo(float).tiny)
            precision_i = w * inv_u2[ci]
            mu[k] = (z.T @ precision_i) / Pk[k]
        Kc = np.sum(Pck / Pk[None, :], axis=1)
        dof = np.maximum(dof_ml - Kc, np.finfo(float).tiny)
        numer = np.zeros(cnum, dtype=float)
        for k, (z, w, ci, cim) in enumerate(zip(z_list, w_list, ci_list, cim_list)):
            d2 = np.sum((z - mu[k]) ** 2, axis=1)
            numer += d2 @ (w[:, None] * cim)
        updated = np.sqrt(numer / (axes * dof))
        updated[~est_mask] = uere_vals[~est_mask]
        rel = np.abs(updated[est_mask] - uere_vals[est_mask]) / np.maximum(updated[est_mask], uere_vals[est_mask])
        err = float(np.nanmax(np.nan_to_num(rel, nan=0.0))) if rel.size else 0.0
        uere_vals = updated
        if err < tol:
            break

    dof = np.maximum(dof_ml - Kc, 0.0)
    if np.any(missing):
        uere_vals[missing] = np.nan
        dof[missing] = 0.0
    aicc = 0.0
    for w, ci in zip(w_list, ci_list):
        v = (2.0 * np.pi) * (uere_vals[ci] ** 2 / w)
        aicc += float(np.nansum(np.log(v)))
    aicc *= axes
    finite_est = est_mask & np.isfinite(dof) & (axes * dof > 2)
    if np.any(finite_est):
        dof_aicc = np.maximum(dof_ml - np.sum((Pck / Pk[None, :]) ** 2, axis=1), 0.0)
        aicc += axes * axes * float(np.nansum(((dof_ml + Kc) * dof_aicc / np.maximum(axes * dof_aicc - 2.0, np.finfo(float).tiny))[finite_est]))
    else:
        aicc = float("inf")
    class_uere = {c: float(uere_vals[i]) for c, i in class_index.items()}
    class_dof = {c: float(dof[i]) for c, i in class_index.items()}
    finite_vals = [v for v in class_uere.values() if np.isfinite(v)]
    est = float(np.nanmedian(finite_vals)) if finite_vals else 1.0
    total_dof = float(np.nansum([v for v in class_dof.values() if np.isfinite(v)]))
    return {
        "UERE": {"horizontal": est, "class": class_uere},
        "DOF": {"horizontal": total_dof, "class": class_dof},
        "AICc": {"horizontal": float(aicc)},
        "Zsq": {"horizontal": float("nan")},
        "VAR.Zsq": {"horizontal": float("nan")},
        "N": {"horizontal": total_dof},
    }
