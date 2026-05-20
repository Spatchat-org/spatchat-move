"""Parity-focused translation of ctmm 1.3.0 ``R/uere.R`` helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import Telemetry
from .uere_ops import uere, uere_fit, uere_set

DOP_LIST = {
    "unknown": {"axes": None, "geo": None, "DOP": None, "VAR": None, "COV": None, "COV.geo": None, "units": None},
    "horizontal": {"axes": ["x", "y"], "geo": ["longitude", "latitude"], "DOP": "HDOP", "VAR": "VAR.xy", "COV": ["COV.x.x", "COV.x.y", "COV.y.y"], "COV.geo": ["COV.major", "COV.minor", "COV.angle"], "units": "distance"},
    "vertical": {"axes": ["z"], "geo": ["z"], "DOP": "VDOP", "VAR": "VAR.z", "COV": None, "COV.geo": None, "units": "distance"},
    "speed": {"axes": ["vx", "vy"], "geo": ["speed", "heading"], "DOP": "SDOP", "VAR": "VAR.v", "COV": ["COV.vx.vx", "COV.vx.vy", "COV.vy.vy"], "COV.geo": None, "units": "speed"},
    "frequency": {"axes": ["f"], "geo": None, "DOP": None, "VAR": None, "COV": None, "COV.geo": None, "units": "frequency"},
    "mass": {"axes": ["m"], "geo": None, "DOP": None, "VAR": None, "COV": None, "COV.geo": None, "units": "mass"},
}


def is_calibrated(data, type: str = "horizontal"):
    if isinstance(data, list):
        vals = [is_calibrated(d, type=type) for d in data]
        return float(np.mean(vals)) if vals else 0.0
    u = uere(data)
    dof = u.get("DOF", {}) if isinstance(u, dict) else {}
    if isinstance(dof.get("class"), dict):
        vals = [v for v in dof["class"].values() if v is not None]
        return float(np.mean([np.isfinite(v) and v > 0 for v in vals])) if vals else 0.0
    val = dof.get(type, dof.get("horizontal", 0.0))
    return float(np.isfinite(val) and float(val) > 0)


def DOP_match(axes):
    axes = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    for name, spec in DOP_LIST.items():
        if name == "unknown":
            continue
        if spec["axes"] == axes:
            return name
    return "unknown"


def get_dop_types(data):
    tracks = data if isinstance(data, list) else [data]
    names = set()
    for d in tracks:
        if isinstance(d, Telemetry):
            names.update(d.data.columns)
        elif hasattr(d, "columns"):
            names.update(d.columns)
    out = []
    for name, spec in DOP_LIST.items():
        if name == "unknown":
            continue
        axes = set(spec["axes"] or [])
        geo = set(spec["geo"] or [])
        if axes.issubset(names) or geo.issubset(names):
            out.append(name)
    return out


def classnames(object):
    u = object.get("UERE", object) if isinstance(object, dict) else {}
    class_map = u.get("class") if isinstance(u, dict) else None
    if isinstance(class_map, dict):
        return list(class_map.keys())
    return ["all"]


def classnames_set(object, value):
    out = dict(object) if isinstance(object, dict) else {"UERE": object}
    u = out.setdefault("UERE", {})
    if not isinstance(u, dict):
        u = {"horizontal": u}
        out["UERE"] = u
    vals = list(value)
    current = u.get("class", {})
    if not isinstance(current, dict):
        current = {}
    u["class"] = {str(v): float(current.get(v, 1.0)) for v in vals}
    return out


def typenames_set(object, value):
    out = dict(object) if isinstance(object, dict) else {"UERE": object}
    u = out.setdefault("UERE", {})
    if not isinstance(u, dict):
        u = {"horizontal": u}
        out["UERE"] = u
    for v in value:
        u.setdefault(str(v), 1.0)
    return out


def get_class(data):
    df = data.data if isinstance(data, Telemetry) else pd.DataFrame(data)
    if "class" in df.columns:
        return df["class"].astype(str).to_numpy()
    return np.repeat("all", len(df))


def get_class_mat(data, classes=None):
    cls = get_class(data)
    if classes is None:
        classes = list(dict.fromkeys(cls.tolist())) or ["all"]
    idx = {str(c): i for i, c in enumerate(classes)}
    mat = np.zeros((cls.size, len(classes)), dtype=float)
    for i, c in enumerate(cls):
        if str(c) in idx:
            mat[i, idx[str(c)]] = 1.0
    return mat


def get_UERE_DOF(x):
    if isinstance(x, dict):
        dof = x.get("DOF", 0.0)
        if isinstance(dof, dict):
            return float(dof.get("horizontal", 0.0))
        try:
            return float(dof)
        except Exception:
            return 0.0
    return 0.0


def uere_null(data):
    tracks = data if isinstance(data, list) else [data]
    classes = []
    for tr in tracks:
        classes.extend(get_class(tr).tolist())
    classes = list(dict.fromkeys(classes)) or ["all"]
    types = get_dop_types(data) or ["horizontal"]
    class_uere = {c: 1.0 for c in classes}
    class_dof = {c: 0.0 for c in classes}
    return {
        "UERE": {typ: 1.0 for typ in types} | {"horizontal": 1.0, "class": class_uere},
        "DOF": {typ: 0.0 for typ in types} | {"horizontal": 0.0, "class": class_dof},
        "AICc": {typ: float("inf") for typ in types},
        "Zsq": {typ: float("inf") for typ in types},
        "VAR.Zsq": {typ: float("inf") for typ in types},
        "N": {typ: 0.0 for typ in types},
    }


def try_assign_uere(data, UERE, TYPE: str = "horizontal"):
    spec = DOP_LIST.get(TYPE, DOP_LIST["horizontal"])
    df = data.data.copy() if isinstance(data, Telemetry) else pd.DataFrame(data).copy()
    if isinstance(UERE, dict) and isinstance(UERE.get("UERE"), dict):
        UERE = UERE["UERE"]
    if "class" in df.columns:
        cls = df["class"].astype(str).to_numpy()
    else:
        cls = np.repeat("all", len(df))
    if isinstance(UERE, dict) and isinstance(UERE.get("class"), dict):
        vals = np.asarray([float(UERE["class"].get(c, UERE.get(TYPE, UERE.get("horizontal", 1.0)))) for c in cls], dtype=float)
    elif isinstance(UERE, dict):
        vals = np.full(len(df), float(UERE.get(TYPE, UERE.get("horizontal", 1.0))), dtype=float)
    else:
        vals = np.full(len(df), float(UERE), dtype=float)
    dop_name = spec.get("DOP")
    dop = pd.to_numeric(df[dop_name], errors="coerce").to_numpy(dtype=float) if dop_name in df.columns else np.ones(len(df), dtype=float)
    var = (vals * np.where(np.isfinite(dop), dop, 1.0)) ** 2 / max(len(spec.get("axes") or [1]), 1)
    var_name = spec.get("VAR")
    if var_name:
        df[var_name] = var
    cov = spec.get("COV")
    if cov and len(cov) == 3:
        df[cov[0]] = var
        df[cov[1]] = 0.0
        df[cov[2]] = var
    if isinstance(data, Telemetry):
        return Telemetry(df, id_col=data.id_col, time_col=data.time_col, x_col=data.x_col, y_col=data.y_col, crs=data.crs, metadata=dict(data.metadata))
    return df


def get_error(data, CTMM, circle: bool = False, DIM: bool = False, calibrate: bool = True):
    del DIM
    model = CTMM.params if hasattr(CTMM, "params") else CTMM
    df = data.data if isinstance(data, Telemetry) else pd.DataFrame(data)
    axes = model.get("axes", ("x", "y"))
    typ = DOP_match(axes)
    spec = DOP_LIST.get(typ, DOP_LIST["horizontal"])
    err = model.get("error", False)
    if not err:
        return np.zeros(len(df), dtype=float)
    if spec.get("VAR") in df.columns:
        out = pd.to_numeric(df[spec["VAR"]], errors="coerce").to_numpy(dtype=float)
    elif spec.get("COV") and all(c in df.columns for c in spec["COV"]):
        cov = spec["COV"]
        vx = pd.to_numeric(df[cov[0]], errors="coerce").to_numpy(dtype=float)
        vy = pd.to_numeric(df[cov[2]], errors="coerce").to_numpy(dtype=float)
        out = (vx + vy) / 2.0 if circle else np.maximum(vx, vy)
    else:
        if calibrate and isinstance(data, Telemetry):
            u = uere(data)
            uval = u.get("UERE", {}).get("horizontal", 1.0) if isinstance(u, dict) else 1.0
        elif isinstance(err, (int, float)):
            uval = float(err)
        else:
            uval = 1.0
        dop_name = spec.get("DOP")
        dop = pd.to_numeric(df[dop_name], errors="coerce").to_numpy(dtype=float) if dop_name in df.columns else np.ones(len(df), dtype=float)
        out = (float(uval) * np.where(np.isfinite(dop), dop, 1.0)) ** 2 / max(len(spec.get("axes") or [1]), 1)
    return np.maximum(np.nan_to_num(out, nan=0.0, posinf=np.inf, neginf=0.0), 0.0)


def uere_type(data, precision: float = 0.5, trace: bool = False, type: str = "horizontal"):
    del trace, type
    return uere_fit(data, precision=precision)


def M2(data):
    arr = np.asarray(data, dtype=float)
    return float(np.nanmean(arr * arr))


def summary_UERE(object, *args, **kwargs):
    del args, kwargs
    return object


def summary_UERE_list(object, *args, **kwargs):
    return [summary_UERE(o, *args, **kwargs) for o in object]


def residuals_calibration(data, *args, **kwargs):
    del args, kwargs
    return data


__all__ = [
    "DOP_LIST",
    "DOP_match",
    "M2",
    "classnames",
    "classnames_set",
    "get_UERE_DOF",
    "get_class",
    "get_class_mat",
    "get_dop_types",
    "get_error",
    "is_calibrated",
    "residuals_calibration",
    "summary_UERE",
    "summary_UERE_list",
    "try_assign_uere",
    "typenames_set",
    "uere",
    "uere_fit",
    "uere_null",
    "uere_set",
    "uere_type",
]
