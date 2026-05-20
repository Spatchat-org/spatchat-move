"""Parity-focused translation of ctmm 1.3.0 ``R/parameters.R`` helpers."""

from __future__ import annotations

import numpy as np

from .ctmm_dynamics import get_states
from .types import CTMMModel


def id_parameters(model: CTMMModel, profile: bool = False, linear: bool = False, **kwargs):
    del profile, linear, kwargs
    names = list((model.params.get("features") or []))
    if not names:
        names = []
        if model.params.get("tau"):
            names.extend([f"tau {k}" for k in model.params["tau"].keys()])
        sig = model.params.get("sigma")
        if hasattr(sig, "par"):
            names.extend(list(sig.par.keys()))
        if model.params.get("omega", 0):
            names.append("omega")
    return {"NAMES": names, "K": len(names), "parscale": np.ones(len(names)), "lower": np.full(len(names), -np.inf), "upper": np.full(len(names), np.inf)}


def get_parameters(model: CTMMModel, names=None, **kwargs):
    del kwargs
    if names is None:
        names = id_parameters(model)["NAMES"]
    out = {}
    for name in names:
        if name.startswith("tau "):
            out[name] = float(model.params.get("tau", {}).get(name.split(" ", 1)[1], np.nan))
        elif name == "omega":
            out[name] = float(model.params.get("omega", np.nan))
        elif hasattr(model.params.get("sigma"), "par") and name in model.params["sigma"].par:
            out[name] = float(model.params["sigma"].par[name])
        else:
            out[name] = model.params.get(name, np.nan)
    return out


def set_parameters(model: CTMMModel, values, names=None, **kwargs):
    del kwargs
    out = CTMMModel(model.model, dict(model.params))
    vals = values
    if isinstance(values, dict):
        items = values.items()
    else:
        if names is None:
            names = id_parameters(model)["NAMES"]
        items = zip(names, vals)
    tau = dict(out.params.get("tau", {}) or {})
    for name, val in items:
        if name.startswith("tau "):
            tau[name.split(" ", 1)[1]] = float(val)
        else:
            out.params[name] = val
    if tau:
        out.params["tau"] = dict(sorted(tau.items(), key=lambda kv: float(kv[1]), reverse=True))
        out.params["tau_list"] = [float(v) for v in out.params["tau"].values()]
    return out


def getter(model, names=None, **kwargs):
    return get_parameters(model, names=names, **kwargs)


def profiled_var(*args, **kwargs):
    del args, kwargs
    return []


def clean_parameters(values):
    return values


def copy_parameters(source: CTMMModel, target: CTMMModel):
    out = CTMMModel(target.model, dict(target.params))
    out.params.update(source.params)
    return out


__all__ = [
    "clean_parameters",
    "copy_parameters",
    "get_parameters",
    "get_states",
    "getter",
    "id_parameters",
    "profiled_var",
    "set_parameters",
]
