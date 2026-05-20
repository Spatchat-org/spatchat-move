"""CTMM dynamics helpers from ctmm 1.3.0 ``R/ctmm.R`` (``get.states``, ``continuity``, ``get.taus``)."""

from __future__ import annotations

import numpy as np

_TAU_NAMES_DEFAULT = ("position", "velocity", "acceleration")


def get_states(CTMM: dict) -> list[str]:
    s = CTMM.get("dynamics")
    if not s or s is False or s == "stationary":
        return []
    inner = CTMM.get(s)
    if isinstance(inner, dict):
        return list(dict.fromkeys(inner.keys()))
    return []


def continuity(CTMM: dict) -> int:
    tau = CTMM.get("tau")
    if not tau:
        return 0
    if isinstance(tau, dict):
        return sum(1 for v in tau.values() if float(v) > 0)
    arr = np.asarray(tau, dtype=float).ravel()
    return int(np.sum(arr > 0))


def _normalize_tau_dict(CTMM: dict) -> list[float]:
    """Sort ``tau`` decreasing; return values in that order (R ``CTMM$tau`` order)."""
    tau = CTMM.get("tau")
    if tau is None:
        CTMM["tau"] = {}
        return []
    if isinstance(tau, dict):
        d = {str(k): float(v) for k, v in tau.items()}
    else:
        arr = np.asarray(tau, dtype=float).ravel()
        names = list(CTMM.get("tau_feature_names") or list(_TAU_NAMES_DEFAULT[: arr.size]))
        if len(names) < arr.size:
            names = [f"t{i}" for i in range(arr.size)]
        d = {names[i]: float(arr[i]) for i in range(arr.size)}
    ordered = dict(sorted(d.items(), key=lambda kv: kv[1], reverse=True))
    CTMM["tau"] = ordered
    return list(ordered.values())


def _cov_rownames(CTMM: dict, *, simplify: bool) -> list[str]:
    if simplify:
        return []
    cov = CTMM.get("COV")
    if cov is None:
        return []
    rn = CTMM.get("COV_rownames")
    if rn is not None:
        return [str(x) for x in rn]
    if hasattr(cov, "index"):
        return [str(x) for x in cov.index.tolist()]
    return []


def get_taus(CTMM: dict, *, zeroes: bool = False, simplify: bool = False) -> dict:
    """``get.taus`` — precompute ``f``, ``f.nu``, ``Omega2``, Jacobians (mutates ``CTMM``)."""
    for s in get_states(CTMM):
        sub = CTMM.get(s)
        if isinstance(sub, dict):
            get_taus(sub, zeroes=zeroes, simplify=simplify)

    vals = _normalize_tau_dict(CTMM)
    if simplify:
        d = {k: v for k, v in CTMM["tau"].items() if v > 0}
        CTMM["tau"] = d
        vals = list(d.values())

    K = len(CTMM["tau"]) if zeroes else continuity(CTMM)
    CTMM["K"] = K

    VARS = _cov_rownames(CTMM, simplify=simplify)
    omega_num = float(CTMM.get("omega") or 0)
    omega_flag = bool(CTMM.get("omega")) or omega_num != 0
    omega_in_cov = "omega" in VARS

    if not vals:
        CTMM["tau_names"] = None
        return CTMM

    t1, t2 = vals[0], vals[1] if len(vals) > 1 else None

    if K == 1 and np.isfinite(t1):
        CTMM["tau_names"] = "tau position"
        CTMM["TAU"] = t1
    elif K > 1 and np.isposinf(t1) and t2 is not None:
        CTMM["tau_names"] = "tau velocity"
        CTMM["TAU"] = t2
        CTMM["Omega2"] = 1.0 / t2
        CTMM["J.Omega2"] = -1.0 / (t2**2)
        f = np.array([1.0 / t1, 1.0 / t2], dtype=float)
        CTMM["f"] = f.tolist()
        mf = float(np.mean(f))
        CTMM["f.nu"] = [mf, float((f[1] - f[0]) / 2)]
        CTMM["TfOmega2"] = 2 * CTMM["f.nu"][0] / CTMM["Omega2"]
    elif K > 1 and all(x in VARS for x in ("tau position", "tau velocity", "omega")):
        f = np.array([1.0 / vals[0], 1.0 / vals[1]], dtype=float)
        om = omega_num
        CTMM["tau_names"] = ["tau position", "tau velocity", "omega"]
        CTMM["f"] = f.tolist()
        CTMM["f.nu"] = [float(f[0]), float(f[1]), om]
        CTMM["Omega2"] = float(np.prod(f)) + om**2
        fnu = np.array(CTMM["f.nu"], dtype=float)
        CTMM["TAU"] = (np.array([1.0, 1.0, 2 * np.pi]) / fnu).tolist()
        CTMM["J.nu.tau"] = np.diag(np.concatenate([-f**2, [1.0]]))
        CTMM["J.TAU.tau"] = np.diag([1.0, 1.0, -2 * np.pi / om**2])
        v = np.array([f[1], f[0], 2 * om], dtype=float)
        CTMM["J.Omega2"] = (v @ CTMM["J.nu.tau"]).tolist()
    elif K > 1 and t2 is not None and (t1 > t2 or all(x in VARS for x in ("tau position", "tau velocity"))):
        tauv = np.array([t1, t2], dtype=float)
        f = 1.0 / tauv
        CTMM["tau_names"] = ["tau position", "tau velocity"]
        CTMM["TAU"] = tauv.tolist()
        CTMM["f"] = f.tolist()
        CTMM["f.nu"] = [float(np.mean(f)), float((f[1] - f[0]) / 2)]
        CTMM["Omega2"] = float(np.prod(f))
        CTMM["TfOmega2"] = 2 * CTMM["f.nu"][0] / CTMM["Omega2"]
        CTMM["J.f.tau"] = (-np.diag(f**2)).tolist()
        CTMM["J.tau.f"] = (-np.diag(tauv**2)).tolist()
        CTMM["J.nu.tau"] = np.array(
            [[-f[0] ** 2 / 2, -f[1] ** 2 / 2], [f[0] ** 2 / 2, -f[1] ** 2 / 2]], dtype=float
        )
        CTMM["J.Omega2"] = (-CTMM["Omega2"] / tauv).tolist()
    elif K > 1 and t2 is not None and t1 == t2 and not omega_flag and not omega_in_cov:
        tau_c = t1
        f = np.array([1.0 / tau_c, 1.0 / tau_c], dtype=float)
        CTMM["tau_names"] = ["tau"]
        CTMM["TAU"] = tau_c
        CTMM["f"] = f.tolist()
        CTMM["f.nu"] = [float(f[0]), 0.0]
        CTMM["Omega2"] = float(np.prod(f))
        CTMM["TfOmega2"] = 2 * CTMM["f.nu"][0] / CTMM["Omega2"]
        CTMM["J.tau.f"] = -tau_c**2
        CTMM["J.f.tau"] = (-(f**2)).tolist()
        CTMM["J.Omega2"] = -2.0 / tau_c**3
    elif K > 1 and (omega_flag or omega_in_cov) and t2 is not None:
        f = np.array([1.0 / t1, 1.0 / t2], dtype=float)
        om = omega_num
        CTMM["tau_names"] = ["tau", "omega"]
        CTMM["f"] = f.tolist()
        CTMM["f.nu"] = [float(np.mean(f)), om]
        fnu = np.array(CTMM["f.nu"], dtype=float)
        CTMM["Omega2"] = float(np.sum(fnu**2))
        CTMM["TfOmega2"] = 2 * CTMM["f.nu"][0] / CTMM["Omega2"]
        CTMM["TAU"] = (np.array([1.0, 2 * np.pi]) / fnu).tolist()
        CTMM["J.nu.tau"] = np.diag([-f[0] ** 2, 1.0])
        CTMM["J.TAU.tau"] = np.diag([1.0, -2 * np.pi / om**2])
        CTMM["J.Omega2"] = (2 * fnu @ CTMM["J.nu.tau"]).tolist()
    else:
        CTMM["tau_names"] = None

    return CTMM


__all__ = ["continuity", "get_states", "get_taus"]
