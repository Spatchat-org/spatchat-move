# app.py
import os
import json
import re
import time
import threading
import shutil
import random
import sys
import zipfile
import inspect
import uuid
import base64
import html
from dataclasses import asdict, is_dataclass

import gradio as gr
import pandas as pd
from gradio.context import LocalContext
import numpy as np
import matplotlib

matplotlib.use("Agg", force=True)

# ---- Local modules ----
from storage import (
    get_cached_df, set_cached_df,
    get_cached_headers, set_cached_headers,
    clear_all_results,
    mcp_results, kde_results, akde_results,
    requested_percents, requested_kde_percents, requested_akde_percents,
    save_all_mcps_zip,
    set_locoh_results, get_locoh_results,
    set_dbbmm_results, get_dbbmm_results,
    requested_dbbmm_percents,
    set_akde_results, get_akde_results,
    set_current_session, set_output_dir, get_output_dir,
    get_repro_manifest, set_repro_manifest,
    delete_session,
)

from llm_utils import ask_llm, ask_llm_stream

try:
    from spatchat_gate import metered_chat
except ImportError:  # launched without spatchat-gateway's client/ on PYTHONPATH
    def metered_chat(**_kwargs):
        return lambda fn: fn

from crs_utils import parse_crs_input
from map_utils import render_empty_map
from coords_utils import looks_like_latlon, looks_invalid_latlon, parse_levels_from_text
from map_layers import build_preview_map, build_results_map
from schema_detect import (
    detect_and_standardize,
    parse_metadata_command,
    try_apply_user_mapping,
    ID_COL, TS_COL
)
from dataset_context import build_dataset_context

from methods.locoh import compute_locoh, LoCoHParams
from methods.dbbmm import compute_dbbmm, DBBMMParams
from methods.kde import add_kdes, KDEParams
from methods.akde import add_akdes, AKDEParams
from movement_analysis import (
    run_autocorrelation_analysis,
    run_displacement_analysis,
    run_hmm_state_analysis,
    run_step_length_analysis,
    run_turning_angle_analysis,
)

print("Starting SpatChat: Home Range Analysis (app.py) — handlers only")

os.environ["GRADIO_SHOW_API"] = "0"
REPRO_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "reproducible_scripts")

import numpy as _np
import pandas as _pd
from datetime import datetime as _dt, date as _date

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _component_accepts_kw(component, kw_name: str) -> bool:
    try:
        sig = inspect.signature(component)
    except (TypeError, ValueError):
        return False
    return kw_name in sig.parameters


def _json_safe(x):
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (_np.bool_,)):
        return bool(x)
    if isinstance(x, (_pd.Timestamp, _dt, _date)):
        return x.isoformat()
    if isinstance(x, _np.ndarray):
        return [_json_safe(v) for v in x.tolist()]
    if isinstance(x, (list, tuple, set)):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    return x


def _home_range_help() -> str:
    return (
        "Estimators available: MCP, KDE, AKDE, LoCoH, dBBMM.\n\n"
        "Examples:\n"
        "• I want 100 MCP\n"
        "• I want 95 KDE\n"
        "• I want AKDE\n"
        "• I want AKDE 50 and 95\n"
        "• I want LoCoH 50 and 95\n"
        "• I want dBBMM 95\n\n"
        "Ask me about parameter options anytime."
    )


def _movement_analysis_help() -> str:
    return (
        "Movement behavior analyses available: displacement, step lengths, turning angles, autocorrelation, and HMM behavioral states.\n\n"
        "Examples:\n"
        "• run displacement analysis\n"
        "• analyze step lengths and turning angles\n"
        "• run autocorrelation diagnostics\n"
        "• run hmm states\n"
        "• run movement analysis states=3\n"
    )


def _room_scope_nudge() -> str:
    return (
        "This room is for wildlife movement and home-range analysis. "
        "You can ask about MCP, KDE, AKDE, LoCoH, dBBMM, movement plots, or behavioral states."
    )


def _plain_params_dict(params) -> dict:
    if params is None:
        return {}
    if is_dataclass(params):
        data = asdict(params)
    elif hasattr(params, "__dict__"):
        data = dict(vars(params))
    else:
        return {}
    return {k: v for k, v in data.items() if v is not None}


def _merge_repro_manifest(
    mcp_list,
    kde_list,
    kde_params,
    akde_list,
    akde_params,
    locoh_requested,
    locoh_params,
    dbbmm_list,
    dbbmm_params,
    movement_requests,
):
    manifest = get_repro_manifest()
    if mcp_list:
        manifest["mcp"]["percents"] = sorted(set(manifest["mcp"].get("percents", [])) | set(int(x) for x in mcp_list))
    if kde_list:
        manifest["kde"]["percents"] = sorted(set(manifest["kde"].get("percents", [])) | set(int(x) for x in kde_list))
        manifest["kde"]["params"] = _plain_params_dict(kde_params)
    if akde_list:
        manifest["akde"]["percents"] = sorted(set(manifest["akde"].get("percents", [])) | set(int(x) for x in akde_list))
        manifest["akde"]["params"] = _plain_params_dict(akde_params)
    if locoh_requested:
        manifest["locoh"]["requested"] = True
        manifest["locoh"]["params"] = _plain_params_dict(locoh_params)
    if dbbmm_list:
        manifest["dbbmm"]["percents"] = sorted(set(manifest["dbbmm"].get("percents", [])) | set(int(x) for x in dbbmm_list))
        manifest["dbbmm"]["params"] = _plain_params_dict(dbbmm_params)
    movement = manifest.get("movement", {})
    for key in ("displacement", "step_lengths", "turning_angles", "autocorrelation", "hmm"):
        if movement_requests.get(key):
            movement[key] = True
    if movement_requests.get("hmm"):
        movement["states"] = int(movement_requests.get("states", movement.get("states", 3)))
    manifest["movement"] = movement
    set_repro_manifest(manifest)


def _write_repro_scripts(output_dir: str) -> list[str]:
    manifest = get_repro_manifest()
    script_dir = os.path.join(output_dir, "reproducible_scripts")
    os.makedirs(script_dir, exist_ok=True)
    paths = []
    df_current = get_cached_df()
    if df_current is not None:
        df_current.to_csv(os.path.join(script_dir, "input_data.csv"), index=False)
    config_path = os.path.join(script_dir, "repro_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    paths.append(config_path)

    if os.path.exists(os.path.join(REPRO_TEMPLATE_DIR, "README.txt")):
        target = os.path.join(script_dir, "README.txt")
        shutil.copyfile(os.path.join(REPRO_TEMPLATE_DIR, "README.txt"), target)
        paths.append(target)

    def copy_ctmm_py_runtime() -> list[str]:
        src_root = os.path.join(os.path.dirname(__file__), "methods", "ctmm_py")
        dst_root = os.path.join(script_dir, "ctmm_py")
        if not os.path.exists(src_root):
            return []
        if os.path.exists(dst_root):
            shutil.rmtree(dst_root)
        copied = []
        skip_dirs = {"__pycache__"}
        skip_files: set[str] = set()
        for root, dirs, files in os.walk(src_root):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            rel_dir = os.path.relpath(root, src_root)
            target_dir = dst_root if rel_dir == "." else os.path.join(dst_root, rel_dir)
            os.makedirs(target_dir, exist_ok=True)
            for file in files:
                if not file.endswith(".py") or file.endswith(".pyc") or file in skip_files:
                    continue
                src = os.path.join(root, file)
                dst = os.path.join(target_dir, file)
                shutil.copyfile(src, dst)
                copied.append(dst)
        return copied

    template_map = {
        "mcp": ("mcp_reproduce.txt", bool(manifest.get("mcp", {}).get("percents"))),
        "kde": ("kde_reproduce.txt", bool(manifest.get("kde", {}).get("percents"))),
        "akde": ("akde_reproduce.txt", bool(manifest.get("akde", {}).get("percents"))),
        "locoh": ("locoh_reproduce.txt", bool(manifest.get("locoh", {}).get("requested"))),
        "dbbmm": ("dbbmm_reproduce.txt", bool(manifest.get("dbbmm", {}).get("percents"))),
        "displacement": ("displacement_reproduce.txt", bool(manifest.get("movement", {}).get("displacement"))),
        "step_lengths": ("step_lengths_reproduce.txt", bool(manifest.get("movement", {}).get("step_lengths"))),
        "turning_angles": ("turning_angles_reproduce.txt", bool(manifest.get("movement", {}).get("turning_angles"))),
        "autocorrelation": ("autocorrelation_reproduce.txt", bool(manifest.get("movement", {}).get("autocorrelation"))),
        "hmm": ("hmm_reproduce.txt", bool(manifest.get("movement", {}).get("hmm"))),
    }
    for _, (filename, needed) in template_map.items():
        if not needed:
            continue
        src = os.path.join(REPRO_TEMPLATE_DIR, filename)
        if not os.path.exists(src):
            continue
        dst = os.path.join(script_dir, filename)
        shutil.copyfile(src, dst)
        paths.append(dst)
        if filename == "akde_reproduce.txt":
            paths.extend(copy_ctmm_py_runtime())
    if any(manifest.get("movement", {}).get(key) for key in ("displacement", "step_lengths", "turning_angles", "autocorrelation", "hmm")):
        helper_src = os.path.join(REPRO_TEMPLATE_DIR, "movement_common.py.txt")
        if os.path.exists(helper_src):
            helper_dst = os.path.join(script_dir, "movement_common.py.txt")
            shutil.copyfile(helper_src, helper_dst)
            paths.append(helper_dst)
    return paths


def parse_kv_tokens(text: str) -> dict:
    toks = {}
    for m in re.finditer(r'([A-Za-z_]+)\s*=\s*([^\s]+)', text):
        k = m.group(1).lower()
        v = m.group(2).strip().rstrip(",")
        toks[k] = v
    return toks


def _summarize_locoh(res: dict, params: LoCoHParams) -> str:
    lines = [f"LoCoH ({params.method}) complete. Areas (km²):"]
    got = False
    for animal_id, data in (res.get("animals") or {}).items():
        parts = [f"{it['isopleth']}%: {it['area_sq_km']:.2f}" for it in data.get("isopleths", [])]
        if parts:
            got = True
            lines.append(f"- {animal_id}: " + ", ".join(parts))
    return "\n".join(lines) if got else "LoCoH finished, but no polygons were built."


SESSIONS_ROOT = "sessions"
os.makedirs(SESSIONS_ROOT, exist_ok=True)

_PENDING_QUESTION_DEFAULTS = {
    "need_id": False,
    "need_ts": False,
    "ts_prompted": False,
    "id_prompted": False,
}
_PENDING_BY_SESSION: dict[str, dict[str, bool]] = {}
_LAST_DATA_SESSION_ID: str | None = None
_GRADIO_SESSION_MAP: dict[str, str] = {}


def _current_gradio_session_hash() -> str | None:
    try:
        request = LocalContext.request.get()
    except LookupError:
        request = None
    if request is None:
        return None
    session_hash = getattr(request, "session_hash", None)
    if not session_hash:
        return None
    return str(session_hash).strip() or None


def _register_gradio_session(session_id: str | None) -> None:
    sid = (session_id or "").strip()
    if not sid:
        return
    request_hash = _current_gradio_session_hash()
    if request_hash:
        _GRADIO_SESSION_MAP[request_hash] = sid


def _cleanup_session_resources(session_id: str | None) -> None:
    sid = (session_id or "").strip()
    if not sid:
        return
    _PENDING_BY_SESSION.pop(sid, None)
    global _LAST_DATA_SESSION_ID
    if _LAST_DATA_SESSION_ID == sid:
        _LAST_DATA_SESSION_ID = None
    delete_session(sid)
    session_root = os.path.join(SESSIONS_ROOT, sid)
    if os.path.exists(session_root):
        shutil.rmtree(session_root, ignore_errors=True)
    for key, value in list(_GRADIO_SESSION_MAP.items()):
        if value == sid:
            _GRADIO_SESSION_MAP.pop(key, None)


def _cleanup_current_browser_session() -> None:
    request_hash = _current_gradio_session_hash()
    if not request_hash:
        return
    sid = _GRADIO_SESSION_MAP.pop(request_hash, None)
    if sid:
        _cleanup_session_resources(sid)


def _prune_stale_sessions(max_age_seconds: int = 60 * 60 * 12) -> None:
    cutoff = time.time() - max_age_seconds
    try:
        for entry in os.scandir(SESSIONS_ROOT):
            if not entry.is_dir():
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    _cleanup_session_resources(entry.name)
            except OSError:
                continue
    except OSError:
        pass


def _pending_questions(session_id: str) -> dict[str, bool]:
    if session_id not in _PENDING_BY_SESSION:
        _PENDING_BY_SESSION[session_id] = dict(_PENDING_QUESTION_DEFAULTS)
    return _PENDING_BY_SESSION[session_id]


def _activate_session(session_id: str | None) -> str:
    sid = (session_id or "").strip() or str(uuid.uuid4())
    session_root = os.path.join(SESSIONS_ROOT, sid)
    upload_dir = os.path.join(session_root, "uploads")
    output_dir = os.path.join(session_root, "outputs")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    set_current_session(sid)
    set_output_dir(output_dir)
    _register_gradio_session(sid)
    return sid


def _remember_dataset_session(session_id: str | None) -> None:
    global _LAST_DATA_SESSION_ID
    sid = (session_id or "").strip()
    if sid:
        _LAST_DATA_SESSION_ID = sid


def _ensure_dataset_session(session_id: str | None) -> str:
    sid = _activate_session(session_id)
    if get_cached_df() is not None or _restore_session_dataframe(sid):
        return sid
    return sid


def _session_dataframe_path(session_id: str) -> str:
    return os.path.join(SESSIONS_ROOT, session_id, "uploads", "_spatchat_cached_df.csv")


def _persist_session_dataframe(session_id: str, df: pd.DataFrame | None) -> None:
    if df is None:
        return
    path = _session_dataframe_path(session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _restore_session_dataframe(session_id: str) -> bool:
    path = _session_dataframe_path(session_id)
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
        set_cached_df(df)
        set_cached_headers(list(df.columns))
        _remember_dataset_session(session_id)
        return True
    except Exception:
        return False


def _find_latest_dataset_session() -> str | None:
    best_sid = None
    best_mtime = None
    try:
        for entry in os.scandir(SESSIONS_ROOT):
            if not entry.is_dir():
                continue
            path = _session_dataframe_path(entry.name)
            if not os.path.exists(path):
                continue
            mtime = os.path.getmtime(path)
            if best_mtime is None or mtime > best_mtime:
                best_sid = entry.name
                best_mtime = mtime
    except Exception:
        return None
    return best_sid


def _current_dataset_context(session_id: str | None = None):
    session_id = _ensure_dataset_session(session_id)
    df = get_cached_df()
    try:
        return build_dataset_context(df)
    except Exception:
        return {"empty": True}


def _get_ready_dataframe(session_id: str | None) -> tuple[str, pd.DataFrame | None]:
    sid = _ensure_dataset_session(session_id)
    df = get_cached_df()
    if df is None:
        _restore_session_dataframe(sid)
        df = get_cached_df()
    return sid, df


_DEFAULT_ANIMAL_ID = "Animal_1"


def _ensure_default_animal_id(df: pd.DataFrame | None) -> tuple[pd.DataFrame | None, bool]:
    if df is None:
        return None, False
    out = df.copy()
    assumed = ID_COL not in out.columns
    if assumed:
        out[ID_COL] = _DEFAULT_ANIMAL_ID
        return out, True

    values = out[ID_COL].astype("string").str.strip()
    missing = values.isna() | values.eq("") | values.str.lower().isin({"nan", "none", "null", "na"})
    assumed = bool(missing.all())
    out[ID_COL] = values.where(~missing, _DEFAULT_ANIMAL_ID).astype(str)
    return out, assumed

# --- permissive level parser (allows 100) -----------------------------------
def _parse_levels_allow_100(text: str) -> list[int]:
    raw = re.findall(r'\b(100|[1-9]?[0-9])\b', text)
    out, seen = [], set()
    for tok in raw:
        p = int(tok)
        if 1 <= p <= 100 and p not in seen:
            seen.add(p)
            out.append(p)
    return out

# --- clear any stale state at app start ---------------------------------------------
def _reset_session_state():
    try:
        clear_all_results()
    except Exception:
        pass
    try:
        set_cached_df(None)
    except Exception:
        pass
    try:
        set_cached_headers([])
    except Exception:
        pass

# --- detect "parameter question" (no levels, no key=val, no action verb) ------------
_PARAM_VERBS = ("run", "compute", "calculate", "do", "make", "generate", "plot", "draw", "want", "need", "give")


def _is_parameter_question(msg: str, keyword: str) -> bool:
    s = msg.lower()
    if keyword not in s:
        return False
    if any(v in s for v in _PARAM_VERBS):
        return False
    if re.search(r"\b(100|[1-9]?[0-9])\b", s):
        return False
    if "=" in s:
        return False
    return ("param" in s) or ("option" in s) or ("argument" in s)

# --- NEW: generic "parameters?" / "options?" overview detector -----------------------
def _wants_parameters_overview(msg: str) -> bool:
    s = msg.lower().strip()
    if any(v in s for v in _PARAM_VERBS):
        return False
    return any(w in s for w in ("param", "option", "argument", "how to set", "available settings"))

# --- parse human distances like "300m", "0.5km", "1k" to meters ---------------------
def _parse_distance_to_meters(s: str) -> float:
    s = s.strip().lower()
    if s.endswith("km"):
        return float(s[:-2]) * 1000.0
    if s.endswith("k"):
        return float(s[:-1]) * 1000.0
    if s.endswith("m"):
        return float(s[:-1])
    return float(s)

# --- one source of truth for parameter help ------------------------------------
def _parameters_markdown(which: str | None = None) -> str:
    mcp = (
        "**MCP (Minimum Convex Polygon)**\n"
        "• `isopleths`: UD contours to export (e.g., 50,95). Example: `mcp 50,95`\n"
    )
    kde = (
        "**KDE (Kernel Density Estimation)**\n"
        "• `isopleths`: UD contours to export (e.g., 50,95). Example: `kde 95`\n"
        "• `bw` / `bandwidth` / `h` **(meters or km)**: kernel bandwidth in UTM meters. Examples: `bw=300m`, `bw=0.5km`\n"
        "• `kernel`: one of `gaussian`, `epanechnikov`, `tophat`, `exponential`, `linear`, `cosine`. Example: `kernel=epanechnikov`\n"
        "• `gres` / `grid_res` **(meters or km)**: grid cell size for the raster (optional). Example: `gres=50m`\n"
        "Examples: `kde 95 bw=300m kernel=epanechnikov`, `kde 50,95 bw=0.5km`\n"
    )
    akde = (
        "**AKDE (Autocorrelated Kernel Density Estimation)**\n"
        "• `isopleths`: UD contours (e.g., 50,95). Default is `95` if omitted. Examples: `akde`, `akde 95`\n"
        "• `model`: one of `auto`, `ou`, `ouf`. Example: `akde 95 model=ou`\n"
        "• `min_points`: minimum relocations required per track. Default 15\n"
        "• `variogram_res`: subsampling factor for faster variogram estimation. Default 1\n"
        "• `variogram_dt`: lag-bin width in seconds (optional)\n"
        "• `bw` / `bandwidth` / `h` **(meters or km)**: optional manual bandwidth override\n"
        "• `gres` / `grid_res` **(meters or km)**: grid cell size for the raster (optional)\n"
        "Examples: `akde`, `akde 95 model=ouf`, `akde 50,95 gres=50m`, `akde 95 bw=300m`\n"
    )
    locoh = (
        "**LoCoH (Local Convex Hull)**\n"
        "• `method`: `k` (neighbors), `a` (radius), or `r` (adaptive)\n"
        "• `k` (int): number of nearest neighbors (for method=`k`). Default 10\n"
        "• `a` (float): distance threshold in meters (for method=`a`)\n"
        "• `r` (float): adaptive radius in meters (for method=`r`)\n"
        "• `isopleths`: UD contours (e.g., 50,95). Examples: `locoh k=10 isopleths=50,95`, `locoh a=1500 95`\n"
    )
    dbbmm = (
        "**dBBMM (Distance-based Brownian Bridge)**\n"
        "• `le`/`locerr`/`sigma` (m): GPS location error. Default 30\n"
        "• `window`/`w` (int): sliding window size. Default 31\n"
        "• `margin`/`m` (int): points trimmed at each end. Default 11\n"
        "• `res`/`resolution` (m): raster cell size. Default 50\n"
        "• `buf`/`buffer` (m): buffer around track. Default 1000\n"
        "• `subs`/`substeps` (int): interpolation substeps. Default 40\n"
        "• `isopleths`: UD contours (e.g., 50,95). Example: `dbbmm 95 res=75 buf=1500`\n"
    )
    all_text = (
        "Here are the parameter options:\n\n" + mcp + "\n" + kde + "\n" + akde + "\n" + locoh + "\n" + dbbmm
        + "\nTip: bare numbers for `bw`, `gres`, etc. are treated as meters; add `km` to use kilometers."
    )
    if which is None:
        return all_text
    which = which.lower()
    if which == "mcp":
        return mcp
    if which == "kde":
        return kde
    if which == "akde":
        return akde
    if which == "locoh":
        return locoh
    if which == "dbbmm":
        return dbbmm
    return all_text


def _wants_movement_help(msg: str) -> bool:
    s = msg.lower()
    return (
        any(term in s for term in ("movement analysis", "behavior analysis", "behaviour analysis", "hidden markov", "hmm", "turning angle", "step length", "autocorrelation", "displacement"))
        and any(word in s for word in ("parameter", "parameters", "option", "options", "help"))
    )


def _parse_hmm_state_count(text: str) -> int:
    m = re.search(r"\bstates?\s*=\s*(\d+)\b", text.lower())
    if m:
        return max(2, min(6, int(m.group(1))))
    m = re.search(r"\b(\d+)\s+states?\b", text.lower())
    if m:
        return max(2, min(6, int(m.group(1))))
    return 3


def _parse_movement_requests(text: str) -> dict:
    s = text.lower()
    explicit = {
        "displacement": "displacement" in s,
        "step_lengths": "step length" in s or "step lengths" in s,
        "turning_angles": "turning angle" in s or "turning angles" in s,
        "autocorrelation": "autocorrelation" in s or re.search(r"\bacf\b", s) is not None,
        "hmm": (
            "hidden markov" in s
            or re.search(r"\bhmm\b", s) is not None
            or ("behavior" in s and "state" in s)
            or ("behaviour" in s and "state" in s)
        ),
    }
    has_explicit_method = any(explicit.values())
    wants_all = (
        not has_explicit_method
        and (
            "movement analysis" in s
            or "movement analyses" in s
            or "behavior analysis" in s
            or "behaviour analysis" in s
            or "movement patterns" in s
        )
    )
    requests = {
        "displacement": wants_all or explicit["displacement"],
        "step_lengths": wants_all or explicit["step_lengths"],
        "turning_angles": wants_all or explicit["turning_angles"],
        "autocorrelation": wants_all or explicit["autocorrelation"],
        "hmm": wants_all or explicit["hmm"],
        "states": _parse_hmm_state_count(text),
    }
    return requests


def _movement_interpretation_response(text: str) -> str | None:
    s = text.lower()
    asks_explanation = (
        "?" in text
        or any(phrase in s for phrase in (
            "what does",
            "what do",
            "tell me",
            "interpret",
            "interpretation",
            "how to interpret",
            "how do i interpret",
            "what is the meaning",
            "explain",
            "how to read",
        ))
    )
    if not asks_explanation:
        return None

    if "hidden markov" in s or re.search(r"\bhmm\b", s) is not None or ("behavior" in s and "state" in s) or ("behaviour" in s and "state" in s):
        return (
            "Hidden Markov state analysis groups movement steps into recurring behavioral states based on the movement features in the track, "
            "here mainly step length and turning behavior.\n\n"
            "A state with longer steps and straighter movement often suggests directed travel or transit, while a state with shorter steps and "
            "more variable turning often suggests localized search, resting, or area-restricted use. The state numbers themselves do not have "
            "fixed biological meanings, so interpret them by comparing each state's mean step length, turning-angle pattern, and when or where "
            "that state appears along the track.\n\n"
            "In this room, the main files to read are the state assignment table and the HMM summary table. Start with `hmm_behavior_state_summary.csv`: "
            "look for which states have the highest mean step length and which have larger average turning angles, then compare those states back to time, "
            "location, habitat, or known behavior."
        )

    if "displacement" in s:
        return (
            "Displacement describes how far an animal is from a reference point over time, usually the starting location or a previous fix.\n\n"
            "Rising displacement suggests net movement away from the origin, flat sections suggest localized use, and repeated rises and drops can suggest "
            "commuting, excursions, or returns to familiar areas. Interpret it together with timestamps and track segments rather than as a single summary number."
        )

    if "step length" in s:
        return (
            "Step length is the distance between consecutive fixes. Larger values usually indicate faster or more directed movement, while smaller values "
            "suggest slower movement, resting, or localized activity. Interpret the distribution, not just the mean, and check whether long steps cluster "
            "at certain times or in certain parts of the landscape."
        )

    if "turning angle" in s:
        return (
            "Turning angle measures how sharply the animal changes direction between steps. Angles near zero indicate straighter travel, while larger absolute "
            "angles indicate more tortuous movement. A pattern of high turning with short steps often points to localized search or area-restricted behavior."
        )

    if "autocorrelation" in s or re.search(r"\bacf\b", s) is not None:
        return (
            "Autocorrelation shows whether movement values remain similar across nearby time steps. Strong short-lag autocorrelation means consecutive fixes are "
            "not independent, which is common in movement data and matters for estimator choice. Persistent autocorrelation is one reason methods like AKDE can "
            "be more appropriate than assuming independent locations."
        )

    return None


def _requested_timestamp_dependent_analyses(
    akde_list: list[int],
    dbbmm_list: list[int],
    movement_requests: dict,
) -> list[str]:
    needed = []
    if akde_list:
        needed.append("AKDE")
    if dbbmm_list:
        needed.append("dBBMM")
    if movement_requests.get("displacement"):
        needed.append("displacement analysis")
    if movement_requests.get("step_lengths"):
        needed.append("step-length analysis")
    if movement_requests.get("turning_angles"):
        needed.append("turning-angle analysis")
    if movement_requests.get("autocorrelation"):
        needed.append("autocorrelation diagnostics")
    if movement_requests.get("hmm"):
        needed.append("HMM behavioral-state analysis")
    return needed


def _is_level_only_followup(text: str) -> bool:
    s = (text or "").strip().lower()
    if not s:
        return False
    s = s.replace("%", " ")
    s = re.sub(r"\band\b", " ", s)
    s = re.sub(r"[,\s]+", " ", s).strip()
    if not s:
        return False
    tokens = s.split()
    return bool(tokens) and all(re.fullmatch(r"(100|[1-9]?[0-9])", tok) for tok in tokens)


def _followup_estimator_from_history(history: list[dict]) -> str | None:
    keywords = ("akde", "dbbmm", "locoh", "kde", "mcp")
    for item in reversed(history or []):
        content = str((item or {}).get("content") or "").lower()
        for keyword in keywords:
            if re.search(rf"\b{keyword}\b", content):
                return keyword
    return None




def _run_movement_analyses(df: pd.DataFrame, requests: dict, output_dir: str) -> list[str]:
    msgs = []
    if requests.get("displacement"):
        res = run_displacement_analysis(df, output_dir)
        msgs.append(res["message"])
    if requests.get("step_lengths"):
        res = run_step_length_analysis(df, output_dir)
        msgs.append(res["message"])
    if requests.get("turning_angles"):
        res = run_turning_angle_analysis(df, output_dir)
        msgs.append(res["message"])
    if requests.get("autocorrelation"):
        res = run_autocorrelation_analysis(df, output_dir)
        msgs.append(res["message"])
    if requests.get("hmm"):
        res = run_hmm_state_analysis(df, output_dir, n_states=int(requests.get("states", 3)))
        msgs.append(res["message"])
    return msgs


def _file_to_browser_src(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }.get(ext)
    if mime is None:
        return None
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _csv_table_html(path: str, max_rows: int = 24) -> str | None:
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return "<div class=\"spatchat-figure-table-empty\">No rows available.</div>"
    table_df = df.head(max_rows).copy()
    cols = list(table_df.columns)
    head_html = "".join(f"<th>{html.escape(str(col))}</th>" for col in cols)
    body_rows = []
    for _, row in table_df.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                text = ""
            elif isinstance(value, float):
                text = f"{value:.2f}"
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    more_note = ""
    if len(df) > len(table_df):
        more_note = f"<div class=\"spatchat-figure-table-note\">Showing first {len(table_df)} of {len(df)} rows.</div>"
    return (
        "<div class=\"spatchat-figure-table-wrap\">"
        f"{more_note}"
        "<table class=\"spatchat-figure-table\">"
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _collect_figure_entries(output_dir: str, akde_data: dict | None = None) -> list[dict]:
    figures = []
    seen_ids = set()

    def add_figure(fig_id: str, title: str, subtitle: str, path: str | None):
        if not path or fig_id in seen_ids:
            return
        src = _file_to_browser_src(path)
        if not src:
            return
        seen_ids.add(fig_id)
        figures.append({
            "id": fig_id,
            "title": title,
            "subtitle": subtitle,
            "path": path,
            "src": src,
        })

    def add_table(fig_id: str, title: str, subtitle: str, path: str | None):
        if not path or fig_id in seen_ids:
            return
        table_html = _csv_table_html(path)
        if not table_html:
            return
        seen_ids.add(fig_id)
        figures.append({
            "id": fig_id,
            "title": title,
            "subtitle": subtitle,
            "path": path,
            "tableHtml": table_html,
        })

    movement_specs = [
        ("displacement", "Displacement", "Net displacement and cumulative movement", ("movement_analysis", "displacement_plots.png")),
        ("step-lengths", "Step Lengths", "Observation series and distribution", ("movement_analysis", "step_length_plots.png")),
        ("turning-angles", "Turning Angles", "Observation series and distribution", ("movement_analysis", "turning_angle_plots.png")),
        ("autocorrelation", "Autocorrelation", "Step-length and displacement diagnostics", ("movement_analysis", "autocorrelation_plots.png")),
        ("hmm-states", "Behavioral States", "Hidden Markov state assignments and step lengths", ("movement_analysis", "hmm_behavior_states.png")),
    ]
    for fig_id, title, subtitle, rel_path_parts in movement_specs:
        add_figure(fig_id, title, subtitle, os.path.join(output_dir, *rel_path_parts))

    add_table(
        "home-range-areas",
        "Home Range Areas",
        "Summary table from home_range_areas.csv",
        os.path.join(output_dir, "home_range_areas.csv"),
    )

    for animal_id, percent_data in (akde_data or {}).items():
        if not isinstance(percent_data, dict):
            continue
        variogram_path = None
        model = None
        tau_pos_s = None
        tau_vel_s = None
        for result in percent_data.values():
            meta = (result or {}).get("meta") or {}
            variogram_path = variogram_path or meta.get("variogram_plot")
            model = model or meta.get("model")
            tau_pos_s = tau_pos_s or meta.get("tau_pos_s")
            tau_vel_s = tau_vel_s or meta.get("tau_vel_s")
        if variogram_path:
            subtitle_bits = []
            if model:
                subtitle_bits.append(f"Model: {model.upper()}")
            if tau_pos_s:
                subtitle_bits.append(f"tau_pos={tau_pos_s / 3600.0:.2f} h")
            if tau_vel_s:
                subtitle_bits.append(f"tau_vel={tau_vel_s / 3600.0:.2f} h")
            subtitle = " | ".join(subtitle_bits) if subtitle_bits else "Empirical variogram and fitted autocorrelation model"
            add_figure(
                f"akde-variogram-{animal_id}",
                f"AKDE Variogram: {animal_id}",
                subtitle,
                variogram_path,
            )

    return figures


def _merge_figure_history(existing: list[dict] | None, discovered: list[dict]) -> tuple[list[dict], int]:
    merged = list(existing or [])
    index_by_id = {fig.get("id"): idx for idx, fig in enumerate(merged)}
    active_index = len(merged) - 1 if merged else 0
    for fig in discovered:
        fig_id = fig.get("id")
        if fig_id in index_by_id:
            merged[index_by_id[fig_id]] = fig
            active_index = index_by_id[fig_id]
        else:
            index_by_id[fig_id] = len(merged)
            merged.append(fig)
            active_index = len(merged) - 1
    return merged, active_index


_EMPTY_PLOTS_HTML = '<div class="spatchat-empty-state"><svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 15l4-5 3 3 5-7"/></svg><div class="spatchat-empty-title">No plots yet</div><div class="spatchat-empty-text">Upload a dataset, then ask for an analysis such as &ldquo;run movement analysis&rdquo; or &ldquo;95% KDE&rdquo;. Plots and tables will appear here.</div></div>'


def _render_figure_viewer(figures: list[dict] | None, active_index: int = 0) -> str:
    figures = list(figures or [])
    if not figures:
        return '<div class="spatchat-figure-root" data-payload=""></div>' + _EMPTY_PLOTS_HTML

    latest = max(0, min(int(active_index or 0), len(figures) - 1))
    cards = []
    for idx, fig in enumerate(figures):
        title = html.escape(str(fig.get("title") or "Figure"), quote=True)
        subtitle = html.escape(str(fig.get("subtitle") or ""), quote=True)
        src = html.escape(str(fig.get("src") or ""), quote=True)
        table_html = str(fig.get("tableHtml") or "")
        if src:
            preview = f'<img class="spatchat-thumb-img" alt="{title}" src="{src}" loading="lazy" />'
        else:
            preview = f'<div class="spatchat-thumb-table">{table_html}</div>'
        badge = '<span class="spatchat-thumb-badge">New</span>' if idx == latest and len(figures) > 1 else ""
        cards.append(
            f'<button type="button" class="spatchat-thumb" data-title="{title}" data-subtitle="{subtitle}" '
            f'onclick="return window.spatchatGallery ? window.spatchatGallery.open(this) : false;" aria-label="Enlarge {title}">'
            f'<span class="spatchat-thumb-preview">{preview}</span>'
            f'<span class="spatchat-thumb-meta"><span class="spatchat-thumb-title">{title}</span>'
            f'<span class="spatchat-thumb-sub">{subtitle}</span></span>{badge}'
            f'<span class="spatchat-thumb-zoom">Enlarge</span></button>'
        )
    count = len(figures)
    label = "1 plot or table" if count == 1 else f"{count} plots and tables"
    return (
        '<div class="spatchat-gallery">'
        f'<div class="spatchat-gallery-bar"><span>{label}</span><span class="spatchat-gallery-hint">Click any item to enlarge</span></div>'
        f'<div class="spatchat-gallery-grid">{"".join(cards)}</div>'
        '</div>'
    )


# --------------------------------------------------------------------------------------
# Upload flow
# --------------------------------------------------------------------------------------
def handle_upload_initial(file, session_id):
    session_id = _activate_session(session_id)
    clear_all_results()

    upload_dir = os.path.join(SESSIONS_ROOT, session_id, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    filename = os.path.join(upload_dir, os.path.basename(file))
    shutil.copy(file, filename)

    try:
        df = pd.read_csv(filename)
        set_cached_df(df)
        set_cached_headers(list(df.columns))
        _remember_dataset_session(session_id)
        _persist_session_dataframe(session_id, df)
    except Exception as e:
        print(f"[upload] failed to read CSV: {e}", file=sys.stderr)
        return [
            [],
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            render_empty_map(),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            session_id,
        ]

    pending = _pending_questions(session_id)
    for k in pending:
        pending[k] = False

    cached_headers = get_cached_headers()
    lower_cols = [c.lower() for c in cached_headers]

    found_x = found_y = None
    latlon_guess = None
    try:
        guess = looks_like_latlon(get_cached_df(), cached_headers)
        if isinstance(guess, tuple) and len(guess) == 3:
            latlon_guess, found_x, found_y = guess
    except Exception:
        pass

    if "latitude" in lower_cols and "longitude" in lower_cols:
        lat_col = cached_headers[lower_cols.index("latitude")]
        lon_col = cached_headers[lower_cols.index("longitude")]

        if looks_invalid_latlon(get_cached_df(), lat_col, lon_col):
            return [[{"role": "assistant", "content":
                "CSV uploaded. Your coordinates do not appear to be latitude/longitude. "
                "Please specify X (easting), Y (northing), and the CRS/UTM zone below "
                "(e.g., 'UTM 10T' or 'EPSG:32610')."}],
                gr.update(choices=cached_headers, value=lon_col, visible=True),
                gr.update(choices=cached_headers, value=lat_col, visible=True),
                gr.update(visible=True),
                render_empty_map(),
                gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                gr.update(visible=True), gr.update(visible=False), session_id]

        df0 = get_cached_df().copy()
        df0["longitude"] = df0[lon_col]
        df0["latitude"] = df0[lat_col]

        from schema_detect import detect_id_column, detect_timestamp_column, ID_COL, TS_COL
        src_id = detect_id_column(df0)
        src_ts = detect_timestamp_column(df0)

        df1, _ = detect_and_standardize(df0)
        df1, single_id_assumed = _ensure_default_animal_id(df1)
        set_cached_df(df1)
        _remember_dataset_session(session_id)
        _persist_session_dataframe(session_id, df1)
        map_html = build_preview_map(df1)

        id_found = (ID_COL in df1.columns)
        ts_found = (TS_COL in df1.columns)
        id_note = f"• **ID column**: `{src_id}`" if src_id else f"• **ID column**: `not detected`; using `{_DEFAULT_ANIMAL_ID}` for all rows"
        ts_note = f"• **Timestamp column**: `{src_ts}`" if src_ts else "• **Timestamp column**: `not detected`"

        tips = []
        if not id_found and not single_id_assumed:
            tips.append("If your data has one, say: **“ID column is <your_col>”** or **“no id”**.")
        if not ts_found:
            tips.append("If your data has one, say: **“Timestamp column is <your_col>”** or **“no timestamp”**.")

        msg = (
            "CSV uploaded. Latitude and longitude detected.\n\n"
            "Detected schema:\n"
            f"• **Longitude**: `{lon_col}`\n"
            f"• **Latitude**: `{lat_col}`\n"
            f"{id_note}\n"
            f"{ts_note}\n\n"
        )
        if tips:
            msg += "You can correct me in chat:\n" + "\n".join(f"  - {t}" for t in tips) + "\n\n"
        msg += _home_range_help()

        return [[{"role": "assistant", "content": msg}],
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                map_html,
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False), session_id]

    if latlon_guess:
        df = get_cached_df()
        df0 = df.copy()
        df0["longitude"] = df0[found_x] if latlon_guess == "lonlat" else df0[found_y]
        df0["latitude"] = df0[found_y] if latlon_guess == "lonlat" else df0[found_x]

        from schema_detect import detect_id_column, detect_timestamp_column, ID_COL, TS_COL
        src_id = detect_id_column(df0)
        src_ts = detect_timestamp_column(df0)

        df1, _ = detect_and_standardize(df0)
        df1, single_id_assumed = _ensure_default_animal_id(df1)
        set_cached_df(df1)
        _remember_dataset_session(session_id)
        _persist_session_dataframe(session_id, df1)
        map_html = build_preview_map(df1)

        id_found = (ID_COL in df1.columns)
        ts_found = (TS_COL in df1.columns)
        id_note = f"• **ID column**: `{src_id}`" if src_id else f"• **ID column**: `not detected`; using `{_DEFAULT_ANIMAL_ID}` for all rows"
        ts_note = f"• **Timestamp column**: `{src_ts}`" if src_ts else "• **Timestamp column**: `not detected`"

        tips = []
        if not id_found and not single_id_assumed:
            tips.append("If your data has one, say: **“ID column is <your_col>”** or **“no id”**.")
        if not ts_found:
            tips.append("If your data has one, say: **“Timestamp column is <your_col>”** or **“no timestamp”**.")

        msg = (
            f"CSV uploaded. `{found_x}`/`{found_y}` interpreted as longitude/latitude.\n\n"
            "Detected schema:\n"
            f"• **Longitude** source: `{found_x if latlon_guess == 'lonlat' else found_y}`\n"
            f"• **Latitude** source: `{found_y if latlon_guess == 'lonlat' else found_x}`\n"
            f"{id_note}\n"
            f"{ts_note}\n\n"
        )
        if tips:
            msg += "You can correct me in chat:\n" + "\n".join(f"  - {t}" for t in tips) + "\n\n"
        msg += _home_range_help()

        return [[{"role": "assistant", "content": msg}],
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                map_html,
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False), session_id]

    return [[{"role": "assistant", "content":
        "CSV uploaded. Your coordinates do not appear to be latitude/longitude. "
        "Please specify X (easting), Y (northing), and the CRS/UTM zone below."}],
        gr.update(choices=cached_headers, value=found_x, visible=True),
        gr.update(choices=cached_headers, value=found_y, visible=True),
        gr.update(visible=True),
        render_empty_map(),
        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
        gr.update(visible=True), gr.update(visible=False), session_id]


def handle_upload_confirm(x_col, y_col, crs_text, session_id):
    _activate_session(session_id)
    df = get_cached_df()
    if df is None:
        return "<p>No data loaded. Please upload a CSV first.</p>"
    df = df.copy()

    if x_col not in df.columns or y_col not in df.columns:
        return "<p>Selected coordinate columns not found in data.</p>"

    if x_col.lower() in ["longitude", "lon"] and y_col.lower() in ["latitude", "lat"]:
        try:
            lon_ok = df[x_col].astype(float).between(-180, 180).all()
            lat_ok = df[y_col].astype(float).between(-90, 90).all()
        except Exception:
            lon_ok = lat_ok = False

        if lon_ok and lat_ok:
            df["longitude"] = df[x_col]
            df["latitude"] = df[y_col]
        else:
            if not str(crs_text).strip():
                return "<p>Your columns are named lon/lat but values are not geographic. Please enter a CRS (e.g., 'UTM 10T' or 'EPSG:32610').</p>"
            try:
                epsg = parse_crs_input(crs_text)
                from pyproj import Transformer
                transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
                df["longitude"], df["latitude"] = transformer.transform(df[x_col].values, df[y_col].values)
            except Exception as e:
                return f"<p>Failed to convert coordinates: {e}</p>"
    else:
        if not str(crs_text).strip():
            return "<p>Please enter a CRS or UTM zone before confirming (e.g., 'UTM 10T' or 'EPSG:32610').</p>"
        try:
            epsg = parse_crs_input(crs_text)
            from pyproj import Transformer
            transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
            df["longitude"], df["latitude"] = transformer.transform(df[x_col].values, df[y_col].values)
        except Exception as e:
            return f"<p>Failed to convert coordinates: {e}</p>"

    df, _ = detect_and_standardize(df)
    df, _ = _ensure_default_animal_id(df)
    set_cached_df(df)
    _remember_dataset_session(session_id)
    _persist_session_dataframe(session_id, df)
    return build_preview_map(df)


def confirm_and_hint(x_col, y_col, crs_text, chat_history, session_id):
    session_id = _activate_session(session_id)
    map_html = handle_upload_confirm(x_col, y_col, crs_text, session_id)
    guidance = _home_range_help()
    chat = list(chat_history)
    chat.append({"role": "assistant", "content": guidance})
    return map_html, chat, session_id


MAX_UPLOAD_MB = 20
_DATASHEET_MAX_ROWS = 500
_NEXT_TAB: dict[str, str] = {}
_DATA_TAB_RE = re.compile(
    r"\b(analy[sz]e|summari[sz]e|describe|show|view|see|display|preview|look at|open)\b.*\b(data|dataset|datasheet|table|rows|columns|csv)\b"
    r"|\b(datasheet|data ?frame|data preview)\b",
    re.IGNORECASE,
)


_EMPTY_DATA_HTML = (
    '<div class="spatchat-empty-state"><svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/>'
    '<path d="M3 10h18M9 4v16"/></svg><div class="spatchat-empty-title">No data loaded</div>'
    '<div class="spatchat-empty-text">Click + in the chat or choose &ldquo;Try sample data&rdquo; to load a movement CSV. '
    'Your table will appear here.</div></div>'
)


def _datasheet_df_updates(df):
    if df is None:
        return gr.update(value=None, visible=False), gr.update(visible=True)
    return gr.update(value=df.head(_DATASHEET_MAX_ROWS), visible=True), gr.update(visible=False)


def _datasheet_update(session_id):
    _activate_session(session_id)
    return _datasheet_df_updates(get_cached_df())


# ---- dataset library: every uploaded dataset keeps its own map, plots and table ----
def _new_library() -> dict:
    return {"items": [], "active": None}


def _dataset_selector_update(library: dict):
    items = library.get("items") or []
    return gr.update(
        choices=[(it["name"], it["id"]) for it in items],
        value=library.get("active"),
        visible=bool(items),
    )


def _register_dataset(label, session_id, library, figure_state, map_html):
    library = dict(library or _new_library())
    items = list(library.get("items") or [])
    _activate_session(session_id)
    df = get_cached_df()
    if df is None or any(it.get("ref") is df for it in items):
        library["items"] = items
        return library, _dataset_selector_update(library)
    base = os.path.basename(str(label or "Dataset")) or "Dataset"
    name, n = base, 2
    existing = {it["name"] for it in items}
    while name in existing:
        name = f"{base} ({n})"
        n += 1
    item = {
        "id": uuid.uuid4().hex[:8],
        "name": name,
        "df": df,
        "ref": df,
        "figures": list(figure_state or []),
        "map_html": map_html,
    }
    items.append(item)
    library["items"] = items
    library["active"] = item["id"]
    return library, _dataset_selector_update(library)


def _sync_active_dataset(session_id, library, figure_state, map_html):
    library = dict(library or _new_library())
    items = list(library.get("items") or [])
    _activate_session(session_id)
    df = get_cached_df()
    for idx, it in enumerate(items):
        if it["id"] == library.get("active"):
            it = dict(it)
            if df is not None:
                it["df"] = df
                it["ref"] = df
            it["figures"] = list(figure_state or [])
            if isinstance(map_html, str) and map_html:
                it["map_html"] = map_html
            items[idx] = it
    library["items"] = items
    return library


def _switch_dataset(selected_id, library, session_id, chat_history):
    library = dict(library or _new_library())
    item = next((it for it in library.get("items") or [] if it["id"] == selected_id), None)
    if item is None:
        return (gr.skip(),) * 7 + (library,)
    _activate_session(session_id)
    df = item["df"]
    set_cached_df(df)
    set_cached_headers(list(df.columns))
    _remember_dataset_session(session_id)
    _persist_session_dataframe(session_id, df)
    library["active"] = item["id"]
    figures = list(item.get("figures") or [])
    chat = list(chat_history or []) + [{"role": "assistant", "content": f"Switched to **{item['name']}**. Its map, plots and table are shown on the right."}]
    sheet, sheet_empty = _datasheet_df_updates(df)
    return (
        chat,
        item.get("map_html") or render_empty_map(),
        _render_figure_viewer(figures),
        figures,
        sheet,
        sheet_empty,
        _select_tab("map"),
        library,
    )


def _chip_js(text: str) -> str:
    return (
        "() => { const ta = document.querySelector('#spatchat-user-input textarea, #spatchat-user-input input');"
        " if (!ta) return; ta.focus(); ta.value = " + json.dumps(text) + ";"
        " ta.dispatchEvent(new Event('input', { bubbles: true }));"
        " setTimeout(() => { const b = document.querySelector('#spatchat-send-btn'); if (b) b.click(); }, 80); }"
    )


_SUGGESTION_CHIPS = [
    "Run movement analysis",
    "95% KDE",
    "100% MCP",
    "AKDE 95",
    "Behavioral states (HMM)",
]


def _select_tab(name: str):
    return gr.Tabs(selected=name)


def _consume_next_tab(session_id):
    tab = _NEXT_TAB.pop(session_id or "", None)
    return _select_tab(tab) if tab else gr.skip()


def _handle_upload_initial_ui(file, chat_history, session_id, figure_state):
    try:
        too_big = file is not None and os.path.getsize(file) > MAX_UPLOAD_MB * 1024 * 1024
    except OSError:
        too_big = False
    if too_big:
        next_chat = list(chat_history or []) + [{"role": "assistant", "content": f"That file is too large. Please upload a CSV under {MAX_UPLOAD_MB} MB."}]
        return (next_chat, *([gr.skip()] * 13))
    result = handle_upload_initial(file, session_id)
    empty_figures = []
    next_chat = list(chat_history or [])
    uploaded_chat = list(result[0] or [])
    if uploaded_chat:
        next_chat.extend(uploaded_chat)
    return (next_chat, gr.update(value=""), *result[1:], _render_figure_viewer(empty_figures), empty_figures)


_SAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "demo_movement.csv")


def _load_sample_data_ui(chat_history, session_id, figure_state):
    return _handle_upload_initial_ui(_SAMPLE_DATA_PATH, chat_history, session_id, figure_state)


def _confirm_and_hint_ui(x_col, y_col, crs_text, chat_history, session_id):
    map_html, chat, session_id = confirm_and_hint(x_col, y_col, crs_text, chat_history, session_id)
    return map_html, chat, gr.update(value=""), session_id


def _is_skip(value) -> bool:
    return value is None or (isinstance(value, dict) and "value" not in value)


def _handle_chat_ui(chat_history, user_message, session_id, figure_state):
    active_session_id = _activate_session(session_id)
    baseline_figures = _render_figure_viewer(figure_state)
    next_tab = "data" if _DATA_TAB_RE.search(str(user_message or "")) else None
    for update in handle_chat(chat_history, user_message, active_session_id, figure_state):
        if isinstance(update, (tuple, list)):
            update = tuple(update)
            figs = update[4] if len(update) > 4 else None
            figs_html = figs.get("value") if isinstance(figs, dict) else figs
            if isinstance(figs_html, str) and figs_html != baseline_figures:
                next_tab = "plots"
            elif next_tab is None and len(update) > 1 and not _is_skip(update[1]):
                next_tab = "map"
            yield (*update, active_session_id)
        else:
            yield update
    if next_tab:
        _NEXT_TAB[active_session_id] = next_tab


def _chat_denied(markdown, chat_history, user_message, session_id, figure_state):
    history = list(chat_history or []) + [
        {"role": "user", "content": str(user_message or "")},
        {"role": "assistant", "content": markdown},
    ]
    return history, gr.skip(), gr.skip(), _status_clear_update(), gr.skip(), figure_state, session_id


def _status_message(label: str, started_at: float | None = None) -> str:
    elapsed = ""
    if started_at is not None:
        seconds = max(0, int(time.time() - started_at))
        elapsed = f"<span class='spatchat-status-time'>Working for {seconds}s</span>"
    dots = "<span class='spatchat-typing-dots'><span></span><span></span><span></span></span>"
    return (
        f"<span class='spatchat-status'>{dots}{label}</span>"
        + (f" {elapsed}" if elapsed else "")
    )


def _status_clear_update():
    return gr.update(value="")


def _run_with_status(session_id: str, status_label: str, fn):
    started_at = time.time()
    yield gr.skip(), gr.skip(), gr.skip(), gr.update(value=_status_message(status_label, started_at)), gr.skip(), gr.skip()

    result_box = {}

    def _target():
        try:
            _activate_session(session_id)
            result_box["result"] = fn()
        except Exception as exc:  # pragma: no cover
            result_box["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()

    last_second = -1
    while worker.is_alive():
        current_second = int(time.time() - started_at)
        if current_second != last_second:
            yield gr.skip(), gr.skip(), gr.skip(), gr.update(value=_status_message(status_label, started_at)), gr.skip(), gr.skip()
            last_second = current_second
        time.sleep(0.15)

    worker.join()
    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("result")


def _stream_llm_reply(chat_history, prior_history, user_message, session_id, context_safe):
    started_at = time.time()
    assistant_message = None

    if ask_llm_stream is not None:
        yield gr.skip(), gr.skip(), gr.skip(), gr.update(value=_status_message("Thinking", started_at)), gr.skip(), gr.skip()
        try:
            streamed_any = False
            last_second = -1
            for chunk in ask_llm_stream(prior_history, user_message, context=context_safe):
                if not chunk:
                    current_second = int(time.time() - started_at)
                    if current_second != last_second:
                        yield gr.skip(), gr.skip(), gr.skip(), gr.update(value=_status_message("Thinking", started_at)), gr.skip(), gr.skip()
                        last_second = current_second
                    continue
                streamed_any = True
                if assistant_message is None:
                    assistant_message = {"role": "assistant", "content": ""}
                    chat_history.append(assistant_message)
                else:
                    assistant_message["content"] = assistant_message["content"].rstrip("▌")
                assistant_message["content"] += str(chunk) + "▌"
                yield chat_history, gr.skip(), gr.skip(), gr.update(value=_status_message("Thinking", started_at)), gr.skip(), gr.skip()

            if streamed_any and assistant_message is not None:
                assistant_message["content"] = assistant_message["content"].rstrip("▌").strip()
                if not assistant_message["content"]:
                    assistant_message["content"] = "How can I help you?"
                if _room_scope_nudge() not in assistant_message["content"]:
                    assistant_message["content"] = assistant_message["content"] + "\n\n" + _room_scope_nudge()
                yield chat_history, gr.skip(), gr.skip(), _status_clear_update(), gr.skip(), gr.skip()
                return

            if assistant_message is not None and chat_history and chat_history[-1] is assistant_message:
                chat_history.pop()
        except Exception:
            if assistant_message is not None and chat_history and chat_history[-1] is assistant_message:
                chat_history.pop()

    yield gr.skip(), gr.skip(), gr.skip(), gr.update(value=_status_message("Thinking", started_at)), gr.skip(), gr.skip()
    try:
        _, llm_output = ask_llm(prior_history, user_message, context=context_safe)
    except Exception:
        llm_output = None

    reply = (llm_output or "").strip() or "How can I help you?"
    reply = reply.rstrip()
    if _room_scope_nudge() not in reply:
        reply = f"{reply}\n\n{_room_scope_nudge()}"
    chat_history.append({"role": "assistant", "content": reply})
    yield chat_history, gr.skip(), gr.skip(), _status_clear_update(), gr.skip(), gr.skip()
    return

# --------------------------------------------------------------------------------------
# Analysis + chat handler
# --------------------------------------------------------------------------------------
def handle_chat(chat_history, user_message, session_id, figure_state):
    session_id, initial_df = _get_ready_dataframe(session_id)
    chat_history = list(chat_history or [])
    figure_state = list(figure_state or [])
    msg = (user_message or "").strip()
    if not msg:
        chat_history.append({"role": "assistant", "content": "How can I help you?"})
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return

    prior_history = list(chat_history)
    chat_history.append({"role": "user", "content": msg})
    yield chat_history, gr.skip(), gr.skip(), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state

    pending = _pending_questions(session_id)
    normalized_msg = re.sub(r"\bbbmm\b", "dbbmm", msg, flags=re.IGNORECASE)
    if re.fullmatch(r"\s*(please\s+)?(run\s+|do\s+)?analy[sz]e(\s+(the|my)\s+(data|dataset|movement(\s+data)?))?\s*[.!]*\s*", msg, flags=re.IGNORECASE):
        normalized_msg = "movement analysis"

    cmd = parse_metadata_command(normalized_msg)
    if cmd:
        session_id, df = _get_ready_dataframe(session_id)
        if df is None or "latitude" not in df or "longitude" not in df:
            chat_history.append({"role": "assistant", "content": "Please upload a CSV first."})
            yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
            return

        df2, msg = try_apply_user_mapping(df, cmd)
        df2, _ = _ensure_default_animal_id(df2)
        set_cached_df(df2)
        _remember_dataset_session(session_id)
        _persist_session_dataframe(session_id, df2)

        pending["need_id"] = (ID_COL not in df2.columns)
        pending["need_ts"] = (TS_COL not in df2.columns)

        follow = []
        if not pending["need_id"] and not pending["need_ts"]:
            follow.append("Great — ID and timestamps detected.")
        elif pending["need_id"] and not pending["id_prompted"]:
            follow.append("I couldn’t detect an individual ID column. If your data has one, say: “ID column is tag_id”.")
            pending["id_prompted"] = True
        elif pending["need_ts"] and not pending["ts_prompted"]:
            follow.append("I couldn’t detect a timestamp column. If your data has one, say: “Timestamp column is datetime”.")
            pending["ts_prompted"] = True

        chat_history.append({"role": "assistant", "content": msg + (" " + " ".join(follow) if follow else "")})
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return

    if _wants_parameters_overview(normalized_msg):
        chat_history.append({"role": "assistant", "content": _parameters_markdown(None)})
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return

    if _is_parameter_question(normalized_msg, "mcp"):
        chat_history.append({"role": "assistant", "content": _parameters_markdown("mcp")})
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return
    if _is_parameter_question(normalized_msg, "kde"):
        chat_history.append({"role": "assistant", "content": _parameters_markdown("kde")})
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return
    if _is_parameter_question(normalized_msg, "akde"):
        chat_history.append({"role": "assistant", "content": _parameters_markdown("akde")})
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return
    if _is_parameter_question(normalized_msg, "locoh"):
        chat_history.append({"role": "assistant", "content": _parameters_markdown("locoh")})
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return
    if _is_parameter_question(normalized_msg, "dbbmm"):
        chat_history.append({"role": "assistant", "content": _parameters_markdown("dbbmm")})
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return
    if _wants_movement_help(normalized_msg):
        chat_history.append({"role": "assistant", "content": _movement_analysis_help()})
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return

    movement_explanation = _movement_interpretation_response(normalized_msg)
    if movement_explanation:
        chat_history.append({"role": "assistant", "content": movement_explanation})
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return

    # Parse intents/keywords
    mcp_list, kde_list, akde_list = [], [], []
    locoh_requested = False
    locoh_params = None
    dbbmm_list = []
    dbbmm_params = None
    kde_params = KDEParams()
    akde_params = AKDEParams()
    movement_requests = _parse_movement_requests(normalized_msg)

    warned_about_kde_100 = False

    # Fallback: keyword parse
    parse_msg = normalized_msg
    msg_lower = normalized_msg.lower()
    has_mcp = re.search(r"\bmcp\b", msg_lower) is not None
    has_kde = re.search(r"\bkde\b", msg_lower) is not None
    has_akde = re.search(r"\bakde\b", msg_lower) is not None
    has_locoh = re.search(r"\blocoh\b", msg_lower) is not None
    has_dbbmm = re.search(r"\bdbbmm\b", msg_lower) is not None
    if _is_level_only_followup(normalized_msg):
        followup_estimator = _followup_estimator_from_history(prior_history)
        if followup_estimator is not None:
            parse_msg = f"{followup_estimator} {normalized_msg}"
            msg_lower = parse_msg.lower()
            has_mcp = re.search(r"\bmcp\b", msg_lower) is not None
            has_kde = re.search(r"\bkde\b", msg_lower) is not None
            has_akde = re.search(r"\bakde\b", msg_lower) is not None
            has_locoh = re.search(r"\blocoh\b", msg_lower) is not None
            has_dbbmm = re.search(r"\bdbbmm\b", msg_lower) is not None
    home_range_request_count = sum(1 for flag in (has_mcp, has_kde, has_akde, has_locoh, has_dbbmm) if flag)
    if has_mcp:
        parsed = _parse_levels_allow_100(parse_msg)
        mcp_list = parsed or [95]

    if has_kde:
        kde_list = parse_levels_from_text(parse_msg) or [95]
        toks = parse_kv_tokens(parse_msg)
        bw_m = None
        if "bw" in toks or "bandwidth" in toks or "h" in toks:
            v = toks.get("bw", toks.get("bandwidth", toks.get("h")))
            try:
                bw_m = _parse_distance_to_meters(v)
            except Exception:
                bw_m = None

        kernel = toks.get("kernel", "gaussian").lower()
        gres_m = None
        if "gres" in toks or "grid_res" in toks:
            v = toks.get("gres", toks.get("grid_res"))
            try:
                gres_m = _parse_distance_to_meters(v)
            except Exception:
                gres_m = None

        kde_params = KDEParams(bandwidth_m=bw_m, kernel=kernel, grid_res_m=gres_m, grid_size=200)

    if has_akde:
        looks_like_levels = bool(re.search(r"\b(100|[1-9]?[0-9])\b", parse_msg))
        looks_like_kv = "=" in parse_msg
        looks_actiony = any(v in msg_lower for v in _PARAM_VERBS)
        bare_akde = msg_lower.strip() == "akde"
        grouped_home_range_request = home_range_request_count > 1

        if looks_like_levels or looks_like_kv or looks_actiony or bare_akde or grouped_home_range_request:
            parsed = _parse_levels_allow_100(parse_msg)
            akde_list = parsed or [95]
            toks = parse_kv_tokens(parse_msg)

            bw_m = None
            if "bw" in toks or "bandwidth" in toks or "h" in toks:
                v = toks.get("bw", toks.get("bandwidth", toks.get("h")))
                try:
                    bw_m = _parse_distance_to_meters(v)
                except Exception:
                    bw_m = None

            gres_m = None
            if "gres" in toks or "grid_res" in toks:
                v = toks.get("gres", toks.get("grid_res"))
                try:
                    gres_m = _parse_distance_to_meters(v)
                except Exception:
                    gres_m = None

            def _get_float(keys, default=None):
                for k in keys:
                    if k in toks:
                        try:
                            return float(toks[k])
                        except Exception:
                            pass
                return default

            def _get_int(keys, default):
                for k in keys:
                    if k in toks:
                        try:
                            return int(toks[k])
                        except Exception:
                            pass
                return int(default)

            model = toks.get("model", "auto").lower()
            if model not in {"auto", "ou", "ouf"}:
                model = "auto"

            akde_params = AKDEParams(
                bandwidth_m=bw_m,
                grid_res_m=gres_m,
                grid_size=200,
                extent_buffer_mult=3.0,
                min_points=_get_int(["min_points"], 15),
                variogram_fast=True,
                variogram_res=_get_int(["variogram_res"], 1),
                variogram_dt=_get_float(["variogram_dt"], None),
                model=model,
                use_effective_n=True,
                estimate_velocity_tau=True,
                smooth=True,
                cores=_get_int(["cores", "core"], 0),
            )

    if has_locoh:
        locoh_requested = True
        toks = parse_kv_tokens(parse_msg)
        method = "k"
        k = int(toks.get("k", 10))
        a = toks.get("a")
        r = toks.get("r")
        if a is not None:
            method = "a"
            try:
                a = float(a)
            except Exception:
                a = None
        elif r is not None:
            method = "r"
            try:
                r = float(r)
            except Exception:
                r = None
        iso_str = toks.get("isopleths")
        if iso_str:
            iso = tuple(int(s) for s in re.split(r"[,\s]+", iso_str) if s)
        else:
            parsed = _parse_levels_allow_100(parse_msg)
            iso = tuple(parsed) if parsed else (95,)
        locoh_params = LoCoHParams(method=method, k=k, a=a, r=r, isopleths=iso)

    if has_dbbmm:
        looks_like_levels = bool(re.search(r"\b(100|[1-9]?[0-9])\b", parse_msg))
        looks_like_kv = "=" in parse_msg
        looks_actiony = any(v in msg_lower for v in _PARAM_VERBS)
        grouped_home_range_request = home_range_request_count > 1
        if looks_like_levels or looks_like_kv or looks_actiony or grouped_home_range_request:
            dbbmm_list = _parse_levels_allow_100(parse_msg) or [95]
            toks = parse_kv_tokens(parse_msg)

            def _get_float(keys, default):
                for k in keys:
                    if k in toks:
                        try:
                            return float(toks[k])
                        except Exception:
                            pass
                return float(default)

            def _get_int(keys, default):
                for k in keys:
                    if k in toks:
                        try:
                            return int(toks[k])
                        except Exception:
                            pass
                return int(default)

            dbbmm_params = DBBMMParams(
                location_error_m=_get_float(["le", "locerr", "sigma"], 30.0),
                window_size=_get_int(["window", "w"], 31),
                margin=_get_int(["margin", "m"], 11),
                raster_resolution_m=_get_float(["res", "resolution"], 50.0),
                buffer_m=_get_float(["buf", "buffer"], 1000.0),
                n_substeps=_get_int(["subs", "substeps"], 40),
                isopleths=tuple(dbbmm_list),
            )

    # If not an analysis request, answer naturally
    if not mcp_list and not kde_list and not akde_list and not locoh_requested and not dbbmm_list and not any(movement_requests[k] for k in ("displacement", "step_lengths", "turning_angles", "autocorrelation", "hmm")):
        if pending["need_id"] and not pending["id_prompted"]:
            chat_history.append({"role": "assistant", "content":
                                 "I couldn’t detect an individual ID column. If your data has one, say: “ID column is tag_id”. "
                                 "Otherwise, you can still proceed - I’ll treat all rows as one animal."})
            pending["id_prompted"] = True
            yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
            return

        if pending["need_ts"] and not pending["ts_prompted"]:
            chat_history.append({"role": "assistant", "content":
                                 "I couldn’t detect a timestamp column. If your data has one, say: “Timestamp column is datetime”. "
                                 "Otherwise, you can proceed - I’ll plot points without drawing tracks."})
            pending["ts_prompted"] = True
            yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
            return

        context_raw = _current_dataset_context(session_id)
        context_safe = _json_safe(context_raw)
        yield from _stream_llm_reply(chat_history, prior_history, msg, session_id, context_safe)
        return

    # Must have lon/lat prepared
    session_id, df = _get_ready_dataframe(session_id)
    if df is None or "latitude" not in df or "longitude" not in df:
        chat_history.append({"role": "assistant", "content": "Please upload a CSV first (with latitude/longitude)."})
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return

    df, single_id_assumed_for_analysis = _ensure_default_animal_id(df)
    set_cached_df(df)
    _persist_session_dataframe(session_id, df)

    timestamp_required = _requested_timestamp_dependent_analyses(akde_list, dbbmm_list, movement_requests)
    has_timestamp = "timestamp" in df.columns and pd.to_datetime(df["timestamp"], errors="coerce", utc=True).notna().any()
    if timestamp_required and not has_timestamp:
        requested_text = ", ".join(timestamp_required)
        chat_history.append({
            "role": "assistant",
            "content": (
                f"I can’t complete {requested_text} with this dataset because it appears to contain presence-only locations without timestamps. "
                "Those analyses require timestamped movement tracks, so please upload data with a valid timestamp column."
            ),
        })
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return

    def _run_requested_analyses():
        df_local = get_cached_df()
        df_local, _ = _ensure_default_animal_id(df_local)
        set_cached_df(df_local)
        results_exist = False
        local_warned_about_kde_100 = warned_about_kde_100

        local_kde_list = list(kde_list)
        if local_kde_list:
            if 100 in local_kde_list or any("100" in s for s in msg.split()):
                local_warned_about_kde_100 = True
            local_kde_list = [min(k, 99) for k in local_kde_list]

        local_locoh_result = None
        local_locoh_error = None
        local_dbbmm_result = None
        local_akde_ran = False
        movement_msgs = []

        if mcp_list:
            from methods.mcp import add_mcps

            add_mcps(df_local, mcp_list)
            requested_percents.update(mcp_list)
            _merge_repro_manifest(mcp_list, [], None, [], None, False, None, [], None, {})
            results_exist = True

        if local_kde_list:
            add_kdes(df_local, local_kde_list, params=kde_params)
            requested_kde_percents.update(local_kde_list)
            _merge_repro_manifest([], local_kde_list, kde_params, [], None, False, None, [], None, {})
            results_exist = True

        if akde_list:
            try:
                if "timestamp" not in df_local.columns:
                    raise ValueError("AKDE requires a timestamp column.")
                add_akdes(df_local, akde_list, params=akde_params)
                requested_akde_percents.update(akde_list)
                _merge_repro_manifest([], [], None, akde_list, akde_params, False, None, [], None, {})
                local_akde_ran = True
                results_exist = True
            except Exception as exc:
                movement_msgs.append(f"AKDE error: {exc}")

        if locoh_requested:
            try:
                df_locoh = df_local.rename(columns={"longitude": "lon", "latitude": "lat"})
                local_locoh_result = compute_locoh(
                    df=df_locoh,
                    id_col="animal_id",
                    x_col="lon",
                    y_col="lat",
                    params=locoh_params or LoCoHParams(),
                )
                set_locoh_results(local_locoh_result)
                _merge_repro_manifest([], [], None, [], None, True, locoh_params or LoCoHParams(), [], None, {})
                results_exist = True
            except Exception as exc:
                local_locoh_error = str(exc)

        if dbbmm_list:
            try:
                if "timestamp" not in df_local.columns:
                    raise ValueError("dBBMM requires a timestamp column to model movement between fixes.")
                local_dbbmm_result = compute_dbbmm(
                    df=df_local,
                    id_col="animal_id",
                    x_col="longitude",
                    y_col="latitude",
                    ts_col="timestamp",
                    params=dbbmm_params,
                    outputs_dir=get_output_dir(),
                )
                set_dbbmm_results(local_dbbmm_result)
                requested_dbbmm_percents.update(dbbmm_list)
                _merge_repro_manifest([], [], None, [], None, False, None, dbbmm_list, dbbmm_params, {})
                results_exist = True
            except Exception as exc:
                movement_msgs.append(f"dBBMM error: {exc}")

        if any(movement_requests[k] for k in ("displacement", "step_lengths", "turning_angles", "autocorrelation", "hmm")):
            try:
                movement_msgs.extend(_run_movement_analyses(df_local, movement_requests, get_output_dir()))
                _merge_repro_manifest([], [], None, [], None, False, None, [], None, movement_requests)
                results_exist = True
            except Exception as exc:
                movement_msgs.append(f"Movement analysis error: {exc}")

        effective_locoh = local_locoh_result if local_locoh_result is not None else get_locoh_results()
        effective_dbbmm = local_dbbmm_result if local_dbbmm_result is not None else get_dbbmm_results()
        effective_akde = get_akde_results()

        map_html = build_results_map(
            df_local,
            mcp_results=mcp_results,
            kde_results=kde_results,
            requested_percents=requested_percents,
            requested_kde_percents=requested_kde_percents,
            akde_results=effective_akde,
            requested_akde_percents=requested_akde_percents,
            locoh_result=effective_locoh,
            dbbmm_result=effective_dbbmm,
        )

        msgs = []
        if single_id_assumed_for_analysis:
            msgs.append(f"No animal ID column was detected, so all rows were treated as `{_DEFAULT_ANIMAL_ID}`.")
        if requested_percents:
            msgs.append(f"MCP home ranges ({', '.join(str(p) for p in sorted(requested_percents))}%) calculated.")
        if requested_kde_percents:
            msgs.append(f"KDE home ranges ({', '.join(str(p) for p in sorted(requested_kde_percents))}%) calculated (raster and contours).")
        if local_akde_ran:
            msgs.append(f"AKDE home ranges ({', '.join(str(p) for p in sorted(set(akde_list)))}%) calculated (raster and contours).")
        if local_warned_about_kde_100:
            msgs.append("Note: KDE at 100% is not supported and has been replaced by 99%.")
        if local_locoh_result:
            msgs.append(f"LoCoH ({(locoh_params or LoCoHParams()).method}) complete.")
        if local_locoh_error:
            msgs.append(f"LoCoH error: {local_locoh_error}")
        if dbbmm_list and local_dbbmm_result:
            toks = parse_kv_tokens(parse_msg)
            used_defaults = not any(k in toks for k in ("le", "locerr", "sigma", "window", "w", "margin", "m", "res", "resolution", "buf", "buffer", "subs", "substeps"))
            default_note = " using default settings" if used_defaults else ""
            msgs.append(f"dBBMM UDs computed ({', '.join(str(p) for p in sorted(set(dbbmm_list)))}% isopleths){default_note}. Raster and contours added.")
        msgs.extend(movement_msgs)
        if results_exist:
            msgs.append("_The download button is below the chat controls in the left column._")

        _write_repro_scripts(get_output_dir())
        archive_path = save_all_mcps_zip()
        discovered_figures = _collect_figure_entries(get_output_dir(), akde_data=effective_akde)
        next_figures, active_index = _merge_figure_history(figure_state, discovered_figures)
        assistant_text = " ".join(msgs) if msgs else "Done."
        return map_html, archive_path, results_exist, assistant_text, next_figures, active_index

    try:
        map_html, archive_path, results_exist, assistant_text, next_figures, active_index = yield from _run_with_status(
            session_id,
            "Thinking",
            _run_requested_analyses,
        )
    except Exception as exc:
        chat_history.append({"role": "assistant", "content": f"Analysis error: {exc}"})
        yield chat_history, gr.skip(), gr.update(visible=False), _status_clear_update(), gr.update(value=_render_figure_viewer(figure_state)), figure_state
        return

    chat_history.append({"role": "assistant", "content": assistant_text})
    yield (
        chat_history,
        gr.update(value=map_html),
        gr.update(value=archive_path, visible=results_exist),
        _status_clear_update(),
        gr.update(value=_render_figure_viewer(next_figures, active_index)),
        next_figures,
    )

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
_reset_session_state()
_prune_stale_sessions()

_SPLITTER_HEAD = """
<script>
window.spatchatBeginResize = function(event) {
  event.preventDefault();
  const workarea = document.getElementById("spatchat-workarea");
  const sidebar = document.getElementById("spatchat-sidebar");
  const mapcol = document.getElementById("spatchat-mapcol");
  if (!workarea || !sidebar || !mapcol) return false;

  const rect = workarea.getBoundingClientRect();
  const minWidth = 80;
  const maxWidth = Math.max(minWidth, rect.width - 80);
  let latestClientX = event.clientX;
  let rafId = 0;

  function apply() {
    rafId = 0;
    const next = Math.min(Math.max(latestClientX - rect.left, minWidth), maxWidth);
    const value = `${Math.round(next)}px`;
    sidebar.style.width = value;
    sidebar.style.flexBasis = value;
    sidebar.style.maxWidth = value;
    mapcol.style.flex = "1 1 auto";
  }

  function scheduleApply(clientX) {
    latestClientX = clientX;
    if (!rafId) {
      rafId = window.requestAnimationFrame(apply);
    }
  }

  function onMove(moveEvent) { scheduleApply(moveEvent.clientX); }

  function onUp() {
    document.body.classList.remove("spatchat-resizing");
    if (rafId) {
      window.cancelAnimationFrame(rafId);
      rafId = 0;
    }
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
  }

  document.body.classList.add("spatchat-resizing");
  scheduleApply(event.clientX);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  return false;
};

window.spatchatScrollChatToBottom = () => {
  const root = document.getElementById("spatchat-chatbot");
  if (!root) return;
  const candidates = [root, ...root.querySelectorAll("div")];
  let best = null;
  let bestScore = -1;
  for (const el of candidates) {
    const score = (el.scrollHeight || 0) - (el.clientHeight || 0);
    if (score > bestScore + 20) {
      best = el;
      bestScore = score;
    }
  }
  if (best) {
    best.scrollTop = best.scrollHeight;
  }
};

window.spatchatWatchChatScroll = () => {
  const root = document.getElementById("spatchat-chatbot");
  if (!root || root.__spatchatScrollInit) return;
  root.__spatchatScrollInit = true;
  const scrollNow = () => requestAnimationFrame(() => window.spatchatScrollChatToBottom && window.spatchatScrollChatToBottom());
  const observer = new MutationObserver(scrollNow);
  observer.observe(root, { childList: true, subtree: true, characterData: true });
  scrollNow();
};

window.spatchatFitViewport = function() {
  const workarea = document.getElementById("spatchat-workarea");
  if (!workarea) return false;
  const top = workarea.getBoundingClientRect().top;
  const available = Math.max(360, Math.round(window.innerHeight - top - 12));
  workarea.style.height = available + "px";
  return true;
};

window.spatchatLockSidebarScrollX = function() {
  const sidebar = document.getElementById("spatchat-sidebar");
  if (!sidebar || sidebar.__spatchatScrollXLock) return !!sidebar;
  sidebar.__spatchatScrollXLock = true;
  sidebar.addEventListener("scroll", () => {
    if (sidebar.scrollLeft !== 0) sidebar.scrollLeft = 0;
  });
  return true;
};

(function initFitViewportWithRetry() {
  let resizeTimer = 0;
  function onResize() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => window.spatchatFitViewport && window.spatchatFitViewport(), 100);
  }
  function tryInit() {
    window.spatchatLockSidebarScrollX && window.spatchatLockSidebarScrollX();
    if (window.spatchatFitViewport()) {
      window.addEventListener("resize", onResize);
      return;
    }
    let attempts = 0;
    const timer = setInterval(() => {
      attempts += 1;
      window.spatchatLockSidebarScrollX && window.spatchatLockSidebarScrollX();
      if (window.spatchatFitViewport() || attempts > 40) {
        clearInterval(timer);
        window.addEventListener("resize", onResize);
      }
    }, 250);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", tryInit, { once: true });
  } else {
    tryInit();
  }
})();

window.spatchatGallery = (() => {
  let items = [];
  let idx = 0;
  let box = null;

  function ensure() {
    if (box) return box;
    box = document.createElement("div");
    box.id = "spatchat-lightbox";
    box.innerHTML =
      '<div class="sl-backdrop" data-sl="close"></div>' +
      '<section class="sl-panel" role="dialog" aria-modal="true" aria-label="Enlarged figure">' +
      '<header class="sl-head"><div class="sl-copy"><div class="sl-title"></div><div class="sl-sub"></div></div>' +
      '<div class="sl-nav"><span class="sl-count"></span>' +
      '<button type="button" data-sl="prev" aria-label="Previous">&larr;</button>' +
      '<button type="button" data-sl="next" aria-label="Next">&rarr;</button>' +
      '<button type="button" data-sl="close" class="sl-close" aria-label="Close">&times;</button></div></header>' +
      '<div class="sl-body"></div></section>';
    box.addEventListener("click", (event) => {
      const target = event.target.closest("[data-sl]");
      if (!target) return;
      const action = target.dataset.sl;
      if (action === "close") close();
      if (action === "prev") show(idx - 1);
      if (action === "next") show(idx + 1);
    });
    document.body.appendChild(box);
    document.addEventListener("keydown", (event) => {
      if (!box.classList.contains("open")) return;
      if (event.key === "Escape") close();
      if (event.key === "ArrowLeft") show(idx - 1);
      if (event.key === "ArrowRight") show(idx + 1);
    });
    return box;
  }

  function show(i) {
    if (!items.length) return;
    idx = Math.max(0, Math.min(items.length - 1, i));
    const item = items[idx];
    const b = ensure();
    b.querySelector(".sl-title").textContent = item.title;
    b.querySelector(".sl-sub").textContent = item.subtitle;
    b.querySelector(".sl-count").textContent = (idx + 1) + " / " + items.length;
    b.querySelector('[data-sl="prev"]').disabled = idx <= 0;
    b.querySelector('[data-sl="next"]').disabled = idx >= items.length - 1;
    const body = b.querySelector(".sl-body");
    body.innerHTML = "";
    if (item.src) {
      const img = document.createElement("img");
      img.src = item.src;
      img.alt = item.title;
      body.appendChild(img);
    } else {
      const table = document.createElement("div");
      table.className = "sl-table";
      table.innerHTML = item.table;
      body.appendChild(table);
    }
    body.scrollTop = 0;
  }

  function open(el) {
    const gallery = el.closest(".spatchat-gallery");
    if (!gallery) return false;
    const thumbs = Array.from(gallery.querySelectorAll(".spatchat-thumb"));
    items = thumbs.map((t) => ({
      title: t.dataset.title || "Figure",
      subtitle: t.dataset.subtitle || "",
      src: (t.querySelector("img") || {}).src || "",
      table: (t.querySelector(".spatchat-thumb-table") || {}).innerHTML || "",
    }));
    ensure().classList.add("open");
    document.body.classList.add("spatchat-lightbox-open");
    show(thumbs.indexOf(el));
    return false;
  }

  function close() {
    if (box) box.classList.remove("open");
    document.body.classList.remove("spatchat-lightbox-open");
  }

  return { open, close };
})();

window.spatchatApplyTheme = function(theme) {
  const dark = theme === "dark";
  document.documentElement.classList.toggle("spatchat-dark", dark);
  document.body.classList.toggle("spatchat-dark", dark);
  document.body.classList.toggle("dark", dark);
  try { localStorage.setItem("spatchat-theme", dark ? "dark" : "light"); } catch (e) {}
};
window.spatchatToggleTheme = function() {
  window.spatchatApplyTheme(document.body.classList.contains("spatchat-dark") ? "light" : "dark");
};
(function initSpatchatTheme() {
  let saved = null;
  try { saved = localStorage.getItem("spatchat-theme"); } catch (e) {}
  const apply = () => window.spatchatApplyTheme(saved === "dark" ? "dark" : "light");
  if (document.body) apply(); else document.addEventListener("DOMContentLoaded", apply, { once: true });
})();

window.spatchatFigureViewer = (() => {
  const selector = ".spatchat-figure-root[data-payload]";
  const layerId = "spatchat-figure-layer";
  let layerObserver = null;
  let activeInteraction = null;
  const minWidth = 240;
  const minHeight = 160;

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function parsePayload(root) {
    try {
      return JSON.parse(root.dataset.payload || "{}");
    } catch (error) {
      return {};
    }
  }

  function refreshStateFromPayload(root) {
    const payload = parsePayload(root);
    const key = root.dataset.payload || "";
    if (root.__spatchatFigurePayload === key && root.__spatchatFigureState) {
      return root.__spatchatFigureState;
    }
    const previous = root.__spatchatFigureState || {};
    const viewportWidth = window.innerWidth || 1280;
    const viewportHeight = window.innerHeight || 800;
    root.__spatchatFigurePayload = key;
    root.__spatchatFigureState = {
      figures: Array.isArray(payload.figures) ? payload.figures : [],
      activeIndex: Number.isFinite(payload.activeIndex) ? payload.activeIndex : (previous.activeIndex || 0),
      isOpen: typeof payload.isOpen === "boolean" ? payload.isOpen : (previous.isOpen ?? true),
      isMinimized: false,
      width: Number.isFinite(payload.width) ? payload.width : (previous.width || clamp(Math.round(viewportWidth * 0.40), minWidth, 720)),
      height: Number.isFinite(payload.height) ? payload.height : (previous.height || clamp(Math.round(viewportHeight * 0.60), minHeight, viewportHeight - 80)),
      x: Number.isFinite(payload.x) ? payload.x : (previous.x ?? null),
      y: Number.isFinite(payload.y) ? payload.y : (previous.y ?? null),
    };
    return root.__spatchatFigureState;
  }

  function syncState(root, state) {
    root.__spatchatFigureState = state;
    root.__spatchatFigurePayload = "";
    root.dataset.payload = JSON.stringify(state);
  }

  function fitToViewport(state) {
    const viewportWidth = window.innerWidth || 1280;
    const viewportHeight = window.innerHeight || 800;
    const maxWidth = Math.max(minWidth, viewportWidth - 8);
    const maxHeight = Math.max(minHeight, viewportHeight - 8);
    const visibleMargin = 72;
    state.width = clamp(state.width || Math.round(viewportWidth * 0.40), minWidth, maxWidth);
    state.height = clamp(state.height || Math.round(viewportHeight * 0.60), minHeight, maxHeight);
    if (!Number.isFinite(state.x) || !Number.isFinite(state.y)) {
      state.x = Math.round((viewportWidth - state.width) / 2);
      state.y = Math.round((viewportHeight - state.height) / 2);
    }
    state.x = clamp(state.x, visibleMargin - state.width, viewportWidth - visibleMargin);
    state.y = clamp(state.y, visibleMargin - state.height, viewportHeight - visibleMargin);
  }

  function applyModalRect(modal, state) {
    if (!modal) return;
    modal.style.width = `${Math.round(state.width)}px`;
    modal.style.height = `${Math.round(state.height)}px`;
    modal.style.left = `${Math.round(state.x)}px`;
    modal.style.top = `${Math.round(state.y)}px`;
  }

  function applyAction(root, action) {
    const state = refreshStateFromPayload(root);
    const figures = Array.isArray(state.figures) ? state.figures : [];
    if (!figures.length && action !== "open") return;
    if (action === "back") {
      state.activeIndex = clamp((state.activeIndex || 0) - 1, 0, figures.length - 1);
    }
    if (action === "forward") {
      state.activeIndex = clamp((state.activeIndex || 0) + 1, 0, figures.length - 1);
    }
    if (action === "open") {
      state.isOpen = true;
      state.isMinimized = false;
    }
    if (action === "close") {
      state.isOpen = false;
      state.isMinimized = false;
    }
    fitToViewport(state);
    syncState(root, state);
    render(root);
  }

  function rootFromElement(element) {
    return element ? element.closest(".spatchat-figure-root") : null;
  }

  function stopActiveInteraction() {
    if (!activeInteraction) return;
    window.removeEventListener("mousemove", activeInteraction.onMove);
    window.removeEventListener("mouseup", activeInteraction.onUp);
    if (activeInteraction.bodyClass) {
      document.body.classList.remove(activeInteraction.bodyClass);
    }
    const shield = activeInteraction.root ? activeInteraction.root.querySelector(".spatchat-figure-interaction-shield") : null;
    if (shield) {
      shield.classList.remove("is-active");
    }
    activeInteraction = null;
  }

  function startInteraction(root, bodyClass, onMove, onUp) {
    stopActiveInteraction();
    activeInteraction = { root, bodyClass, onMove, onUp };
    if (bodyClass) {
      document.body.classList.add(bodyClass);
    }
    const shield = root ? root.querySelector(".spatchat-figure-interaction-shield") : null;
    if (shield) {
      shield.classList.add("is-active");
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  function beginDrag(event, root) {
    if (!event || event.button !== 0) return;
    if (event.target.closest("[data-action]")) return;
    const modal = root.querySelector(".spatchat-figure-modal");
    if (!modal) return;
    const state = refreshStateFromPayload(root);
    fitToViewport(state);
    const startX = event.clientX;
    const startY = event.clientY;
    const originX = state.x;
    const originY = state.y;
    event.preventDefault();

    function onMove(moveEvent) {
      state.x = originX + (moveEvent.clientX - startX);
      state.y = originY + (moveEvent.clientY - startY);
      const viewportWidth = window.innerWidth || 1280;
      const viewportHeight = window.innerHeight || 800;
      const visibleMargin = 72;
      state.x = clamp(state.x, visibleMargin - state.width, viewportWidth - visibleMargin);
      state.y = clamp(state.y, visibleMargin - state.height, viewportHeight - visibleMargin);
      applyModalRect(modal, state);
    }

    function onUp() {
      stopActiveInteraction();
      fitToViewport(state);
      syncState(root, state);
      render(root);
    }

    startInteraction(root, "spatchat-figure-moving", onMove, onUp);
    return false;
  }

  function beginResize(event, root, direction) {
    if (!event || event.button !== 0) return;
    const modal = root.querySelector(".spatchat-figure-modal");
    if (!modal) return;
    const state = refreshStateFromPayload(root);
    fitToViewport(state);
    const startX = event.clientX;
    const startY = event.clientY;
    const originX = state.x;
    const originY = state.y;
    const originWidth = state.width;
    const originHeight = state.height;
    const viewportWidth = window.innerWidth || 1280;
    const viewportHeight = window.innerHeight || 800;
    const visibleMargin = 72;
    event.preventDefault();
    event.stopPropagation();

    function onMove(moveEvent) {
      const dx = moveEvent.clientX - startX;
      const dy = moveEvent.clientY - startY;
      if (direction.includes("e")) {
        state.width = clamp(originWidth + dx, minWidth, viewportWidth + originX - visibleMargin);
      }
      if (direction.includes("s")) {
        state.height = clamp(originHeight + dy, minHeight, viewportHeight + originY - visibleMargin);
      }
      if (direction.includes("w")) {
        const nextX = clamp(originX + dx, visibleMargin - originWidth, originX + originWidth - minWidth);
        state.width = clamp(originWidth - (nextX - originX), minWidth, viewportWidth - visibleMargin - nextX);
        state.x = nextX;
      }
      if (direction.includes("n")) {
        const nextY = clamp(originY + dy, visibleMargin - originHeight, originY + originHeight - minHeight);
        state.height = clamp(originHeight - (nextY - originY), minHeight, viewportHeight - visibleMargin - nextY);
        state.y = nextY;
      }
      state.x = clamp(state.x, visibleMargin - state.width, viewportWidth - visibleMargin);
      state.y = clamp(state.y, visibleMargin - state.height, viewportHeight - visibleMargin);
      applyModalRect(modal, state);
    }

    function onUp() {
      stopActiveInteraction();
      fitToViewport(state);
      syncState(root, state);
      render(root);
    }

    startInteraction(root, "spatchat-figure-resizing", onMove, onUp);
    return false;
  }

  function render(root) {
    const state = refreshStateFromPayload(root);
    const figures = Array.isArray(state.figures) ? state.figures : [];
    const launcher = root.querySelector(".spatchat-figure-launcher");
    const modal = root.querySelector(".spatchat-figure-modal");
    const backdrop = root.querySelector(".spatchat-figure-modal-backdrop");
    const title = root.querySelector(".spatchat-figure-card-title");
    const meta = root.querySelector(".spatchat-figure-card-meta");
    const image = root.querySelector(".spatchat-figure-card-image");
    const table = root.querySelector(".spatchat-figure-card-table");
    const count = root.querySelector(".spatchat-figure-modal-count");
    const back = root.querySelector('[data-action="back"]');
    const forward = root.querySelector('[data-action="forward"]');
    if (!figures.length) {
      stopActiveInteraction();
      root.style.display = "none";
      return;
    }
    root.style.display = "";
    state.activeIndex = clamp(Number.isFinite(state.activeIndex) ? state.activeIndex : 0, 0, figures.length - 1);
    const active = figures[state.activeIndex];
    if (launcher) {
      launcher.textContent = `Plots/Tables (${figures.length})`;
      launcher.style.display = !state.isOpen ? "inline-flex" : "none";
    }
    if (modal) {
      modal.classList.toggle("is-hidden", !state.isOpen);
      if (!state.isOpen) {
        stopActiveInteraction();
      }
      fitToViewport(state);
      applyModalRect(modal, state);
    }
    if (backdrop) {
      backdrop.classList.add("is-hidden");
    }
    if (title) title.textContent = active.title || "Figure";
    if (meta) meta.textContent = active.subtitle || "";
    if (image) {
      image.src = active.src || "";
      image.alt = active.title || "Figure";
      image.style.display = active.src ? "block" : "none";
    }
    if (table) {
      table.innerHTML = active.tableHtml || "";
      table.style.display = active.tableHtml ? "block" : "none";
    }
    if (count) count.textContent = `${state.activeIndex + 1} / ${figures.length}`;
    if (back) back.disabled = state.activeIndex <= 0;
    if (forward) forward.disabled = state.activeIndex >= figures.length - 1;
    syncState(root, state);
  }

  function init(root) {
    if (!root) return;
    if (!root.__spatchatFigureInit) {
      root.__spatchatFigureInit = true;
      window.addEventListener("resize", () => render(root));
    }
    render(root);
  }

  function initForLayer() {
    const layer = document.getElementById(layerId);
    if (!layer) return;
    layer.querySelectorAll(selector).forEach(init);
  }

  function watchLayer() {
    const layer = document.getElementById(layerId);
    if (!layer) return;
    if (layerObserver) {
      layerObserver.disconnect();
    }
    layerObserver = new MutationObserver(() => initForLayer());
    layerObserver.observe(layer, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      window.spatchatWatchChatScroll && window.spatchatWatchChatScroll();
      initForLayer();
      watchLayer();
    }, { once: true });
  } else {
    window.spatchatWatchChatScroll && window.spatchatWatchChatScroll();
    initForLayer();
    watchLayer();
  }

  function handleAction(element, event) {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    const root = rootFromElement(element);
    if (!root) return false;
    applyAction(root, element.dataset.action);
    return false;
  }

  function handleDrag(element, event) {
    const root = rootFromElement(element);
    if (!root) return false;
    return beginDrag(event, root);
  }

  function handleResize(element, event) {
    const root = rootFromElement(element);
    if (!root) return false;
    return beginResize(event, root, element.dataset.resize || "se");
  }

  return { initForLayer, render, handleAction, handleDrag, handleResize };
})();

window.spatchatInitMapDropzone = function() {
  const mapcol = document.getElementById("spatchat-mapcol");
  const mapHost = document.getElementById("spatchat-map");
  if (!mapcol || !mapHost) return false;
  if (mapHost.__spatchatDropzoneInit) return true;
  mapHost.__spatchatDropzoneInit = true;

  function hasFiles(event) {
    return event.dataTransfer && Array.from(event.dataTransfer.types || []).includes("Files");
  }

  function dropFiles(files) {
    if (!files || !files.length) return;
    const hiddenInput = document.querySelector('#spatchat-input-row input[type="file"]');
    if (!hiddenInput) return;
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    hiddenInput.files = dt.files;
    hiddenInput.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function bindTarget(target) {
    target.addEventListener("dragover", (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      mapcol.classList.add("spatchat-map-dragover");
    });
    target.addEventListener("dragleave", () => {
      mapcol.classList.remove("spatchat-map-dragover");
    });
    target.addEventListener("drop", (event) => {
      if (!hasFiles(event)) return;
      event.preventDefault();
      mapcol.classList.remove("spatchat-map-dragover");
      dropFiles(event.dataTransfer.files);
    });
  }

  // Covers the mapcol area outside the map iframe (padding/margins).
  bindTarget(mapcol);

  // The Leaflet map itself renders inside a same-origin iframe, which does
  // not bubble native drag/drop events to the parent document, so it needs
  // its own listeners. The iframe element is re-used across map updates but
  // its contentDocument is replaced on every reload, so re-wire on "load".
  function wireIframeDoc(iframe) {
    if (!iframe) return;
    const wire = () => {
      let doc;
      try {
        doc = iframe.contentDocument;
      } catch (e) {
        return;
      }
      if (!doc || doc.__spatchatDropWired) return;
      doc.__spatchatDropWired = true;
      bindTarget(doc);
    };
    iframe.addEventListener("load", wire);
    wire();
  }

  const existingIframe = mapHost.querySelector("iframe");
  if (existingIframe) wireIframeDoc(existingIframe);

  const observer = new MutationObserver(() => {
    const iframe = mapHost.querySelector("iframe");
    if (iframe) wireIframeDoc(iframe);
  });
  observer.observe(mapHost, { childList: true, subtree: true });

  return true;
};

(function initMapDropzoneWithRetry() {
  function setUploadTooltip() {
    const btn = document.getElementById("spatchat-file-input");
    if (btn && !btn.title) {
      btn.title = "Upload movement CSV (.csv or .txt) — or drag a file onto the map";
    }
    return !!btn;
  }

  function tryInit() {
    document.addEventListener("dragover", (event) => {
      if (event.dataTransfer && Array.from(event.dataTransfer.types || []).includes("Files")) {
        event.preventDefault();
      }
    });
    document.addEventListener("drop", (event) => {
      if (event.dataTransfer && Array.from(event.dataTransfer.types || []).includes("Files") && !event.defaultPrevented) {
        event.preventDefault();
      }
    });
    const dropzoneReady = window.spatchatInitMapDropzone();
    const tooltipReady = setUploadTooltip();
    if (dropzoneReady && tooltipReady) return;
    let attempts = 0;
    const timer = setInterval(() => {
      attempts += 1;
      const dzOk = window.spatchatInitMapDropzone();
      const ttOk = setUploadTooltip();
      if ((dzOk && ttOk) || attempts > 40) {
        clearInterval(timer);
      }
    }, 250);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", tryInit, { once: true });
  } else {
    tryInit();
  }
})();
</script>
"""

with gr.Blocks(title="SpatChat: Home Range Analysis") as demo:
    session_state = gr.State(None)
    library_state = gr.State(_new_library())
    figure_state = gr.State([])
    image_kwargs = {
        "value": "logo_long1.png",
        "show_label": False,
        "type": "filepath",
        "elem_id": "logo-img",
    }
    if _component_accepts_kw(gr.Image, "buttons"):
        image_kwargs["buttons"] = []
    gr.HTML("""
    <style>
    :root {
        --spatchat-connect-bg: #f5f8f5;
        --spatchat-surface: #ffffff;
        --spatchat-surface-alt: #eef3ee;
        --spatchat-border: rgba(16, 36, 26, 0.1);
        --spatchat-text: #10241a;
        --spatchat-text-secondary: #5c6b63;
        --spatchat-text-subtle: #8a978f;
        --spatchat-accent: #1f7a4f;
        --spatchat-accent-hover: #17603d;
        --spatchat-accent-soft: #e3f2e8;
        --spatchat-shadow: 0 1px 3px rgba(16, 36, 26, 0.08), 0 1px 2px rgba(16, 36, 26, 0.06);
    }
    :host,
    html,
    body,
    gradio-app,
    body > gradio-app,
    .gradio-container,
    .gradio-container > .main,
    .gradio-container .main,
    #spatchat-workarea,
    #spatchat-sidebar,
    #spatchat-mapcol,
    #spatchat-splitter {
        background: var(--spatchat-connect-bg) !important;
        color: var(--spatchat-text) !important;
    }
    .gradio-container {
        max-width: 100% !important;
        padding-left: 4px !important;
        padding-right: 4px !important;
        overflow: visible !important;
    }
    #spatchat-workarea {
        overflow-x: auto;
        overflow-y: hidden;
        height: 80vh;
        padding-bottom: 8px;
        box-sizing: border-box;
    }
    #spatchat-workarea > .gradio-row {
        min-width: 900px;
        min-height: 0;
        height: 100%;
        align-items: stretch;
        flex-wrap: nowrap;
        gap: 0 !important;
    }
    #spatchat-sidebar {
        min-width: 0 !important;
        max-width: 1200px;
        flex: 0 0 420px !important;
        width: 420px;
        height: 100%;
        display: flex !important;
        flex-direction: column !important;
        flex-wrap: nowrap !important;
        overflow-y: auto;
        overflow-x: clip;
        padding-right: 4px !important;
        font-size: 15px;
        scrollbar-color: #c9d3cb var(--spatchat-surface-alt);
        scrollbar-width: thin;
    }
    #spatchat-sidebar::-webkit-scrollbar {
        width: 12px;
    }
    #spatchat-sidebar::-webkit-scrollbar-track {
        background: var(--spatchat-surface-alt);
        border-radius: 999px;
    }
    #spatchat-sidebar::-webkit-scrollbar-thumb {
        background: #c9d3cb;
        border-radius: 999px;
        border: 2px solid var(--spatchat-surface-alt);
    }
    #spatchat-sidebar::-webkit-scrollbar-thumb:hover {
        background: #b5c1b7;
    }
    #spatchat-sidebar::-webkit-scrollbar-button:single-button {
        display: block;
        height: 12px;
        background-color: #c9d3cb;
        border-radius: 999px;
        border: 2px solid var(--spatchat-surface-alt);
        background-repeat: no-repeat;
        background-position: center;
        background-size: 7px 7px;
    }
    #spatchat-sidebar::-webkit-scrollbar-button:single-button:vertical:decrement {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 6.5 5 3.5 8 6.5' fill='none' stroke='%23516358' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }
    #spatchat-sidebar::-webkit-scrollbar-button:single-button:vertical:increment {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 3.5 5 6.5 8 3.5' fill='none' stroke='%23516358' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }
    #spatchat-sidebar > div {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    #spatchat-sidebar .form {
        background: transparent !important;
    }
    #spatchat-sidebar label span,
    #spatchat-sidebar .label-wrap span {
        color: var(--spatchat-text-secondary) !important;
    }
    #spatchat-chatbot,
    #spatchat-input-row,
    #spatchat-user-input,
    #spatchat-file-input,
    #spatchat-x-col,
    #spatchat-y-col,
    #spatchat-crs,
    #spatchat-confirm,
    #spatchat-download {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
    #spatchat-chatbot {
        padding-left: 0 !important;
        padding-right: 0 !important;
        --chatbot-body-text-size: 16px;
        font-size: 16px !important;
        flex: 1 1 auto !important;
        height: auto !important;
        min-height: 0 !important;
    }
    #spatchat-chatbot > div,
    #spatchat-chatbot .wrap,
    #spatchat-chatbot .bubble-wrap,
    #spatchat-chatbot .message-wrap,
    #spatchat-chatbot .message,
    #spatchat-chatbot .panel-wrap {
        margin-left: 0 !important;
        margin-right: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    #spatchat-chatbot .panel-wrap,
    #spatchat-chatbot .wrap,
    #spatchat-chatbot .bubble-wrap,
    #spatchat-chatbot .message-wrap,
    #spatchat-chatbot .message,
    #spatchat-chatbot .message-row.panel,
    #spatchat-chatbot .message-row.panel.user-row,
    #spatchat-chatbot .message-row.panel.bot-row,
    #spatchat-chatbot .wrapper {
        background: var(--spatchat-connect-bg) !important;
        color: var(--spatchat-text) !important;
    }
    #spatchat-chatbot,
    #spatchat-chatbot > div {
        background: var(--spatchat-connect-bg) !important;
        border: none !important;
        box-shadow: none !important;
    }
    #spatchat-chatbot .md,
    #spatchat-chatbot .prose,
    #spatchat-chatbot .md *,
    #spatchat-chatbot .prose * {
        color: var(--spatchat-text) !important;
    }
    #spatchat-chatbot code,
    #spatchat-chatbot pre {
        background: var(--spatchat-surface-alt) !important;
        color: var(--spatchat-text) !important;
    }
    #spatchat-chatbot,
    #spatchat-chatbot > div,
    #spatchat-chatbot .panel-wrap,
    #spatchat-chatbot .wrap {
        scrollbar-color: #c9d3cb var(--spatchat-connect-bg);
        scrollbar-width: thin;
    }
    #spatchat-chatbot::-webkit-scrollbar,
    #spatchat-chatbot > div::-webkit-scrollbar,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar,
    #spatchat-chatbot .wrap::-webkit-scrollbar {
        width: 12px;
    }
    #spatchat-chatbot::-webkit-scrollbar-track,
    #spatchat-chatbot > div::-webkit-scrollbar-track,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar-track,
    #spatchat-chatbot .wrap::-webkit-scrollbar-track {
        background: var(--spatchat-connect-bg);
        border-radius: 999px;
    }
    #spatchat-chatbot::-webkit-scrollbar-thumb,
    #spatchat-chatbot > div::-webkit-scrollbar-thumb,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar-thumb,
    #spatchat-chatbot .wrap::-webkit-scrollbar-thumb {
        background: #c9d3cb;
        border-radius: 999px;
        border: 2px solid var(--spatchat-connect-bg);
    }
    #spatchat-chatbot::-webkit-scrollbar-thumb:hover,
    #spatchat-chatbot > div::-webkit-scrollbar-thumb:hover,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar-thumb:hover,
    #spatchat-chatbot .wrap::-webkit-scrollbar-thumb:hover {
        background: #b5c1b7;
    }
    #spatchat-chatbot::-webkit-scrollbar-button:single-button,
    #spatchat-chatbot > div::-webkit-scrollbar-button:single-button,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar-button:single-button,
    #spatchat-chatbot .wrap::-webkit-scrollbar-button:single-button {
        display: block;
        height: 12px;
        background-color: #c9d3cb;
        border-radius: 999px;
        border: 2px solid var(--spatchat-connect-bg);
        background-repeat: no-repeat;
        background-position: center;
        background-size: 7px 7px;
    }
    #spatchat-chatbot::-webkit-scrollbar-button:single-button:vertical:decrement,
    #spatchat-chatbot > div::-webkit-scrollbar-button:single-button:vertical:decrement,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar-button:single-button:vertical:decrement,
    #spatchat-chatbot .wrap::-webkit-scrollbar-button:single-button:vertical:decrement {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 6.5 5 3.5 8 6.5' fill='none' stroke='%23516358' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }
    #spatchat-chatbot::-webkit-scrollbar-button:single-button:vertical:increment,
    #spatchat-chatbot > div::-webkit-scrollbar-button:single-button:vertical:increment,
    #spatchat-chatbot .panel-wrap::-webkit-scrollbar-button:single-button:vertical:increment,
    #spatchat-chatbot .wrap::-webkit-scrollbar-button:single-button:vertical:increment {
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><path d='M2 3.5 5 6.5 8 3.5' fill='none' stroke='%23516358' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/></svg>");
    }
    #spatchat-input-row {
        align-items: center !important;
        gap: 8px !important;
        flex-wrap: nowrap !important;
        background: var(--spatchat-surface) !important;
        border: 1px solid var(--spatchat-border) !important;
        border-top: none !important;
        border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
        padding: 6px 8px !important;
        margin-top: -8px !important;
    }
    #spatchat-input-row .form {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        background: transparent !important;
        border: none !important;
    }
    #spatchat-chatbot {
        border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
        overflow: hidden !important;
    }
    #spatchat-chatbot > div {
        border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
    }
    #spatchat-file-input {
        flex: 0 0 34px !important;
        width: 34px !important;
        height: 34px !important;
        min-width: 34px !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 50% !important;
        border: none !important;
        box-shadow: none !important;
        background: var(--spatchat-accent) !important;
        color: #ffffff !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: background 0.15s ease, transform 0.1s ease !important;
    }
    #spatchat-file-input:hover {
        background: var(--spatchat-accent-hover) !important;
    }
    #spatchat-file-input:active {
        transform: scale(0.92) !important;
    }
    #spatchat-sample-btn {
        align-self: flex-start !important;
        margin-top: 6px !important;
        padding: 4px 10px !important;
        min-width: 0 !important;
        background: transparent !important;
        color: var(--spatchat-accent) !important;
        border: none !important;
        box-shadow: none !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        transition: background 0.15s ease !important;
    }
    #spatchat-sample-btn:hover {
        background: var(--spatchat-accent-soft) !important;
        border-radius: var(--radius-md) !important;
    }
    #spatchat-user-input,
    #spatchat-user-input > div,
    #spatchat-user-input .wrap,
    #spatchat-user-input label {
        background: transparent !important;
    }
    #spatchat-user-input {
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
    }
    #spatchat-user-input .input-container {
        align-items: center !important;
    }
    #spatchat-user-input textarea,
    #spatchat-user-input input {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        min-height: 34px !important;
        box-sizing: border-box !important;
        padding: 8px 4px !important;
        line-height: 20px !important;
        color: var(--spatchat-text) !important;
    }
    #spatchat-user-input textarea {
        max-height: 200px !important;
        overflow-y: auto !important;
        resize: vertical !important;
    }
    #spatchat-user-input input {
        overflow: hidden !important;
    }
    #spatchat-user-input textarea::placeholder,
    #spatchat-user-input input::placeholder {
        color: var(--spatchat-text-subtle) !important;
        text-align: left !important;
    }
    #spatchat-chatbot .message-row.panel,
    #spatchat-chatbot .message-row.panel.user-row,
    #spatchat-chatbot .message-row.panel.bot-row {
        background: transparent !important;
        margin: 0 !important;
        padding: 6px 0 !important;
    }
    #spatchat-chatbot .message-row.panel.user-row,
    #spatchat-chatbot .message-row.panel.bot-row {
        background: transparent !important;
    }
    #spatchat-chatbot .message {
        width: 100% !important;
    }
    #spatchat-chatbot .flex-wrap.user,
    #spatchat-chatbot .flex-wrap.bot {
        border: none !important;
        box-shadow: none !important;
        padding: 8px 12px !important;
    }
    #spatchat-chatbot .flex-wrap.bot {
        background: transparent !important;
        border-bottom-left-radius: var(--radius-md) !important;
    }
    #spatchat-chatbot .flex-wrap.user {
        background: var(--spatchat-accent-soft) !important;
        border-bottom-right-radius: var(--radius-md) !important;
    }
    #spatchat-chatbot .message-row.panel.user-row {
        align-self: stretch !important;
        width: 100% !important;
        background: transparent !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .message {
        display: flex !important;
        justify-content: flex-end !important;
        width: 100% !important;
        margin-bottom: 0 !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .message > div {
        display: block !important;
        width: max-content !important;
        max-width: min(72%, 680px) !important;
        height: auto !important;
        flex: 0 1 auto !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        align-self: flex-end !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .user {
        display: inline-block !important;
        vertical-align: top !important;
        width: max-content !important;
        max-width: 100% !important;
        height: auto !important;
        flex: 0 1 auto !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        align-self: flex-end !important;
        background: var(--spatchat-accent-soft) !important;
        color: var(--spatchat-text) !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: var(--radius-md) !important;
        border-bottom-right-radius: 0 !important;
        padding: 8px 12px !important;
        white-space: normal !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
        text-align: left !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user {
        display: block !important;
        width: max-content !important;
        max-width: 100% !important;
        min-width: max-content !important;
        height: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .user *,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user * {
        text-align: left !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .user > div,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user > div,
    #spatchat-chatbot .message-row.panel.user-row .user > div > div,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user > div > div,
    #spatchat-chatbot .message-row.panel.user-row .user .wrapper,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user .wrapper,
    #spatchat-chatbot .message-row.panel.user-row .user .md,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user .md,
    #spatchat-chatbot .message-row.panel.user-row .user .prose,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user .prose,
    #spatchat-chatbot .message-row.panel.user-row .user p,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user p {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        width: auto !important;
        max-width: 100% !important;
        min-width: 0 !important;
        height: auto !important;
        min-height: 0 !important;
        flex-grow: 0 !important;
        gap: 0 !important;
        margin: 0 !important;
        white-space: normal !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .user .md pre,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user .md pre {
        white-space: pre !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .user .md > *,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user .md > *,
    #spatchat-chatbot .message-row.panel.user-row .user .prose > *,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user .prose > * {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    #spatchat-chatbot .message-row.panel.user-row .user div:empty,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user div:empty,
    #spatchat-chatbot .message-row.panel.user-row .user p:empty,
    #spatchat-chatbot .message-row.panel.user-row .flex-wrap.user p:empty {
        display: none !important;
    }
    #spatchat-chatbot .spatchat-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--spatchat-text-secondary);
        font-weight: 500;
    }
    #spatchat-chatbot .spatchat-status-time {
        color: var(--spatchat-text-secondary);
        font-size: 0.92em;
    }
    #spatchat-status {
        min-height: 24px;
        margin: 4px 0 8px 0 !important;
        color: var(--spatchat-text-secondary);
        font-size: 14px;
    }
    #spatchat-status .spatchat-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--spatchat-text-secondary);
        font-weight: 500;
    }
    #spatchat-status .spatchat-status-time {
        color: var(--spatchat-text-secondary);
        font-size: 0.92em;
    }
    .spatchat-typing-dots {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .spatchat-typing-dots span {
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: var(--spatchat-accent);
        animation: spatchat-typing-bounce 1.2s ease-in-out infinite;
    }
    .spatchat-typing-dots span:nth-child(2) {
        animation-delay: 0.15s;
    }
    .spatchat-typing-dots span:nth-child(3) {
        animation-delay: 0.3s;
    }
    @keyframes spatchat-typing-bounce {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
        30% { transform: translateY(-3px); opacity: 1; }
    }
    #spatchat-splitter {
        min-width: 10px;
        max-width: 10px;
        width: 10px;
        flex: 0 0 10px !important;
        position: relative;
        padding: 0 !important;
        margin: 0 !important;
        height: 100%;
        min-height: 0;
        align-self: stretch !important;
        background: transparent !important;
        overflow: visible !important;
    }
    #spatchat-splitter > div,
    #spatchat-splitter .gradio-html,
    #spatchat-splitter .gradio-html > div {
        height: 100%;
        min-height: 0;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
    }
    #spatchat-splitter .spatchat-splitter-handle {
        position: absolute;
        inset: 0;
        display: block;
        width: 100%;
        height: 100%;
        cursor: col-resize;
        user-select: none;
        background: transparent;
    }
    #spatchat-splitter .spatchat-splitter-handle::before {
        content: "";
        position: absolute;
        top: 8px;
        bottom: 8px;
        left: 50%;
        width: 1px;
        transform: translateX(-50%);
        background: var(--spatchat-border);
        border-radius: 999px;
        transition: background 120ms ease, width 120ms ease;
    }
    #spatchat-splitter:hover .spatchat-splitter-handle::before,
    body.spatchat-resizing #spatchat-splitter .spatchat-splitter-handle::before {
        width: 2px;
        background: var(--spatchat-accent);
    }
    body.spatchat-resizing,
    body.spatchat-resizing * {
        cursor: col-resize !important;
        user-select: none !important;
    }
    #spatchat-mapcol {
        min-width: 0 !important;
        flex: 1 1 auto !important;
        width: auto !important;
        height: 100%;
        position: relative;
        overflow: visible;
    }
    #spatchat-mapcol.spatchat-map-dragover {
        outline: 3px dashed var(--spatchat-accent);
        outline-offset: -3px;
        border-radius: var(--radius-lg);
    }
    #spatchat-mapcol.spatchat-map-dragover::after {
        content: "Drop CSV to upload";
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(31, 122, 79, 0.14);
        color: var(--spatchat-accent-hover);
        font-size: 13px;
        font-weight: 600;
        pointer-events: none;
        z-index: 500;
        border-radius: var(--radius-lg);
    }
    #spatchat-map {
        min-height: 0;
        height: 100%;
        overflow: visible;
        width: 100%;
        padding-bottom: 10px;
        box-sizing: border-box;
    }
    #spatchat-map > div {
        height: 100%;
    }
    #spatchat-map iframe {
        width: 100% !important;
        height: 100% !important;
        min-height: 0;
        border: none;
    }
    body.spatchat-resizing #spatchat-map iframe {
        pointer-events: none !important;
    }
    #spatchat-figure-layer {
        position: relative;
        width: 0;
        height: 0;
        z-index: 2000;
        background: transparent !important;
        pointer-events: auto;
        overflow: visible !important;
    }
    #spatchat-figure-layer > div,
    #spatchat-figure-layer .spatchat-figure-root {
        width: 0;
        height: 0;
        background: transparent !important;
        overflow: visible !important;
    }
    .spatchat-figure-root {
        position: relative;
        pointer-events: auto;
    }
    .spatchat-figure-launcher {
        position: fixed;
        right: 22px;
        bottom: 22px;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 11px 16px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 999px;
        background: rgba(9, 13, 21, 0.96);
        color: #f4f7fb;
        pointer-events: auto;
        box-shadow: 0 18px 44px rgba(0, 0, 0, 0.38);
        font-size: 13px;
        font-weight: 700;
        cursor: pointer;
    }
    .spatchat-figure-modal-backdrop {
        position: fixed;
        inset: 0;
        background: transparent;
        backdrop-filter: none;
        pointer-events: none;
    }
    .spatchat-figure-interaction-shield {
        position: fixed;
        inset: 0;
        background: transparent;
        pointer-events: none;
        z-index: 1999;
    }
    .spatchat-figure-interaction-shield.is-active {
        pointer-events: auto;
        cursor: inherit;
    }
    .spatchat-figure-modal-backdrop.is-hidden {
        display: none;
    }
    .spatchat-figure-modal {
        position: fixed;
        left: clamp(12px, calc(50vw - 280px), calc(100vw - 572px));
        top: clamp(12px, calc(50vh - 230px), calc(100vh - 472px));
        width: min(560px, calc(100vw - 24px));
        height: min(460px, calc(100vh - 24px));
        max-width: calc(100vw - 24px);
        max-height: calc(100vh - 24px);
        margin: 0;
        display: flex;
        flex-direction: column;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(9, 13, 21, 0.97);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.46);
        color: #f4f7fb;
        overflow: hidden;
        pointer-events: auto;
        transform: none;
        z-index: 2001;
    }
    .spatchat-figure-modal.is-hidden {
        display: none;
    }
    .spatchat-figure-modal-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding: 6px 10px;
        background: linear-gradient(135deg, rgba(36, 51, 82, 0.94), rgba(11, 15, 25, 0.94));
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        cursor: move;
        user-select: none;
    }
    .spatchat-figure-modal-copy {
        display: flex;
        flex-direction: column;
        gap: 0;
        min-width: 0;
    }
    .spatchat-figure-modal-title {
        font-size: 12px;
        font-weight: 700;
        line-height: 1.15;
    }
    .spatchat-figure-modal-count {
        font-size: 10px;
        line-height: 1.1;
        color: rgba(226, 233, 245, 0.74);
    }
    .spatchat-figure-modal-nav {
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .spatchat-figure-modal-btn {
        width: 24px;
        height: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border: none;
        border-radius: 7px;
        background: rgba(255, 255, 255, 0.08);
        color: #f4f7fb;
        font-size: 13px;
        font-weight: 700;
        cursor: pointer;
    }
    .spatchat-figure-modal-btn:disabled {
        opacity: 0.35;
        cursor: default;
    }
    .spatchat-figure-modal-btn-icon svg {
        width: 14px;
        height: 14px;
        stroke: currentColor;
        stroke-width: 1.8;
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .spatchat-figure-modal-btn-close {
        font-size: 16px;
        line-height: 0.7;
        padding-bottom: 6px;
    }
    .spatchat-figure-modal-body {
        flex: 1 1 auto;
        overflow: auto;
        padding: 18px;
        display: flex;
        flex-direction: column;
    }
    .spatchat-figure-resize-handle {
        position: absolute;
        z-index: 3;
        background: transparent;
    }
    .spatchat-figure-resize-handle.is-n {
        left: 12px;
        right: 12px;
        top: -4px;
        height: 8px;
        cursor: ns-resize;
    }
    .spatchat-figure-resize-handle.is-s {
        left: 12px;
        right: 12px;
        bottom: -4px;
        height: 8px;
        cursor: ns-resize;
    }
    .spatchat-figure-resize-handle.is-e {
        top: 12px;
        bottom: 12px;
        right: -4px;
        width: 8px;
        cursor: ew-resize;
    }
    .spatchat-figure-resize-handle.is-w {
        top: 12px;
        bottom: 12px;
        left: -4px;
        width: 8px;
        cursor: ew-resize;
    }
    .spatchat-figure-resize-handle.is-ne,
    .spatchat-figure-resize-handle.is-nw,
    .spatchat-figure-resize-handle.is-se,
    .spatchat-figure-resize-handle.is-sw {
        width: 18px;
        height: 18px;
    }
    .spatchat-figure-resize-handle.is-ne {
        top: -4px;
        right: -4px;
        cursor: nesw-resize;
    }
    .spatchat-figure-resize-handle.is-nw {
        top: -4px;
        left: -4px;
        cursor: nwse-resize;
    }
    .spatchat-figure-resize-handle.is-se {
        right: -4px;
        bottom: -4px;
        cursor: nwse-resize;
    }
    .spatchat-figure-resize-handle.is-sw {
        left: -4px;
        bottom: -4px;
        cursor: nesw-resize;
    }
    .spatchat-figure-resize-handle.is-se::before {
        content: "";
        position: absolute;
        right: 7px;
        bottom: 7px;
        width: 11px;
        height: 11px;
        border-right: 2px solid rgba(244, 247, 251, 0.72);
        border-bottom: 2px solid rgba(244, 247, 251, 0.72);
        border-bottom-right-radius: 2px;
    }
    .spatchat-figure-card {
        display: flex;
        flex-direction: column;
        flex: 1 1 auto;
        gap: 10px;
        min-height: 0;
        padding: 16px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .spatchat-figure-card-title {
        font-size: 16px;
        font-weight: 700;
        line-height: 1.25;
    }
    .spatchat-figure-card-meta {
        font-size: 13px;
        color: rgba(226, 233, 245, 0.74);
        line-height: 1.35;
    }
    .spatchat-figure-card-image {
        width: 100%;
        flex: 1 1 auto;
        min-height: 0;
        height: 100%;
        max-height: none;
        object-fit: contain;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
    }
    .spatchat-figure-card-table {
        flex: 1 1 auto;
        min-height: 0;
        overflow: auto;
    }
    .spatchat-figure-table-wrap {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .spatchat-figure-table-note {
        font-size: 11px;
        color: rgba(226, 233, 245, 0.72);
    }
    .spatchat-figure-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
    }
    .spatchat-figure-table th,
    .spatchat-figure-table td {
        padding: 7px 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        text-align: left;
        white-space: nowrap;
    }
    .spatchat-figure-table th {
        position: sticky;
        top: 0;
        background: rgba(13, 18, 29, 0.98);
        z-index: 1;
        font-weight: 700;
    }
    .spatchat-figure-table-empty {
        font-size: 12px;
        color: rgba(226, 233, 245, 0.72);
    }
    #spatchat-x-col,
    #spatchat-y-col,
    #spatchat-crs {
        background: var(--spatchat-surface) !important;
        border: 1px solid var(--spatchat-border) !important;
        border-radius: var(--radius-md) !important;
        padding: 8px 10px !important;
        margin-top: 8px !important;
        box-shadow: var(--spatchat-shadow);
    }
    #spatchat-confirm {
        margin-top: 8px !important;
        background: var(--spatchat-accent) !important;
        color: #ffffff !important;
        border: none !important;
        transition: background 0.15s ease, transform 0.1s ease !important;
    }
    #spatchat-confirm:hover {
        background: var(--spatchat-accent-hover) !important;
    }
    #spatchat-confirm:active {
        transform: scale(0.98) !important;
    }
    #spatchat-download {
        display: flex;
        justify-content: center;
        width: 100%;
        margin-top: 10px;
    }
    #spatchat-download button {
        min-width: 220px;
        width: 100%;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px;
        text-align: center;
        background: var(--spatchat-accent) !important;
        color: #ffffff !important;
        border: none !important;
        transition: background 0.15s ease, transform 0.1s ease !important;
    }
    #spatchat-download button:hover {
        background: var(--spatchat-accent-hover) !important;
    }
    #spatchat-download button:active {
        transform: scale(0.98) !important;
    }
    footer {
        display: none !important;
    }
    #spatchat-header-row {
        align-items: center !important;
        justify-content: space-between !important;
        flex-wrap: nowrap !important;
        gap: 16px !important;
        background: var(--spatchat-accent-soft) !important;
        border-radius: var(--radius-lg, 12px) !important;
        padding: 6px 18px !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 1000 !important;
    }
    #logo-img,
    #logo-img > div,
    #logo-img button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        justify-content: flex-start !important;
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
    }
    #logo-img img {
        height: 32px;
        margin: 6px 0;
        border-radius: 6px;
    }
    #spatchat-navbar {
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
    }
    body.spatchat-resizing,
    body.spatchat-resizing * {
        cursor: col-resize !important;
        user-select: none !important;
    }
    body.spatchat-figure-moving,
    body.spatchat-figure-moving * {
        cursor: move !important;
        user-select: none !important;
    }
    body.spatchat-figure-resizing,
    body.spatchat-figure-resizing * {
        user-select: none !important;
    }
    @media (max-width: 900px) {
        .spatchat-figure-launcher {
            right: 14px;
            bottom: 14px;
        }
        .spatchat-figure-modal {
            left: 12px;
            top: 12px;
            width: calc(100vw - 24px);
            height: calc(100vh - 24px);
            max-height: calc(100vh - 24px);
        }
        .spatchat-figure-modal-head {
            padding: 14px;
            align-items: flex-start;
            flex-direction: column;
        }
        .spatchat-figure-modal-nav {
            width: 100%;
            justify-content: flex-end;
        }
        .spatchat-figure-modal-body {
            padding: 14px;
        }
        .spatchat-figure-card-image {
            height: 100%;
        }
    }
    #spatchat-share-btn {
        transition: background 0.15s ease, transform 0.1s ease;
    }
    #spatchat-share-btn:hover {
        background: #17603d !important;
    }
    #spatchat-share-btn:active {
        transform: scale(0.96);
    }
    .spatchat-map-empty-hint {
        position: absolute !important;
        inset: 0 !important;
        display: flex !important;
        align-items: flex-end !important;
        justify-content: center !important;
        padding-bottom: 8% !important;
        pointer-events: none !important;
        z-index: 500 !important;
    }
    .spatchat-map-empty-hint-card {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        gap: 4px !important;
        padding: 8px 14px !important;
        border-radius: 10px !important;
        background: #ffffff !important;
        border: 1px solid rgba(16, 36, 26, 0.12) !important;
        color: #10241a !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-align: center !important;
        box-shadow: 0 8px 24px rgba(16, 36, 26, 0.16) !important;
    }
    .spatchat-map-empty-hint-card * {
        color: inherit !important;
    }
    .spatchat-map-empty-hint-icon {
        font-size: 16px !important;
    }
    .spatchat-map-empty-hint-sub {
        font-size: 10px !important;
        font-weight: 500 !important;
        color: #5c6b63 !important;
    }
    </style>
    """)
    gr.HTML("""
    <style>
    /* ---- Tabs (Map / Plots / Datasheet) ---- */
    #spatchat-mapcol { flex: 1 1 0 !important; width: 0 !important; min-width: 0 !important; overflow: hidden !important; }
    #spatchat-tabs { height: 100%; width: 100%; display: flex; flex-direction: column; background: transparent !important; }
    #spatchat-tabs > .tab-wrapper, #spatchat-tabs > .tab-container { flex: 0 0 auto; }
    #spatchat-tabs > div[role="tabpanel"], #spatchat-tabs .tabitem {
        flex: 1 1 auto; min-height: 0; height: calc(100% - 44px); overflow: auto;
        background: var(--spatchat-surface) !important; border-radius: 0 0 var(--radius-lg) var(--radius-lg);
        padding: 8px !important;
    }
    #spatchat-tabs button[role="tab"].selected { color: var(--spatchat-accent) !important; border-color: var(--spatchat-accent) !important; }
    #spatchat-datasheet { height: 100%; }
    #spatchat-tabs .tabitem { padding: 0 !important; margin-top: 4px; }
    #spatchat-tabs .tabitem:not(#spatchat-plots-pane):not(#spatchat-data-pane) { background: transparent !important; overflow: hidden !important; }
    #spatchat-tabs .tabitem:not(#spatchat-plots-pane):not(#spatchat-data-pane) > .column,
    #spatchat-tabs .tabitem:not(#spatchat-plots-pane):not(#spatchat-data-pane) #spatchat-map { height: 100% !important; width: 100% !important; padding-bottom: 0 !important; }
    #spatchat-tabs div.tabitem:not(#spatchat-plots-pane):not(#spatchat-data-pane):not(#x) { padding: 0 !important; }
    #spatchat-tabs #spatchat-map .html-container { padding: 0 !important; height: 100% !important; }
    #spatchat-tabs #spatchat-map .prose, #spatchat-tabs #spatchat-map .prose > div:not(.spatchat-map-empty-hint), #spatchat-tabs #spatchat-map .prose > div:not(.spatchat-map-empty-hint) > div,
    #spatchat-tabs #spatchat-map iframe { height: 100% !important; width: 100% !important; }
    #spatchat-plots-pane, #spatchat-data-pane { padding: 8px !important; }
    /* ---- Plots tab: dock the figure viewer instead of a floating window ---- */
    #spatchat-figure-layer, #spatchat-figure-layer > div, #spatchat-figure-layer .spatchat-figure-root {
        width: 100% !important; height: 100% !important; overflow: visible !important;
    }
    #spatchat-figure-layer .spatchat-figure-launcher,
    #spatchat-figure-layer .spatchat-figure-modal-backdrop,
    #spatchat-figure-layer .spatchat-figure-resize-handle,
    #spatchat-figure-layer .spatchat-figure-modal-btn-close { display: none !important; }
    #spatchat-figure-layer .spatchat-figure-modal {
        position: static !important; display: flex !important; width: 100% !important; height: 100% !important;
        max-width: none !important; max-height: none !important; box-shadow: none !important;
        border-radius: var(--radius-lg) !important;
    }
    #spatchat-figure-layer .spatchat-figure-modal-head { cursor: default !important; }
    #spatchat-figure-layer .spatchat-figure-card-image { max-width: 100% !important; height: auto !important; }
    #spatchat-tabs, #spatchat-tabs .tabitem, #spatchat-plots-pane, #spatchat-plots-pane * { min-width: 0 !important; }
    #spatchat-tabs, #spatchat-tabs .tabitem, #spatchat-plots-pane, #spatchat-plots-pane * { max-width: 100% !important; }
    #spatchat-tabs .tabitem { overflow: auto !important; align-items: flex-start; }
    #spatchat-figure-layer, #spatchat-figure-layer > div, #spatchat-figure-layer .spatchat-figure-root,
    #spatchat-figure-layer .spatchat-figure-modal { height: auto !important; }
    #spatchat-figure-layer .spatchat-figure-modal-body { overflow: auto; }
    #spatchat-figure-layer .spatchat-figure-modal { background: var(--spatchat-surface-alt) !important; border: 1px solid var(--spatchat-border) !important; color: var(--spatchat-text) !important; }
    #spatchat-figure-layer .spatchat-figure-modal-head { background: var(--spatchat-accent-soft) !important; border-bottom: 1px solid var(--spatchat-border) !important; }
    #spatchat-figure-layer .spatchat-figure-modal-title, #spatchat-figure-layer .spatchat-figure-modal-count,
    #spatchat-figure-layer .spatchat-figure-card-title, #spatchat-figure-layer .spatchat-figure-card-meta { color: var(--spatchat-text) !important; }
    #spatchat-figure-layer .spatchat-figure-card { background: var(--spatchat-surface) !important; }
    /* ---- Professional polish ---- */
    :root { --spatchat-font: "Inter", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif; }
    html, body, gradio-app, .gradio-container, .gradio-container * { font-family: var(--spatchat-font) !important; }
    .gradio-container code, .gradio-container pre, .gradio-container pre * { font-family: ui-monospace, "Cascadia Code", Consolas, monospace !important; }
    #spatchat-header-row { background: var(--spatchat-surface) !important; border: 1px solid var(--spatchat-border) !important;
        border-radius: 14px !important; padding: 6px 18px !important; box-shadow: var(--spatchat-shadow); align-items: center !important; }
    #spatchat-navbar a { font-weight: 500 !important; font-size: 14px !important; padding: 6px 2px; border-bottom: 2px solid transparent; transition: border-color .15s, color .15s; }
    #spatchat-navbar a:hover { border-bottom-color: var(--spatchat-accent); color: var(--spatchat-accent) !important; }
    .spatchat-title-row { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin: 22px 4px 14px; }
    .spatchat-title { margin: 0 !important; font-size: 26px !important; font-weight: 700 !important; letter-spacing: -0.02em; color: var(--spatchat-text) !important; line-height: 1.2; }
    .spatchat-subtitle { margin: 4px 0 0 !important; font-size: 14px !important; color: var(--spatchat-text-secondary) !important; }
    #spatchat-share-btn { flex: 0 0 auto; padding: 8px 18px; font-size: 13px; font-weight: 600; background: transparent; color: var(--spatchat-accent);
        border: 1px solid var(--spatchat-accent); border-radius: 8px; cursor: pointer; transition: background .15s, color .15s; }
    #spatchat-share-btn:hover { background: var(--spatchat-accent); color: #fff; }
    #spatchat-tabs button[role="tab"] { font-size: 14px !important; font-weight: 600 !important; padding: 8px 18px !important; color: var(--spatchat-text-secondary) !important; }
    #spatchat-tabs button[role="tab"].selected { color: var(--spatchat-accent) !important; }
    #spatchat-tabs .tabitem { border: 1px solid var(--spatchat-border) !important; box-shadow: var(--spatchat-shadow); border-radius: 12px !important; }
    #spatchat-chatbot { box-shadow: var(--spatchat-shadow) !important; background: var(--spatchat-surface) !important; }
    #spatchat-chatbot .panel-wrap, #spatchat-chatbot .wrap, #spatchat-chatbot .bubble-wrap, #spatchat-chatbot .message-wrap { background: var(--spatchat-surface) !important; }
    #spatchat-input-row { background: var(--spatchat-surface) !important; box-shadow: var(--spatchat-shadow); }
    #spatchat-input-row:focus-within { border-color: var(--spatchat-accent) !important; box-shadow: 0 0 0 3px var(--spatchat-accent-soft); }
    #spatchat-sample-btn { border: 1px solid var(--spatchat-border) !important; border-radius: 8px !important; background: var(--spatchat-surface) !important;
        align-self: stretch !important; padding: 8px 12px !important; font-size: 13px !important; }
    #spatchat-download, #spatchat-confirm { border-radius: 8px !important; font-weight: 600 !important; }
    #spatchat-download { background: var(--spatchat-accent) !important; color: #fff !important; border: none !important; }
    #spatchat-download:hover { background: var(--spatchat-accent-hover) !important; }
    #spatchat-datasheet table { font-size: 13px; }
    #spatchat-datasheet th { background: var(--spatchat-surface-alt) !important; font-weight: 600 !important; }
    /* ---- Round chatbot ---- */
    #spatchat-chatbot, #spatchat-chatbot > div { border-radius: 24px !important; }
    #spatchat-chatbot { border: 1px solid var(--spatchat-border) !important; }
    #spatchat-input-row {
        border: 1px solid var(--spatchat-border) !important; border-radius: 999px !important;
        margin-top: 8px !important; padding: 4px 10px !important;
    }

    /* ================= ChatGPT-style chat ================= */
    #spatchat-sidebar { gap: 0 !important; flex: 0 0 440px !important; width: 440px; }
    #spatchat-chatbot, #spatchat-chatbot > div, #spatchat-chatbot .wrapper, #spatchat-chatbot .panel-wrap,
    #spatchat-chatbot .bubble-wrap, #spatchat-chatbot .message-wrap {
        background: transparent !important; border: none !important; box-shadow: none !important; border-radius: 0 !important;
    }
    #spatchat-chatbot { flex: 1 1 auto !important; height: auto !important; min-height: 0 !important; padding: 0 !important; }
    #spatchat-chatbot .message-wrap { padding: 4px 4px 12px !important; gap: 18px !important; }
    #spatchat-chatbot .message-row { margin: 0 !important; padding: 0 !important; }
    #spatchat-chatbot .message, #spatchat-chatbot .message * { font-size: 15px; line-height: 1.65; }
    #spatchat-chatbot .message { max-width: 100% !important; padding: 0 !important; border: none !important; box-shadow: none !important; }
    /* assistant: plain text on the page, no box */
    #spatchat-chatbot .bot, #spatchat-chatbot .bot.message, #spatchat-chatbot .message.bot {
        background: transparent !important; border: none !important; padding: 0 !important; width: 100% !important;
    }
    /* user: soft right-aligned bubble */
    #spatchat-chatbot .message-row.user-row, #spatchat-chatbot .user-row { justify-content: flex-end !important; }
    #spatchat-chatbot .user, #spatchat-chatbot .user.message, #spatchat-chatbot .message.user {
        background: var(--spatchat-surface-alt) !important; color: var(--spatchat-text) !important;
        border-radius: 20px 20px 6px 20px !important; padding: 10px 16px !important; width: fit-content !important; max-width: 85% !important;
        margin-left: auto !important;
    }
    #spatchat-chatbot .message p { margin: 0 0 10px !important; }
    #spatchat-chatbot .message p:last-child { margin-bottom: 0 !important; }
    #spatchat-chatbot .message ul, #spatchat-chatbot .message ol { margin: 6px 0 10px 4px !important; padding-left: 20px !important; }
    #spatchat-chatbot .message li { margin: 3px 0 !important; }
    #spatchat-chatbot .message code { padding: 1px 6px !important; border-radius: 6px !important; font-size: 13px !important; }
    #spatchat-chatbot .message pre { border-radius: 10px !important; padding: 12px 14px !important; border: 1px solid var(--spatchat-border) !important; }
    #spatchat-chatbot .icon-button-wrapper { opacity: 0.6; }
    #spatchat-status { min-height: 0 !important; padding: 0 6px !important; color: var(--spatchat-text-secondary) !important; font-size: 13px !important; }

    /* suggestion chips */
    #spatchat-chips { display: flex !important; flex-wrap: wrap !important; gap: 8px !important; padding: 8px 2px 10px !important; background: transparent !important; border: none !important; }
    #spatchat-chips > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
    .spatchat-chip, #spatchat-sample-btn.spatchat-chip {
        width: auto !important; min-width: 0 !important; flex: 0 0 auto !important; padding: 6px 14px !important; margin: 0 !important;
        font-size: 13px !important; font-weight: 500 !important; line-height: 1.3 !important; border-radius: 999px !important;
        background: var(--spatchat-surface) !important; color: var(--spatchat-text) !important; border: 1px solid var(--spatchat-border) !important;
        box-shadow: none !important; align-self: auto !important; transition: background .15s, border-color .15s, color .15s;
    }
    .spatchat-chip:hover { background: var(--spatchat-accent-soft) !important; border-color: var(--spatchat-accent) !important; color: var(--spatchat-accent) !important; }

    /* composer */
    #spatchat-input-row {
        background: var(--spatchat-surface) !important; border: 1px solid var(--spatchat-border) !important; border-radius: 28px !important;
        padding: 6px 8px 6px 8px !important; margin: 0 0 10px !important; box-shadow: 0 2px 10px rgba(16,36,26,.10) !important;
        min-height: 54px; align-items: center !important;
    }
    #spatchat-input-row:focus-within { border-color: var(--spatchat-accent) !important; box-shadow: 0 0 0 3px var(--spatchat-accent-soft), 0 2px 10px rgba(16,36,26,.10) !important; }
    #spatchat-user-input textarea, #spatchat-user-input input { font-size: 15px !important; padding: 10px 6px !important; }
    #spatchat-file-input { width: 36px !important; height: 36px !important; min-width: 36px !important; flex: 0 0 36px !important; background: var(--spatchat-surface-alt) !important; color: var(--spatchat-text) !important; font-size: 22px !important; }
    #spatchat-file-input:hover { background: var(--spatchat-accent-soft) !important; color: var(--spatchat-accent) !important; }
    #spatchat-send-btn {
        flex: 0 0 38px !important; width: 38px !important; min-width: 38px !important; height: 38px !important; padding: 0 !important; margin: 0 !important;
        border-radius: 50% !important; border: none !important; background: var(--spatchat-accent) !important; color: #fff !important;
        font-size: 18px !important; font-weight: 700 !important; line-height: 1 !important; display: flex; align-items: center; justify-content: center;
    }
    #spatchat-send-btn:hover { background: var(--spatchat-accent-hover) !important; }

    /* ================= titles / header actions ================= */
    .spatchat-title-actions { display: flex; gap: 8px; align-items: center; }
    #spatchat-theme-btn { padding: 8px 16px; font-size: 13px; font-weight: 600; background: transparent; color: var(--spatchat-text-secondary);
        border: 1px solid var(--spatchat-border); border-radius: 8px; cursor: pointer; transition: background .15s, color .15s; }
    #spatchat-theme-btn:hover { background: var(--spatchat-surface-alt); color: var(--spatchat-text); }
    .spatchat-theme-dark { display: none; }
    body.spatchat-dark .spatchat-theme-light { display: none; }
    body.spatchat-dark .spatchat-theme-dark { display: inline; }

    /* ================= dataset selector ================= */
    #spatchat-dataset-select { flex: 0 0 auto !important; margin-bottom: 8px; background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
    #spatchat-dataset-select label > span, #spatchat-dataset-select .label-wrap span { font-size: 12px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: .04em; color: var(--spatchat-text-secondary) !important; }
    #spatchat-dataset-select input { font-size: 14px !important; font-weight: 600 !important; }
    #spatchat-mapcol { flex-wrap: nowrap !important; flex-direction: column !important; }
    #spatchat-tabs { flex: 1 1 0 !important; min-height: 0 !important; }
    #spatchat-dataset-select { width: 100% !important; min-width: 0 !important; }
    #spatchat-dataset-select .wrap, #spatchat-dataset-select input { background: var(--spatchat-surface) !important; border-radius: 10px !important; }
    #spatchat-dataset-select input { border: 1px solid var(--spatchat-border) !important; padding: 8px 12px !important; }

    /* ================= empty states ================= */
    .spatchat-empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
        gap: 8px; min-height: 360px; padding: 32px 24px; color: var(--spatchat-text-subtle); }
    .spatchat-empty-state svg { color: var(--spatchat-text-subtle); opacity: .8; }
    .spatchat-empty-title { font-size: 16px; font-weight: 600; color: var(--spatchat-text); }
    .spatchat-empty-text { font-size: 14px; max-width: 360px; line-height: 1.55; color: var(--spatchat-text-secondary); }


    /* ================= plot gallery + lightbox ================= */
    .spatchat-gallery { display: flex; flex-direction: column; gap: 10px; width: 100%; }
    .spatchat-gallery-bar { display: flex; justify-content: space-between; align-items: baseline; font-size: 13px; font-weight: 600; color: var(--spatchat-text); padding: 2px 2px 0; }
    .spatchat-gallery-hint { font-weight: 400; color: var(--spatchat-text-secondary); font-size: 12px; }
    .spatchat-gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
    .spatchat-thumb { position: relative; display: flex !important; flex-direction: column; text-align: left !important; align-items: stretch !important; padding: 0 !important; cursor: zoom-in; overflow: hidden;
        background: var(--spatchat-surface) !important; color: var(--spatchat-text) !important; border: 1px solid var(--spatchat-border); border-radius: 12px;
        box-shadow: var(--spatchat-shadow); transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease; font: inherit; }
    .spatchat-thumb:hover { transform: translateY(-2px); border-color: var(--spatchat-accent); box-shadow: 0 6px 18px rgba(16,36,26,.16); }
    .spatchat-thumb-preview { display: block; height: 100px; background: #fff; overflow: hidden; border-bottom: 1px solid var(--spatchat-border); }
    .spatchat-thumb-img { display: block; width: 100%; height: 100%; object-fit: contain; }
    .spatchat-thumb-table { height: 100%; overflow: hidden; font-size: 9px; padding: 6px; color: #10241a; pointer-events: none; }
    .spatchat-thumb-meta { display: flex; flex-direction: column; gap: 2px; padding: 7px 10px 9px; text-align: left; }
    .spatchat-thumb-title { font-size: 12px; font-weight: 600; line-height: 1.3; }
    .spatchat-thumb-sub { font-size: 10.5px; color: var(--spatchat-text-secondary); line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .spatchat-thumb-badge { position: absolute; top: 8px; left: 8px; background: var(--spatchat-accent); color: #fff; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 999px; }
    .spatchat-thumb-zoom { position: absolute; top: 8px; right: 8px; background: rgba(16,36,26,.78); color: #fff; font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 999px; opacity: 0; transition: opacity .12s; }
    .spatchat-thumb:hover .spatchat-thumb-zoom, .spatchat-thumb:focus-visible .spatchat-thumb-zoom { opacity: 1; }
    #spatchat-lightbox { position: fixed; inset: 0; z-index: 100000; display: none; align-items: center; justify-content: center; }
    #spatchat-lightbox.open { display: flex; }
    #spatchat-lightbox .sl-backdrop { position: absolute; inset: 0; background: rgba(8,14,11,.62); backdrop-filter: blur(2px); }
    #spatchat-lightbox .sl-panel { position: relative; display: flex; flex-direction: column; width: min(1100px, 92vw); height: min(86vh, 900px);
        background: var(--spatchat-surface, #fff); color: var(--spatchat-text, #10241a); border-radius: 16px; overflow: hidden;
        box-shadow: 0 24px 80px rgba(0,0,0,.45); border: 1px solid var(--spatchat-border, rgba(0,0,0,.1)); }
    #spatchat-lightbox .sl-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 16px;
        border-bottom: 1px solid var(--spatchat-border, rgba(0,0,0,.1)); background: var(--spatchat-surface-alt, #eef3ee); }
    #spatchat-lightbox .sl-title { font-size: 15px; font-weight: 700; }
    #spatchat-lightbox .sl-sub { font-size: 12px; color: var(--spatchat-text-secondary, #5c6b63); }
    #spatchat-lightbox .sl-nav { display: flex; align-items: center; gap: 6px; }
    #spatchat-lightbox .sl-count { font-size: 12px; color: var(--spatchat-text-secondary, #5c6b63); margin-right: 6px; }
    #spatchat-lightbox .sl-nav button { width: 34px; height: 34px; border-radius: 8px; border: 1px solid var(--spatchat-border, rgba(0,0,0,.12)); cursor: pointer;
        background: var(--spatchat-surface, #fff); color: var(--spatchat-text, #10241a); font-size: 17px; line-height: 1; }
    #spatchat-lightbox .sl-nav button:hover:not(:disabled) { background: var(--spatchat-accent-soft, #e3f2e8); border-color: var(--spatchat-accent, #1f7a4f); }
    #spatchat-lightbox .sl-nav button:disabled { opacity: .35; cursor: default; }
    #spatchat-lightbox .sl-body { flex: 1 1 auto; min-height: 0; overflow: auto; display: flex; align-items: flex-start; justify-content: center; padding: 16px; background: var(--spatchat-surface, #fff); }
    #spatchat-lightbox .sl-body img { max-width: 100%; max-height: 100%; width: auto; height: auto; object-fit: contain; background: #fff; border-radius: 8px; margin: auto; }
    #spatchat-lightbox .sl-table { width: 100%; }
    body.spatchat-lightbox-open { overflow: hidden; }
    .spatchat-thumb-table table, #spatchat-lightbox .sl-table table { width: 100%; border-collapse: collapse; }
    .spatchat-thumb-table th, #spatchat-lightbox .sl-table th { background: var(--spatchat-surface-alt) !important; color: var(--spatchat-text) !important; font-weight: 600; text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--spatchat-border); }
    .spatchat-thumb-table td, #spatchat-lightbox .sl-table td { background: transparent !important; color: var(--spatchat-text) !important; padding: 4px 8px; border-bottom: 1px solid var(--spatchat-border); }
    .spatchat-thumb-table { background: var(--spatchat-surface); }
    /* ================= dark mode ================= */
    .spatchat-dark {
        --spatchat-connect-bg: #0e1512; --spatchat-surface: #16201b; --spatchat-surface-alt: #1e2a24;
        --spatchat-border: rgba(255,255,255,0.12); --spatchat-text: #e8efe9; --spatchat-text-secondary: #a6b4ab; --spatchat-text-subtle: #7f8f85;
        --spatchat-accent: #3fb27f; --spatchat-accent-hover: #5cc796; --spatchat-accent-soft: #1d3a2d;
        --spatchat-shadow: 0 1px 3px rgba(0,0,0,.5), 0 1px 2px rgba(0,0,0,.4);
    }
    body.spatchat-dark #spatchat-navbar a { color: var(--spatchat-text) !important; }
    body.spatchat-dark #spatchat-navbar a:hover { color: var(--spatchat-accent) !important; }
    body.spatchat-dark #spatchat-input-row { box-shadow: 0 2px 12px rgba(0,0,0,.5) !important; }
    body.spatchat-dark #spatchat-logo-img img, body.spatchat-dark #logo-img img { background: #fff; border-radius: 8px; padding: 2px 6px; }
    body.spatchat-dark .spatchat-map-empty-hint-card { background: var(--spatchat-surface) !important; color: var(--spatchat-text) !important; border-color: var(--spatchat-border) !important; }
    body.spatchat-dark #spatchat-datasheet, body.spatchat-dark #spatchat-datasheet * { color: var(--spatchat-text); }
    body.spatchat-dark #spatchat-datasheet th { background: var(--spatchat-surface-alt) !important; }
    body.spatchat-dark #spatchat-datasheet td { background: var(--spatchat-surface) !important; }
    body.spatchat-dark #spatchat-figure-layer .spatchat-figure-card-image { background: #fff; border-radius: 8px; }
    body.spatchat-dark .spatchat-figure-table th, body.spatchat-dark .spatchat-figure-table td { color: var(--spatchat-text) !important; }
    body.spatchat-dark #spatchat-sidebar, body.spatchat-dark #spatchat-mapcol, body.spatchat-dark #spatchat-workarea, body.spatchat-dark #spatchat-splitter { background: var(--spatchat-connect-bg) !important; }
    </style>
    """)
    with gr.Row(elem_id="spatchat-header-row"):
        gr.Image(scale=0, min_width=0, **image_kwargs)
        gr.HTML(elem_id="spatchat-navbar", scale=0, min_width=0, value="""
        <nav style="display: flex; gap: 22px; align-items: center; font-size: 15px; font-weight: 600;">
          <a href="https://spatchat.org/" target="_blank" rel="noopener" style="color: #10241a; text-decoration: none;">Home</a>
          <a href="https://spatchat.org/about" target="_blank" rel="noopener" style="color: #10241a; text-decoration: none;">About</a>
          <a href="https://spatchat.org/browse" target="_blank" rel="noopener" style="color: #1f7a4f; text-decoration: none;">Browse Rooms</a>
          <a href="https://spatchat.org/join" target="_blank" rel="noopener" style="color: #10241a; text-decoration: none;">Join</a>
        </nav>
        """)
    gr.HTML("""
    <div class="spatchat-title-row">
      <div>
        <h1 class="spatchat-title">Animal Movement and Home Range Analysis</h1>
        <p class="spatchat-subtitle">Upload tracking data, estimate home ranges, and explore movement behavior through conversation.</p>
      </div>
      <div class="spatchat-title-actions">
      <button id="spatchat-theme-btn" type="button" onclick="window.spatchatToggleTheme && window.spatchatToggleTheme()">
        <span class="spatchat-theme-light">Dark mode</span><span class="spatchat-theme-dark">Light mode</span>
      </button>
      <button id="spatchat-share-btn" onclick="
        navigator.clipboard.writeText('https://spatchat.org/browse/?room=move');
        this.textContent = 'Link copied';
        setTimeout(() => { this.textContent = 'Share'; }, 1500);
      ">Share</button>
      </div>
    </div>
    """)

    with gr.Row(elem_id="spatchat-workarea"):
        with gr.Column(scale=3, min_width=0, elem_id="spatchat-sidebar"):
            chatbot_kwargs = {
                "label": "Spatchat",
                "show_label": False,
                "layout": "bubble",
                "value": [{"role": "assistant", "content": "Hi, I'm Spatchat! This room helps you analyze home ranges and movement behavior from movement data.\n\nUpload a movement CSV to begin: it should include coordinates, and can also include timestamps and animal IDs for track-aware analyses.\nThis room can:\n- estimate home ranges with MCP, KDE, AKDE, LoCoH, and dBBMM\n- analyze movement patterns using displacement, step lengths, turning angles, and autocorrelation diagnostics\n- identify behavioral states with a hidden Markov model"}],
                "elem_id": "spatchat-chatbot",
            }
            if _component_accepts_kw(gr.Chatbot, "type"):
                chatbot_kwargs["type"] = "messages"
            if _component_accepts_kw(gr.Chatbot, "buttons"):
                chatbot_kwargs["buttons"] = []
            if _component_accepts_kw(gr.Chatbot, "feedback_options"):
                chatbot_kwargs["feedback_options"] = None
            chatbot = gr.Chatbot(**chatbot_kwargs)
            status_output = gr.HTML(value="", visible=True, elem_id="spatchat-status")
            with gr.Row(elem_id="spatchat-chips"):
                sample_btn = gr.Button("Try sample data", elem_id="spatchat-sample-btn", size="sm", elem_classes=["spatchat-chip"], scale=0, min_width=0)
                chip_buttons = [
                    gr.Button(label, size="sm", elem_classes=["spatchat-chip"], scale=0, min_width=0)
                    for label in _SUGGESTION_CHIPS
                ]
            with gr.Row(elem_id="spatchat-input-row"):
                file_input = gr.UploadButton(
                    "+",
                    file_types=[".csv", ".txt"],
                    elem_id="spatchat-file-input",
                    scale=0,
                    min_width=44,
                )
                user_input = gr.Textbox(label="", show_label=False, placeholder="Message Spatchat...", lines=1, elem_id="spatchat-user-input", scale=1)
                send_btn = gr.Button("\u2191", elem_id="spatchat-send-btn", scale=0, min_width=40)
            x_col = gr.Dropdown(label="X column", choices=[], visible=False, elem_id="spatchat-x-col")
            y_col = gr.Dropdown(label="Y column", choices=[], visible=False, elem_id="spatchat-y-col")
            crs_text = gr.Text(label="CRS (e.g. '32633', '33N', or 'EPSG:32633')", visible=False, elem_id="spatchat-crs")
            confirm_btn = gr.Button("Confirm Coordinate Settings", visible=False, elem_id="spatchat-confirm")
            download_btn = gr.DownloadButton("Download results", value=None, visible=False, elem_id="spatchat-download")
        with gr.Column(scale=0, min_width=14, elem_id="spatchat-splitter"):
            gr.HTML("<div class='spatchat-splitter-handle' onmousedown='return window.spatchatBeginResize ? window.spatchatBeginResize(event) : false;' title='Drag to resize panels'></div>")
        with gr.Column(scale=5, min_width=0, elem_id="spatchat-mapcol"):
            dataset_selector = gr.Dropdown(label="Dataset", show_label=False, container=False, choices=[], value=None, visible=False, interactive=True, elem_id="spatchat-dataset-select", scale=0)
            with gr.Tabs(elem_id="spatchat-tabs") as main_tabs:
                with gr.Tab("Map", id="map"):
                    map_output = gr.HTML(label="Map Preview", value=render_empty_map(), show_label=False, elem_id="spatchat-map")
                with gr.Tab("Plots", id="plots", elem_id="spatchat-plots-pane"):
                    figure_output = gr.HTML(value=_render_figure_viewer([]), elem_id="spatchat-figure-layer")
                with gr.Tab("Datasheet", id="data", elem_id="spatchat-data-pane"):
                    datasheet_empty = gr.HTML(value=_EMPTY_DATA_HTML, visible=True)
                    datasheet = gr.Dataframe(label=f"Data (first {_DATASHEET_MAX_ROWS} rows)", interactive=False, wrap=False, visible=False, elem_id="spatchat-datasheet")
    gr.HTML("""
    <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid rgba(16,36,26,0.1); font-size: 11px; color: #5c6b63; text-align: center;">
      © 2025 Ho Yi Wan &amp; Logan Hysen &middot; Cite: Wan, H.Y. &amp; Hysen, L. (2025). <i>Spatchat: Animal Movement and Home Range Analysis.</i>
    </div>
    """)

    demo.queue(max_size=16)

    file_input.upload(
        fn=_handle_upload_initial_ui,
        inputs=[file_input, chatbot, session_state, figure_state],
        outputs=[chatbot, status_output, x_col, y_col, crs_text, map_output, x_col, y_col, crs_text, confirm_btn, download_btn, session_state, figure_output, figure_state]
    ).then(_datasheet_update, inputs=session_state, outputs=[datasheet, datasheet_empty]).then(
        lambda f, sid, lib, figs, mh: _register_dataset(f, sid, lib, figs, mh),
        inputs=[file_input, session_state, library_state, figure_state, map_output],
        outputs=[library_state, dataset_selector],
    ).then(lambda: _select_tab("map"), inputs=None, outputs=main_tabs)
    sample_btn.click(
        fn=_load_sample_data_ui,
        inputs=[chatbot, session_state, figure_state],
        outputs=[chatbot, status_output, x_col, y_col, crs_text, map_output, x_col, y_col, crs_text, confirm_btn, download_btn, session_state, figure_output, figure_state]
    ).then(_datasheet_update, inputs=session_state, outputs=[datasheet, datasheet_empty]).then(
        lambda sid, lib, figs, mh: _register_dataset("Sample data", sid, lib, figs, mh),
        inputs=[session_state, library_state, figure_state, map_output],
        outputs=[library_state, dataset_selector],
    ).then(lambda: _select_tab("map"), inputs=None, outputs=main_tabs)
    confirm_btn.click(fn=_confirm_and_hint_ui, inputs=[x_col, y_col, crs_text, chatbot, session_state], outputs=[map_output, chatbot, status_output, session_state]).then(_datasheet_update, inputs=session_state, outputs=[datasheet, datasheet_empty]).then(
        _sync_active_dataset,
        inputs=[session_state, library_state, figure_state, map_output],
        outputs=library_state,
        show_progress="hidden",
    ).then(lambda: _select_tab("map"), inputs=None, outputs=main_tabs)
    gr.on(
        triggers=[user_input.submit, send_btn.click],
        fn=metered_chat(room="move", message_index=1, deny=_chat_denied)(_handle_chat_ui),
        inputs=[chatbot, user_input, session_state, figure_state],
        outputs=[chatbot, map_output, download_btn, status_output, figure_output, figure_state, session_state],
        show_progress="hidden",
        js="""
        (chatHistory, message, sessionState, figureState) => {
            const text = message ?? "";
            setTimeout(() => {
                const input = document.querySelector('#spatchat-user-input textarea, #spatchat-user-input input');
                if (input) {
                    input.value = "";
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                }
            }, 0);
            return [chatHistory, text, sessionState, figureState];
        }
        """,
    ).then(_datasheet_update, inputs=session_state, outputs=[datasheet, datasheet_empty], show_progress="hidden").then(
        _consume_next_tab, inputs=session_state, outputs=main_tabs, show_progress="hidden"
    ).then(
        _sync_active_dataset,
        inputs=[session_state, library_state, figure_state, map_output],
        outputs=library_state,
        show_progress="hidden",
    )
    for _chip, _label in zip(chip_buttons, _SUGGESTION_CHIPS):
        _chip.click(fn=None, js=_chip_js(_label))
    dataset_selector.input(
        _switch_dataset,
        inputs=[dataset_selector, library_state, session_state, chatbot],
        outputs=[chatbot, map_output, figure_output, figure_state, datasheet, datasheet_empty, main_tabs, library_state],
    )
    demo.unload(_cleanup_current_browser_session)


def _launch_demo(blocks: gr.Blocks) -> None:
    launch_kwargs = {
        "share": False,
        "head": _SPLITTER_HEAD,
        "server_name": os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        "server_port": int(os.environ.get("PORT", 7860))
    }
    try:
        sig = inspect.signature(gr.Blocks.launch)
        params = getattr(sig, "parameters", {}) or {}
    except (ValueError, TypeError):
        params = {}

    available = params.keys() if hasattr(params, "keys") else []
    if "max_file_size" in available:
        launch_kwargs["max_file_size"] = f"{MAX_UPLOAD_MB}mb"
    if "ssr_mode" in available:
        launch_kwargs["ssr_mode"] = False

    blocks.launch(**launch_kwargs)


if os.environ.get("SPATCHAT_SKIP_LAUNCH") != "1":
    _launch_demo(demo)
