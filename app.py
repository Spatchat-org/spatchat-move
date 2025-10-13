# app.py
import os
import json
import re
import time
import shutil
import random
import sys
import zipfile

import gradio as gr
import pandas as pd
import numpy as np

# ---- Local modules ----
from storage import (
    get_cached_df, set_cached_df,
    get_cached_headers, set_cached_headers,
    clear_all_results,
    mcp_results, kde_results,
    requested_percents, requested_kde_percents,
    save_all_mcps_zip, 
    set_locoh_results, get_locoh_results,
    set_dbbmm_results, get_dbbmm_results, 
    requested_dbbmm_percents
)

from llm_utils import ask_llm
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

from estimators.locoh import compute_locoh, LoCoHParams
from estimators.dbbmm import compute_dbbmm, DBBMMParams
from estimators.kde import add_kdes, KDEParams  # params-enabled KDE

print("Starting SpatChat: Home Range Analysis (app.py) — handlers only")

import numpy as _np
import pandas as _pd
from datetime import datetime as _dt, date as _date

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
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
        "Estimators available: MCP, KDE, LoCoH, dBBMM.\n\n"
        "Examples:\n"
        "• I want 100 MCP\n"
        "• I want 95 KDE\n"
        "• I want LoCoH 50 and 95\n"
        "• I want dBBMM 95\n\n"
        "Ask me about parameter options anytime."
    )

def parse_kv_tokens(text: str) -> dict:
    """
    Parse key=value tokens without breaking comma-separated lists.
    Examples:
      "locoh k=10 a=1500 isopleths=50,95"
      "dbbmm 95 le=20 res=75 buf=1500 window=31 margin=11 subs=40"
      "kde 95 bw=300m kernel=epanechnikov gres=50m"
    """
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

# Track per-upload “pending questions”
PENDING_QUESTIONS = {
    "need_id": False,
    "need_ts": False,
    "ts_prompted": False,
    "id_prompted": False,
}

def _current_dataset_context():
    df = get_cached_df()
    try:
        return build_dataset_context(df)
    except Exception:
        return {"empty": True}

# --- permissive level parser for MCP (allows 100) -----------------------------------
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
_PARAM_VERBS = ("run", "compute", "calculate", "do", "make", "generate", "plot", "draw", "want")
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
    # generic ask without an estimator keyword
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

# --- NEW: one source of truth for parameter help ------------------------------------
def _parameters_markdown(which: str | None = None) -> str:
    mcp = (
        "**MCP (Minimum Convex Polygon)**\n"
        "• `isopleths`: UD contours to export (e.g., 50,95). Example: `mcp 50,95`\n"
    )
    kde = (
        "**KDE (Kernel Density Estimation)**\n"
        "• `isopleths`: UD contours to export (e.g., 50,95). Example: `kde 95`\n"
        "• `bw` / `bandwidth` / `h` **(meters or km)**: kernel bandwidth in UTM meters. "
        "Examples: `bw=300m`, `bw=0.5km` (500 m)\n"
        "• `kernel`: one of `gaussian`, `epanechnikov`, `tophat`, `exponential`, `linear`, `cosine`. "
        "Example: `kernel=epanechnikov`\n"
        "• `gres` / `grid_res` **(meters or km)**: grid cell size for the raster (optional). Example: `gres=50m`\n"
        "Examples: `kde 95 bw=300m kernel=epanechnikov`,  `kde 50,95 bw=0.5km`\n"
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
        "Here are the parameter options:\n\n" + mcp + "\n" + kde + "\n" + locoh + "\n" + dbbmm
        + "\nTip: bare numbers for `bw`, `gres`, etc. are treated as meters; add `km` to use kilometers."
    )
    if which is None:
        return all_text
    which = which.lower()
    if which == "mcp": return mcp
    if which == "kde": return kde
    if which == "locoh": return locoh
    if which == "dbbmm": return dbbmm
    return all_text

# --------------------------------------------------------------------------------------
# Upload flow
# --------------------------------------------------------------------------------------
def handle_upload_initial(file):
    clear_all_results()

    os.makedirs("uploads", exist_ok=True)
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)

    try:
        df = pd.read_csv(filename)
        set_cached_df(df)
        set_cached_headers(list(df.columns))
    except Exception as e:
        print(f"[upload] failed to read CSV: {e}", file=sys.stderr)
        return [
            [],  # chatbot messages
            gr.update(visible=False),  # x
            gr.update(visible=False),  # y
            gr.update(visible=False),  # crs
            render_empty_map(),        # map
            gr.update(visible=False),  # x
            gr.update(visible=False),  # y
            gr.update(visible=False),  # crs
            gr.update(visible=False),  # confirm
            gr.update(visible=False),  # download
        ]

    # Reset pending flags
    for k in PENDING_QUESTIONS:
        PENDING_QUESTIONS[k] = False

    cached_headers = get_cached_headers()
    lower_cols = [c.lower() for c in cached_headers]

    # heuristic lon/lat candidates
    found_x = found_y = None
    latlon_guess = None
    try:
        guess = looks_like_latlon(get_cached_df(), cached_headers)
        if isinstance(guess, tuple) and len(guess) == 3:
            latlon_guess, found_x, found_y = guess
    except Exception:
        pass

    # A) explicit latitude/longitude
    if "latitude" in lower_cols and "longitude" in lower_cols:
        lat_col = cached_headers[lower_cols.index("latitude")]
        lon_col = cached_headers[lower_cols.index("longitude")]

        if looks_invalid_latlon(get_cached_df(), lat_col, lon_col):
            return [
                [{"role": "assistant", "content":
                  "CSV uploaded. Your coordinates do not appear to be latitude/longitude. "
                  "Please specify X (easting), Y (northing), and the CRS/UTM zone below "
                  "(e.g., 'UTM 10T' or 'EPSG:32610')."}],
                gr.update(choices=cached_headers, value=lon_col, visible=True),
                gr.update(choices=cached_headers, value=lat_col, visible=True),
                gr.update(visible=True),
                render_empty_map(),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
            ]

        df0 = get_cached_df().copy()
        df0["longitude"] = df0[lon_col]
        df0["latitude"]  = df0[lat_col]

        from schema_detect import detect_id_column, detect_timestamp_column, ID_COL, TS_COL
        src_id = detect_id_column(df0)
        src_ts = detect_timestamp_column(df0)

        df1, _ = detect_and_standardize(df0)
        set_cached_df(df1)
        map_html = build_preview_map(df1)

        id_found  = (ID_COL in df1.columns)
        ts_found  = (TS_COL in df1.columns)
        id_note   = f"• **ID column**: `{src_id}`" if src_id else "• **ID column**: `not detected`"
        ts_note   = f"• **Timestamp column**: `{src_ts}`" if src_ts else "• **Timestamp column**: `not detected`"

        tips = []
        if not id_found:
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

        return [
            [{"role": "assistant", "content": msg}],
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            map_html,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ]

    # B) heuristic lon/lat guess
    if latlon_guess:
        df = get_cached_df()
        df0 = df.copy()
        df0["longitude"] = df0[found_x] if latlon_guess == "lonlat" else df0[found_y]
        df0["latitude"]  = df0[found_y] if latlon_guess == "lonlat" else df0[found_x]

        from schema_detect import detect_id_column, detect_timestamp_column, ID_COL, TS_COL
        src_id = detect_id_column(df0)
        src_ts = detect_timestamp_column(df0)

        df1, _ = detect_and_standardize(df0)
        set_cached_df(df1)
        map_html = build_preview_map(df1)

        id_found = (ID_COL in df1.columns)
        ts_found = (TS_COL in df1.columns)
        id_note  = f"• **ID column**: `{src_id}`" if src_id else "• **ID column**: `not detected`"
        ts_note  = f"• **Timestamp column**: `{src_ts}`" if src_ts else "• **Timestamp column**: `not detected`"

        tips = []
        if not id_found:
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

        return [
            [{"role": "assistant", "content": msg}],
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            map_html,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ]

    # C) need user to pick X/Y and CRS
    return [
        [{"role": "assistant", "content":
          "CSV uploaded. Your coordinates do not appear to be latitude/longitude. "
          "Please specify X (easting), Y (northing), and the CRS/UTM zone below."}],
        gr.update(choices=cached_headers, value=found_x, visible=True),
        gr.update(choices=cached_headers, value=found_y, visible=True),
        gr.update(visible=True),
        render_empty_map(),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
    ]


def handle_upload_confirm(x_col, y_col, crs_text):
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
            df["latitude"]  = df[y_col]
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
    set_cached_df(df)
    return build_preview_map(df)

def confirm_and_hint(x_col, y_col, crs_text, chat_history):
    map_html = handle_upload_confirm(x_col, y_col, crs_text)
    guidance = _home_range_help()
    chat = list(chat_history)
    chat.append({"role": "assistant", "content": guidance})
    return map_html, chat

# --------------------------------------------------------------------------------------
# Analysis + chat handler
# --------------------------------------------------------------------------------------
def handle_chat(chat_history, user_message):
    """
    Handles user chat. Detects estimator requests or parameter questions.
    """
    chat_history = list(chat_history)

    # Mapping commands (id/timestamp)
    cmd = parse_metadata_command(user_message)
    if cmd:
        df = get_cached_df()
        if df is None or "latitude" not in df or "longitude" not in df:
            chat_history.append({"role": "assistant", "content": "Please upload a CSV first."})
            return chat_history, gr.update(), gr.update(visible=False)

        df2, msg = try_apply_user_mapping(df, cmd)
        set_cached_df(df2)

        PENDING_QUESTIONS["need_id"] = (ID_COL not in df2.columns)
        PENDING_QUESTIONS["need_ts"] = (TS_COL not in df2.columns)

        follow = []
        if not PENDING_QUESTIONS["need_id"] and not PENDING_QUESTIONS["need_ts"]:
            follow.append("Great — ID and timestamps detected.")
        elif PENDING_QUESTIONS["need_id"] and not PENDING_QUESTIONS["id_prompted"]:
            follow.append("I couldn’t detect an individual ID column. If your data has one, say: “ID column is tag_id”.")
            PENDING_QUESTIONS["id_prompted"] = True
        elif PENDING_QUESTIONS["need_ts"] and not PENDING_QUESTIONS["ts_prompted"]:
            follow.append("I couldn’t detect a timestamp column. If your data has one, say: “Timestamp column is datetime”.")
            PENDING_QUESTIONS["ts_prompted"] = True

        chat_history.append({"role": "assistant", "content": msg + (" " + " ".join(follow) if follow else "")})
        return chat_history, gr.update(), gr.update(visible=False)

    # --- NEW: Generic "parameters?" overview (before LLM!) ---------------------------
    if _wants_parameters_overview(user_message):
        chat_history.append({"role": "assistant", "content": _parameters_markdown(None)})
        return chat_history, gr.update(), gr.update(visible=False)

    # --- NEW: Estimator-specific parameter asks (no action verbs) --------------------
    if _is_parameter_question(user_message, "mcp"):
        chat_history.append({"role": "assistant", "content": _parameters_markdown("mcp")})
        return chat_history, gr.update(), gr.update(visible=False)
    if _is_parameter_question(user_message, "kde"):
        chat_history.append({"role": "assistant", "content": _parameters_markdown("kde")})
        return chat_history, gr.update(), gr.update(visible=False)
    if _is_parameter_question(user_message, "locoh"):
        chat_history.append({"role": "assistant", "content": _parameters_markdown("locoh")})
        return chat_history, gr.update(), gr.update(visible=False)
    if _is_parameter_question(user_message, "dbbmm"):
        chat_history.append({"role": "assistant", "content": _parameters_markdown("dbbmm")})
        return chat_history, gr.update(), gr.update(visible=False)

    # --- Previously added: dBBMM parameter Q&A also stays (redundant but harmless) ---
    if _is_parameter_question(user_message, "dbbmm"):
        chat_history.append({"role": "assistant", "content": _parameters_markdown("dbbmm")})
        return chat_history, gr.update(), gr.update(visible=False)

    # Normal tool-intent call (with dataset context)
    context_raw = _current_dataset_context()
    context_safe = _json_safe(context_raw)
    tool, llm_output = ask_llm(chat_history, user_message, context=context_safe)

    # -------------------------
    # Parse intents/keywords
    # -------------------------
    mcp_list, kde_list = [], []
    locoh_requested = False
    locoh_params = None
    dbbmm_list = []
    dbbmm_params = None
    kde_params = KDEParams()

    warned_about_kde_100 = False
    if tool and tool.get("tool") == "home_range":
        method = tool.get("method")
        raw_levels = tool.get("levels", [95])
        levels = [int(p) for p in raw_levels if 1 <= int(p) <= 100]

        if method == "mcp":
            mcp_list = levels

        elif method == "kde":
            if 100 in levels:
                warned_about_kde_100 = True
            kde_list = [min(p, 99) for p in levels]

        elif method == "locoh":
            locoh_requested = True
            locoh_params = LoCoHParams(method=tool.get("locoh_method", "k"),
                                       k=int(tool.get("k", 10)),
                                       a=tool.get("a"),
                                       r=tool.get("r"),
                                       isopleths=tuple(levels or (95,)))
        elif method == "dbbmm":
            dbbmm_list = levels or [95]
            dbbmm_params = DBBMMParams(isopleths=tuple(dbbmm_list))

    # Fallback: keyword parse
    msg_lower = user_message.lower()
    if "mcp" in msg_lower:
        parsed = _parse_levels_allow_100(user_message)
        mcp_list = parsed or [95]

    # KDE with meters/km bandwidth + kernel + grid res
    if "kde" in msg_lower:
        kde_list = parse_levels_from_text(user_message)  # existing (clamped later)

        toks = parse_kv_tokens(user_message)
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

        kde_params = KDEParams(
            bandwidth_m=bw_m,
            kernel=kernel,
            grid_res_m=gres_m,
            grid_size=200
        )

    # LoCoH keywords
    if "locoh" in msg_lower:
        locoh_requested = True
        toks = parse_kv_tokens(user_message)
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
            parsed = _parse_levels_allow_100(user_message)
            iso = tuple(parsed) if parsed else (95,)
        locoh_params = LoCoHParams(method=method, k=k, a=a, r=r, isopleths=iso)

    # dBBMM keywords (only arm if user gave levels, key=vals, or action verb)
    if "dbbmm" in msg_lower:
        looks_like_levels = bool(re.search(r"\b(100|[1-9]?[0-9])\b", user_message))
        looks_like_kv     = "=" in user_message
        looks_actiony     = any(v in msg_lower for v in _PARAM_VERBS)
        if looks_like_levels or looks_like_kv or looks_actiony:
            dbbmm_list = _parse_levels_allow_100(user_message) or [95]
            toks = parse_kv_tokens(user_message)

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
    if not mcp_list and not kde_list and not locoh_requested and not dbbmm_list:
        if PENDING_QUESTIONS["need_id"] and not PENDING_QUESTIONS["id_prompted"]:
            chat_history.append({"role": "assistant", "content":
                                 "I couldn’t detect an individual ID column. If your data has one, say: “ID column is tag_id”. "
                                 "Otherwise, you can still proceed — I’ll treat all rows as one animal."})
            PENDING_QUESTIONS["id_prompted"] = True
            return chat_history, gr.update(), gr.update(visible=False)

        if PENDING_QUESTIONS["need_ts"] and not PENDING_QUESTIONS["ts_prompted"]:
            chat_history.append({"role": "assistant", "content":
                                 "I couldn’t detect a timestamp column. If your data has one, say: “Timestamp column is datetime”. "
                                 "Otherwise, you can proceed — I’ll plot points without drawing tracks."})
            PENDING_QUESTIONS["ts_prompted"] = True
            return chat_history, gr.update(), gr.update(visible=False)

        if llm_output:
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": llm_output})
            return chat_history, gr.update(), gr.update(visible=False)

        chat_history.append({"role": "assistant", "content": "How can I help you? Please upload a CSV for analysis or ask a question."})
        return chat_history, gr.update(), gr.update(visible=False)

    # Must have lon/lat prepared
    df = get_cached_df()
    if df is None or "latitude" not in df or "longitude" not in df:
        clear_all_results()
        chat_history.append({"role": "assistant", "content": "Please upload a CSV first (with latitude/longitude)."})
        return chat_history, gr.update(), gr.update(visible=False)

    results_exist = False
    if kde_list:
        if 100 in kde_list or any("100" in s for s in user_message.split()):
            warned_about_kde_100 = True
        kde_list = [min(k, 99) for k in kde_list]

    locoh_result = None
    locoh_error = None
    dbbmm_result = None

    # Run analyses
    if mcp_list:
        from estimators.mcp import add_mcps
        add_mcps(df, mcp_list)
        requested_percents.update(mcp_list)
        results_exist = True

    if kde_list:
        add_kdes(df, kde_list, params=kde_params)
        requested_kde_percents.update(kde_list)
        results_exist = True

    if locoh_requested:
        try:
            df_locoh = df.rename(columns={"longitude": "lon", "latitude": "lat"})
            locoh_result = compute_locoh(
                df=df_locoh,
                id_col="animal_id",
                x_col="lon",
                y_col="lat",
                params=locoh_params or LoCoHParams()
            )
            set_locoh_results(locoh_result)
            results_exist = True
        except Exception as e:
            locoh_error = str(e)

    if dbbmm_list:
        try:
            if "timestamp" not in df.columns:
                raise ValueError("dBBMM requires a timestamp column to model movement between fixes.")
            dbbmm_result = compute_dbbmm(
                df=df,
                id_col="animal_id",
                x_col="longitude",
                y_col="latitude",
                ts_col="timestamp",
                params=dbbmm_params,
                outputs_dir="outputs",
            )
            set_dbbmm_results(dbbmm_result)
            requested_dbbmm_percents.update(dbbmm_list)
            results_exist = True
        except Exception as e:
            chat_history.append({"role": "assistant", "content": f"dBBMM error: {e}"})

    effective_locoh  = locoh_result  if locoh_result  is not None else get_locoh_results()
    effective_dbbmm  = dbbmm_result  if dbbmm_result  is not None else get_dbbmm_results()

    map_html = build_results_map(
        df,
        mcp_results=mcp_results,
        kde_results=kde_results,
        requested_percents=requested_percents,
        requested_kde_percents=requested_kde_percents,
        locoh_result=effective_locoh,
        dbbmm_result=effective_dbbmm
    )

    msgs = []
    if requested_percents:
        msgs.append(f"MCP home ranges ({', '.join(str(p) for p in sorted(requested_percents))}%) calculated.")
    if requested_kde_percents:
        msgs.append(f"KDE home ranges ({', '.join(str(p) for p in sorted(requested_kde_percents))}%) calculated (raster & contours).")
    if warned_about_kde_100:
        msgs.append("Note: KDE at 100% is not supported and has been replaced by 99%.")
    if locoh_result:
        msgs.append(_summarize_locoh(locoh_result, locoh_params or LoCoHParams()))
    if locoh_error:
        msgs.append(f"LoCoH error: {locoh_error}")
    if dbbmm_list:
        msgs.append(f"dBBMM UDs computed ({', '.join(str(p) for p in sorted(set(dbbmm_list))) }% isopleths). Raster + contours added.")
    if results_exist:
        msgs.append("_The download button is below the preview map._")

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": " ".join(msgs) if msgs else "Done."})

    archive_path = save_all_mcps_zip()
    return chat_history, gr.update(value=map_html), gr.update(value=archive_path, visible=results_exist)

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
_reset_session_state()

with gr.Blocks(title="SpatChat: Home Range Analysis") as demo:
    gr.Image(
        value="logo_long1.png",
        show_label=False,
        show_download_button=False,
        show_share_button=False,
        type="filepath",
        elem_id="logo-img"
    )
    gr.HTML("""
    <style>
    #logo-img img {
        height: 90px;
        margin: 10px 50px 10px 10px;
        border-radius: 6px;
    }
    </style>
    """)
    gr.Markdown("## 🏠 SpatChat: Home Range Analysis {hr}  🦊🦉🐢")
    gr.HTML("""
    <div style="margin-top: -10px; margin-bottom: 15px;">
      <input type="text" value="https://spatchat.org/browse/?room=hr" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">
        📋 Copy Share Link
      </button>
      <div style="margin-top: 10px; font-size: 14px;">
        <b>Share:</b>
        <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=hr" target="_blank">🐦 Twitter</a> |
        <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=hr" target="_blank">📘 Facebook</a>
      </div>
    </div>
    """)
    gr.Markdown("""
        <div style="font-size: 14px;">
        © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
        If you use Spatchat in research, please cite:<br>
        <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>SpatChat: Home Range Analysis.</i>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="SpatChat",
                show_label=True,
                type="messages",
                value=[{"role": "assistant", "content": "Welcome to SpatChat! Please upload a CSV containing coordinates (lat/lon or UTM) and optional timestamp/animal_id to begin."}]
            )
            user_input = gr.Textbox(
                label="Ask Spatchat",
                placeholder="Type commands...",
                lines=1
            )
            file_input = gr.File(
                label="Upload Movement CSV (.csv or .txt only)",
                file_types=[".csv", ".txt"]
            )
            x_col = gr.Dropdown(label="X column", choices=[], visible=False)
            y_col = gr.Dropdown(label="Y column", choices=[], visible=False)
            crs_text = gr.Text(label="CRS (e.g. '32633', '33N', or 'EPSG:32633')", visible=False)
            confirm_btn = gr.Button("Confirm Coordinate Settings", visible=False)
        with gr.Column(scale=3):
            map_output = gr.HTML(label="Map Preview", value=render_empty_map(), show_label=False)
            download_btn = gr.DownloadButton(
                "📥 Download Results",
                value=None,
                visible=False
            )

    demo.queue(max_size=16)

    file_input.change(
        fn=handle_upload_initial,
        inputs=file_input,
        outputs=[
            chatbot, x_col, y_col, crs_text, map_output,
            x_col, y_col, crs_text, confirm_btn, download_btn
        ]
    )
    confirm_btn.click(
        fn=confirm_and_hint,
        inputs=[x_col, y_col, crs_text, chatbot],
        outputs=[map_output, chatbot]
    )
    user_input.submit(
        fn=handle_chat,
        inputs=[chatbot, user_input],
        outputs=[chatbot, map_output, download_btn]
    )
    user_input.submit(lambda *args: "", inputs=None, outputs=user_input)

demo.launch(ssr_mode=False)
