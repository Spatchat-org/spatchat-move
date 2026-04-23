# storage.py
import contextvars
import json
import os
import shutil
import zipfile
from collections.abc import MutableMapping, MutableSet

import pandas as pd
from shapely.geometry import mapping


_CURRENT_SESSION = contextvars.ContextVar("spatchat_hr_session", default="__default__")
_SESSION_STATE: dict[str, dict] = {}


def _new_state() -> dict:
    return {
        "mcp_results": {},
        "kde_results": {},
        "akde_results": {},
        "locoh_results": None,
        "dbbmm_results": {},
        "requested_percents": set(),
        "requested_kde_percents": set(),
        "requested_akde_percents": set(),
        "requested_dbbmm_percents": set(),
        "cached_df": None,
        "cached_headers": [],
        "last_detection_summary": "",
        "output_dir": "outputs",
        "repro_manifest": {
            "mcp": {"percents": []},
            "kde": {"percents": [], "params": {}},
            "akde": {"percents": [], "params": {}},
            "locoh": {"requested": False, "params": {}},
            "dbbmm": {"percents": [], "params": {}},
            "movement": {
                "displacement": False,
                "step_lengths": False,
                "turning_angles": False,
                "autocorrelation": False,
                "hmm": False,
                "states": 3,
            },
        },
    }


def _state() -> dict:
    key = _CURRENT_SESSION.get()
    if key not in _SESSION_STATE:
        _SESSION_STATE[key] = _new_state()
    return _SESSION_STATE[key]


def set_current_session(session_id: str | None) -> None:
    _CURRENT_SESSION.set((session_id or "__default__").strip() or "__default__")
    _state()


def delete_session(session_id: str | None) -> None:
    sid = (session_id or "").strip()
    if not sid or sid == "__default__":
        return
    _SESSION_STATE.pop(sid, None)
    if _CURRENT_SESSION.get() == sid:
        _CURRENT_SESSION.set("__default__")


def set_output_dir(path: str | None) -> None:
    state = _state()
    state["output_dir"] = path or "outputs"


def get_output_dir() -> str:
    return _state().get("output_dir") or "outputs"


def get_repro_manifest() -> dict:
    manifest = _state().get("repro_manifest")
    if not isinstance(manifest, dict):
        manifest = _new_state()["repro_manifest"]
        _state()["repro_manifest"] = manifest
    return manifest


def set_repro_manifest(manifest: dict | None) -> None:
    _state()["repro_manifest"] = manifest if isinstance(manifest, dict) else _new_state()["repro_manifest"]


class _SessionDictProxy(MutableMapping):
    def __init__(self, key: str):
        self.key = key

    def _target(self) -> dict:
        return _state()[self.key]

    def __getitem__(self, item):
        return self._target()[item]

    def __setitem__(self, item, value):
        self._target()[item] = value

    def __delitem__(self, item):
        del self._target()[item]

    def __iter__(self):
        return iter(self._target())

    def __len__(self):
        return len(self._target())

    def clear(self):
        self._target().clear()

    def update(self, *args, **kwargs):
        self._target().update(*args, **kwargs)

    def copy(self):
        return self._target().copy()

    def items(self):
        return self._target().items()

    def keys(self):
        return self._target().keys()

    def values(self):
        return self._target().values()

    def __contains__(self, item):
        return item in self._target()

    def __repr__(self):
        return repr(self._target())


class _SessionSetProxy(MutableSet):
    def __init__(self, key: str):
        self.key = key

    def _target(self) -> set:
        return _state()[self.key]

    def __contains__(self, item):
        return item in self._target()

    def __iter__(self):
        return iter(self._target())

    def __len__(self):
        return len(self._target())

    def add(self, value):
        self._target().add(value)

    def discard(self, value):
        self._target().discard(value)

    def clear(self):
        self._target().clear()

    def update(self, *others):
        self._target().update(*others)

    def copy(self):
        return self._target().copy()

    def __repr__(self):
        return repr(self._target())


mcp_results = _SessionDictProxy("mcp_results")
kde_results = _SessionDictProxy("kde_results")
akde_results = _SessionDictProxy("akde_results")
dbbmm_results = _SessionDictProxy("dbbmm_results")
requested_percents = _SessionSetProxy("requested_percents")
requested_kde_percents = _SessionSetProxy("requested_kde_percents")
requested_akde_percents = _SessionSetProxy("requested_akde_percents")
requested_dbbmm_percents = _SessionSetProxy("requested_dbbmm_percents")


def get_cached_df():
    return _state()["cached_df"]


def set_cached_df(df):
    _state()["cached_df"] = df


def get_cached_headers():
    return list(_state()["cached_headers"])


def set_cached_headers(headers):
    _state()["cached_headers"] = list(headers or [])


def set_detection_summary(text: str):
    _state()["last_detection_summary"] = text or ""


def get_detection_summary() -> str:
    return _state()["last_detection_summary"]


def get_locoh_results():
    return _state()["locoh_results"]


def set_locoh_results(res: dict | None):
    _state()["locoh_results"] = res


def get_dbbmm_results():
    return dbbmm_results


def set_dbbmm_results(res: dict | None):
    dbbmm_results.clear()
    if isinstance(res, dict):
        dbbmm_results.update(res)


def get_akde_results():
    return akde_results


def set_akde_results(res: dict | None):
    akde_results.clear()
    if isinstance(res, dict):
        akde_results.update(res)


def clear_all_results():
    mcp_results.clear()
    kde_results.clear()
    akde_results.clear()
    dbbmm_results.clear()
    requested_percents.clear()
    requested_kde_percents.clear()
    requested_akde_percents.clear()
    requested_dbbmm_percents.clear()
    _state()["locoh_results"] = None
    _state()["repro_manifest"] = _new_state()["repro_manifest"]

    outdir = get_output_dir()
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)


def _write_mcp_assets(rows_accum: list[tuple], outdir: str):
    features = []
    for animal, percents in mcp_results.items():
        for percent, v in percents.items():
            area = float(v.get("area", 0.0))
            rows_accum.append((animal, f"MCP-{percent}", area))
            poly = v.get("polygon")
            if poly is not None:
                features.append({
                    "type": "Feature",
                    "properties": {"animal_id": str(animal), "percent": int(percent), "area_km2": area},
                    "geometry": mapping(poly)
                })

    if features:
        with open(os.path.join(outdir, "mcps_all.geojson"), "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)


def _write_kde_assets(rows_accum: list[tuple], outdir: str):
    index = {"animals": {}}
    any_kde = False
    features_all = []
    to_delete_paths = set()
    outdir_abs = os.path.abspath(outdir)

    for animal, percents in kde_results.items():
        index["animals"][animal] = {}
        for percent, v in percents.items():
            any_kde = True
            area = float(v.get("area", 0.0))
            rows_accum.append((animal, f"KDE-{percent}", area))
            index["animals"][animal][str(percent)] = {
                "area_km2": area,
                "geotiff": v.get("geotiff"),
                "geojson": v.get("geojson"),
            }

            feat_list = []
            gj_path = v.get("geojson")
            if gj_path and os.path.exists(gj_path):
                try:
                    with open(gj_path, "r", encoding="utf-8") as f:
                        gj = json.load(f)
                    if isinstance(gj, dict) and gj.get("type") == "FeatureCollection":
                        feat_list = gj.get("features", []) or []
                    elif isinstance(gj, dict) and gj.get("type") == "Feature":
                        feat_list = [gj]
                    gj_abs = os.path.abspath(gj_path)
                    if gj_abs.startswith(outdir_abs + os.sep):
                        to_delete_paths.add(gj_abs)
                except Exception:
                    feat_list = []

            if not feat_list:
                contour = v.get("contour")
                if contour is not None:
                    try:
                        feat_list = [{"type": "Feature", "properties": {}, "geometry": mapping(contour)}]
                    except Exception:
                        feat_list = []

            for feat in feat_list:
                if not isinstance(feat, dict):
                    continue
                props = feat.setdefault("properties", {})
                props["animal_id"] = str(animal)
                props["percent"] = int(percent)
                props["area_km2"] = area
                features_all.append(feat)

    if any_kde:
        with open(os.path.join(outdir, "kde_index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    if features_all:
        with open(os.path.join(outdir, "kdes_all.geojson"), "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features_all}, f)
        for p in sorted(to_delete_paths):
            try:
                os.remove(p)
            except OSError:
                pass


def _write_akde_assets(rows_accum: list[tuple], outdir: str):
    index = {"animals": {}}
    any_akde = False
    features_all = []
    to_delete_paths = set()
    outdir_abs = os.path.abspath(outdir)

    for animal, percents in akde_results.items():
        index["animals"][animal] = {}
        for percent, v in percents.items():
            any_akde = True
            area = float(v.get("area", 0.0))
            rows_accum.append((animal, f"AKDE-{percent}", area))
            index["animals"][animal][str(percent)] = {
                "area_km2": area,
                "geotiff": v.get("geotiff"),
                "geojson": v.get("geojson"),
                "meta": v.get("meta"),
            }

            feat_list = []
            gj_path = v.get("geojson")
            if gj_path and os.path.exists(gj_path):
                try:
                    with open(gj_path, "r", encoding="utf-8") as f:
                        gj = json.load(f)
                    if isinstance(gj, dict) and gj.get("type") == "FeatureCollection":
                        feat_list = gj.get("features", []) or []
                    elif isinstance(gj, dict) and gj.get("type") == "Feature":
                        feat_list = [gj]
                    else:
                        feat_list = [{"type": "Feature", "properties": {}, "geometry": gj}]
                    gj_abs = os.path.abspath(gj_path)
                    if gj_abs.startswith(outdir_abs + os.sep):
                        to_delete_paths.add(gj_abs)
                except Exception:
                    feat_list = []

            if not feat_list:
                contour = v.get("contour")
                if contour is not None:
                    try:
                        feat_list = [{"type": "Feature", "properties": {}, "geometry": mapping(contour)}]
                    except Exception:
                        feat_list = []

            for feat in feat_list:
                if not isinstance(feat, dict):
                    continue
                props = feat.setdefault("properties", {})
                props["animal_id"] = str(animal)
                props["percent"] = int(percent)
                props["area_km2"] = area
                features_all.append(feat)

    if any_akde:
        with open(os.path.join(outdir, "akde_index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    if features_all:
        with open(os.path.join(outdir, "akdes_all.geojson"), "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features_all}, f)
        for p in sorted(to_delete_paths):
            try:
                os.remove(p)
            except OSError:
                pass


def _write_locoh_assets(rows_accum: list[tuple], outdir: str):
    locoh_result = get_locoh_results()
    if not locoh_result or not isinstance(locoh_result, dict):
        return

    with open(os.path.join(outdir, "locoh_results.json"), "w", encoding="utf-8") as f:
        json.dump(locoh_result, f)

    animals = (locoh_result.get("animals") or {})
    envelope_features = []
    facets_features = []

    for animal_id, data in animals.items():
        for item in data.get("isopleths", []):
            iso = int(item.get("isopleth"))
            area = float(item.get("area_sq_km", 0.0))
            rows_accum.append((animal_id, f"LoCoH-{iso}", area))
            gj = item.get("geometry")
            if gj:
                envelope_features.append({
                    "type": "Feature",
                    "properties": {"animal_id": str(animal_id), "isopleth": iso, "area_km2": area},
                    "geometry": gj,
                })

        for fct in (data.get("facets") or []):
            facets_features.append({
                "type": "Feature",
                "properties": {
                    "animal_id": str(animal_id),
                    "cum_percent": int(fct.get("cum_percent", 0)),
                    "area_km2": float(fct.get("area_sq_km", 0.0)),
                },
                "geometry": fct.get("geometry"),
            })

    if envelope_features:
        with open(os.path.join(outdir, "locoh_envelopes_all.geojson"), "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": envelope_features}, f)

    if facets_features:
        with open(os.path.join(outdir, "locoh_facets_all.geojson"), "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": facets_features}, f)


def _write_dbbmm_assets(rows_accum: list[tuple], outdir: str):
    index = {"animals": {}}
    any_bb = False
    features_all = []

    for animal, data in dbbmm_results.items():
        any_bb = True
        if isinstance(data, dict):
            geotiff = data.get("geotiff")
            iso_list = data.get("isopleths", []) or []
        else:
            geotiff = getattr(data, "geotiff", None)
            iso_list = getattr(data, "isopleths", []) or []

        index["animals"][animal] = {"geotiff": geotiff, "isopleths": []}
        for item in iso_list:
            p = int(item.get("percent"))
            area = float(item.get("area_sq_km", 0.0))
            index["animals"][animal]["isopleths"].append({"percent": p, "area_km2": area})
            rows_accum.append((animal, f"dBBMM-{p}", area))
            gj = item.get("geometry")
            if gj:
                features_all.append({
                    "type": "Feature",
                    "properties": {"animal_id": str(animal), "percent": p, "area_km2": area},
                    "geometry": gj,
                })

    if any_bb:
        with open(os.path.join(outdir, "dbbmm_index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    if features_all:
        with open(os.path.join(outdir, "dbbmms_all.geojson"), "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features_all}, f)


def save_all_mcps_zip():
    outdir = get_output_dir()
    os.makedirs(outdir, exist_ok=True)
    rows: list[tuple] = []

    _write_mcp_assets(rows, outdir)
    _write_kde_assets(rows, outdir)
    _write_akde_assets(rows, outdir)
    _write_locoh_assets(rows, outdir)
    _write_dbbmm_assets(rows, outdir)

    if rows:
        df = pd.DataFrame(rows, columns=["animal_id", "type", "area_km2"])
        df.sort_values(["animal_id", "type"], inplace=True)
        df.to_csv(os.path.join(outdir, "home_range_areas.csv"), index=False)

    archive = os.path.join(outdir, "spatchat_results.zip")
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(outdir):
            for file in files:
                if file.endswith(".zip"):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, outdir)
                zipf.write(full_path, arcname=rel_path)

    print("ZIP written:", archive)
    return archive
