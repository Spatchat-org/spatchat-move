# map_layers.py

import os
import random

import numpy as np
import folium
import rasterio
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPolygon

from map_utils import (
    apply_map_control_patches,
    fit_map_to_bounds,
)

from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)

from rasterio.transform import array_bounds

from branca.element import MacroElement, Template


# =============================================================================
# Geometry helper
# =============================================================================

def _add_polygon_geom(
    layer,
    geom,
    *,
    color,
    popup,
    fill=True,
    fill_opacity=0.2,
    weight=2,
    opacity=1.0,
    dash_array=None,
):

    def add_one(poly):

        if poly.is_empty:
            return

        kwargs = {
            "locations": [
                (lat, lon)
                for lon, lat
                in poly.exterior.coords
            ],
            "color": color,
            "fill": fill,
            "fill_opacity": fill_opacity,
            "weight": weight,
            "opacity": opacity,
            "popup": popup,
        }

        if dash_array:
            kwargs[
                "dash_array"
            ] = dash_array

        layer.add_child(
            folium.Polygon(
                **kwargs
            )
        )

    if isinstance(
        geom,
        MultiPolygon
    ):

        for poly in geom.geoms:
            add_one(
                poly
            )

    elif isinstance(
        geom,
        Polygon
    ):

        add_one(
            geom
        )


# =============================================================================
# Raster helper
# =============================================================================

def _raster_for_webmap(
    tif_path
):
    """
    Read a scientific raster and return an in-memory EPSG:4326 copy
    suitable for Folium/Leaflet display.

    The original GeoTIFF is not modified.

    Returns
    -------
    arr
        Raster array in WGS84 geographic coordinates.

    bounds_ll
        Leaflet-compatible bounds:
        [[south, west], [north, east]]
    """

    with rasterio.open(
        tif_path
    ) as src:

        arr = (
            src.read(
                1
            )
            .astype(
                np.float32
            )
        )

        if src.crs is None:

            raise ValueError(
                f"Raster has no CRS: {tif_path}"
            )

        dst_crs = (
            "EPSG:4326"
        )

        # ---------------------------------------------------------------------
        # Raster is already WGS84
        # ---------------------------------------------------------------------

        if (
            src.crs.to_string().upper()
            ==
            dst_crs
        ):

            bounds = (
                src.bounds
            )

            bounds_ll = [
                [
                    bounds.bottom,
                    bounds.left
                ],
                [
                    bounds.top,
                    bounds.right
                ],
            ]

            return (
                arr,
                bounds_ll
            )

        # ---------------------------------------------------------------------
        # Reproject a display copy to WGS84
        # ---------------------------------------------------------------------

        (
            dst_transform,
            dst_width,
            dst_height
        ) = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds
        )

        dst_arr = np.full(
            (
                dst_height,
                dst_width
            ),
            np.nan,
            dtype=np.float32
        )

        reproject(
            source=arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        (
            west,
            south,
            east,
            north
        ) = array_bounds(
            dst_height,
            dst_width,
            dst_transform
        )

        bounds_ll = [
            [
                south,
                west
            ],
            [
                north,
                east
            ],
        ]

        return (
            dst_arr,
            bounds_ll
        )


def _raster_to_rgba(
    arr,
    *,
    alpha_power=0.35,
    alpha_max=190.0,
):
    """
    Convert a raster array to a plasma-colored RGBA image.

    Positive finite raster values are displayed.
    Zero, negative, NaN, and infinite values are transparent.
    """

    arr = np.asarray(
        arr,
        dtype=np.float32
    )

    valid = (
        np.isfinite(
            arr
        )
        &
        (
            arr > 0
        )
    )

    if np.any(
        valid
    ):

        max_value = float(
            np.nanmax(
                arr[
                    valid
                ]
            )
        )

    else:

        max_value = 0.0

    if (
        np.isfinite(
            max_value
        )
        and max_value > 0
    ):

        arr_norm = (
            arr
            / max_value
        )

    else:

        arr_norm = np.zeros_like(
            arr,
            dtype=np.float32
        )

    arr_norm = np.clip(
        arr_norm,
        0.0,
        1.0
    )

    cmap = plt.get_cmap(
        "plasma"
    )

    rgba = (
        cmap(
            arr_norm
        )
        * 255
    ).astype(
        np.uint8
    )

    alpha = np.where(
        valid,
        np.power(
            arr_norm,
            alpha_power
        )
        * alpha_max,
        0.0
    ).astype(
        np.uint8
    )

    img = np.dstack(
        [
            rgba[
                :,
                :,
                0
            ],
            rgba[
                :,
                :,
                1
            ],
            rgba[
                :,
                :,
                2
            ],
            alpha,
        ]
    )

    return img


# =============================================================================
# Base map
# =============================================================================

def _base_map(
    center_lat,
    center_lon,
    control_scale=True,
    zoom=9
):

    m = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=zoom,
        control_scale=control_scale
    )

    folium.TileLayer(
        "OpenStreetMap"
    ).add_to(
        m
    )

    folium.TileLayer(
        "CartoDB positron",
        attr="CartoDB"
    ).add_to(
        m
    )

    folium.TileLayer(
        tiles=(
            "https://{s}.tile.opentopomap.org/"
            "{z}/{y}/{x}.png"
        ),
        attr="OpenTopoMap",
        name="Topographic"
    ).add_to(
        m
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/"
            "{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Satellite"
    ).add_to(
        m
    )

    return m


# =============================================================================
# Layer-control styling
# =============================================================================

def _add_layer_control_style_patch(
    m
):

    root = (
        m.get_root()
    )

    if getattr(
        root,
        "_spatchat_layer_control_style_patch_added",
        False
    ):

        return

    setattr(
        root,
        "_spatchat_layer_control_style_patch_added",
        True
    )

    patch = MacroElement()

    patch._template = Template(
        """
        {% macro html(this, kwargs) %}
        <style>

          .leaflet-control-layers {
            background: rgba(30, 30, 30, 0.9) !important;
            color: #f5f5f5 !important;
            border-radius: 8px !important;
            border: none !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35) !important;
          }

          .leaflet-control-layers-toggle {
            background-color: rgba(30, 30, 30, 0.9) !important;
            border-radius: 8px !important;
            filter: invert(1) brightness(1.25);
          }

          .leaflet-control-layers-expanded {
            padding: 0 !important;
            background: rgba(30, 30, 30, 0.9) !important;
          }

          .leaflet-control-layers-list,
          .leaflet-control-layers form,
          .leaflet-control-layers .leaflet-control-layers-base,
          .leaflet-control-layers .leaflet-control-layers-overlays,
          .leaflet-control-layers .leaflet-control-layers-separator {
            background: rgba(30, 30, 30, 0.9) !important;
            color: inherit !important;
            border-color: rgba(255, 255, 255, 0.08) !important;
          }

          .leaflet-control-layers label {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 7px 10px;
            margin: 0;
            color: #f5f5f5 !important;
            font-size: 12px;
            line-height: 1.3;
          }

          .leaflet-control-layers label:hover {
            background: rgba(255, 255, 255, 0.05);
          }

          .leaflet-control-layers input[type="checkbox"],
          .leaflet-control-layers input[type="radio"] {
            accent-color: #8ec5ff;
          }

          .leaflet-control-layers .leaflet-control-layers-separator {
            margin: 0;
          }

          .leaflet-control-layers,
          .leaflet-control-layers .leaflet-control-layers-list {
            scrollbar-color:
              rgba(255, 255, 255, 0.18)
              rgba(255, 255, 255, 0.06);
            scrollbar-width: thin;
          }

          .leaflet-control-layers::-webkit-scrollbar,
          .leaflet-control-layers
          .leaflet-control-layers-list::-webkit-scrollbar {
            width: 10px;
            height: 10px;
          }

          .leaflet-control-layers::-webkit-scrollbar-track,
          .leaflet-control-layers
          .leaflet-control-layers-list::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.06);
          }

          .leaflet-control-layers::-webkit-scrollbar-thumb,
          .leaflet-control-layers
          .leaflet-control-layers-list::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.18);
            border-radius: 999px;
          }

          .leaflet-control-layers::-webkit-scrollbar-thumb:hover,
          .leaflet-control-layers
          .leaflet-control-layers-list::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.26);
          }

        </style>
        {% endmacro %}
        """
    )

    root.add_child(
        patch
    )


# =============================================================================
# Preview map
# =============================================================================

def build_preview_map(
    df
):

    has_timestamp = (
        "timestamp"
        in df.columns
    )

    m = _base_map(
        df[
            "latitude"
        ].mean(),
        df[
            "longitude"
        ].mean(),
        control_scale=True,
        zoom=9
    )

    points_layer = folium.FeatureGroup(
        name="Points",
        show=True
    )

    lines_layer = folium.FeatureGroup(
        name="Tracks",
        show=True
    )

    animal_ids = (
        df[
            "animal_id"
        ].unique()
        if "animal_id" in df.columns
        else [
            "sample"
        ]
    )

    color_map = {
        aid:
            f"#{random.randint(0, 0xFFFFFF):06x}"
        for aid
        in animal_ids
    }

    for animal in animal_ids:

        track = (
            df[
                df[
                    "animal_id"
                ]
                == animal
            ]
            if "animal_id"
            in df.columns
            else df
        )

        if has_timestamp:

            track = (
                track.sort_values(
                    "timestamp"
                )
            )

            coords = list(
                zip(
                    track[
                        "latitude"
                    ],
                    track[
                        "longitude"
                    ]
                )
            )

            if len(
                coords
            ) > 1:

                folium.PolyLine(
                    coords,
                    color=color_map[
                        animal
                    ],
                    weight=2.5,
                    opacity=0.8,
                    popup=str(
                        animal
                    )
                ).add_to(
                    lines_layer
                )

        for _, row in track.iterrows():

            label = (
                f"{animal}"
                +
                (
                    f"<br>{row['timestamp']}"
                    if (
                        has_timestamp
                        and row.get(
                            "timestamp"
                        )
                        is not None
                    )
                    else ""
                )
            )

            folium.CircleMarker(
                location=[
                    row[
                        "latitude"
                    ],
                    row[
                        "longitude"
                    ]
                ],
                radius=3,
                popup=label,
                color=color_map[
                    animal
                ],
                fill=True,
                fill_opacity=0.7
            ).add_to(
                points_layer
            )

    points_layer.add_to(
        m
    )

    if has_timestamp:

        lines_layer.add_to(
            m
        )

    folium.LayerControl(
        collapsed=False
    ).add_to(
        m
    )

    _add_layer_control_style_patch(
        m
    )

    apply_map_control_patches(
        m
    )

    m = fit_map_to_bounds(
        m,
        df
    )

    return (
        m._repr_html_()
    )


# =============================================================================
# Points and tracks
# =============================================================================

def make_points_tracks_layers(
    df,
    color_map
):

    points_layer = folium.FeatureGroup(
        name="Points",
        show=True
    )

    tracks_layer = folium.FeatureGroup(
        name="Tracks",
        show=True
    )

    has_timestamp = (
        "timestamp"
        in df.columns
    )

    animal_ids = (
        df[
            "animal_id"
        ].unique()
        if "animal_id"
        in df.columns
        else [
            "sample"
        ]
    )

    for animal in animal_ids:

        track = (
            df[
                df[
                    "animal_id"
                ]
                == animal
            ]
            if "animal_id"
            in df.columns
            else df
        )

        color = color_map[
            animal
        ]

        if has_timestamp:

            track = track.sort_values(
                "timestamp"
            )

            coords = list(
                zip(
                    track[
                        "latitude"
                    ],
                    track[
                        "longitude"
                    ]
                )
            )

            if len(
                coords
            ) > 1:

                folium.PolyLine(
                    coords,
                    color=color,
                    weight=2.5,
                    opacity=0.8,
                    popup=(
                        f"{animal} Track"
                    )
                ).add_to(
                    tracks_layer
                )

        for _, row in track.iterrows():

            folium.CircleMarker(
                location=[
                    row[
                        "latitude"
                    ],
                    row[
                        "longitude"
                    ]
                ],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{animal}"
            ).add_to(
                points_layer
            )

    return (
        points_layer,
        tracks_layer
    )


# =============================================================================
# MCP
# =============================================================================

def make_mcp_layers(
    mcp_results,
    requested_percents,
    animal_ids,
    color_map
):

    layers = []

    for percent in requested_percents:

        for animal in animal_ids:

            if (
                animal in mcp_results
                and percent
                in mcp_results[
                    animal
                ]
            ):

                v = (
                    mcp_results[
                        animal
                    ][
                        percent
                    ]
                )

                layer = folium.FeatureGroup(
                    name=(
                        f"{animal} "
                        f"MCP {percent}%"
                    ),
                    show=True
                )

                layer.add_child(
                    folium.Polygon(
                        locations=[
                            (
                                lat,
                                lon
                            )
                            for lon, lat
                            in np.array(
                                v[
                                    "polygon"
                                ].exterior.coords
                            )
                        ],
                        color=color_map[
                            animal
                        ],
                        fill=True,
                        fill_opacity=(
                            0.15
                            +
                            0.15
                            * (
                                percent
                                / 100
                            )
                        ),
                        popup=(
                            f"{animal} "
                            f"MCP {percent}%"
                        )
                    )
                )

                layers.append(
                    layer
                )

    return layers


# =============================================================================
# KDE
# =============================================================================

def make_kde_layers(
    kde_results,
    requested_kde_percents,
    animal_ids,
    color_map
):

    layers = []

    for animal in animal_ids:

        kde_percs = [
            p
            for p
            in requested_kde_percents
            if (
                animal
                in kde_results
                and p
                in kde_results[
                    animal
                ]
            )
        ]

        # ---------------------------------------------------------------------
        # Raster
        # ---------------------------------------------------------------------

        if kde_percs:

            max_perc = max(
                kde_percs
            )

            v = (
                kde_results[
                    animal
                ][
                    max_perc
                ]
            )

            tif_path = (
                v.get(
                    "geotiff"
                )
            )

            if (
                tif_path
                and os.path.exists(
                    tif_path
                )
            ):

                raster_layer = (
                    folium.FeatureGroup(
                        name=(
                            f"{animal} "
                            "KDE Raster"
                        ),
                        show=True
                    )
                )

                try:

                    (
                        arr,
                        bounds_ll
                    ) = _raster_for_webmap(
                        tif_path
                    )

                    img = _raster_to_rgba(
                        arr,
                        alpha_power=0.35,
                        alpha_max=190.0,
                    )

                    raster_layer.add_child(
                        folium.raster_layers.ImageOverlay(
                            image=img,
                            bounds=bounds_ll,
                            opacity=1.0,
                            interactive=False
                        )
                    )

                    layers.append(
                        raster_layer
                    )

                except Exception as exc:

                    print(
                        f"[map] KDE raster display error "
                        f"for animal={animal}: {exc}"
                    )

        # ---------------------------------------------------------------------
        # Contours
        # ---------------------------------------------------------------------

        for percent in kde_percs:

            v = (
                kde_results[
                    animal
                ][
                    percent
                ]
            )

            contour_layer = (
                folium.FeatureGroup(
                    name=(
                        f"{animal} KDE "
                        f"{percent}% Contour"
                    ),
                    show=True
                )
            )

            contour = (
                v.get(
                    "contour"
                )
            )

            if contour:

                _add_polygon_geom(
                    contour_layer,
                    contour,
                    color=color_map[
                        animal
                    ],
                    fill=True,
                    fill_opacity=0.2,
                    popup=(
                        f"{animal} KDE "
                        f"{percent}% Contour"
                    )
                )

            layers.append(
                contour_layer
            )

    return layers


# =============================================================================
# AKDE
# =============================================================================

def make_akde_layers(
    akde_results,
    requested_akde_percents,
    animal_ids,
    color_map
):

    layers = []

    for animal in animal_ids:

        akde_percs = [
            p
            for p
            in requested_akde_percents
            if (
                animal
                in akde_results
                and p
                in akde_results[
                    animal
                ]
            )
        ]

        # ---------------------------------------------------------------------
        # Raster
        # ---------------------------------------------------------------------

        if akde_percs:

            max_perc = max(
                akde_percs
            )

            v = (
                akde_results[
                    animal
                ][
                    max_perc
                ]
            )

            tif_path = (
                v.get(
                    "geotiff"
                )
            )

            if (
                tif_path
                and os.path.exists(
                    tif_path
                )
            ):

                raster_layer = (
                    folium.FeatureGroup(
                        name=(
                            f"{animal} "
                            "AKDE Raster"
                        ),
                        show=True
                    )
                )

                try:

                    (
                        arr,
                        bounds_ll
                    ) = _raster_for_webmap(
                        tif_path
                    )

                    img = _raster_to_rgba(
                        arr,
                        alpha_power=0.35,
                        alpha_max=190.0,
                    )

                    raster_layer.add_child(
                        folium.raster_layers.ImageOverlay(
                            image=img,
                            bounds=bounds_ll,
                            opacity=1.0,
                            interactive=False
                        )
                    )

                    layers.append(
                        raster_layer
                    )

                except Exception as exc:

                    print(
                        f"[map] AKDE raster display error "
                        f"for animal={animal}: {exc}"
                    )

        # ---------------------------------------------------------------------
        # Main contours
        # ---------------------------------------------------------------------

        for percent in akde_percs:

            v = (
                akde_results[
                    animal
                ][
                    percent
                ]
            )

            contour_layer = (
                folium.FeatureGroup(
                    name=(
                        f"{animal} AKDE "
                        f"{percent}% Contour"
                    ),
                    show=True
                )
            )

            contour = (
                v.get(
                    "contour"
                )
            )

            area = float(
                v.get(
                    "area",
                    0.0
                )
            )

            ci = (
                v.get(
                    "area_ci95_km2"
                )
            )

            if (
                isinstance(
                    ci,
                    (
                        list,
                        tuple
                    )
                )
                and len(
                    ci
                )
                == 2
            ):

                popup = (
                    f"{animal} AKDE "
                    f"{percent}% Contour"
                    f"<br>Area: "
                    f"{area:.2f} km²"
                    f"<br>95% CI: "
                    f"[{float(ci[0]):.2f}, "
                    f"{float(ci[1]):.2f}] km²"
                )

            else:

                popup = (
                    f"{animal} AKDE "
                    f"{percent}% Contour"
                    f"<br>Area: "
                    f"{area:.2f} km²"
                )

            if contour:

                _add_polygon_geom(
                    contour_layer,
                    contour,
                    color=color_map[
                        animal
                    ],
                    fill=True,
                    fill_opacity=0.2,
                    popup=popup
                )

            layers.append(
                contour_layer
            )

            # -----------------------------------------------------------------
            # AKDE CI contours
            # -----------------------------------------------------------------

            ci_contours = (
                v.get(
                    "ci_contours"
                )
                or {}
            )

            if ci_contours:

                ci_layer = (
                    folium.FeatureGroup(
                        name=(
                            f"{animal} AKDE "
                            f"{percent}% "
                            "95% CI Contours"
                        ),
                        show=True
                    )
                )

                ci_values = {
                    "low":
                        v.get(
                            "ci_low_km2"
                        ),

                    "high":
                        v.get(
                            "ci_high_km2"
                        ),
                }

                for (
                    label,
                    ci_contour
                ) in ci_contours.items():

                    ci_area = (
                        ci_values.get(
                            label
                        )
                    )

                    area_text = (
                        f"{float(ci_area):.2f} km²"
                        if ci_area is not None
                        else "not available"
                    )

                    ci_popup = (
                        f"{animal} AKDE "
                        f"{percent}% 95% CI "
                        f"{label} contour"
                        f"<br>Area: "
                        f"{area_text}"
                    )

                    _add_polygon_geom(
                        ci_layer,
                        ci_contour,
                        color=color_map[
                            animal
                        ],
                        fill=False,
                        fill_opacity=0.0,
                        weight=(
                            2
                            if label
                            == "low"
                            else 3
                        ),
                        opacity=(
                            0.9
                            if label
                            == "low"
                            else 0.75
                        ),
                        dash_array=(
                            "5,5"
                            if label
                            == "low"
                            else "10,6"
                        ),
                        popup=ci_popup
                    )

                layers.append(
                    ci_layer
                )

    return layers


# =============================================================================
# LoCoH isopleths
# =============================================================================

def make_locoh_layers(
    locoh_result: dict,
    animal_ids,
    color_map,
    name_prefix: str = "LoCoH"
):

    layers = []

    animals_dict = (
        locoh_result.get(
            "animals",
            {}
        )
        if locoh_result
        else {}
    )

    for animal in animal_ids:

        data = (
            animals_dict.get(
                str(
                    animal
                )
            )
            or animals_dict.get(
                animal
            )
        )

        if not data:
            continue

        for item in data.get(
            "isopleths",
            []
        ):

            iso = int(
                item[
                    "isopleth"
                ]
            )

            gj_geom = (
                item[
                    "geometry"
                ]
            )

            area_km2 = float(
                item.get(
                    "area_sq_km",
                    0.0
                )
            )

            feature = {
                "type":
                    "Feature",

                "properties": {
                    "animal_id":
                        str(
                            animal
                        ),

                    "isopleth":
                        iso,

                    "area_km2":
                        round(
                            area_km2,
                            3
                        ),
                },

                "geometry":
                    gj_geom,
            }

            layer = (
                folium.FeatureGroup(
                    name=(
                        f"{animal} "
                        f"{name_prefix} "
                        f"{iso}%"
                    ),
                    show=True
                )
            )

            folium.GeoJson(
                data=feature,
                name=(
                    f"{name_prefix} "
                    f"{iso}% — "
                    f"{animal}"
                ),
                style_function=(
                    lambda _feat,
                    iso=iso,
                    animal=animal:
                    {
                        "fillOpacity":
                            (
                                0.45
                                if iso
                                != 95
                                else 0.25
                            ),

                        "weight":
                            2,

                        "color":
                            color_map[
                                animal
                            ],
                    }
                ),
                tooltip=folium.GeoJsonTooltip(
                    fields=[
                        "animal_id",
                        "isopleth",
                        "area_km2"
                    ],
                    aliases=[
                        "Animal",
                        "Isopleth (%)",
                        "Area (km²)"
                    ],
                    localize=True,
                ),
            ).add_to(
                layer
            )

            layers.append(
                layer
            )

    return layers


# =============================================================================
# LoCoH facets
# =============================================================================

def make_locoh_facets_layers(
    locoh_result: dict,
    animal_ids,
    color_map,
    name_prefix: str = "LoCoH facets"
):

    bins = [
        20,
        40,
        60,
        80,
        95,
        100
    ]

    palette = {
        20:
            "#d7301f",

        40:
            "#ef6548",

        60:
            "#fc8d59",

        80:
            "#fdbb84",

        95:
            "#fdd49e",

        100:
            "#fee8c8",
    }

    def color_for(
        pct: int
    ) -> str:

        for b in bins:

            if pct <= b:

                return palette[
                    b
                ]

        return palette[
            100
        ]

    layers = []

    animals_dict = (
        locoh_result.get(
            "animals",
            {}
        )
        if locoh_result
        else {}
    )

    for animal in animal_ids:

        data = (
            animals_dict.get(
                str(
                    animal
                )
            )
            or animals_dict.get(
                animal
            )
        )

        if not data:
            continue

        facets = (
            data.get(
                "facets",
                []
            )
        )

        if not facets:
            continue

        features = []

        for f in facets:

            pct = int(
                f.get(
                    "cum_percent",
                    100
                )
            )

            area = float(
                f.get(
                    "area_sq_km",
                    0.0
                )
            )

            features.append(
                {
                    "type":
                        "Feature",

                    "properties": {
                        "animal_id":
                            str(
                                animal
                            ),

                        "cum_percent":
                            pct,

                        "area_km2":
                            round(
                                area,
                                3
                            ),

                        "_fill":
                            color_for(
                                pct
                            ),
                    },

                    "geometry":
                        f[
                            "geometry"
                        ],
                }
            )

        fc = {
            "type":
                "FeatureCollection",

            "features":
                features,
        }

        layer = (
            folium.FeatureGroup(
                name=(
                    f"{animal} "
                    f"{name_prefix}"
                ),
                show=True
            )
        )

        folium.GeoJson(
            data=fc,
            name=(
                f"{name_prefix} — "
                f"{animal}"
            ),
            style_function=(
                lambda feat:
                {
                    "fillOpacity":
                        0.6,

                    "weight":
                        1,

                    "color":
                        "#222222",

                    "fillColor":
                        feat[
                            "properties"
                        ][
                            "_fill"
                        ],
                }
            ),
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "animal_id",
                    "cum_percent",
                    "area_km2"
                ],
                aliases=[
                    "Animal",
                    "Cum. %",
                    "Hull area (km²)"
                ],
                localize=True,
            ),
        ).add_to(
            layer
        )

        layers.append(
            layer
        )

    return layers


# =============================================================================
# dBBMM
# =============================================================================

def make_dbbmm_layers(
    dbbmm_results: dict,
    animal_ids,
    color_map,
    name_prefix: str = "dBBMM"
):

    layers = []

    for animal in animal_ids:

        data = (
            dbbmm_results.get(
                str(
                    animal
                )
            )
            or dbbmm_results.get(
                animal
            )
        )

        if not data:
            continue

        if isinstance(
            data,
            dict
        ):

            tif_path = (
                data.get(
                    "geotiff"
                )
            )

            iso_list = (
                data.get(
                    "isopleths"
                )
                or []
            )

        else:

            tif_path = (
                getattr(
                    data,
                    "geotiff",
                    None
                )
            )

            iso_list = (
                getattr(
                    data,
                    "isopleths",
                    []
                )
                or []
            )

        color = (
            color_map[
                animal
            ]
        )

        # ---------------------------------------------------------------------
        # Raster
        # ---------------------------------------------------------------------

        if (
            tif_path
            and os.path.exists(
                tif_path
            )
        ):

            raster_layer = (
                folium.FeatureGroup(
                    name=(
                        f"{animal} "
                        f"{name_prefix} Raster"
                    ),
                    show=True
                )
            )

            try:

                (
                    arr,
                    bounds_ll
                ) = _raster_for_webmap(
                    tif_path
                )

                img = _raster_to_rgba(
                    arr,
                    alpha_power=0.35,
                    alpha_max=190.0,
                )

                raster_layer.add_child(
                    folium.raster_layers.ImageOverlay(
                        image=img,
                        bounds=bounds_ll,
                        opacity=0.80,
                        interactive=False
                    )
                )

                layers.append(
                    raster_layer
                )

            except Exception as exc:

                print(
                    f"[map] dBBMM raster display error "
                    f"for animal={animal}: {exc}"
                )

        # ---------------------------------------------------------------------
        # Isopleths
        # ---------------------------------------------------------------------

        for item in iso_list:

            percent = int(
                item.get(
                    "percent"
                )
            )

            geom = (
                item.get(
                    "geometry"
                )
            )

            area_km2 = float(
                item.get(
                    "area_sq_km",
                    0.0
                )
            )

            layer = (
                folium.FeatureGroup(
                    name=(
                        f"{animal} "
                        f"{name_prefix} "
                        f"{percent}%"
                    ),
                    show=True
                )
            )

            if geom:

                folium.GeoJson(
                    data={
                        "type":
                            "Feature",

                        "properties": {
                            "animal_id":
                                str(
                                    animal
                                ),

                            "percent":
                                percent,

                            "area_km2":
                                round(
                                    area_km2,
                                    3
                                ),
                        },

                        "geometry":
                            geom,
                    },
                    style_function=(
                        lambda _feat,
                        color=color:
                        {
                            "fillOpacity":
                                0.25,

                            "weight":
                                2,

                            "color":
                                color,
                        }
                    ),
                    tooltip=folium.GeoJsonTooltip(
                        fields=[
                            "animal_id",
                            "percent",
                            "area_km2"
                        ],
                        aliases=[
                            "Animal",
                            "Isopleth (%)",
                            "Area (km²)"
                        ],
                        localize=True,
                    ),
                ).add_to(
                    layer
                )

            layers.append(
                layer
            )

    return layers


# =============================================================================
# Composition entrypoint
# =============================================================================

def build_results_map(
    df,
    mcp_results,
    kde_results,
    requested_percents,
    requested_kde_percents,
    akde_results=None,
    requested_akde_percents=None,
    locoh_result=None,
    dbbmm_result=None,
):

    m = _base_map(
        df[
            "latitude"
        ].mean(),
        df[
            "longitude"
        ].mean(),
        control_scale=True,
        zoom=9
    )

    animal_ids = (
        df[
            "animal_id"
        ].unique()
        if "animal_id"
        in df.columns
        else [
            "sample"
        ]
    )

    color_map = {
        aid:
            f"#{random.randint(0, 0xFFFFFF):06x}"
        for aid
        in animal_ids
    }

    # -------------------------------------------------------------------------
    # Points and tracks
    # -------------------------------------------------------------------------

    (
        points_layer,
        tracks_layer
    ) = make_points_tracks_layers(
        df,
        color_map
    )

    points_layer.add_to(
        m
    )

    tracks_layer.add_to(
        m
    )

    # -------------------------------------------------------------------------
    # MCP
    # -------------------------------------------------------------------------

    for layer in make_mcp_layers(
        mcp_results
        or {},
        requested_percents
        or [],
        animal_ids,
        color_map
    ):

        layer.add_to(
            m
        )

    # -------------------------------------------------------------------------
    # KDE
    # -------------------------------------------------------------------------

    for layer in make_kde_layers(
        kde_results
        or {},
        requested_kde_percents
        or [],
        animal_ids,
        color_map
    ):

        layer.add_to(
            m
        )

    # -------------------------------------------------------------------------
    # AKDE
    # -------------------------------------------------------------------------

    for layer in make_akde_layers(
        akde_results
        or {},
        requested_akde_percents
        or [],
        animal_ids,
        color_map
    ):

        layer.add_to(
            m
        )

    # -------------------------------------------------------------------------
    # LoCoH
    # -------------------------------------------------------------------------

    if locoh_result is not None:

        for layer in make_locoh_layers(
            locoh_result,
            animal_ids,
            color_map,
            name_prefix="LoCoH"
        ):

            layer.add_to(
                m
            )

        for layer in make_locoh_facets_layers(
            locoh_result,
            animal_ids,
            color_map,
            name_prefix="LoCoH facets"
        ):

            layer.add_to(
                m
            )

    # -------------------------------------------------------------------------
    # dBBMM
    # -------------------------------------------------------------------------

    if dbbmm_result is not None:

        for layer in make_dbbmm_layers(
            dbbmm_result,
            animal_ids,
            color_map,
            name_prefix="dBBMM"
        ):

            layer.add_to(
                m
            )

    # -------------------------------------------------------------------------
    # Controls
    # -------------------------------------------------------------------------

    folium.LayerControl(
        collapsed=False
    ).add_to(
        m
    )

    _add_layer_control_style_patch(
        m
    )

    apply_map_control_patches(
        m
    )

    m = fit_map_to_bounds(
        m,
        df
    )

    return (
        m._repr_html_()
    )