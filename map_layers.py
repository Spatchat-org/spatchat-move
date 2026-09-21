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
                for lon, lat in poly.exterior.coords
            ],
            "color": color,
            "fill": fill,
            "fill_opacity": fill_opacity,
            "weight": weight,
            "opacity": opacity,
            "popup": popup,
        }

        if dash_array:
            kwargs["dash_array"] = dash_array

        layer.add_child(
            folium.Polygon(**kwargs)
        )

    if isinstance(geom, MultiPolygon):

        for poly in geom.geoms:
            add_one(poly)

    elif isinstance(geom, Polygon):

        add_one(geom)


# =============================================================================
# Raster helpers
# =============================================================================

def _raster_for_webmap(tif_path):
    """
    Read a scientific raster and create an in-memory EPSG:4326
    version for Leaflet/Folium display.

    The original GeoTIFF is not changed.

    Returns
    -------
    arr
        Raster array in WGS84.

    bounds_ll
        Leaflet bounds:
        [[south, west], [north, east]]
    """

    with rasterio.open(tif_path) as src:

        arr = (
            src.read(1)
            .astype(np.float32)
        )

        if src.crs is None:
            raise ValueError(
                f"Raster has no CRS: {tif_path}"
            )

        dst_crs = "EPSG:4326"

        # ---------------------------------------------------------------------
        # Already WGS84
        # ---------------------------------------------------------------------

        if src.crs.to_string().upper() == dst_crs:

            bounds = src.bounds

            return (
                arr,
                [
                    [
                        bounds.bottom,
                        bounds.left,
                    ],
                    [
                        bounds.top,
                        bounds.right,
                    ],
                ],
            )

        # ---------------------------------------------------------------------
        # Reproject a temporary display copy
        # ---------------------------------------------------------------------

        (
            dst_transform,
            dst_width,
            dst_height,
        ) = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
        )

        dst_arr = np.full(
            (
                dst_height,
                dst_width,
            ),
            np.nan,
            dtype=np.float32,
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
            north,
        ) = array_bounds(
            dst_height,
            dst_width,
            dst_transform,
        )

        bounds_ll = [
            [
                south,
                west,
            ],
            [
                north,
                east,
            ],
        ]

        return (
            dst_arr,
            bounds_ll,
        )


def _raster_to_rgba(
    arr,
    *,
    alpha_power=0.35,
    alpha_max=190.0,
):
    """
    Convert raster values to a plasma RGBA image.

    Positive finite values are displayed.
    Zero, negative, NaN, and infinite values are transparent.
    """

    arr = np.asarray(
        arr,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(arr)
        & (arr > 0)
    )

    if np.any(valid):

        max_value = float(
            np.nanmax(
                arr[valid]
            )
        )

    else:

        max_value = 0.0

    if (
        np.isfinite(max_value)
        and max_value > 0
    ):

        arr_norm = (
            arr
            / max_value
        )

    else:

        arr_norm = np.zeros_like(
            arr,
            dtype=np.float32,
        )

    arr_norm = np.clip(
        arr_norm,
        0.0,
        1.0,
    )

    cmap = plt.get_cmap(
        "plasma"
    )

    rgba = (
        cmap(arr_norm)
        * 255
    ).astype(
        np.uint8
    )

    alpha = np.where(
        valid,
        np.power(
            arr_norm,
            alpha_power,
        )
        * alpha_max,
        0.0,
    ).astype(
        np.uint8
    )

    return np.dstack(
        [
            rgba[:, :, 0],
            rgba[:, :, 1],
            rgba[:, :, 2],
            alpha,
        ]
    )


# =============================================================================
# Base map
# =============================================================================

def _base_map(
    center_lat,
    center_lon,
    control_scale=True,
    zoom=9,
):
    """
    Create the Leaflet base map.

    A blank background is included as one of the mutually exclusive
    base-map choices.
    """

    m = folium.Map(
        location=[
            center_lat,
            center_lon,
        ],
        zoom_start=zoom,
        control_scale=control_scale,
        tiles=None,
    )

    # -------------------------------------------------------------------------
    # Blank base layer
    #
    # A transparent 1x1 PNG is used for all tiles. The map container
    # background is forced to white by the CSS patch below.
    # -------------------------------------------------------------------------

    blank_tile = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVQIHWP4z8DwHwAFgAI/ScL6WQAAAABJRU5ErkJggg=="
    )

    folium.TileLayer(
        tiles=blank_tile,
        name="Blank",
        attr="Blank background",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # -------------------------------------------------------------------------
    # OpenStreetMap
    # -------------------------------------------------------------------------

    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)

    # -------------------------------------------------------------------------
    # CartoDB Positron
    # -------------------------------------------------------------------------

    folium.TileLayer(
        "CartoDB positron",
        name="CartoDB Positron",
        attr="CartoDB",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # -------------------------------------------------------------------------
    # Topographic
    # -------------------------------------------------------------------------

    folium.TileLayer(
        tiles=(
            "https://{s}.tile.opentopomap.org/"
            "{z}/{x}/{y}.png"
        ),
        attr="OpenTopoMap",
        name="Topographic",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # -------------------------------------------------------------------------
    # Satellite
    # -------------------------------------------------------------------------

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/"
            "{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    return m


# =============================================================================
# Layer-control appearance
# =============================================================================

def _add_layer_control_style_patch(m):

    root = m.get_root()

    if getattr(
        root,
        "_spatchat_layer_control_style_patch_added",
        False,
    ):
        return

    setattr(
        root,
        "_spatchat_layer_control_style_patch_added",
        True,
    )

    patch = MacroElement()

    patch._template = Template(
        """
        {% macro html(this, kwargs) %}

        <style>

        /* White canvas used by the Blank base map */
        .leaflet-container {
            background: #ffffff !important;
        }

        .leaflet-control-layers {
            background: rgba(30, 30, 30, 0.92) !important;
            color: #f5f5f5 !important;
            border-radius: 8px !important;
            border: none !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35) !important;

            /*
             * Prevent very long layer lists from taking over the map.
             */
            max-height: 78vh;
            overflow-y: auto;
            min-width: 220px;
        }

        .leaflet-control-layers-toggle {
            background-color: rgba(30, 30, 30, 0.92) !important;
            border-radius: 8px !important;
            filter: invert(1) brightness(1.25);
        }

        .leaflet-control-layers-expanded {
            padding: 0 !important;
            background: rgba(30, 30, 30, 0.92) !important;
        }

        .leaflet-control-layers-list,
        .leaflet-control-layers form,
        .leaflet-control-layers .leaflet-control-layers-base,
        .leaflet-control-layers .leaflet-control-layers-overlays,
        .leaflet-control-layers .leaflet-control-layers-separator {
            background: rgba(30, 30, 30, 0.92) !important;
            color: inherit !important;
            border-color: rgba(255, 255, 255, 0.08) !important;
        }

        .leaflet-control-layers label {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
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
                rgba(255, 255, 255, 0.22)
                rgba(255, 255, 255, 0.06);
            scrollbar-width: thin;
        }

        .leaflet-control-layers::-webkit-scrollbar,
        .leaflet-control-layers-list::-webkit-scrollbar {
            width: 9px;
        }

        .leaflet-control-layers::-webkit-scrollbar-track,
        .leaflet-control-layers-list::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.06);
        }

        .leaflet-control-layers::-webkit-scrollbar-thumb,
        .leaflet-control-layers-list::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.20);
            border-radius: 999px;
        }

        </style>

        {% endmacro %}
        """
    )

    root.add_child(patch)


# =============================================================================
# Group layers by animal
# =============================================================================

def _add_grouped_layer_control_patch(
    m,
    animal_ids,
):
    """
    Convert the flat Leaflet overlay list into collapsible groups
    for each animal.

    Layer names are expected to begin with the animal ID, e.g.:

        147 MCP 95%
        147 KDE Raster
        147 KDE 50% Contour
        147 LoCoH 95%

    Points and Tracks remain outside the animal groups.

    Also adds explicit:
        All layers on
        All layers off
    buttons.
    """

    root = m.get_root()

    if getattr(
        root,
        "_spatchat_grouped_layer_patch_added",
        False,
    ):
        return

    setattr(
        root,
        "_spatchat_grouped_layer_patch_added",
        True,
    )

    patch = MacroElement()

    patch.animal_ids = [
        str(x)
        for x in animal_ids
    ]

    patch._template = Template(
        """
        {% macro html(this, kwargs) %}

        <style>

        /* ========================================================
           Global layer buttons
        ======================================================== */

        .spatchat-layer-buttons {
            display: flex;
            gap: 5px;
            padding: 8px;
            flex-wrap: wrap;
            border-bottom:
                1px solid rgba(255,255,255,0.10);
        }

        .spatchat-layer-button {
            border: 0;
            border-radius: 5px;
            padding: 5px 8px;
            cursor: pointer;
            font-size: 11px;
            line-height: 1.2;
            color: #ffffff;
            background: rgba(255,255,255,0.14);
        }

        .spatchat-layer-button:hover {
            background: rgba(255,255,255,0.25);
        }


        /* ========================================================
           Animal sections
        ======================================================== */

        .spatchat-animal-group {
            margin: 0;
            border-top:
                1px solid rgba(255,255,255,0.08);
        }

        .spatchat-animal-header {
            cursor: pointer;
            user-select: none;
            padding: 7px 9px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-weight: 600;
            color: #f5f5f5;
        }

        .spatchat-animal-header:hover {
            background: rgba(255,255,255,0.07);
        }

        .spatchat-animal-arrow {
            display: inline-block;
            width: 12px;
            font-size: 10px;
        }

        .spatchat-animal-group:not(.open)
        .spatchat-animal-arrow::before {
            content: "▶";
        }

        .spatchat-animal-group.open
        .spatchat-animal-arrow::before {
            content: "▼";
        }

        .spatchat-animal-body {
            display: none;
            padding-left: 8px;
            padding-bottom: 4px;
        }

        .spatchat-animal-group.open
        .spatchat-animal-body {
            display: block;
        }


        /* ========================================================
           Optional buttons within each individual
        ======================================================== */

        .spatchat-animal-actions {
            display: flex;
            gap: 4px;
            padding: 3px 8px 5px 8px;
        }

        .spatchat-animal-action {
            border: 0;
            border-radius: 4px;
            padding: 3px 6px;
            cursor: pointer;
            font-size: 10px;
            color: #eeeeee;
            background: rgba(255,255,255,0.10);
        }

        .spatchat-animal-action:hover {
            background: rgba(255,255,255,0.20);
        }

        </style>


        <script>

        (function() {

            const animalIds =
                {{ this.animal_ids | tojson }};


            function stripAnimalPrefix(
                element,
                prefix
            ) {

                /*
                 * Leaflet/Folium can place the text inside nested spans.
                 * Walk all text nodes and remove the first matching prefix.
                 */

                const walker =
                    document.createTreeWalker(
                        element,
                        NodeFilter.SHOW_TEXT
                    );

                let node;

                while (
                    node = walker.nextNode()
                ) {

                    const value =
                        node.nodeValue || "";

                    const trimmed =
                        value.trimStart();

                    if (
                        trimmed.startsWith(
                            prefix
                        )
                    ) {

                        node.nodeValue =
                            value.replace(
                                prefix,
                                ""
                            );

                        return;
                    }
                }
            }


            function setCheckboxes(
                container,
                checked
            ) {

                const inputs =
                    container.querySelectorAll(
                        'input[type="checkbox"]'
                    );

                inputs.forEach(
                    function(input) {

                        if (
                            input.checked
                            !== checked
                        ) {

                            /*
                             * click(), rather than merely changing checked,
                             * lets Leaflet actually add/remove the layer.
                             */

                            input.click();
                        }
                    }
                );
            }


            function setupLayerGroups() {

                const control =
                    document.querySelector(
                        '.leaflet-control-layers'
                    );

                if (!control) {

                    setTimeout(
                        setupLayerGroups,
                        100
                    );

                    return;
                }


                const overlays =
                    control.querySelector(
                        '.leaflet-control-layers-overlays'
                    );

                if (!overlays) {

                    setTimeout(
                        setupLayerGroups,
                        100
                    );

                    return;
                }


                if (
                    overlays.dataset.spatchatGrouped
                    === "1"
                ) {
                    return;
                }


                overlays.dataset.spatchatGrouped =
                    "1";


                /* ====================================================
                   Save the original direct labels first
                ==================================================== */

                const labels =
                    Array.from(
                        overlays.children
                    ).filter(
                        element =>
                            element.tagName
                            === "LABEL"
                    );


                /* ====================================================
                   Global All On / All Off controls
                ==================================================== */

                const buttons =
                    document.createElement(
                        "div"
                    );

                buttons.className =
                    "spatchat-layer-buttons";


                const allOn =
                    document.createElement(
                        "button"
                    );

                allOn.type =
                    "button";

                allOn.className =
                    "spatchat-layer-button";

                allOn.textContent =
                    "All layers on";


                const allOff =
                    document.createElement(
                        "button"
                    );

                allOff.type =
                    "button";

                allOff.className =
                    "spatchat-layer-button";

                allOff.textContent =
                    "All layers off";


                allOn.addEventListener(
                    "click",
                    function(event) {

                        event.preventDefault();
                        event.stopPropagation();

                        setCheckboxes(
                            overlays,
                            true
                        );
                    }
                );


                allOff.addEventListener(
                    "click",
                    function(event) {

                        event.preventDefault();
                        event.stopPropagation();

                        setCheckboxes(
                            overlays,
                            false
                        );
                    }
                );


                buttons.appendChild(
                    allOn
                );

                buttons.appendChild(
                    allOff
                );


                overlays.insertBefore(
                    buttons,
                    overlays.firstChild
                );


                /* ====================================================
                   Create animal groups
                ==================================================== */

                animalIds.forEach(
                    function(animalId) {

                        const prefix =
                            animalId + " ";


                        const matching =
                            labels.filter(
                                function(label) {

                                    return (
                                        label.textContent
                                        .trim()
                                        .startsWith(
                                            prefix
                                        )
                                    );
                                }
                            );


                        if (
                            matching.length
                            === 0
                        ) {
                            return;
                        }


                        const group =
                            document.createElement(
                                "div"
                            );

                        group.className =
                            "spatchat-animal-group";


                        const header =
                            document.createElement(
                                "div"
                            );

                        header.className =
                            "spatchat-animal-header";


                        const arrow =
                            document.createElement(
                                "span"
                            );

                        arrow.className =
                            "spatchat-animal-arrow";


                        const title =
                            document.createElement(
                                "span"
                            );

                        title.textContent =
                            "Animal "
                            + animalId;


                        header.appendChild(
                            arrow
                        );

                        header.appendChild(
                            title
                        );


                        const body =
                            document.createElement(
                                "div"
                            );

                        body.className =
                            "spatchat-animal-body";


                        /* --------------------------------------------
                           Per-animal On / Off buttons
                        -------------------------------------------- */

                        const actions =
                            document.createElement(
                                "div"
                            );

                        actions.className =
                            "spatchat-animal-actions";


                        const animalOn =
                            document.createElement(
                                "button"
                            );

                        animalOn.type =
                            "button";

                        animalOn.className =
                            "spatchat-animal-action";

                        animalOn.textContent =
                            "All on";


                        const animalOff =
                            document.createElement(
                                "button"
                            );

                        animalOff.type =
                            "button";

                        animalOff.className =
                            "spatchat-animal-action";

                        animalOff.textContent =
                            "All off";


                        animalOn.addEventListener(
                            "click",
                            function(event) {

                                event.preventDefault();
                                event.stopPropagation();

                                setCheckboxes(
                                    body,
                                    true
                                );
                            }
                        );


                        animalOff.addEventListener(
                            "click",
                            function(event) {

                                event.preventDefault();
                                event.stopPropagation();

                                setCheckboxes(
                                    body,
                                    false
                                );
                            }
                        );


                        actions.appendChild(
                            animalOn
                        );

                        actions.appendChild(
                            animalOff
                        );


                        body.appendChild(
                            actions
                        );


                        /* --------------------------------------------
                           Move layers into this animal section
                        -------------------------------------------- */

                        matching.forEach(
                            function(label) {

                                stripAnimalPrefix(
                                    label,
                                    prefix
                                );

                                body.appendChild(
                                    label
                                );
                            }
                        );


                        header.addEventListener(
                            "click",
                            function(event) {

                                event.preventDefault();

                                group.classList.toggle(
                                    "open"
                                );
                            }
                        );


                        group.appendChild(
                            header
                        );

                        group.appendChild(
                            body
                        );

                        overlays.appendChild(
                            group
                        );
                    }
                );


                /* ====================================================
                   Open first animal group by default
                ==================================================== */

                const first =
                    overlays.querySelector(
                        ".spatchat-animal-group"
                    );

                if (first) {
                    first.classList.add(
                        "open"
                    );
                }

            }


            if (
                document.readyState
                === "loading"
            ) {

                document.addEventListener(
                    "DOMContentLoaded",
                    setupLayerGroups
                );

            } else {

                /*
                 * Give Leaflet a moment to construct its LayerControl.
                 */

                setTimeout(
                    setupLayerGroups,
                    50
                );
            }

        })();

        </script>

        {% endmacro %}
        """
    )

    root.add_child(patch)


# =============================================================================
# Preview map
# =============================================================================

def build_preview_map(df):

    has_timestamp = (
        "timestamp"
        in df.columns
    )

    m = _base_map(
        df["latitude"].mean(),
        df["longitude"].mean(),
        control_scale=True,
        zoom=9,
    )

    points_layer = folium.FeatureGroup(
        name="Points",
        show=True,
    )

    lines_layer = folium.FeatureGroup(
        name="Tracks",
        show=True,
    )

    animal_ids = (
        df["animal_id"].unique()
        if "animal_id" in df.columns
        else ["sample"]
    )

    color_map = {
        aid:
            f"#{random.randint(0, 0xFFFFFF):06x}"
        for aid in animal_ids
    }

    for animal in animal_ids:

        track = (
            df[
                df["animal_id"]
                == animal
            ]
            if "animal_id" in df.columns
            else df
        )

        if has_timestamp:

            track = track.sort_values(
                "timestamp"
            )

            coords = list(
                zip(
                    track["latitude"],
                    track["longitude"],
                )
            )

            if len(coords) > 1:

                folium.PolyLine(
                    coords,
                    color=color_map[animal],
                    weight=2.5,
                    opacity=0.8,
                    popup=str(animal),
                ).add_to(lines_layer)

        for _, row in track.iterrows():

            label = (
                f"{animal}"
                +
                (
                    f"<br>{row['timestamp']}"
                    if (
                        has_timestamp
                        and row.get("timestamp")
                        is not None
                    )
                    else ""
                )
            )

            folium.CircleMarker(
                location=[
                    row["latitude"],
                    row["longitude"],
                ],
                radius=3,
                popup=label,
                color=color_map[animal],
                fill=True,
                fill_opacity=0.7,
            ).add_to(points_layer)

    points_layer.add_to(m)

    if has_timestamp:
        lines_layer.add_to(m)

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    _add_layer_control_style_patch(m)

    apply_map_control_patches(m)

    m = fit_map_to_bounds(
        m,
        df,
    )

    return m._repr_html_()


# =============================================================================
# Points and tracks
# =============================================================================

def make_points_tracks_layers(
    df,
    color_map,
):

    points_layer = folium.FeatureGroup(
        name="Points",
        show=True,
    )

    tracks_layer = folium.FeatureGroup(
        name="Tracks",
        show=True,
    )

    has_timestamp = (
        "timestamp"
        in df.columns
    )

    animal_ids = (
        df["animal_id"].unique()
        if "animal_id" in df.columns
        else ["sample"]
    )

    for animal in animal_ids:

        track = (
            df[
                df["animal_id"]
                == animal
            ]
            if "animal_id" in df.columns
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
                    track["latitude"],
                    track["longitude"],
                )
            )

            if len(coords) > 1:

                folium.PolyLine(
                    coords,
                    color=color,
                    weight=2.5,
                    opacity=0.8,
                    popup=f"{animal} Track",
                ).add_to(tracks_layer)

        for _, row in track.iterrows():

            folium.CircleMarker(
                location=[
                    row["latitude"],
                    row["longitude"],
                ],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{animal}",
            ).add_to(points_layer)

    return (
        points_layer,
        tracks_layer,
    )


# =============================================================================
# MCP
# =============================================================================

def make_mcp_layers(
    mcp_results,
    requested_percents,
    animal_ids,
    color_map,
):

    layers = []

    for percent in requested_percents:

        for animal in animal_ids:

            if (
                animal in mcp_results
                and percent
                in mcp_results[animal]
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
                    show=True,
                )

                polygon = v[
                    "polygon"
                ]

                _add_polygon_geom(
                    layer,
                    polygon,
                    color=color_map[animal],
                    fill=True,
                    fill_opacity=(
                        0.15
                        +
                        0.15
                        * (
                            percent / 100
                        )
                    ),
                    popup=(
                        f"{animal} "
                        f"MCP {percent}%"
                    ),
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
    color_map,
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
                in kde_results[animal]
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
                        show=True,
                    )
                )

                try:

                    (
                        arr,
                        bounds_ll,
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
                            interactive=False,
                        )
                    )

                    layers.append(
                        raster_layer
                    )

                except Exception as exc:

                    print(
                        "[map] KDE raster display "
                        f"error for animal={animal}: "
                        f"{exc}"
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

            contour_layer = folium.FeatureGroup(
                name=(
                    f"{animal} KDE "
                    f"{percent}% Contour"
                ),
                show=True,
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
                    color=color_map[animal],
                    fill=True,
                    fill_opacity=0.2,
                    popup=(
                        f"{animal} KDE "
                        f"{percent}% Contour"
                    ),
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
    color_map,
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
                in akde_results[animal]
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

                raster_layer = folium.FeatureGroup(
                    name=(
                        f"{animal} "
                        "AKDE Raster"
                    ),
                    show=True,
                )

                try:

                    (
                        arr,
                        bounds_ll,
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
                            interactive=False,
                        )
                    )

                    layers.append(
                        raster_layer
                    )

                except Exception as exc:

                    print(
                        "[map] AKDE raster display "
                        f"error for animal={animal}: "
                        f"{exc}"
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

            contour_layer = folium.FeatureGroup(
                name=(
                    f"{animal} AKDE "
                    f"{percent}% Contour"
                ),
                show=True,
            )

            contour = (
                v.get(
                    "contour"
                )
            )

            area = float(
                v.get(
                    "area",
                    0.0,
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
                        tuple,
                    )
                )
                and len(ci)
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
                    color=color_map[animal],
                    fill=True,
                    fill_opacity=0.2,
                    popup=popup,
                )

            layers.append(
                contour_layer
            )

            # -----------------------------------------------------------------
            # Confidence interval contours
            # -----------------------------------------------------------------

            ci_contours = (
                v.get(
                    "ci_contours"
                )
                or {}
            )

            if ci_contours:

                ci_layer = folium.FeatureGroup(
                    name=(
                        f"{animal} AKDE "
                        f"{percent}% "
                        "95% CI Contours"
                    ),
                    show=True,
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
                    ci_contour,
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
                        f"<br>Area: {area_text}"
                    )

                    _add_polygon_geom(
                        ci_layer,
                        ci_contour,
                        color=color_map[animal],
                        fill=False,
                        fill_opacity=0.0,
                        weight=(
                            2
                            if label == "low"
                            else 3
                        ),
                        opacity=(
                            0.9
                            if label == "low"
                            else 0.75
                        ),
                        dash_array=(
                            "5,5"
                            if label == "low"
                            else "10,6"
                        ),
                        popup=ci_popup,
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
    name_prefix: str = "LoCoH",
):

    layers = []

    animals_dict = (
        locoh_result.get(
            "animals",
            {},
        )
        if locoh_result
        else {}
    )

    for animal in animal_ids:

        data = (
            animals_dict.get(
                str(animal)
            )
            or animals_dict.get(
                animal
            )
        )

        if not data:
            continue

        for item in data.get(
            "isopleths",
            [],
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
                    0.0,
                )
            )

            feature = {
                "type":
                    "Feature",

                "properties": {
                    "animal_id":
                        str(animal),

                    "isopleth":
                        iso,

                    "area_km2":
                        round(
                            area_km2,
                            3,
                        ),
                },

                "geometry":
                    gj_geom,
            }

            layer = folium.FeatureGroup(
                name=(
                    f"{animal} "
                    f"{name_prefix} "
                    f"{iso}%"
                ),
                show=True,
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
                                if iso != 95
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
                        "area_km2",
                    ],
                    aliases=[
                        "Animal",
                        "Isopleth (%)",
                        "Area (km²)",
                    ],
                    localize=True,
                ),
            ).add_to(layer)

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
    name_prefix: str = "LoCoH facets",
):

    bins = [
        20,
        40,
        60,
        80,
        95,
        100,
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

    def color_for(pct):

        for b in bins:

            if pct <= b:
                return palette[b]

        return palette[100]

    layers = []

    animals_dict = (
        locoh_result.get(
            "animals",
            {},
        )
        if locoh_result
        else {}
    )

    for animal in animal_ids:

        data = (
            animals_dict.get(
                str(animal)
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
                [],
            )
        )

        if not facets:
            continue

        features = []

        for f in facets:

            pct = int(
                f.get(
                    "cum_percent",
                    100,
                )
            )

            area = float(
                f.get(
                    "area_sq_km",
                    0.0,
                )
            )

            features.append(
                {
                    "type":
                        "Feature",

                    "properties": {
                        "animal_id":
                            str(animal),

                        "cum_percent":
                            pct,

                        "area_km2":
                            round(
                                area,
                                3,
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

        layer = folium.FeatureGroup(
            name=(
                f"{animal} "
                f"{name_prefix}"
            ),
            show=True,
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
                    "area_km2",
                ],
                aliases=[
                    "Animal",
                    "Cum. %",
                    "Hull area (km²)",
                ],
                localize=True,
            ),
        ).add_to(layer)

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
    name_prefix: str = "dBBMM",
):

    layers = []

    for animal in animal_ids:

        data = (
            dbbmm_results.get(
                str(animal)
            )
            or dbbmm_results.get(
                animal
            )
        )

        if not data:
            continue

        if isinstance(
            data,
            dict,
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
                    None,
                )
            )

            iso_list = (
                getattr(
                    data,
                    "isopleths",
                    [],
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

            raster_layer = folium.FeatureGroup(
                name=(
                    f"{animal} "
                    f"{name_prefix} Raster"
                ),
                show=True,
            )

            try:

                (
                    arr,
                    bounds_ll,
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
                        interactive=False,
                    )
                )

                layers.append(
                    raster_layer
                )

            except Exception as exc:

                print(
                    "[map] dBBMM raster display "
                    f"error for animal={animal}: "
                    f"{exc}"
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
                    0.0,
                )
            )

            layer = folium.FeatureGroup(
                name=(
                    f"{animal} "
                    f"{name_prefix} "
                    f"{percent}%"
                ),
                show=True,
            )

            if geom:

                folium.GeoJson(
                    data={
                        "type":
                            "Feature",

                        "properties": {
                            "animal_id":
                                str(animal),

                            "percent":
                                percent,

                            "area_km2":
                                round(
                                    area_km2,
                                    3,
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
                            "area_km2",
                        ],
                        aliases=[
                            "Animal",
                            "Isopleth (%)",
                            "Area (km²)",
                        ],
                        localize=True,
                    ),
                ).add_to(layer)

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
        df["latitude"].mean(),
        df["longitude"].mean(),
        control_scale=True,
        zoom=9,
    )

    animal_ids = (
        df["animal_id"].unique()
        if "animal_id" in df.columns
        else ["sample"]
    )

    color_map = {
        aid:
            f"#{random.randint(0, 0xFFFFFF):06x}"
        for aid in animal_ids
    }

    # -------------------------------------------------------------------------
    # Points and tracks
    # -------------------------------------------------------------------------

    (
        points_layer,
        tracks_layer,
    ) = make_points_tracks_layers(
        df,
        color_map,
    )

    points_layer.add_to(m)
    tracks_layer.add_to(m)

    # -------------------------------------------------------------------------
    # MCP
    # -------------------------------------------------------------------------

    for layer in make_mcp_layers(
        mcp_results or {},
        requested_percents or [],
        animal_ids,
        color_map,
    ):

        layer.add_to(m)

    # -------------------------------------------------------------------------
    # KDE
    # -------------------------------------------------------------------------

    for layer in make_kde_layers(
        kde_results or {},
        requested_kde_percents or [],
        animal_ids,
        color_map,
    ):

        layer.add_to(m)

    # -------------------------------------------------------------------------
    # AKDE
    # -------------------------------------------------------------------------

    for layer in make_akde_layers(
        akde_results or {},
        requested_akde_percents or [],
        animal_ids,
        color_map,
    ):

        layer.add_to(m)

    # -------------------------------------------------------------------------
    # LoCoH
    # -------------------------------------------------------------------------

    if locoh_result is not None:

        for layer in make_locoh_layers(
            locoh_result,
            animal_ids,
            color_map,
            name_prefix="LoCoH",
        ):

            layer.add_to(m)

        for layer in make_locoh_facets_layers(
            locoh_result,
            animal_ids,
            color_map,
            name_prefix="LoCoH facets",
        ):

            layer.add_to(m)

    # -------------------------------------------------------------------------
    # dBBMM
    # -------------------------------------------------------------------------

    if dbbmm_result is not None:

        for layer in make_dbbmm_layers(
            dbbmm_result,
            animal_ids,
            color_map,
            name_prefix="dBBMM",
        ):

            layer.add_to(m)

    # -------------------------------------------------------------------------
    # Layer control
    # -------------------------------------------------------------------------

    folium.LayerControl(
        collapsed=False,
    ).add_to(m)

    # Styling, including white Blank-map canvas
    _add_layer_control_style_patch(
        m
    )

    # Group analytical layers by animal and add explicit
    # All Layers On / All Layers Off controls
    _add_grouped_layer_control_patch(
        m,
        animal_ids,
    )

    # Keep any other custom Spatchat map-control behavior
    apply_map_control_patches(
        m
    )

    # -------------------------------------------------------------------------
    # Initial map extent
    # -------------------------------------------------------------------------

    m = fit_map_to_bounds(
        m,
        df,
    )

    return m._repr_html_()