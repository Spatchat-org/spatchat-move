# map_utils.py
import folium
import numpy as np
from branca.element import MacroElement, Template


def _add_layer_control_style_patch(m):
    root = m.get_root()
    if getattr(root, "_spatchat_layer_control_style_patch_added", False):
        return
    setattr(root, "_spatchat_layer_control_style_patch_added", True)
    patch = MacroElement()
    patch._template = Template("""
    {% macro html(this, kwargs) %}
    <style>
      .leaflet-control-layers,
      .leaflet-control-layers.spatchat-draggable-control {
        background: rgba(30, 30, 30, 0.9) !important;
        color: #f5f5f5 !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35) !important;
      }
      .leaflet-control-layers-toggle {
        background-color: transparent !important;
        border-radius: 8px !important;
        filter: invert(1) brightness(1.25);
      }
      .leaflet-control-layers-expanded {
        padding: 0 !important;
        background: transparent !important;
      }
      .leaflet-control-layers-list,
      .leaflet-control-layers form,
      .leaflet-control-layers .leaflet-control-layers-base,
      .leaflet-control-layers .leaflet-control-layers-overlays,
      .leaflet-control-layers .leaflet-control-layers-separator {
        background: transparent !important;
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
        scrollbar-color: rgba(255, 255, 255, 0.18) rgba(255, 255, 255, 0.06);
        scrollbar-width: thin;
      }
      .leaflet-control-layers::-webkit-scrollbar,
      .leaflet-control-layers .leaflet-control-layers-list::-webkit-scrollbar {
        width: 10px;
        height: 10px;
      }
      .leaflet-control-layers::-webkit-scrollbar-track,
      .leaflet-control-layers .leaflet-control-layers-list::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.06);
      }
      .leaflet-control-layers::-webkit-scrollbar-thumb,
      .leaflet-control-layers .leaflet-control-layers-list::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.18);
        border-radius: 999px;
      }
      .leaflet-control-layers::-webkit-scrollbar-thumb:hover,
      .leaflet-control-layers .leaflet-control-layers-list::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.26);
      }
      .leaflet-control-layers.spatchat-draggable-control {
        z-index: 9998 !important;
        width: 280px;
        min-width: 220px;
        max-width: calc(100% - 28px);
        min-height: 120px;
        max-height: calc(100% - 28px);
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }
      .leaflet-control-layers.spatchat-draggable-control .leaflet-control-layers-list {
        flex: 1 1 auto;
        min-height: 0;
        overflow: auto;
      }
      .spatchat-layer-control-handle {
        padding: 8px 10px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
        cursor: move;
        user-select: none;
        background: rgba(255, 255, 255, 0.08);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }
      .spatchat-layer-control-tools {
        display: flex;
        padding: 8px 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.04);
      }
      .spatchat-layer-control-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        appearance: none;
        border: 1px solid rgba(255, 255, 255, 0.18);
        background: rgba(24, 24, 24, 0.92) !important;
        color: #ffffff !important;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 600;
        line-height: 1.2;
        cursor: pointer;
      }
      .spatchat-layer-control-btn:hover {
        background: rgba(48, 48, 48, 0.98) !important;
      }
      .spatchat-dock-resize {
        position: absolute;
        right: 0;
        bottom: 0;
        width: 18px;
        height: 18px;
        cursor: nwse-resize;
        z-index: 10000;
        user-select: none;
      }
      .spatchat-dock-resize::before {
        content: "";
        position: absolute;
        right: 4px;
        bottom: 4px;
        width: 10px;
        height: 10px;
        border-right: 2px solid rgba(255, 255, 255, 0.65);
        border-bottom: 2px solid rgba(255, 255, 255, 0.65);
        border-bottom-right-radius: 2px;
      }
    </style>
    <script>
    (function() {
      function getLeafletMap() {
        if (!window.L) return null;
        for (var key in window) {
          try {
            if (window[key] instanceof L.Map) return window[key];
          } catch (err) {}
        }
        return null;
      }

      function ensureMapSizing(map) {
        if (!map || map.__spatchatSizePatched) return;
        map.__spatchatSizePatched = true;
        function refresh() {
          try { map.invalidateSize(false); } catch (err) {}
        }
        if (document.readyState === 'loading') {
          window.addEventListener('load', function() {
            requestAnimationFrame(refresh);
            setTimeout(refresh, 120);
            setTimeout(refresh, 400);
          }, { once: true });
        } else {
          requestAnimationFrame(refresh);
          setTimeout(refresh, 120);
          setTimeout(refresh, 400);
        }
      }

      function enableDrag(container, handle, mapContainer) {
        if (!container || !handle || !mapContainer || handle.dataset.spatchatDragBound === '1') return;
        handle.dataset.spatchatDragBound = '1';
        handle.addEventListener('mousedown', function(event) {
          if (event.button !== 0) return;
          event.preventDefault();
          event.stopPropagation();

          var mapRect = mapContainer.getBoundingClientRect();
          var rect = container.getBoundingClientRect();
          if (container.parentElement !== mapContainer) mapContainer.appendChild(container);
          container.style.position = 'absolute';
          container.style.left = (rect.left - mapRect.left) + 'px';
          container.style.top = (rect.top - mapRect.top) + 'px';
          container.style.right = 'auto';
          container.style.bottom = 'auto';

          var startX = event.clientX;
          var startY = event.clientY;
          var startLeft = rect.left - mapRect.left;
          var startTop = rect.top - mapRect.top;

          function onMove(moveEvent) {
            var nextLeft = startLeft + (moveEvent.clientX - startX);
            var nextTop = startTop + (moveEvent.clientY - startY);
            var maxLeft = Math.max(0, mapContainer.clientWidth - container.offsetWidth);
            var maxTop = Math.max(0, mapContainer.clientHeight - container.offsetHeight);
            container.style.left = Math.max(0, Math.min(nextLeft, maxLeft)) + 'px';
            container.style.top = Math.max(0, Math.min(nextTop, maxTop)) + 'px';
          }

          function onUp() {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
          }

          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });
      }

      function enableResize(container, handle, mapContainer, options) {
        if (!container || !handle || !mapContainer || handle.dataset.spatchatResizeBound === '1') return;
        handle.dataset.spatchatResizeBound = '1';
        var minWidth = (options && options.minWidth) || 220;
        var minHeight = (options && options.minHeight) || 120;
        handle.addEventListener('mousedown', function(event) {
          if (event.button !== 0) return;
          event.preventDefault();
          event.stopPropagation();

          var mapRect = mapContainer.getBoundingClientRect();
          var rect = container.getBoundingClientRect();
          if (container.parentElement !== mapContainer) mapContainer.appendChild(container);
          container.style.position = 'absolute';
          container.style.left = (rect.left - mapRect.left) + 'px';
          container.style.top = (rect.top - mapRect.top) + 'px';

          var startX = event.clientX;
          var startY = event.clientY;
          var startWidth = rect.width;
          var startHeight = rect.height;
          var startLeft = rect.left - mapRect.left;
          var startTop = rect.top - mapRect.top;

          function onMove(moveEvent) {
            var maxWidth = Math.max(minWidth, mapContainer.clientWidth - startLeft);
            var maxHeight = Math.max(minHeight, mapContainer.clientHeight - startTop);
            var nextWidth = Math.max(minWidth, Math.min(startWidth + (moveEvent.clientX - startX), maxWidth));
            var nextHeight = Math.max(minHeight, Math.min(startHeight + (moveEvent.clientY - startY), maxHeight));
            container.style.width = Math.round(nextWidth) + 'px';
            container.style.height = Math.round(nextHeight) + 'px';
            container.style.maxWidth = 'none';
            container.style.maxHeight = 'none';
          }

          function onUp() {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
          }

          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        });
      }

      function initLayerControlDrag(map) {
        var mapContainer = map && map.getContainer ? map.getContainer() : null;
        if (!mapContainer) return false;
        var layerControl = mapContainer.querySelector('.leaflet-control-layers') || document.querySelector('.leaflet-control-layers');
        if (!layerControl) return false;
        layerControl.classList.add('spatchat-draggable-control');
        if (!layerControl.style.width) layerControl.style.width = '280px';

        var handle = layerControl.querySelector('.spatchat-layer-control-handle');
        if (!handle) {
          handle = document.createElement('div');
          handle.className = 'spatchat-layer-control-handle';
          handle.textContent = 'Layers';
          layerControl.insertBefore(handle, layerControl.firstChild);
        }

        var overlaysList = layerControl.querySelector('.leaflet-control-layers-overlays');
        var overlayInputs = Array.from(layerControl.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]'));
        var tools = layerControl.querySelector('.spatchat-layer-control-tools');
        if (!overlayInputs.length) {
          if (tools) tools.remove();
        } else if (!tools) {
          tools = document.createElement('div');
          tools.className = 'spatchat-layer-control-tools';
          var toggleBtn = document.createElement('button');
          toggleBtn.type = 'button';
          toggleBtn.className = 'spatchat-layer-control-btn';
          toggleBtn.textContent = 'All layers on/off';
          toggleBtn.addEventListener('click', function(event) {
            event.preventDefault();
            event.stopPropagation();
            var inputs = Array.from(layerControl.querySelectorAll('.leaflet-control-layers-overlays input[type="checkbox"]'));
            var anyUnchecked = inputs.some(function(input) { return !input.checked; });
            inputs.forEach(function(input) {
              if (!!input.checked !== anyUnchecked) input.click();
            });
          });
          tools.appendChild(toggleBtn);
          if (overlaysList) overlaysList.insertAdjacentElement('beforebegin', tools);
        }

        var resize = layerControl.querySelector('.spatchat-dock-resize');
        if (!resize) {
          resize = document.createElement('div');
          resize.className = 'spatchat-dock-resize';
          resize.title = 'Resize layers dock';
          layerControl.appendChild(resize);
        }

        enableDrag(layerControl, handle, mapContainer);
        enableResize(layerControl, resize, mapContainer, { minWidth: 220, minHeight: 120 });
        return true;
      }

      function initFloatingUi() {
        var map = getLeafletMap();
        if (!map) return false;
        ensureMapSizing(map);
        return initLayerControlDrag(map);
      }

      initFloatingUi();
      window.addEventListener('load', initFloatingUi);
      var attempts = 0;
      var timer = setInterval(function() {
        if (initFloatingUi() || attempts >= 20) clearInterval(timer);
        attempts += 1;
      }, 500);
    })();
    </script>
    {% endmacro %}
    """)
    root.add_child(patch)


def apply_map_control_patches(m):
    _add_layer_control_style_patch(m)


def render_empty_map():
    m = folium.Map(location=[20, 0], zoom_start=2, control_scale=True, tiles=None)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", attr="CartoDB").add_to(m)
    folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", attr="OpenTopoMap", name="Topographic").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    apply_map_control_patches(m)
    return m._repr_html_()


def fit_map_to_bounds(m, df):
    min_lat, max_lat = df["latitude"].min(), df["latitude"].max()
    min_lon, max_lon = df["longitude"].min(), df["longitude"].max()
    if np.isfinite([min_lat, max_lat, min_lon, max_lon]).all():
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return m
