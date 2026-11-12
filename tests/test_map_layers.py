import json
import os
import re
import sys
import time
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import map_layers  # noqa: E402


def _track(n, animal="A", with_time=True, seed=0):
    rng = np.random.default_rng(seed)
    data = {
        "latitude": 26.27 + np.cumsum(rng.normal(0, 1e-4, n)),
        "longitude": -80.11 + np.cumsum(rng.normal(0, 1e-4, n)),
        "animal_id": animal,
    }
    if with_time:
        data["timestamp"] = pd.date_range("2026-02-17", periods=n, freq="15min")
    return pd.DataFrame(data)


def _page(html):
    """The map's own HTML (folium wraps it in an iframe's srcdoc)."""
    import html as _html

    m = re.search(r'srcdoc="([^"]*)"', html)
    return _html.unescape(m.group(1)) if m else html


def _points_in(page):
    """The [lat, lon] / [lat, lon, "time"] arrays handed to the canvas layers."""
    found = re.findall(r"var pts = (\[.*?\]);\s*var label", page, flags=re.S)
    return [json.loads(chunk) for chunk in found]


class AllPointsTests(unittest.TestCase):
    def test_every_fix_is_drawn_with_no_limit(self):
        df = _track(12463)
        arrays = _points_in(_page(map_layers.build_preview_map(df)))
        self.assertEqual(len(arrays), 1)
        self.assertEqual(len(arrays[0]), 12463)

    def test_a_35000_point_track_is_fully_drawn_and_quick_to_build(self):
        df = _track(35000)
        started = time.perf_counter()
        html = map_layers.build_preview_map(df)
        elapsed = time.perf_counter() - started
        self.assertEqual(len(_points_in(_page(html))[0]), 35000)
        self.assertLess(elapsed, 20.0)          # the one-marker-per-fix version took ~45 s
        self.assertLess(len(html), 10_000_000)  # ... and produced a ~47 MB page

    def test_the_gps_collar_page_stays_small(self):
        html = map_layers.build_preview_map(_track(12463))
        self.assertLess(len(html), 4_000_000)    # was 16.7 MB

    def test_points_keep_their_coordinates_and_timestamps(self):
        df = _track(5)
        points = _points_in(_page(map_layers.build_preview_map(df)))[0]
        for got, (_, row) in zip(points, df.iterrows()):
            self.assertAlmostEqual(got[0], row["latitude"], places=6)
            self.assertAlmostEqual(got[1], row["longitude"], places=6)
            self.assertEqual(got[2], str(row["timestamp"]))

    def test_tracks_without_timestamps_still_draw_every_point(self):
        df = _track(300, with_time=False)
        points = _points_in(_page(map_layers.build_preview_map(df)))[0]
        self.assertEqual(len(points), 300)
        self.assertTrue(all(len(p) == 2 for p in points))

    def test_each_animal_gets_its_own_layer_with_all_of_its_fixes(self):
        df = pd.concat([_track(400, "Ada", seed=1), _track(250, "Bo", seed=2)], ignore_index=True)
        arrays = _points_in(_page(map_layers.build_preview_map(df)))
        self.assertEqual(sorted(len(a) for a in arrays), [250, 400])

    def test_no_point_limit_is_advertised(self):
        html = map_layers.build_preview_map(_track(12463))
        self.assertNotIn("showing", html)

    def test_results_map_points_layer_also_draws_every_fix(self):
        import folium

        df = _track(7000)
        points, tracks = map_layers.make_points_tracks_layers(df, {"A": "#ff7f0e"})
        m = folium.Map()
        points.add_to(m)
        arrays = _points_in(m.get_root().render())
        self.assertEqual(sum(len(a) for a in arrays), 7000)

    def test_point_labels_are_escaped(self):
        df = _track(3, animal="</script><b>x")
        page = _page(map_layers.build_preview_map(df))
        label = re.search(r"var label = (.*?);", page).group(1)
        self.assertNotIn("</script>", label)
        self.assertNotIn("<b>", label)


class AnimalIdEscapingTests(unittest.TestCase):
    """Animal IDs come from the user's CSV and end up in popups, layer names and tooltips, which Leaflet inserts as HTML."""

    EVIL = "</script><img src=x onerror=alert(1)>"
    ESCAPED = "&lt;/script&gt;&lt;img src=x onerror=alert(1)&gt;"

    def test_preview_map_escapes_the_id_everywhere(self):
        page = _page(map_layers.build_preview_map(_track(20, animal=self.EVIL)))
        self.assertNotIn(self.EVIL, page)
        self.assertIn(self.ESCAPED, page)

    def test_results_map_escapes_the_id_in_popups_layer_names_and_tooltips(self):
        import tempfile

        import storage
        from methods.kde import KDEParams, add_kdes
        from methods.mcp import add_mcps
        from schema_detect import detect_and_standardize

        df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data", "demo_movement.csv"))
        df, _ = detect_and_standardize(df)
        df["animal_id"] = "Fox1"  # the analyses write files named after the ID, so run them with a safe one
        original = storage.get_output_dir
        storage.get_output_dir = lambda: tempfile.mkdtemp()
        try:
            add_mcps(df, [95])
            add_kdes(df, [95], params=KDEParams())
            for results in (storage.mcp_results, storage.kde_results):
                results[self.EVIL] = results.pop("Fox1")  # ... then label them with the hostile ID for the map
            df["animal_id"] = self.EVIL
            html = map_layers.build_results_map(
                df,
                mcp_results=storage.mcp_results,
                kde_results=storage.kde_results,
                requested_percents={95},
                requested_kde_percents={95},
            )
        finally:
            storage.get_output_dir = original
            for results in (storage.mcp_results, storage.kde_results):
                results.clear()
        page = _page(html)
        self.assertNotIn(self.EVIL, page)
        self.assertIn(self.ESCAPED + " MCP 95%", page)          # layer name and popup
        self.assertIn(self.ESCAPED + " KDE 95% Contour", page)

    def test_ordinary_ids_are_left_exactly_as_they_were(self):
        self.assertEqual(map_layers._esc("Fox-1"), "Fox-1")
        self.assertEqual(map_layers._esc("156323"), "156323")
        self.assertEqual(map_layers._esc(156323), "156323")


if __name__ == "__main__":
    unittest.main()
