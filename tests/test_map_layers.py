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


if __name__ == "__main__":
    unittest.main()
