import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import map_layers  # noqa: E402


def _track(n, animal="A"):
    rng = np.random.default_rng(0)
    lat = 26.27 + np.cumsum(rng.normal(0, 1e-4, n))
    lon = -80.11 + np.cumsum(rng.normal(0, 1e-4, n))
    ts = pd.date_range("2026-02-17", periods=n, freq="15min")
    return pd.DataFrame({"latitude": lat, "longitude": lon, "timestamp": ts, "animal_id": animal})


class ThinPointsTests(unittest.TestCase):
    def test_small_tracks_are_left_alone(self):
        df = _track(900)
        self.assertEqual(len(map_layers._thin_points(df, 1500)), 900)

    def test_large_tracks_are_capped_and_keep_first_and_last_fix(self):
        df = _track(12463)
        thinned = map_layers._thin_points(df, 1500)
        self.assertLessEqual(len(thinned), 1500)
        self.assertEqual(thinned.index[0], 0)
        self.assertEqual(thinned.index[-1], 12462)

    def test_budget_is_shared_between_animals(self):
        self.assertEqual(map_layers._points_budget(1), map_layers.MAX_MAP_POINTS)
        self.assertEqual(map_layers._points_budget(2), map_layers.MAX_MAP_POINTS // 2)
        self.assertGreaterEqual(map_layers._points_budget(100), 300)


class PreviewMapSizeTests(unittest.TestCase):
    def test_gps_collar_sized_track_gives_a_small_page_and_keeps_the_full_track(self):
        # ~12,000 fixes used to produce a ~17 MB page that took ~16 s to build.
        html = map_layers.build_preview_map(_track(12463))
        self.assertLess(len(html), 6_000_000)
        self.assertIn("showing 1,500 of 12,463", html.replace("&nbsp;", " "))

    def test_small_track_is_not_relabelled(self):
        html = map_layers.build_preview_map(_track(200))
        self.assertNotIn("showing", html)


if __name__ == "__main__":
    unittest.main()
