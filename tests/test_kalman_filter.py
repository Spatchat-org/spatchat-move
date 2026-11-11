import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from methods.ctmm_py import models as M  # noqa: E402


def _reference_filter(x, errv, components):
    """The original element-by-element 2-state Kalman filter, kept here as the reference for bit-for-bit checks."""
    g00, g01, g10, g11, s00, s01, s11, omega2 = components
    n = x.size
    state0 = 0.0
    state1 = 0.0
    cov00 = 1.0
    cov01 = 0.0
    cov11 = omega2
    err = np.empty(n, dtype=float)
    var = np.empty(n, dtype=float)
    tiny = np.finfo(float).tiny
    for i in range(n):
        qi = float(cov00 + errv[i])
        if qi < tiny:
            qi = tiny
        ei = float(x[i] - state0)
        gain0 = cov00 / qi
        gain1 = cov01 / qi
        state0 = state0 + gain0 * ei
        state1 = state1 + gain1 * ei
        cov00_old = cov00
        cov01_old = cov01
        cov11_old = cov11
        cov00 = cov00_old - gain0 * cov00_old
        cov01 = cov01_old - gain0 * cov01_old
        cov11 = cov11_old - gain1 * cov01_old
        err[i] = ei
        var[i] = qi
        if i < n - 1:
            state0, state1 = (
                g00[i] * state0 + g01[i] * state1,
                g10[i] * state0 + g11[i] * state1,
            )
            cov00, cov01, cov11 = (
                g00[i] * g00[i] * cov00 + 2.0 * g00[i] * g01[i] * cov01 + g01[i] * g01[i] * cov11 + s00[i],
                g00[i] * g10[i] * cov00 + (g00[i] * g11[i] + g01[i] * g10[i]) * cov01 + g01[i] * g11[i] * cov11 + s01[i],
                g10[i] * g10[i] * cov00 + 2.0 * g10[i] * g11[i] * cov01 + g11[i] * g11[i] * cov11 + s11[i],
            )
    return err, var


def _reference_ouf(x, t, tau, error_var=None):
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    errv = M._sanitize_error_var(error_var, x.size)
    return _reference_filter(x, errv, M._langevin2_components(np.diff(t), tau))


def _reference_ouomega(x, t, tau, omega, error_var=None):
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    errv = M._sanitize_error_var(error_var, x.size)
    return _reference_filter(x, errv, M._ouomega_components(np.diff(t), tau, omega))


def _times(rng, n):
    if n == 0:
        return np.array([])
    return np.cumsum(rng.choice([900.0, 900.0, 900.0, 1800.0, 5400.0, 86400.0, 3.0], size=n))


def _same(a, b):
    return np.array_equal(a[0], b[0], equal_nan=True) and np.array_equal(a[1], b[1], equal_nan=True)


class KalmanFilterTests(unittest.TestCase):
    def test_ou_f_matches_the_original_filter_exactly(self):
        rng = np.random.default_rng(1)
        for n in (0, 1, 2, 3, 7, 60, 500):
            t = _times(rng, n)
            for tau in ([3e5, 2e3], [5e4, 5e4], [2e6, 50.0], [1e3, 1e3]):
                for errv in (None, np.abs(rng.normal(size=n)) * 5.0):
                    x = rng.normal(size=n) * 100
                    self.assertTrue(_same(M._innovations_ouf_scalar(x, t, tau, error_var=errv),
                                          _reference_ouf(x, t, tau, error_var=errv)), (n, tau))

    def test_ou_omega_matches_the_original_filter_exactly(self):
        rng = np.random.default_rng(2)
        for n in (0, 1, 2, 3, 7, 60, 500):
            t = _times(rng, n)
            for tau, omega in ((3e5, 1e-4), (5e4, 3e-3), (2e6, 0.1), (1e3, 0.0)):
                for errv in (None, np.abs(rng.normal(size=n)) * 5.0):
                    x = rng.normal(size=n) * 100
                    self.assertTrue(_same(M._innovations_ouomega_scalar(x, t, tau, omega, error_var=errv),
                                          _reference_ouomega(x, t, tau, omega, error_var=errv)), (n, tau, omega))

    def test_reusing_cached_covariance_for_new_data_stays_exact(self):
        rng = np.random.default_rng(3)
        t = _times(rng, 300)
        tau = [4e5, 3e3]
        for _ in range(5):  # same times and parameters, different data columns each time
            x = rng.normal(size=300)
            self.assertTrue(_same(M._innovations_ouf_scalar(x, t, tau), _reference_ouf(x, t, tau)))

    def test_missing_error_values_are_treated_as_zero(self):
        rng = np.random.default_rng(4)
        t = _times(rng, 80)
        x = rng.normal(size=80)
        errv = np.where(rng.random(80) < 0.3, np.nan, 2.0)
        self.assertTrue(_same(M._innovations_ouf_scalar(x, t, [1e5, 2e3], error_var=errv),
                              _reference_ouf(x, t, [1e5, 2e3], error_var=errv)))

    def test_callers_cannot_corrupt_the_cache(self):
        t = np.arange(40) * 900.0
        err, var = M._innovations_ouf_scalar(np.ones(40), t, [1e5, 2e3])
        var[:] = -1.0
        err[:] = -1.0
        err2, var2 = M._innovations_ouf_scalar(np.ones(40), t, [1e5, 2e3])
        self.assertTrue((var2 > 0).all())
        self.assertTrue((err2 != -1.0).all())

    def test_long_tracks_filter_quickly(self):
        # 12,000 fixes: the original loop took ~75 ms per call; with the covariance cached each extra data column is cheap.
        rng = np.random.default_rng(5)
        n = 12000
        t = np.cumsum(np.full(n, 900.0))
        cols = [rng.normal(size=n) for _ in range(5)]
        started = time.perf_counter()
        for _ in range(20):
            for x in cols:
                M._innovations_ouf_scalar(x, t, [3e5, 2e3])
        self.assertLess(time.perf_counter() - started, 20.0)


if __name__ == "__main__":
    unittest.main()
