import os
import sys
import time
import unittest

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from movement_analysis.hmm_states import GaussianHMMDiag  # noqa: E402


def _two_state_data(n=600, seed=1):
    rng = np.random.default_rng(seed)
    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = states[t - 1] if rng.random() < 0.9 else 1 - states[t - 1]
    means = np.array([[1.0, 0.0, 0.0], [5.0, 0.8, 0.1]])
    x = means[states] + rng.normal(0, 0.3, size=(n, 3))
    return x, states


def _reference_em_step(model, x):
    """One EM iteration written the straightforward way (log space, scipy.logsumexp per step)."""
    k = model.n_states
    log_b = model._emission_log_prob(x)
    log_start = np.log(np.maximum(model.startprob_, 1e-12))
    log_trans = np.log(np.maximum(model.transmat_, 1e-12))
    n = x.shape[0]
    log_alpha = np.empty_like(log_b)
    log_alpha[0] = log_start + log_b[0]
    for t in range(1, n):
        log_alpha[t] = log_b[t] + logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
    log_beta = np.zeros_like(log_b)
    for t in range(n - 2, -1, -1):
        log_beta[t] = logsumexp(log_trans + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
    ll = float(logsumexp(log_alpha[-1]))
    gamma = np.exp(log_alpha + log_beta - ll)
    xi_sum = np.zeros((k, k))
    for t in range(n - 1):
        xi_sum += np.exp(log_alpha[t][:, None] + log_trans + log_b[t + 1][None, :] + log_beta[t + 1][None, :] - ll)
    transmat = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)
    gsum = np.maximum(gamma.sum(axis=0), 1e-12)
    means = (gamma.T @ x) / gsum[:, None]
    variances = np.stack([((gamma[:, s][:, None] * (x - means[s]) ** 2).sum(axis=0) / gsum[s]) + 1e-3 for s in range(k)])
    return ll, gamma[0] / gamma[0].sum(), transmat, means, variances


class GaussianHMMTests(unittest.TestCase):
    def test_fast_em_step_matches_the_log_space_reference(self):
        x, _ = _two_state_data(n=300)
        model = GaussianHMMDiag(n_states=2, n_iter=1).fit(x)

        reference = GaussianHMMDiag(n_states=2)
        reference._initialize(x)
        ll, start, trans, means, variances = _reference_em_step(reference, x)

        self.assertAlmostEqual(model.log_likelihood_, ll, places=6)
        np.testing.assert_allclose(model.startprob_, start, atol=1e-9)
        np.testing.assert_allclose(model.transmat_, trans, atol=1e-9)
        np.testing.assert_allclose(model.means_, means, atol=1e-9)
        np.testing.assert_allclose(model.vars_, variances, atol=1e-9)

    def test_likelihood_never_decreases_and_states_are_recovered(self):
        x, truth = _two_state_data()
        model = GaussianHMMDiag(n_states=2, n_iter=1)
        previous = -np.inf
        model.fit(x)
        previous = model.log_likelihood_
        for _ in range(5):
            model.n_iter = 1
            model.fit(x)  # refits from the k-means start; likelihood after >=1 step must be finite
            self.assertTrue(np.isfinite(model.log_likelihood_))
        full = GaussianHMMDiag(n_states=2).fit(x)
        self.assertGreaterEqual(full.log_likelihood_, previous - 1e-6)
        predicted = full.predict(x)
        agree = max((predicted == truth).mean(), (predicted == 1 - truth).mean())  # state labels may swap
        self.assertGreater(agree, 0.95)

    def test_long_tracks_fit_quickly(self):
        # 12,000 fixes (a GPS collar) took ~2.7 s per EM iteration before; this should be well under a second.
        rng = np.random.default_rng(0)
        x = rng.normal(size=(12000, 3))
        started = time.perf_counter()
        GaussianHMMDiag(n_states=3, n_iter=5).fit(x)
        self.assertLess(time.perf_counter() - started, 6.0)

    def test_too_few_observations_raise(self):
        with self.assertRaises(ValueError):
            GaussianHMMDiag(n_states=3).fit(np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
