import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from ._shared import build_track_metrics, ensure_output_dir


class GaussianHMMDiag:
    def __init__(self, n_states: int, n_iter: int = 75, tol: float = 1e-4, random_state: int = 0):
        self.n_states = int(n_states)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.startprob_ = None
        self.transmat_ = None
        self.means_ = None
        self.vars_ = None

    def _initialize(self, x: np.ndarray):
        n, d = x.shape
        km = KMeans(n_clusters=self.n_states, n_init=10, random_state=self.random_state)
        labels = km.fit_predict(x)
        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
        self.transmat_ = np.full((self.n_states, self.n_states), 1.0 / max(self.n_states - 1, 1))
        np.fill_diagonal(self.transmat_, 0.85)
        if self.n_states > 1:
            off = 0.15 / (self.n_states - 1)
            self.transmat_[self.transmat_ != 0.85] = off
        self.means_ = np.zeros((self.n_states, d), dtype=float)
        self.vars_ = np.zeros((self.n_states, d), dtype=float)
        global_var = np.var(x, axis=0) + 1e-3
        for state in range(self.n_states):
            xs = x[labels == state]
            if xs.size == 0:
                self.means_[state] = x[np.random.default_rng(self.random_state).integers(0, n)]
                self.vars_[state] = global_var
            else:
                self.means_[state] = xs.mean(axis=0)
                self.vars_[state] = np.var(xs, axis=0) + 1e-3

    def _emission_log_prob(self, x: np.ndarray) -> np.ndarray:
        d = x.shape[1]
        out = np.empty((x.shape[0], self.n_states), dtype=float)
        for state in range(self.n_states):
            var = np.maximum(self.vars_[state], 1e-6)
            diff = x - self.means_[state]
            out[:, state] = -0.5 * (
                np.sum(np.log(2.0 * np.pi * var))
                + np.sum((diff * diff) / var, axis=1)
            )
        return out

    def fit(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[0] < self.n_states + 2:
            raise ValueError("Not enough observations to fit the hidden Markov model.")
        self._initialize(x)

        prev_ll = None
        for _ in range(self.n_iter):
            log_b = self._emission_log_prob(x)
            log_start = np.log(np.maximum(self.startprob_, 1e-12))
            log_trans = np.log(np.maximum(self.transmat_, 1e-12))

            log_alpha = np.empty_like(log_b)
            log_alpha[0] = log_start + log_b[0]
            for t in range(1, x.shape[0]):
                log_alpha[t] = log_b[t] + logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)

            log_beta = np.zeros_like(log_b)
            for t in range(x.shape[0] - 2, -1, -1):
                log_beta[t] = logsumexp(log_trans + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

            log_likelihood = float(logsumexp(log_alpha[-1]))
            log_gamma = log_alpha + log_beta - log_likelihood
            gamma = np.exp(log_gamma)

            xi_sum = np.zeros((self.n_states, self.n_states), dtype=float)
            for t in range(x.shape[0] - 1):
                log_xi_t = (
                    log_alpha[t][:, None]
                    + log_trans
                    + log_b[t + 1][None, :]
                    + log_beta[t + 1][None, :]
                    - log_likelihood
                )
                xi_sum += np.exp(log_xi_t)

            self.startprob_ = gamma[0] / np.sum(gamma[0])
            self.transmat_ = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)

            gamma_sum = np.maximum(gamma.sum(axis=0), 1e-12)
            self.means_ = (gamma.T @ x) / gamma_sum[:, None]
            for state in range(self.n_states):
                diff = x - self.means_[state]
                self.vars_[state] = ((gamma[:, state][:, None] * diff * diff).sum(axis=0) / gamma_sum[state]) + 1e-3

            if prev_ll is not None and abs(log_likelihood - prev_ll) < self.tol:
                break
            prev_ll = log_likelihood
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        log_b = self._emission_log_prob(x)
        log_start = np.log(np.maximum(self.startprob_, 1e-12))
        log_trans = np.log(np.maximum(self.transmat_, 1e-12))
        delta = np.empty_like(log_b)
        psi = np.zeros((x.shape[0], self.n_states), dtype=int)
        delta[0] = log_start + log_b[0]
        for t in range(1, x.shape[0]):
            score = delta[t - 1][:, None] + log_trans
            psi[t] = np.argmax(score, axis=0)
            delta[t] = np.max(score, axis=0) + log_b[t]
        states = np.zeros(x.shape[0], dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(x.shape[0] - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states


def run_hmm_state_analysis(df: pd.DataFrame, output_dir: str, n_states: int = 3) -> dict:
    outdir = ensure_output_dir(output_dir, "movement_analysis")
    metrics = build_track_metrics(df)
    steps = metrics.dropna(subset=["step_length_m", "turning_angle_rad"]).copy()
    if steps.empty:
        raise ValueError("At least three fixes per track are required for HMM state analysis.")

    state_rows = []
    summary_rows = []
    for animal_id, group in steps.groupby("animal_id", dropna=False):
        if len(group) < max(6, n_states + 2):
            continue
        used_states = min(n_states, max(2, len(group) // 4))
        features = np.column_stack([
            np.log1p(group["step_length_m"].to_numpy(dtype=float)),
            np.cos(group["turning_angle_rad"].to_numpy(dtype=float)),
            np.sin(group["turning_angle_rad"].to_numpy(dtype=float)),
        ])
        model = GaussianHMMDiag(n_states=used_states)
        model.fit(features)
        states = model.predict(features)
        labeled = group.copy()
        labeled["behavior_state"] = states + 1
        state_rows.append(labeled)

        for state_id, state_group in labeled.groupby("behavior_state"):
            summary_rows.append({
                "animal_id": animal_id,
                "state": int(state_id),
                "n_steps": int(len(state_group)),
                "mean_step_length_m": float(state_group["step_length_m"].mean()),
                "mean_abs_turning_angle_deg": float(state_group["turning_angle_deg"].abs().mean()),
            })

    if not state_rows:
        raise ValueError("Not enough sequential fixes were available to fit the hidden Markov model.")

    states_df = pd.concat(state_rows, ignore_index=True)
    states_csv = os.path.join(outdir, "hmm_behavior_states.csv")
    states_df.to_csv(states_csv, index=False)

    summary = pd.DataFrame(summary_rows).sort_values(["animal_id", "state"])
    summary_csv = os.path.join(outdir, "hmm_behavior_state_summary.csv")
    summary.to_csv(summary_csv, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for animal_id, group in states_df.groupby("animal_id", dropna=False):
        axes[0].scatter(group["obs_index"], group["behavior_state"], s=20, label=str(animal_id))
        axes[1].plot(group["obs_index"], group["step_length_m"], alpha=0.7, label=str(animal_id))
    axes[0].set_title("Hidden Markov Behavioral States")
    axes[0].set_xlabel("Observation Index")
    axes[0].set_ylabel("State")
    axes[1].set_title("Step Lengths Used by the HMM")
    axes[1].set_xlabel("Observation Index")
    axes[1].set_ylabel("Step Length (m)")
    if states_df["animal_id"].nunique() <= 8:
        axes[0].legend()
        axes[1].legend()
    fig_path = os.path.join(outdir, "hmm_behavior_states.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    return {
        "summary": summary,
        "csv": states_csv,
        "summary_csv": summary_csv,
        "plot": fig_path,
        "message": f"Hidden Markov state analysis complete for {summary['animal_id'].nunique()} track(s).",
    }
