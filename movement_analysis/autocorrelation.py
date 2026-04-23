import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._shared import build_track_metrics, ensure_output_dir


def _acf(values: np.ndarray, max_lag: int) -> list[float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return []
    x = x - np.mean(x)
    denom = np.dot(x, x)
    if denom <= 0:
        return []
    out = []
    for lag in range(1, min(max_lag, x.size - 1) + 1):
        out.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return out


def run_autocorrelation_analysis(df: pd.DataFrame, output_dir: str, max_lag: int = 20) -> dict:
    outdir = ensure_output_dir(output_dir, "movement_analysis")
    metrics = build_track_metrics(df)
    if metrics.empty:
        raise ValueError("No valid coordinates were available for autocorrelation analysis.")

    rows = []
    for animal_id, group in metrics.groupby("animal_id", dropna=False):
        step_acf = _acf(group["step_length_m"].to_numpy(dtype=float), max_lag=max_lag)
        net_acf = _acf(group["net_displacement_m"].to_numpy(dtype=float), max_lag=max_lag)
        for lag, value in enumerate(step_acf, start=1):
            rows.append({"animal_id": animal_id, "metric": "step_length_m", "lag": lag, "autocorrelation": value})
        for lag, value in enumerate(net_acf, start=1):
            rows.append({"animal_id": animal_id, "metric": "net_displacement_m", "lag": lag, "autocorrelation": value})
    acf_df = pd.DataFrame(rows)
    if acf_df.empty:
        raise ValueError("Not enough sequential observations were available for autocorrelation analysis.")

    csv_path = os.path.join(outdir, "autocorrelation_diagnostics.csv")
    acf_df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for animal_id, group in acf_df[acf_df["metric"] == "step_length_m"].groupby("animal_id", dropna=False):
        axes[0].plot(group["lag"], group["autocorrelation"], marker="o", label=str(animal_id))
    for animal_id, group in acf_df[acf_df["metric"] == "net_displacement_m"].groupby("animal_id", dropna=False):
        axes[1].plot(group["lag"], group["autocorrelation"], marker="o", label=str(animal_id))
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Step-Length Autocorrelation")
    axes[1].set_title("Net-Displacement Autocorrelation")
    axes[0].set_xlabel("Lag")
    axes[1].set_xlabel("Lag")
    axes[0].set_ylabel("ACF")
    axes[1].set_ylabel("ACF")
    if acf_df["animal_id"].nunique() <= 8:
        axes[0].legend()
        axes[1].legend()
    fig_path = os.path.join(outdir, "autocorrelation_plots.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    summary = (
        acf_df.groupby(["animal_id", "metric"], dropna=False)["autocorrelation"]
        .agg(["mean", "max", "min"])
        .reset_index()
        .rename(columns={"mean": "mean_acf", "max": "max_acf", "min": "min_acf"})
    )
    summary_path = os.path.join(outdir, "autocorrelation_summary.csv")
    summary.to_csv(summary_path, index=False)

    return {
        "summary": summary,
        "csv": csv_path,
        "summary_csv": summary_path,
        "plot": fig_path,
        "message": f"Autocorrelation diagnostics complete for {summary['animal_id'].nunique()} track(s).",
    }
