import os

import matplotlib.pyplot as plt
import pandas as pd

from ._shared import build_track_metrics, ensure_output_dir, summarize_numeric


def run_step_length_analysis(df: pd.DataFrame, output_dir: str) -> dict:
    outdir = ensure_output_dir(output_dir, "movement_analysis")
    metrics = build_track_metrics(df)
    steps = metrics.dropna(subset=["step_length_m"]).copy()
    if steps.empty:
        raise ValueError("At least two fixes per track are required for step-length analysis.")

    csv_path = os.path.join(outdir, "step_lengths.csv")
    steps.to_csv(csv_path, index=False)

    summary_rows = []
    for animal_id, group in steps.groupby("animal_id", dropna=False):
        stats = summarize_numeric(group["step_length_m"])
        summary_rows.append({
            "animal_id": animal_id,
            "n_steps": stats["n"],
            "mean_step_length_m": stats["mean"],
            "median_step_length_m": stats["median"],
            "sd_step_length_m": stats["sd"],
            "max_step_length_m": stats["max"],
        })
    summary = pd.DataFrame(summary_rows).sort_values("animal_id")
    summary_path = os.path.join(outdir, "step_length_summary.csv")
    summary.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for animal_id, group in steps.groupby("animal_id", dropna=False):
        axes[0].plot(group["obs_index"], group["step_length_m"], label=str(animal_id))
        axes[1].hist(group["step_length_m"], bins=20, alpha=0.35, label=str(animal_id))
    axes[0].set_title("Step Length by Observation")
    axes[0].set_xlabel("Observation Index")
    axes[0].set_ylabel("Step Length (m)")
    axes[1].set_title("Step Length Distribution")
    axes[1].set_xlabel("Step Length (m)")
    axes[1].set_ylabel("Frequency")
    if steps["animal_id"].nunique() <= 8:
        axes[0].legend()
        axes[1].legend()
    fig_path = os.path.join(outdir, "step_length_plots.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    return {
        "summary": summary,
        "csv": csv_path,
        "summary_csv": summary_path,
        "plot": fig_path,
        "message": f"Step-length analysis complete for {summary.shape[0]} track(s).",
    }
