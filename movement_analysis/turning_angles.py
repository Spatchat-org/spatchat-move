import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._shared import build_track_metrics, ensure_output_dir, summarize_numeric


def run_turning_angle_analysis(df: pd.DataFrame, output_dir: str) -> dict:
    outdir = ensure_output_dir(output_dir, "movement_analysis")
    metrics = build_track_metrics(df)
    turns = metrics.dropna(subset=["turning_angle_deg"]).copy()
    if turns.empty:
        raise ValueError("At least three fixes per track are required for turning-angle analysis.")

    csv_path = os.path.join(outdir, "turning_angles.csv")
    turns.to_csv(csv_path, index=False)

    summary_rows = []
    for animal_id, group in turns.groupby("animal_id", dropna=False):
        stats = summarize_numeric(group["turning_angle_deg"].abs())
        summary_rows.append({
            "animal_id": animal_id,
            "n_turns": stats["n"],
            "mean_abs_turning_angle_deg": stats["mean"],
            "median_abs_turning_angle_deg": stats["median"],
            "sd_abs_turning_angle_deg": stats["sd"],
            "max_abs_turning_angle_deg": stats["max"],
        })
    summary = pd.DataFrame(summary_rows).sort_values("animal_id")
    summary_path = os.path.join(outdir, "turning_angle_summary.csv")
    summary.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    bins = np.linspace(-180, 180, 37)
    for animal_id, group in turns.groupby("animal_id", dropna=False):
        axes[0].plot(group["obs_index"], group["turning_angle_deg"], label=str(animal_id))
        axes[1].hist(group["turning_angle_deg"], bins=bins, alpha=0.35, label=str(animal_id))
    axes[0].set_title("Turning Angle by Observation")
    axes[0].set_xlabel("Observation Index")
    axes[0].set_ylabel("Turning Angle (deg)")
    axes[1].set_title("Turning Angle Distribution")
    axes[1].set_xlabel("Turning Angle (deg)")
    axes[1].set_ylabel("Frequency")
    if turns["animal_id"].nunique() <= 8:
        axes[0].legend()
        axes[1].legend()
    fig_path = os.path.join(outdir, "turning_angle_plots.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    return {
        "summary": summary,
        "csv": csv_path,
        "summary_csv": summary_path,
        "plot": fig_path,
        "message": f"Turning-angle analysis complete for {summary.shape[0]} track(s).",
    }
