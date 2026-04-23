import os

import matplotlib.pyplot as plt
import pandas as pd

from ._shared import build_track_metrics, ensure_output_dir, summarize_numeric


def run_displacement_analysis(df: pd.DataFrame, output_dir: str) -> dict:
    outdir = ensure_output_dir(output_dir, "movement_analysis")
    metrics = build_track_metrics(df)
    if metrics.empty:
        raise ValueError("No valid coordinates were available for displacement analysis.")

    csv_path = os.path.join(outdir, "displacement_metrics.csv")
    metrics.to_csv(csv_path, index=False)

    summary_rows = []
    for animal_id, group in metrics.groupby("animal_id", dropna=False):
        net_stats = summarize_numeric(group["net_displacement_m"])
        path_stats = summarize_numeric(group["cumulative_distance_m"])
        summary_rows.append({
            "animal_id": animal_id,
            "n_fixes": int(len(group)),
            "max_net_displacement_m": net_stats["max"],
            "mean_net_displacement_m": net_stats["mean"],
            "final_cumulative_distance_m": float(group["cumulative_distance_m"].iloc[-1]),
            "mean_cumulative_distance_m": path_stats["mean"],
        })
    summary = pd.DataFrame(summary_rows).sort_values("animal_id")
    summary_path = os.path.join(outdir, "displacement_summary.csv")
    summary.to_csv(summary_path, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for animal_id, group in metrics.groupby("animal_id", dropna=False):
        x = group["obs_index"]
        axes[0].plot(x, group["net_displacement_m"], label=str(animal_id))
        axes[1].plot(x, group["cumulative_distance_m"], label=str(animal_id))
    axes[0].set_title("Net Displacement by Observation")
    axes[0].set_xlabel("Observation Index")
    axes[0].set_ylabel("Net Displacement (m)")
    axes[1].set_title("Cumulative Distance by Observation")
    axes[1].set_xlabel("Observation Index")
    axes[1].set_ylabel("Cumulative Distance (m)")
    if metrics["animal_id"].nunique() <= 8:
        axes[0].legend()
        axes[1].legend()
    fig_path = os.path.join(outdir, "displacement_plots.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    return {
        "summary": summary,
        "csv": csv_path,
        "summary_csv": summary_path,
        "plot": fig_path,
        "message": f"Displacement analysis complete for {summary.shape[0]} track(s).",
    }
